"""
CPU-side mic front-end: RNNoise denoise + voice-activity probability.

PersonaPlex needs the user stream to fall truly silent between turns, otherwise
it stays in listen-mode forever (background noise / speaker bleed keeps the
injected user stream "active").  RNNoise is a tiny GRU denoiser built for VoIP:
negligible CPU, ~0 added latency, and it returns a speech probability per 10 ms
frame for free — so one CPU component gives us both clean audio and a reliable
VAD, leaving the GPU entirely for PersonaPlex and the fast LM.

RNNoise is hardcoded to 48 kHz / 480-sample frames.  Our pipeline is 24 kHz with
1920-sample (80 ms) chunks, which maps to exactly 3840 samples = 8 RNNoise frames
at 48 kHz — a clean integer ratio with no partial-frame buffering.  We resample
per-chunk (stateless, length-exact at the 2x integer ratio) and keep one
persistent RNNoise state across all frames so the GRU's temporal context carries.

Usage:
    fe = RNNoiseFrontend(in_rate=24000)        # raises if librnnoise/soxr missing
    clean, speech_prob = fe.process(pcm_f32)   # pcm_f32 in [-1, 1], length 1920
"""

import ctypes
import numpy as np

try:
    import soxr
except ImportError as e:  # pragma: no cover - environment dependent
    soxr = None
    _SOXR_IMPORT_ERROR = e
else:
    _SOXR_IMPORT_ERROR = None

# RNNoise operates on fixed 480-sample frames at 48 kHz (10 ms).
_FRAME_48K = 480
_RNNOISE_RATE = 48000
# RNNoise expects audio scaled to the int16 range, not [-1, 1].
_INT16_SCALE = 32768.0


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or trim so x has exactly n samples (guards ±1 resampler drift)."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


class RNNoiseFrontend:
    def __init__(self, in_rate: int = 24000, lib_path: str = "librnnoise.so"):
        if soxr is None:
            raise RuntimeError(
                f"soxr is required for resampling but failed to import: {_SOXR_IMPORT_ERROR}. "
                "Install with: pip install soxr"
            )
        self.in_rate = in_rate
        self._up = _RNNOISE_RATE // in_rate  # 2 for 24 kHz

        self._lib = ctypes.CDLL(lib_path)
        # DenoiseState *rnnoise_create(RNNModel *model);  (model = NULL for default)
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        # float rnnoise_process_frame(DenoiseState*, float *out, const float *in);
        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

        self._st = self._lib.rnnoise_create(None)
        if not self._st:
            raise RuntimeError("rnnoise_create returned NULL")

        self._out = np.empty(_FRAME_48K, dtype=np.float32)
        self._fptr = ctypes.POINTER(ctypes.c_float)

    def process(self, pcm: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Denoise one chunk and return (clean_pcm, speech_prob).

        pcm:  float32 in [-1, 1] at self.in_rate (typically 1920 samples = 80 ms).
        returns clean audio at the same rate and length, plus the mean RNNoise
        speech probability over the chunk (0..1).
        """
        n = len(pcm)
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)

        # 24 kHz -> 48 kHz.  Integer 2x ratio keeps the length exact (1920 -> 3840).
        up = soxr.resample(pcm, self.in_rate, _RNNOISE_RATE).astype(np.float32)
        up = _fit(up, n * self._up)

        n_frames = len(up) // _FRAME_48K
        probs = np.empty(n_frames, dtype=np.float32)
        out48 = np.empty(n_frames * _FRAME_48K, dtype=np.float32)

        out_ptr = self._out.ctypes.data_as(self._fptr)
        for i in range(n_frames):
            frame = np.ascontiguousarray(up[i * _FRAME_48K:(i + 1) * _FRAME_48K] * _INT16_SCALE)
            in_ptr = frame.ctypes.data_as(self._fptr)
            probs[i] = self._lib.rnnoise_process_frame(self._st, out_ptr, in_ptr)
            out48[i * _FRAME_48K:(i + 1) * _FRAME_48K] = self._out

        out48 /= _INT16_SCALE
        # 48 kHz -> 24 kHz.
        clean = soxr.resample(out48, _RNNOISE_RATE, self.in_rate).astype(np.float32)
        clean = _fit(clean, n)
        return clean, float(probs.mean()) if n_frames else 0.0

    def close(self) -> None:
        if getattr(self, "_st", None):
            self._lib.rnnoise_destroy(self._st)
            self._st = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
