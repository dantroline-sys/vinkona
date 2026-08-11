"""
Qwen3-TTS — the "qwen3" engine (wraps the official `qwen-tts` package).

Unlike Orpheus (hand-rolled: a llama.cpp token server + our own SNAC decode),
Qwen3-TTS ships the LM and its 12.5 Hz / 16-codebook codec together in one
package, so this engine follows the neutts / chatterbox pattern instead: load the
model in its own venv (qwen3_env: torch + qwen-tts), call its high-level API, get
a waveform back.  Natural-language voice control: each voice NAME resolves to a
style DESCRIPTION passed to the model as `instruct` — the thing the -VoiceDesign
checkpoint is trained for.

Reference usage (Qwen3-TTS official README / model card):

    import torch
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    wavs, sr = model.generate_voice_design(
        text="…", language="English",
        instruct="A warm, deep storyteller voice, slightly fast-paced, high energy")

NOTE on the checkpoint: natural-language voice design needs the *-VoiceDesign*
model (default here), NOT *-Base*.  NOTE on streaming: the package exposes no
public token-level streaming generator yet, so synthesize_stream synthesizes then
chunks — and since the cascade already splits speech into sentence-sized calls,
first-audio latency is per-sentence.  Swap in the real streaming API here if/when
the package ships one (that's the paper's ~97 ms TTFA path).

Written against the documented API but NOT run here (the sandbox has no GPU /
torch / qwen-tts) — it needs a live smoke test on the box after
`pip install qwen-tts` in qwen3_env.
"""

import os
import time
import typing as tp

import numpy as np

DEFAULT_SAMPLE_RATE = 24000     # updated from the model's returned sr at first synth


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [tts-qwen3] {msg}", flush=True)


def _resolve_device(device: str, torch, tries: int = 6, wait: float = 2.0) -> str:
    """--device 'auto' → the best available device_map string; an explicit value
    (e.g. 'cuda:0', 'cpu') passes through unchanged.

    On 'auto', RETRY the CUDA probe before conceding to CPU.  torch.cuda.is_available()
    can transiently return False when the target GPU is saturated by another
    process's kernels at the instant we check — seen on the shared 4090 when the
    fast LM's first-token burst overlaps TTS startup.  For this live engine CPU
    means ~1 min/sentence and cascade timeouts (silence), so a few seconds of
    re-probing is cheap insurance.  Only retry when torch was built with CUDA — a
    CPU-only build is instantly and permanently unavailable, so don't stall it."""
    if device and device != "auto":
        return device
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    cuda_built = bool(getattr(getattr(torch, "version", None), "cuda", None))
    attempts = tries if cuda_built else 1
    for i in range(attempts):
        if torch.cuda.is_available():
            return "cuda"
        if mps is not None and mps.is_available():
            return "mps"
        if i < attempts - 1:
            if i == 0:
                _log(f"CUDA not ready (GPU busy?) — re-probing for up to "
                     f"{int(tries * wait)}s before falling back to CPU ...")
            time.sleep(wait)
    return "cpu"


class Qwen3TTSEngine:
    """Qwen3-TTS via the official package.  Same contract as the other engines
    (sample_rate / voices / synthesize / synthesize_stream)."""

    def __init__(
        self,
        model_repo: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: tp.Optional[str] = None,
        language: str = "English",
        default_voice: str = "vinkona",
        voices: tp.Optional[dict] = None,
        chunk_ms: int = 200,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        gen_kwargs: tp.Optional[dict] = None,
    ):
        import torch                                  # lazy: only this engine's venv has it
        from qwen_tts import Qwen3TTSModel

        self.language = language
        # name → natural-language style description (the `instruct`)
        self._styles = dict(voices or {})
        # A default_voice that isn't a configured style (e.g. the Orpheus default
        # "tara") would be sent as a literal instruct — fall back to a real style.
        self.default_voice = (default_voice if default_voice in self._styles
                              else next(iter(self._styles), default_voice))
        self._gen_kwargs = dict(gen_kwargs or {})
        self.sample_rate = int(sample_rate)
        self._chunk_ms = int(chunk_ms)

        dev = _resolve_device(device, torch)
        # This is a live voice engine — CPU means ~a minute per sentence, so the
        # cascade's 60 s TTS timeout drops it and you get SILENCE, not slow speech.
        # Never let that happen quietly: if we asked for auto/cuda and still landed
        # on CPU, say so loudly (with what CUDA_VISIBLE_DEVICES the process sees) so
        # a masked-away GPU is obvious in the log instead of a mysterious no-audio.
        if str(dev).startswith("cpu") and not str(device).startswith("cpu"):
            _log(f"WARNING: no CUDA visible — loading on CPU (expect ~1 min/sentence, "
                 f"the cascade will time out → silence). "
                 f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')!r}, "
                 f"cuda.is_available={torch.cuda.is_available()}. "
                 f"Set tts.qwen3.device to a real GPU (e.g. 'cuda:0') or fix "
                 f"CUDA_VISIBLE_DEVICES in serve_tts.sh.")
        td = getattr(torch, dtype, None) or torch.float32
        kw = {"device_map": dev, "dtype": td}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        _log(f"loading {model_repo} on {dev} ({dtype}) ...")
        t0 = time.monotonic()
        self._model = Qwen3TTSModel.from_pretrained(model_repo, **kw)
        if not hasattr(self._model, "generate_voice_design"):
            raise RuntimeError(
                "this qwen-tts build has no generate_voice_design(); check the "
                "package version / model repo (need a -VoiceDesign checkpoint)")
        _log(f"ready in {time.monotonic()-t0:.1f}s  voices={self.voices}")

    @property
    def voices(self) -> list:
        return list(self._styles.keys()) or [self.default_voice]

    def resolve_style(self, voice: tp.Optional[str]) -> str:
        """A voice name → its style description; an unknown name is used as a
        literal description, so ad-hoc styles ('a brisk newsreader') work too."""
        voice = voice or self.default_voice
        return self._styles.get(voice, voice)

    # ── engine contract ───────────────────────────────────────────────────────

    def synthesize(self, text: str, voice: tp.Optional[str] = None) -> np.ndarray:
        """One utterance → float32 PCM.  BLOCKING — worker thread."""
        instruct = self.resolve_style(voice)
        wavs, sr = self._model.generate_voice_design(
            text=text, language=self.language, instruct=instruct, **self._gen_kwargs)
        self.sample_rate = int(sr)
        pcm = wavs[0] if len(wavs) else wavs
        if hasattr(pcm, "detach"):                    # a torch tensor
            pcm = pcm.detach().to("cpu").float().numpy()
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(pcm)

    def synthesize_stream(self, text: str, voice: tp.Optional[str] = None):
        """Yield 16-bit PCM byte chunks — same framing as the other engines.

        Synthesize-then-chunk: the package has no public token-level streaming
        generator yet, and the cascade calls this per sentence, so latency is
        per-sentence.  Replace the body with the real streaming API when available."""
        pcm = self.synthesize(text, voice)
        if not len(pcm):
            _log(f"no audio for {len(text)} chars")
            return
        i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
        step = max(1, int(self.sample_rate * self._chunk_ms / 1000))
        for i in range(0, len(i16), step):
            yield i16[i:i + step].tobytes()
