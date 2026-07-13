"""
Chatterbox TTS engine (Resemble AI, MIT) — the low-footprint voice for
machines that can't hold the Orpheus 3B backbone at real time (a 16 GB M2
mini, a CPU-only box): a ~0.5B token model + flow-matching vocoder, ~2-3 GB
loaded, runs on cuda / mps / cpu.  Zero-shot voice cloning from a short
reference clip, an emotion-exaggeration knob, and a built-in default voice so
it works with no reference at all.  Output is 24 kHz, matching the client PCM
pipeline exactly.  (The library watermarks its audio — Perth; inaudible.)

A "voice" here is a reference WAV registered under a name; its conditionals
are computed once per voice *switch*, not per utterance, so sentence-by-
sentence synthesis doesn't re-encode the reference every call.

Usage:
    eng = ChatterboxEngine()                   # device="auto" -> cuda > mps > cpu
    eng.register_voice("vinkona", "voices/vinkona.wav")   # or ref_wav=None: built-in voice
    pcm = eng.synthesize("Hello there.", voice="vinkona")  # float32 @ 24 kHz

synthesize() is blocking (GPU/CPU inference); call it from a worker thread,
never directly on the asyncio event loop.
"""

from pathlib import Path
import typing as tp

import numpy as np

SAMPLE_RATE = 24000


def _resolve_device(device: str) -> str:
    """'auto' -> the best available torch backend: cuda > mps (Apple) > cpu.
    Explicit values pass through untouched."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class ChatterboxEngine:
    def __init__(
        self,
        device: str = "auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ):
        from chatterbox.tts import ChatterboxTTS  # raises ImportError if not installed
        device = _resolve_device(device)
        try:
            self._model = ChatterboxTTS.from_pretrained(device=device)
        except Exception:
            if device == "mps":
                # Some chatterbox/torch pairings trip over mps map_location at
                # load time; CPU still works (slower) rather than not at all.
                print("[tts-chatterbox] mps load failed — falling back to cpu", flush=True)
                device = "cpu"
                self._model = ChatterboxTTS.from_pretrained(device=device)
            else:
                raise
        self.device = device
        self.sample_rate = int(getattr(self._model, "sr", SAMPLE_RATE))
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        self._temperature = temperature
        # The built-in voice ships with the model; keep its conditionals so a
        # ref-less voice still works after a cloned one has been prepared.
        self._builtin_conds = getattr(self._model, "conds", None)
        self._voices: dict[str, tp.Optional[str]] = {}   # name -> ref wav path (None = built-in)
        self._prepared: tp.Optional[str] = None          # whose conditionals are loaded

    def register_voice(self, name: str, ref_wav: tp.Optional[str] = None) -> None:
        """A cloned voice from a ~7-20 s reference clip, or the built-in voice
        when ref_wav is None.  Conditional encoding is deferred to first use."""
        if ref_wav is not None and not Path(ref_wav).exists():
            raise FileNotFoundError(f"reference clip not found: {ref_wav}")
        self._voices[name] = ref_wav

    @property
    def voices(self) -> list[str]:
        return list(self._voices)

    def _prepare(self, voice: str) -> None:
        if self._prepared == voice:
            return
        ref = self._voices[voice]
        if ref is None:
            if self._builtin_conds is None:
                raise RuntimeError("this chatterbox build has no built-in voice — "
                                   "set tts.chatterbox.ref_wav to a reference clip")
            self._model.conds = self._builtin_conds
        else:
            self._model.prepare_conditionals(ref, exaggeration=self._exaggeration)
        self._prepared = voice

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """
        Synthesize one utterance.  Returns float32 PCM at self.sample_rate.
        BLOCKING — run in a worker thread, not on the event loop.
        """
        if voice not in self._voices:
            raise KeyError(f"unknown voice '{voice}'; registered: {self.voices}")
        self._prepare(voice)
        wav = self._model.generate(
            text,
            exaggeration=self._exaggeration,
            cfg_weight=self._cfg_weight,
            temperature=self._temperature,
        )
        pcm = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(pcm)

    def synthesize_stream(self, text: str, voice: str):
        """Yield 16-bit PCM byte chunks.  Chatterbox synthesizes a whole
        utterance at once, so this is a single chunk — same wire format as the
        Orpheus stream."""
        pcm = self.synthesize(text, voice)
        yield (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
