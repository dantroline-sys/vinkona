"""
NeuTTS Air text-to-speech engine — the controllable "mouth" for the cascade.

Unlike PersonaPlex (a conversational model that fights forced tokens), NeuTTS is a
true TTS: it says exactly the text it's given, with human-like prosody, and clones
a voice from a ~3 s reference clip.  On a 4090 it runs ~320x real-time, so a full
sentence synthesizes in ~10 ms — negligible latency, tiny VRAM, GPU mostly free
for the LLM.  Output is 24 kHz, matching the client PCM pipeline exactly.

A "voice" is a reference WAV + its transcript, encoded once at registration.  This
lines up with the persona system: each persona can carry its own cloned voice.

Usage:
    eng = NeuTTSEngine(device="cuda")          # raises if neutts isn't installed
    eng.register_voice("vinkona", "voices/vinkona.wav")   # reads voices/vinkona.txt
    pcm = eng.synthesize("Hello there.", voice="vinkona")  # float32 @ 24 kHz

synthesize() is blocking (GPU/CPU inference); call it from a worker thread, never
directly on the asyncio event loop.
"""

from pathlib import Path
import typing as tp

import numpy as np

SAMPLE_RATE = 24000


class NeuTTSEngine:
    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air",
        codec_repo: str = "neuphonic/neucodec",
        device: str = "cuda",
        codec_device: tp.Optional[str] = None,
    ):
        from neutts import NeuTTS  # raises ImportError if not installed
        self._tts = NeuTTS(
            backbone_repo=backbone_repo,
            backbone_device=device,
            codec_repo=codec_repo,
            codec_device=codec_device or device,
        )
        self.sample_rate = SAMPLE_RATE
        # name -> (ref_codes, ref_text)
        self._voices: dict[str, tuple[tp.Any, str]] = {}

    def register_voice(self, name: str, ref_wav: str, ref_text: tp.Optional[str] = None) -> None:
        """
        Clone a voice from a reference clip.  ref_text is the transcript of the
        clip; if omitted we read a sibling .txt (e.g. vinkona.wav -> vinkona.txt).
        Encoding happens once here, not per utterance.
        """
        if ref_text is None:
            txt_path = Path(ref_wav).with_suffix(".txt")
            if not txt_path.exists():
                raise FileNotFoundError(
                    f"No transcript for {ref_wav}: pass ref_text or create {txt_path}"
                )
            ref_text = txt_path.read_text().strip()
        ref_codes = self._tts.encode_reference(ref_wav)
        self._voices[name] = (ref_codes, ref_text)

    @property
    def voices(self) -> list[str]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """
        Synthesize one utterance.  Returns float32 PCM at self.sample_rate.
        BLOCKING — run in a worker thread, not on the event loop.
        """
        if voice not in self._voices:
            raise KeyError(f"unknown voice '{voice}'; registered: {self.voices}")
        ref_codes, ref_text = self._voices[voice]
        wav = self._tts.infer(text, ref_codes, ref_text)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def synthesize_stream(self, text: str, voice: str):
        """Yield 16-bit PCM byte chunks.  NeuTTS synthesizes a whole utterance at
        once, so this is a single chunk — same wire format as the Orpheus stream."""
        pcm = self.synthesize(text, voice)
        yield (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
