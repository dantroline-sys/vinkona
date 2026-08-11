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
        seed: tp.Optional[int] = None,
        mode: str = "design",
        ref_audio: tp.Optional[str] = None,
        ref_text: tp.Optional[str] = None,
        x_vector_only: bool = True,
        speaker: tp.Optional[str] = None,
    ):
        import torch                                  # lazy: only this engine's venv has it
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        # VoiceDesign *designs a voice afresh from `instruct` on every call*, so with
        # sampling on the timbre drifts sentence-to-sentence.  A fixed RNG seed makes
        # that design reproducible → one stable voice, while keeping sampling's
        # naturalness.  null = don't seed (let it vary).
        self._seed = seed
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

        # Voice MODE (see synthesize).  "design" varies the timbre per call; "clone"
        # and "custom" are FIXED voices (consistent across turns).  Prepare the
        # fixed-voice state here so it's paid once; fall back to "design" (which any
        # -VoiceDesign checkpoint supports) if the chosen mode can't be set up.
        self._mode = (mode or "design").lower()
        self._speaker = speaker
        self._clone_prompt = None
        if self._mode == "clone":
            path = self._resolve_ref(ref_audio)
            if not path:
                _log(f"WARNING: mode=clone but ref_audio {ref_audio!r} not found — "
                     f"using design mode")
                self._mode = "design"
            else:
                try:
                    self._clone_prompt = self._model.create_voice_clone_prompt(
                        ref_audio=path, ref_text=ref_text,
                        x_vector_only_mode=bool(x_vector_only))
                    _log(f"voice-clone prompt built from {path} "
                         f"(x_vector_only={bool(x_vector_only)})")
                except Exception as e:
                    _log(f"WARNING: create_voice_clone_prompt failed "
                         f"({type(e).__name__}: {e}) — using design mode")
                    self._mode = "design"
        elif self._mode == "custom":
            supported = None
            try:
                supported = self._model.get_supported_speakers()
            except Exception:
                pass
            if not self._speaker or (supported and self._speaker not in supported):
                _log(f"WARNING: mode=custom needs a valid speaker (have {self._speaker!r}; "
                     f"supported={supported}) — using design mode")
                self._mode = "design"
        _log(f"ready in {time.monotonic()-t0:.1f}s  mode={self._mode}  voices={self.voices}")

    def _resolve_ref(self, ref: tp.Optional[str]) -> tp.Optional[str]:
        """A ref-audio path as given, or resolved against this file's dir; None if
        it doesn't exist (serve_tts runs from the assistant dir, but be robust)."""
        if not ref:
            return None
        import os
        from pathlib import Path
        for cand in (ref, str(Path(__file__).resolve().parent / ref)):
            if os.path.exists(cand):
                return cand
        return None

    @property
    def voices(self) -> list:
        return list(self._styles.keys()) or [self.default_voice]

    def resolve_style(self, voice: tp.Optional[str]) -> str:
        """A voice NAME → its style description (the `instruct`).

        - a configured name → its description;
        - a MULTI-WORD string → used as an ad-hoc natural-language style, so
          "a brisk newsreader" still works;
        - any other BARE name (e.g. an Orpheus voice like "tara" arriving from a
          persona/config that predates qwen3) → the default style.

        That last case is load-bearing: a bare name is NOT a valid voice-design
        instruct — VoiceDesign would invent a DIFFERENT random voice for it every
        sentence (the "five different people" bug).  Falling back to the default
        style keeps the persona voice stable across turns."""
        voice = voice or self.default_voice
        if voice in self._styles:
            return self._styles[voice]
        if " " in voice.strip():                     # a description, not a name
            return voice
        return self._styles.get(self.default_voice) or voice

    # ── engine contract ───────────────────────────────────────────────────────

    def _generate(self, text: str, voice: tp.Optional[str]):
        """Dispatch to the package call for the active mode → (wavs, sr)."""
        if self._seed is not None:
            # Seed right before generation so a sampled voice is reproducible
            # (stable persona voice across turns).
            self._torch.manual_seed(int(self._seed))
        if self._mode == "clone":
            return self._model.generate_voice_clone(
                text=text, language=self.language,
                voice_clone_prompt=self._clone_prompt, **self._gen_kwargs)
        if self._mode == "custom":
            return self._model.generate_custom_voice(
                text=text, speaker=self._speaker, language=self.language,
                **self._gen_kwargs)
        return self._model.generate_voice_design(          # design (default)
            text=text, language=self.language,
            instruct=self.resolve_style(voice), **self._gen_kwargs)

    def synthesize(self, text: str, voice: tp.Optional[str] = None) -> np.ndarray:
        """One utterance → float32 PCM.  BLOCKING — worker thread."""
        wavs, sr = self._generate(text, voice)
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
