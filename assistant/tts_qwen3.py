"""
Qwen3-TTS — the "qwen3" engine (SCAFFOLDED — see the "STILL NEEDED" note below).

Qwen3-TTS is architecturally the SAME family as our Orpheus engine: a discrete,
multi-codebook language model emits audio-codec tokens, and a neural codec (the
"Qwen3-TTS-Tokenizer-12Hz") turns those tokens back into a waveform.  So it slots
into the exact seam Orpheus uses — a token-streaming HTTP server plus a codec
decode — and reuses the same synthesize()/synthesize_stream() contract and the
same dual-track streaming idea (emit audio as tokens arrive → low TTFA).

WHAT'S REAL HERE (written + unit-tested, no model needed):
  • the streaming HTTP client to a token server (SSE, mirrors Orpheus's _sse);
  • the engine contract (sample_rate / voices / synthesize / synthesize_stream)
    and the PCM framing that streams chunks to the cascade as they decode;
  • Qwen3's natural-language voice control: a voice NAME resolves to a style
    DESCRIPTION ("a warm, slow storyteller with a faint British accent") so the
    cascade keeps addressing voices by name while the model reads the description.

STILL NEEDED to make it actually speak (from the real model card — the two facts
I could not verify and must not fabricate; each is isolated behind one function):
  1. _format_input(text, style): the exact prompt/input string Qwen3-TTS expects
     (Orpheus's analogue is format_prompt()).  ← fill in `_format_input`.
  2. the codec: how a streamed token id maps to codebook frames, and how the 12Hz
     codec decodes frames → PCM (Orpheus's analogues are audio_code /
     stream_to_windows / deinterleave / SnacDecoder).  ← fill in `Qwen3Codec`.

Until those are filled in, selecting engine="qwen3" constructs fine and answers
/health and /voices, but a synthesis call raises a clear, actionable error naming
exactly what's missing — never a silent failure or a wrong noise.

Runs in vinkona_env (numpy + urllib); the codec adds whatever runtime the real
detokenizer needs (likely onnxruntime or torch — TBD from the card).
"""

import json
import time
import typing as tp
import urllib.error
import urllib.request

import numpy as np

DEFAULT_SAMPLE_RATE = 24000     # confirm against the model card


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [tts-qwen3] {msg}", flush=True)


class _NeedsModelCard(NotImplementedError):
    """Raised by the two model-specific stubs so the failure is unambiguous: the
    engine is wired, but this one detail must come from the Qwen3-TTS model card."""


# ── (1) input format — STUB: needs the real prompt spec ───────────────────────

def _format_input(text: str, style: str) -> str:
    """The input string Qwen3-TTS expects, given the text and a natural-language
    voice/style description.  Orpheus's analogue is format_prompt("...","tara");
    Qwen3 uses semantic descriptions instead of preset voice tags, so `style` is a
    sentence, not a name.

    STILL NEEDED: the exact scaffolding tokens / chat template from the model card
    (e.g. does it take a system-style description + text, a single templated
    string, special BOS/EOS markers?).  Fill this in and delete the raise."""
    raise _NeedsModelCard(
        "Qwen3-TTS input format not set yet — provide the model card's prompt/chat "
        "template and complete _format_input() in tts_qwen3.py "
        f"(would format: style={style!r}, text={text[:40]!r}…)")


# ── (2) codec — STUB: needs the 12Hz detokenizer spec ─────────────────────────

class Qwen3Codec:
    """The Qwen3-TTS 12Hz codec/detokenizer: streamed token ids → PCM.

    STILL NEEDED from the model card:
      • the audio-token vocabulary offset + the codebook layout (Qwen3 is
        multi-codebook; Orpheus is 7 codes/frame in a 1+2+4 SNAC layout — Qwen3's
        will differ), i.e. how to turn a run of token ids into decode frames;
      • the detokenizer artifact + runtime (ONNX? torch?) and its output sample
        rate, to turn frames → waveform.
    Fill in `frames_from_tokens` and `decode_frames`, then set `.ready = True`."""

    def __init__(self, codec_path: tp.Optional[str], codec_repo: str,
                 codec_file: tp.Optional[str], sample_rate: int):
        self.sample_rate = int(sample_rate)
        self._path, self._repo, self._file = codec_path, codec_repo, codec_file
        self.ready = False            # flip to True once the two methods are real
        # (Deliberately does NOT load anything yet — no artifact spec to load.)

    def frames_from_tokens(self, token_ids: tp.Iterable[int]) -> tp.Iterator[list]:
        """Group streamed token ids into decode frames at the codec's cadence
        (the analogue of Orpheus's stream_to_windows + deinterleave)."""
        raise _NeedsModelCard(
            "Qwen3-TTS token→frame mapping not set yet — provide the codebook "
            "layout from the model card and complete Qwen3Codec.frames_from_tokens().")

    def decode_frames(self, frame: list) -> bytes:
        """One frame → 16-bit PCM bytes (the analogue of SnacDecoder.decode_window)."""
        raise _NeedsModelCard(
            "Qwen3-TTS codec decode not set yet — provide the 12Hz detokenizer "
            "artifact + runtime from the model card and complete "
            "Qwen3Codec.decode_frames().")


class Qwen3TTSEngine:
    """Token-streaming TTS over a Qwen3-TTS server + the 12Hz codec.  Same contract
    as the Orpheus/NeuTTS/Chatterbox engines."""

    def __init__(
        self,
        lm_url: str,
        default_voice: str = "vinkona",
        voices: tp.Optional[dict] = None,
        codec_path: tp.Optional[str] = None,
        codec_repo: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        codec_file: tp.Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 3500,
        request_timeout_s: int = 300,
        wait_for_lm_s: int = 180,
    ):
        self.lm_url = lm_url.rstrip("/")
        self.sample_rate = int(sample_rate)
        self.default_voice = default_voice
        # name → natural-language style description (Qwen3's voice control)
        self._styles = dict(voices or {})
        self._timeout = request_timeout_s
        self._sampling = {"temperature": float(temperature), "top_p": float(top_p)}
        self._max_tokens = int(max_tokens)
        self._codec = Qwen3Codec(codec_path, codec_repo, codec_file, sample_rate)
        self._wait_for_lm(wait_for_lm_s)

    @property
    def voices(self) -> list:
        """The configured voice NAMES (each maps to a style description)."""
        return list(self._styles.keys()) or [self.default_voice]

    def resolve_style(self, voice: tp.Optional[str]) -> str:
        """A voice name → its natural-language style description.  An unknown name
        is treated as a literal description, so ad-hoc styles work too."""
        voice = voice or self.default_voice
        return self._styles.get(voice, voice)

    # ── token server client (real; mirrors the Orpheus SSE client) ────────────

    def _wait_for_lm(self, budget_s: int) -> None:
        deadline = time.monotonic() + budget_s
        said = 0.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.lm_url + "/health", timeout=3) as r:
                    if r.status == 200:
                        _log(f"qwen3-tts token server is up at {self.lm_url}")
                        return
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() - said > 10:
                _log(f"waiting for the qwen3-tts token server at {self.lm_url} ...")
                said = time.monotonic()
            time.sleep(1)
        _log(f"gave up waiting for {self.lm_url} after {budget_s}s — continuing; "
             f"synthesis will fail until it's up")

    def _sse(self, payload: dict) -> tp.Iterator[dict]:
        """POST /completion with stream=true and yield each SSE JSON chunk.
        Closing this generator (barge-in) closes the connection → the server
        cancels the generation.  Same wire shape as the Orpheus engine."""
        req = urllib.request.Request(self.lm_url + "/completion",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=self._timeout)
        try:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                yield obj
                if obj.get("stop"):
                    break
        finally:
            resp.close()

    def _stream_token_ids(self, text: str, style: str) -> tp.Iterator[int]:
        """Stream generation and yield raw audio-token ids as they arrive.  The
        server contract mirrors the Orpheus tts_lm (return_tokens ids preferred).
        The MEANING of the ids (which are audio vs control, the codebook layout)
        is the codec's job — see Qwen3Codec."""
        payload = {
            "prompt": _format_input(text, style),   # ← model-specific (stub)
            "stream": True,
            "n_predict": self._max_tokens,
            **self._sampling,
            "cache_prompt": True,
            "return_tokens": True,
        }
        for chunk in self._sse(payload):
            for i in (chunk.get("tokens") or []):
                yield int(i)

    # ── engine contract (same as the other engines) ──────────────────────────

    def synthesize_stream(self, text: str, voice: tp.Optional[str] = None):
        """Yield 16-bit PCM byte chunks as tokens decode (dual-track streaming)."""
        if not self._codec.ready:
            raise _NeedsModelCard(
                "The qwen3 TTS engine is wired in but not finished: complete "
                "_format_input() and Qwen3Codec (frames_from_tokens/decode_frames) "
                "in tts_qwen3.py using the Qwen3-TTS model card, then set "
                "Qwen3Codec.ready = True.  See the module docstring for the exact "
                "two facts needed.")
        style = self.resolve_style(voice)
        for frame in self._codec.frames_from_tokens(self._stream_token_ids(text, style)):
            yield self._codec.decode_frames(frame)

    def synthesize(self, text: str, voice: tp.Optional[str] = None) -> np.ndarray:
        """One utterance → float32 PCM.  BLOCKING — worker thread."""
        chunks = [np.frombuffer(c, dtype=np.int16) for c in self.synthesize_stream(text, voice)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        pcm16 = np.concatenate(chunks)
        return np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0)
