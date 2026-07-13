"""
Orpheus TTS on llama.cpp — the "orpheus_gguf" engine.

The Orpheus voices without a heavyweight venv: the Llama-3B backbone runs as a
GGUF on a plain llama-server (the `tts_lm` config tier, started by
serve_tts_lm.sh next to the other LM tiers), and this module turns its token
stream into audio.  No multi-GB engine venv, no Python-version pin —
the whole engine is a
llama-server HTTP client plus a ~50 MB SNAC vocoder decoded with onnxruntime
on the CPU (well under realtime; zero GPU contention with the LMs).

How Orpheus encodes speech: the LM emits <custom_token_N> vocab tokens, 7 per
~85 ms audio frame.  De-interleaved into SNAC's three codebook layers (1+2+4
codes per frame) and decoded, each frame yields 2048 samples of 24 kHz PCM.
The sliding window here mirrors the official orpheus-speech decoder exactly —
decode the last 4 frames, keep the middle 2048 samples — so audio quality
matches the reference implementation by construction (including its baked-in silent head and
tail, which the cascade's trim_silence already handles).

Same synthesize()/synthesize_stream() contract as the other engines; runs in
vinkona_env (numpy + onnxruntime + huggingface_hub, all plain wheels).
"""

import json
import re
import time
import typing as tp
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
PRESET_VOICES = ["tara", "leah", "julia", "jess", "leo", "mia", "zac", "zoe"]

# The Llama-3 tokenizer ends at 128255; Orpheus's audio vocabulary starts right
# after it: token id 128256 is the string "<custom_token_0>".  The official
# decoder maps token number N to a SNAC code as  N - 10 - (position%7)*4096,
# accepting only codes in (0, 4096) — everything else (text tokens, the
# <custom_token_2> end-of-speech marker, prompt scaffolding) falls outside and
# is skipped.  We reproduce that arithmetic bit-for-bit.
CUSTOM_TOKEN_ID0 = 128256
CODE_OFFSET = 10
FRAME_TOKENS = 7          # SNAC codes per audio frame
WINDOW_TOKENS = 28        # official decoder: decode the last 4 frames…
EMIT_SLICE = (2048, 4096)  # …and emit the middle 2048 samples (~85 ms)

CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
_MAX_TOKEN_TEXT = len("<custom_token_9999999>")   # longest possible split tail worth keeping


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [tts-gguf] {msg}", flush=True)


def format_prompt(text: str, voice: str) -> str:
    """The prompt string the finetuned model expects.  This is the detokenized
    form of canopylabs' _format_prompt ids — 128259 <custom_token_3>, then
    "voice: text", then 128009 <|eot_id|> and 128260 <custom_token_4>.  No
    <|begin_of_text|>: llama-server adds BOS itself per the model metadata
    (including it here would double it)."""
    return f"<custom_token_3>{voice}: {text}<|eot_id|><custom_token_4>"


class TokenParser:
    """Extract <custom_token_N> numbers from streamed TEXT chunks, tolerating a
    token string split across two chunks (the unfinished tail is kept and
    prepended to the next feed).  Only used when llama-server is too old for
    return_tokens — the id path needs no parsing at all."""

    def __init__(self):
        self._tail = ""

    def feed(self, chunk: str) -> list:
        text = self._tail + chunk
        out = [int(m.group(1)) for m in CUSTOM_TOKEN_RE.finditer(text)]
        # Keep anything after the last '>' that could be the start of the next
        # token; cap it so stray '<' in non-token text can't grow unbounded.
        rest = text[text.rfind(">") + 1:]
        lt = rest.rfind("<")
        self._tail = rest[lt:] if lt != -1 and len(rest) - lt <= _MAX_TOKEN_TEXT else ""
        return out


def audio_code(n: int, index: int) -> tp.Optional[int]:
    """Token number -> SNAC code for stream position `index`, or None if it
    isn't a valid audio token at that position (official acceptance rule)."""
    code = n - CODE_OFFSET - (index % FRAME_TOKENS) * 4096
    return code if 0 < code < 4096 else None


def stream_to_windows(numbers: tp.Iterable) -> tp.Iterator:
    """Audio-token numbers -> 28-code decode windows, at the official cadence:
    the position counter only advances on ACCEPTED codes, and a window is
    emitted every 7th code once at least 28 have arrived."""
    codes: list = []
    count = 0
    for n in numbers:
        code = audio_code(n, count)
        if code is None:
            continue
        codes.append(code)
        count += 1
        if count % FRAME_TOKENS == 0 and count > WINDOW_TOKENS - 1:
            yield codes[-WINDOW_TOKENS:]


def deinterleave(codes: tp.Sequence) -> tuple:
    """Flat 7-per-frame code list -> SNAC's three codebook layers (1+2+4 per
    frame), shaped [1, T] int64 as the ONNX decoder expects."""
    n = len(codes) // FRAME_TOKENS
    c0 = np.empty((1, n), dtype=np.int64)
    c1 = np.empty((1, 2 * n), dtype=np.int64)
    c2 = np.empty((1, 4 * n), dtype=np.int64)
    for j in range(n):
        f = codes[FRAME_TOKENS * j: FRAME_TOKENS * (j + 1)]
        c0[0, j] = f[0]
        c1[0, 2 * j], c1[0, 2 * j + 1] = f[1], f[4]
        c2[0, 4 * j: 4 * j + 4] = (f[2], f[3], f[5], f[6])
    return c0, c1, c2


class SnacDecoder:
    """The SNAC 24 kHz vocoder decoder via onnxruntime, on CPU by design: a
    window decodes in ~10–25 ms for 85 ms of audio, and keeping it off the GPU
    means TTS never contends with the LM that's generating its tokens."""

    def __init__(self, model_path):
        import onnxruntime as ort            # lazy: only this engine needs it
        so = ort.SessionOptions()
        so.log_severity_level = 3            # errors only
        self._sess = ort.InferenceSession(str(model_path), sess_options=so,
                                          providers=["CPUExecutionProvider"])
        names = [i.name for i in self._sess.get_inputs()]
        if len(names) != 3:                  # audio_codes.0 / .1 / .2, in export order
            raise RuntimeError(f"unexpected SNAC decoder inputs: {names} "
                               f"(need the 3-codebook 24 kHz decoder)")
        self._names = names

    def decode_window(self, codes: tp.Sequence) -> bytes:
        """One 28-code window -> the middle 2048 samples as int16 PCM bytes."""
        feed = dict(zip(self._names, deinterleave(codes)))
        audio = self._sess.run(None, feed)[0]            # [1, 1, samples] float32
        pcm = np.clip(audio[0, 0, EMIT_SLICE[0]:EMIT_SLICE[1]], -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes()


def resolve_snac_model(snac_path, repo: str, filename: str) -> Path:
    """A configured local file wins; then the installer's default drop
    location; then the in-tree HF cache (offline — env.sh pins HF_HOME under
    var/); the network is only touched when none of those have it.  Restarts
    must never re-download or even re-contact the hub."""
    if snac_path:
        p = Path(snac_path)
        if p.exists():
            return p
        _log(f"configured snac_path {p} not found — trying the local copies")
    default = Path(__file__).resolve().parent / "Models" / "snac_24khz_decoder.onnx"
    if default.exists():
        _log(f"SNAC decoder: using {default}")
        return default
    from huggingface_hub import hf_hub_download
    try:                                   # cache hit = no network, no HF warning
        p = Path(hf_hub_download(repo_id=repo, filename=filename, local_files_only=True))
        _log(f"SNAC decoder: using the in-tree cached copy")
        return p
    except Exception:
        pass
    _log(f"fetching SNAC decoder {repo}/{filename} (≈50 MB, one-time — cached in-tree)")
    return Path(hf_hub_download(repo_id=repo, filename=filename))


class OrpheusGGUFEngine:
    def __init__(
        self,
        lm_url: str,
        default_voice: str = "tara",
        snac_path: tp.Optional[str] = None,
        snac_repo: str = "onnx-community/snac_24khz-ONNX",
        snac_file: str = "onnx/decoder_model.onnx",
        temperature: float = 0.6,
        top_p: float = 0.8,
        repeat_penalty: float = 1.3,
        max_tokens: int = 3500,
        request_timeout_s: int = 300,
        wait_for_lm_s: int = 180,
    ):
        self.lm_url = lm_url.rstrip("/")
        self.sample_rate = SAMPLE_RATE
        self.default_voice = default_voice
        self._timeout = request_timeout_s
        self._sampling = {"temperature": float(temperature), "top_p": float(top_p),
                          "repeat_penalty": float(repeat_penalty)}
        self._max_tokens = int(max_tokens)
        self._snac = SnacDecoder(resolve_snac_model(snac_path, snac_repo, snac_file))
        self._wait_for_lm(wait_for_lm_s)

    @property
    def voices(self) -> list:
        return list(PRESET_VOICES)

    # ── llama-server client ──────────────────────────────────────────────────

    def _wait_for_lm(self, budget_s: int) -> None:
        """The tts_lm llama-server starts alongside us and takes ~10–30 s to load
        a 3B GGUF; wait for /health rather than failing the first request.  A
        timeout is a warning, not fatal — synthesis errors say what's wrong."""
        deadline = time.monotonic() + budget_s
        said = 0.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.lm_url + "/health", timeout=3) as r:
                    if r.status == 200:
                        _log(f"tts_lm llama-server is up at {self.lm_url}")
                        return
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() - said > 10:
                _log(f"waiting for the tts_lm llama-server at {self.lm_url} "
                     f"(it loads the Orpheus GGUF on startup) ...")
                said = time.monotonic()
            time.sleep(1)
        _log(f"gave up waiting for {self.lm_url} after {budget_s}s — continuing; "
             f"synthesis will fail until it's up (is serve_tts_lm.sh running?)")

    def _sse(self, payload: dict) -> tp.Iterator:
        """POST /completion with stream=true and yield each SSE JSON chunk.
        Closing this generator (client hang-up / barge-in) closes the HTTP
        connection, which makes llama-server cancel the generation slot."""
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

    def _token_numbers(self, text: str, voice: str) -> tp.Iterator:
        """Stream generation and yield raw <custom_token_N> numbers.  Prefers
        the id path (`return_tokens`, llama-server ≥ b4337: ids arrive exact
        even if the text rendering hides 'special' tokens); falls back to
        regex-parsing the streamed text on older servers."""
        payload = {
            "prompt": format_prompt(text, voice),
            "stream": True,
            "n_predict": self._max_tokens,
            **self._sampling,
            # 128258 = <custom_token_2>, Orpheus's end-of-speech.  The GGUF
            # metadata marks it EOS; the stop string is belt-and-braces for
            # conversions that don't.
            "stop": ["<custom_token_2>"],
            "cache_prompt": True,       # reuse the KV prefix across sentences
            "return_tokens": True,
        }
        parser = TokenParser()
        saw_text = False
        got_ids = False
        for chunk in self._sse(payload):
            ids = chunk.get("tokens")
            if ids:
                got_ids = True
                for i in ids:
                    yield i - CUSTOM_TOKEN_ID0     # non-audio ids go negative and
            else:                                  # are rejected by audio_code()
                content = chunk.get("content") or ""
                saw_text = saw_text or bool(content.strip())
                for n in parser.feed(content):
                    yield n
        self._last_stream_empty_hint = (
            None if got_ids or saw_text else
            "the LM stream had no token ids and no text — llama-server may be "
            "hiding special tokens; upgrade llama.cpp or check tts_lm.model")

    # ── engine contract (same as tts_orpheus / tts_neutts) ───────────────────

    def synthesize(self, text: str, voice: tp.Optional[str] = None) -> np.ndarray:
        """One utterance -> float32 PCM @ 24 kHz.  BLOCKING — worker thread."""
        chunks = [np.frombuffer(c, dtype=np.int16) for c in self.synthesize_stream(text, voice)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        pcm16 = np.concatenate(chunks)
        return np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0)

    def synthesize_stream(self, text: str, voice: tp.Optional[str] = None):
        """Yield 16-bit PCM byte chunks (~85 ms each) as tokens arrive — same
        framing as the other engines' synthesize_stream."""
        voice = voice or self.default_voice
        self._last_stream_empty_hint = None
        n_windows = 0
        for window in stream_to_windows(self._token_numbers(text, voice)):
            n_windows += 1
            yield self._snac.decode_window(window)
        if n_windows == 0:
            hint = self._last_stream_empty_hint or (
                "tokens arrived but none were audio tokens — is tts_lm.model "
                "an Orpheus GGUF (e.g. orpheus-3b-0.1-ft-Q8_0.gguf)?")
            _log(f"no audio for {len(text)} chars: {hint}")
