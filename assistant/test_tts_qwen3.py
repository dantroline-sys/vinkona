"""Tests for the qwen3 TTS engine SCAFFOLD (tts_qwen3.py).

The model-specific decode isn't written yet (it needs the Qwen3-TTS model card),
so these pin the parts that ARE real and must stay correct when the decode is
filled in:
  • the engine contract (sample_rate / voices / resolve_style);
  • the streaming SSE client against a fake token server;
  • the synthesize_stream → synthesize orchestration + PCM framing, exercised
    through a FAKE codec (so the plumbing is proven without the real 12Hz codec);
  • the honest failure: with no real codec, a synth call raises a clear, named
    error rather than silence or noise; and the two model-specific stubs raise.

Needs numpy (like the Orpheus engine test) — skips cleanly without it.

Run inside vinkona_env:  python test_tts_qwen3.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import numpy as np
    import tts_qwen3 as q3
except Exception as e:                       # numpy (or the engine's deps) absent
    print(f"  skip qwen3 engine tests (needs numpy): {e}")
    raise SystemExit(0)


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


class _FakeServer(BaseHTTPRequestHandler):
    """/health OK + a streamed /completion emitting canned token-id chunks."""
    token_ids = []
    last_payload = None

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            _FakeServer.last_payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            _FakeServer.last_payload = None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        ids = list(self.token_ids)
        for i in range(0, len(ids), 2):
            self._chunk({"tokens": ids[i:i + 2]})
        self._chunk({"tokens": [], "stop": True})

    def _chunk(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

    def log_message(self, *a):
        pass


class _FakeCodec:
    """A stand-in 12Hz codec: 1 frame per token id, 2 samples of PCM per frame —
    just enough to prove the engine's streaming/framing orchestration."""
    ready = True
    sample_rate = 24000

    def frames_from_tokens(self, token_ids):
        for i in token_ids:
            yield [int(i)]

    def decode_frames(self, frame):
        return np.array([frame[0] % 100, -(frame[0] % 100)], dtype=np.int16).tobytes()


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_contract_and_voice_style():
    srv, url = _serve()
    try:
        eng = q3.Qwen3TTSEngine(
            lm_url=url, default_voice="vinkona", wait_for_lm_s=5,
            voices={"vinkona": "A warm, calm assistant.",
                    "narrator": "A slow storyteller with a faint British accent."})
        check("sample_rate is exposed", eng.sample_rate == q3.DEFAULT_SAMPLE_RATE)
        check("voices are the configured names", set(eng.voices) == {"vinkona", "narrator"})
        check("a known voice resolves to its description",
              eng.resolve_style("narrator").startswith("A slow storyteller"))
        check("an unknown voice is used as a literal style",
              eng.resolve_style("A brisk newsreader.") == "A brisk newsreader.")
        check("no voice → the default's description",
              eng.resolve_style(None) == "A warm, calm assistant.")
    finally:
        srv.shutdown()


def test_sse_client_parses_stream():
    srv, url = _serve()
    _FakeServer.token_ids = [11, 22, 33, 44]
    try:
        eng = q3.Qwen3TTSEngine(lm_url=url, wait_for_lm_s=5, voices={"v": "x"})
        chunks = list(eng._sse({"prompt": "p", "stream": True}))
        got = [i for c in chunks for i in (c.get("tokens") or [])]
        check("SSE client yields every streamed token id", got == [11, 22, 33, 44])
        check("request marked streaming", _FakeServer.last_payload.get("stream") is True)
        check("request asks for token ids", _FakeServer.last_payload.get("return_tokens") is True)
    finally:
        srv.shutdown()


def test_orchestration_with_fake_codec():
    """With a real codec in place, synthesize_stream/​synthesize should flow
    tokens → frames → PCM.  Prove that plumbing with a fake codec + a stubbed
    input format (the two model-specific pieces), leaving the engine code real."""
    srv, url = _serve()
    _FakeServer.token_ids = [1, 2, 3, 4, 5]
    orig_fmt = q3._format_input
    q3._format_input = lambda text, style: f"[{style}] {text}"     # stand in for the real prompt
    try:
        eng = q3.Qwen3TTSEngine(lm_url=url, wait_for_lm_s=5, voices={"v": "warm"})
        eng._codec = _FakeCodec()
        chunks = list(eng.synthesize_stream("hello", "v"))
        check("one PCM chunk per token frame", len(chunks) == 5)
        pcm = eng.synthesize("hello", "v")
        check("synthesize concatenates to float32 in [-1,1]",
              pcm.dtype == np.float32 and float(np.abs(pcm).max()) <= 1.0)
        check("synthesize produced the expected sample count", pcm.shape == (10,))
    finally:
        q3._format_input = orig_fmt
        srv.shutdown()


def test_honest_failure_when_unfinished():
    srv, url = _serve()
    try:
        eng = q3.Qwen3TTSEngine(lm_url=url, wait_for_lm_s=5, voices={"v": "warm"})
        # The real codec is not ready → a synth call must raise a clear, named error.
        try:
            list(eng.synthesize_stream("hi", "v"))
            raised = False
        except NotImplementedError as e:
            raised = "model card" in str(e).lower() or "qwen3" in str(e).lower()
        check("unfinished engine raises a clear error, not silence", raised)
        # the two model-specific stubs each name what they need
        for fn in (lambda: q3._format_input("t", "s"),
                   lambda: list(q3.Qwen3Codec(None, "r", None, 24000).frames_from_tokens([1])),
                   lambda: q3.Qwen3Codec(None, "r", None, 24000).decode_frames([1])):
            try:
                fn(); ok = False
            except NotImplementedError:
                ok = True
            check("model-specific stub raises NotImplementedError", ok)
    finally:
        srv.shutdown()


def main():
    test_contract_and_voice_style()
    test_sse_client_parses_stream()
    test_orchestration_with_fake_codec()
    test_honest_failure_when_unfinished()
    print(f"\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
