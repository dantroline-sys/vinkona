"""Tests for the orpheus_gguf engine.

Two layers:
  1. Pure-logic checks (prompt format, token parsing, code mapping, window
     cadence, SNAC de-interleave) — need only numpy.
  2. An END-TO-END synthesis against a FAKE llama-server (stdlib HTTP serving
     a canned SSE token stream) through the REAL SNAC ONNX decoder — runs when
     onnxruntime + a decoder file are available (after ./install.sh tts
     orpheus_gguf, or point SNAC_ONNX at one), and is skipped cleanly
     otherwise.  This proves the whole client → parse → window → vocode path
     without a GPU or the 3B model.

Run inside vinkona_env:  python test_tts_orpheus_gguf.py
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import tts_orpheus_gguf as og


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def valid_numbers(n_tokens):
    """Token numbers that decode to valid codes (code 1000+i at each position)."""
    return [(1000 + i) % 4000 + 1 + og.CODE_OFFSET + (i % 7) * 4096 for i in range(n_tokens)]


def pure_logic():
    # format_prompt — canopylabs' scaffolding, no BOS (llama-server adds it)
    p = og.format_prompt("Hello.", "tara")
    check("prompt scaffolding", p == "<custom_token_3>tara: Hello.<|eot_id|><custom_token_4>")
    check("prompt has no BOS", "<|begin_of_text|>" not in p)

    # TokenParser — whole, split, and non-token text
    tp = og.TokenParser()
    check("parse two tokens", tp.feed("<custom_token_10><custom_token_4106>") == [10, 4106])
    check("split token part 1", tp.feed("<custom_tok") == [])
    check("split token part 2", tp.feed("en_99>") == [99])
    check("plain text ignored", tp.feed("hello world") == [])
    tp2 = og.TokenParser()
    check("tail capped on long garbage after '<'", tp2.feed("<" + "x" * 100) == [] and tp2._tail == "")
    check("parser recovers after garbage", tp2.feed("<custom_token_7>") == [7])

    # audio_code — the official acceptance rule
    check("position 0 maps", og.audio_code(11, 0) == 1)
    check("position 1 subtracts 4096", og.audio_code(11 + 4096, 1) == 1)
    check("code 0 rejected", og.audio_code(og.CODE_OFFSET, 0) is None)
    check("eos token (n=2) rejected", og.audio_code(2, 0) is None)
    check("code 4096 rejected", og.audio_code(4096 + og.CODE_OFFSET, 0) is None)
    check("text token (negative n) rejected", og.audio_code(-8256, 3) is None)

    # stream_to_windows — official cadence: first window at 28 codes, then every 7
    wins = list(og.stream_to_windows(valid_numbers(70)))       # 10 frames
    check("10 frames -> 7 windows", len(wins) == 7)
    check("windows are 28 codes", all(len(w) == 28 for w in wins))
    check("21 codes -> no window yet", list(og.stream_to_windows(valid_numbers(21))) == [])
    check("28 codes -> one window", len(list(og.stream_to_windows(valid_numbers(28)))) == 1)
    # invalid tokens must not advance the position counter
    nums = valid_numbers(28)
    with_junk = nums[:5] + [2, -100, 999999] + nums[5:]
    wins2 = list(og.stream_to_windows(with_junk))
    check("junk tokens skipped without breaking framing", len(wins2) == 1 and wins2[0] == wins[0] if wins else False)

    # deinterleave — 7 codes/frame -> SNAC's 1+2+4 layer layout
    c0, c1, c2 = og.deinterleave([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16])
    check("layer 0 layout", c0.tolist() == [[0, 10]])
    check("layer 1 layout", c1.tolist() == [[1, 4, 11, 14]])
    check("layer 2 layout", c2.tolist() == [[2, 3, 5, 6, 12, 13, 15, 16]])
    check("layers are int64", all(a.dtype == np.int64 for a in (c0, c1, c2)))


class _FakeLlama(BaseHTTPRequestHandler):
    """A llama-server just real enough for the engine: /health and a streamed
    /completion whose SSE chunks carry the class-level token payload."""
    mode = "text"          # "text" -> content chunks; "ids" -> return_tokens chunks
    numbers = []           # token numbers to stream
    last_payload = None

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _FakeLlama.last_payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # two tokens per SSE chunk, one split across chunks to test the parser
        ns = list(self.numbers)
        for i in range(0, len(ns), 2):
            batch = ns[i:i + 2]
            if self.mode == "ids":
                obj = {"content": "", "tokens": [n + og.CUSTOM_TOKEN_ID0 for n in batch]}
            else:
                text = "".join(f"<custom_token_{n}>" for n in batch)
                if i == 0 and len(text) > 10:          # split the very first token string
                    self._chunk({"content": text[:10]})
                    text = text[10:]
                obj = {"content": text}
            self._chunk(obj)
        self._chunk({"content": "", "stop": True})

    def _chunk(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

    def log_message(self, *a):
        pass


def find_snac():
    for cand in [os.environ.get("SNAC_ONNX"), "Models/snac_24khz_decoder.onnx"]:
        if cand and Path(cand).exists():
            return cand
    return None


def end_to_end():
    snac = find_snac()
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        snac = None
    if not snac:
        print("  skip end-to-end (needs onnxruntime + a SNAC decoder — run "
              "'./install.sh tts orpheus_gguf' or set SNAC_ONNX=/path/decoder_model.onnx)")
        return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    n_frames = 10
    _FakeLlama.numbers = valid_numbers(n_frames * 7)
    expect = (n_frames - 3) * (og.EMIT_SLICE[1] - og.EMIT_SLICE[0])   # sliding window

    for mode in ("text", "ids"):
        _FakeLlama.mode = mode
        eng = og.OrpheusGGUFEngine(lm_url=url, snac_path=snac, wait_for_lm_s=5)
        chunks = list(eng.synthesize_stream("Hello there.", "tara"))
        check(f"[{mode}] stream yields {n_frames - 3} chunks", len(chunks) == n_frames - 3)
        pcm = eng.synthesize("Hello there.", "tara")
        check(f"[{mode}] sample count {expect}", pcm.shape == (expect,))
        check(f"[{mode}] float32 in [-1,1]",
              pcm.dtype == np.float32 and float(np.abs(pcm).max()) <= 1.0)
        check(f"[{mode}] audio is not silence", float(np.abs(pcm).max()) > 0.0)

    pay = _FakeLlama.last_payload
    check("request streams", pay.get("stream") is True)
    check("request has repetition penalty", pay.get("repeat_penalty", 0) >= 1.1)
    check("request stops on <custom_token_2>", pay.get("stop") == ["<custom_token_2>"])
    check("request asks for token ids", pay.get("return_tokens") is True)
    srv.shutdown()


def main():
    pure_logic()
    end_to_end()
    print(f"\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
