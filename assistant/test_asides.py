#!/usr/bin/env python
"""VIN-WM-01 phase 2 — private first-person asides.

The reply model may wrap up to two notes-to-self in <aside>…</aside>; the bridge
strips them from the spoken stream (exactly as it already strips <think>) and routes
them to chat_logs (role='aside') for the dreaming phase.  This covers:
  * _TagStripper: <think> still dropped; <aside> captured; both safe across the chunk
    boundaries SSE delivers (a tag split over two chunks), and an unclosed span never
    spoken; a bare '<' is not mistaken for a tag.
  * the exact think→aside chain _stream_chat runs.
  * _finish_turn logs each aside AFTER the assistant reply, skips blanks, preserves
    order, and clears them (so a second _finish_turn can't double-log).

    python test_asides.py
"""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent

_a = types.ModuleType("aiohttp")                     # stub so llm_bridge imports
_a.ClientSession = object
_a.ClientTimeout = lambda **k: None
_a.ClientError = type("ClientError", (Exception,), {})
_a.ClientConnectorError = type("ClientConnectorError", (Exception,), {})
sys.modules["aiohttp"] = _a


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = _load("llm_bridge")

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _run_stream(chunks):
    """Reproduce _stream_chat's think→aside composition over a token sequence."""
    think = lb._TagStripper(lb._THINK_OPEN, lb._THINK_CLOSE)          # dropped
    sink = []
    aside = lb._TagStripper(lb._ASIDE_OPEN, lb._ASIDE_CLOSE, capture=sink.append)
    spoken = []
    for c in chunks:
        piece = aside.feed(think.feed(c))
        if piece:
            spoken.append(piece)
    tail = aside.feed(think.flush()) + aside.flush()
    if tail:
        spoken.append(tail)
    return "".join(spoken), sink


def _chunk(s, n):
    return [s[i:i + n] for i in range(0, len(s), n)]


# ── _TagStripper / the stream composition ────────────────────────────────────
def test_stripper():
    spoken, sink = _run_stream(["Just a normal reply."])
    check("plain reply: spoken verbatim, no asides", spoken == "Just a normal reply." and sink == [])

    spoken, sink = _run_stream(["Sure thing.<aside>they seem rushed</aside>"])
    check("aside stripped from speech and captured",
          spoken == "Sure thing." and sink == ["they seem rushed"])

    spoken, sink = _run_stream(["<think>plan the reply</think>Hello.<aside>note</aside> bye"])
    check("<think> dropped, <aside> captured, rest spoken",
          spoken == "Hello. bye" and sink == ["note"])

    # tags split across arbitrary chunk boundaries (the SSE reality)
    full = "He<aside>xx</aside>llo, <think>hmm</think>world"
    for n in (1, 2, 3, 5):
        spoken, sink = _run_stream(_chunk(full, n))
        check(f"split tags reassemble at chunk size {n}",
              spoken == "Hello, world" and sink == ["xx"])

    spoken, sink = _run_stream(["bye", "<aside>unfin", "ished note"])   # never closed
    check("an unclosed aside is captured, never spoken",
          spoken == "bye" and sink == ["unfinished note"])

    spoken, sink = _run_stream(["a < b ", "and c > d"])                 # bare '<' and '>'
    check("a bare '<' is not mistaken for a tag (no text eaten)",
          spoken == "a < b and c > d" and sink == [])

    spoken, sink = _run_stream(["one<aside>A</aside>two<aside>B</aside>three"])
    check("two asides in one turn, both captured, order kept",
          spoken == "onetwothree" and sink == ["A", "B"])


# ── _finish_turn routes asides to the log after the reply, then clears them ───
def test_finish_turn_logs_asides():
    logged = []
    sh = types.SimpleNamespace(
        history=[], offer_spoken_hook=None, big_url="",
        log_hook=lambda role, text: logged.append((role, text)),
        _turn_asides=["they seem rushed", "   ", "I'm unsure about this one"])
    sh._finish_turn = lb.LLMBridge._finish_turn.__get__(sh)

    sh._finish_turn("Sure, done.")
    check("assistant reply logged first, then each non-blank aside in order",
          logged == [("assistant", "Sure, done."),
                     ("aside", "they seem rushed"),
                     ("aside", "I'm unsure about this one")])
    check("asides go to history? no — only the assistant turn does",
          sh.history == [{"role": "assistant", "content": "Sure, done."}])
    check("consumed asides are cleared", sh._turn_asides == [])

    logged.clear()
    sh._finish_turn("next turn")
    check("a later turn with no asides logs only the assistant reply (no double-log)",
          logged == [("assistant", "next turn")])


# ── the prompt instruction is a stable, non-empty directive ──────────────────
def test_instruction_present():
    check("aside instruction exists and names the <aside> channel",
          "<aside>" in lb._ASIDE_INSTRUCTION and "never" in lb._ASIDE_INSTRUCTION.lower())


if __name__ == "__main__":
    test_stripper()
    test_finish_turn_logs_asides()
    test_instruction_present()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        sys.exit(1)
    print(f"ALL OK ({PASS} checks)")
