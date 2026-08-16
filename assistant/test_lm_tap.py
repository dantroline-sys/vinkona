#!/usr/bin/env python
"""Tests for lm_tap.py — the RAM-file live feed of LM context/output.

    python test_lm_tap.py
"""
import importlib.util
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tap = _load("lm_tap")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def main():
    with tempfile.TemporaryDirectory() as td:
        tap.PATH_OVERRIDE = str(Path(td) / "feed.jsonl")

        # roundtrip + ordering (oldest→newest) + src filter
        tap.write("big", "request", "the prompt", call_id="c1", lane="toolsmith",
                  model="qwen", meta={"think": True})
        tap.write("big", "response", "the reply", call_id="c1", elapsed_s=12.34)
        tap.write("fast", "request", "chat context", call_id="c2")
        evs = tap.read()
        check("events round-trip in order",
              [e["kind"] for e in evs] == ["request", "response", "request"])
        check("request carries lane/model/meta",
              evs[0]["lane"] == "toolsmith" and evs[0]["model"] == "qwen"
              and evs[0]["meta"] == {"think": True})
        check("response carries elapsed", evs[1]["elapsed_s"] == 12.34)
        big = tap.read(src="big")
        check("src filter works",
              len(big) == 2 and all(e["src"] == "big" for e in big)
              and len(tap.read(src="fast")) == 1)
        check("n limits from the tail", [e["kind"] for e in tap.read(n=1)] == ["request"])

        # a huge prompt is clipped head+tail, never stored whole
        tap.clear()
        tap.write("big", "request", "H" * 20000 + "TAILMARK")
        e = tap.read()[0]
        check("a huge event is clipped with head and tail kept",
              len(e["text"]) < 13000 and "chars clipped" in e["text"]
              and e["text"].startswith("H") and e["text"].endswith("TAILMARK"))

        # a torn/garbage line is skipped, not fatal
        with open(tap.feed_path(), "a") as f:
            f.write("{not json at all\n")
        tap.write("fast", "response", "after garbage")
        kinds = [e["kind"] for e in tap.read()]
        check("garbage lines are skipped", kinds[-1] == "response" and len(kinds) == 2)

        # the ring trims itself and stays readable
        tap.clear()
        for i in range(400):
            tap.write("big", "response", f"event {i} " + "x" * 8000)
        size = os.path.getsize(tap.feed_path())
        check("the feed trims itself under the cap", size <= tap.MAX_BYTES)
        evs = tap.read(n=5)
        check("the trimmed feed still reads (newest kept)",
              len(evs) == 5 and "event 399" in evs[-1]["text"])

        # clear removes the file; read of a missing feed is []
        tap.clear()
        check("clear empties the feed", tap.read() == []
              and not os.path.exists(tap.feed_path()))

        # writes never raise even when the path is unwritable
        tap.PATH_OVERRIDE = "/nonexistent-dir-xyz/feed.jsonl"
        try:
            tap.write("big", "request", "x")
            ok = True
        except Exception:
            ok = False
        check("a broken feed path never raises", ok and tap.read() == [])
        tap.PATH_OVERRIDE = None

    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
