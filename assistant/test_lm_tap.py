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

        # a normal-large prompt (say 100KB) is stored WHOLE — "all of it"
        tap.clear()
        tap.write("big", "request", "W" * 100_000 + "END")
        check("a large event is stored in full", tap.read()[0]["text"].endswith("END")
              and len(tap.read()[0]["text"]) == 100_003)

        # only a pathological event is clipped (head+tail kept, honestly marked)
        tap.clear()
        tap.write("big", "request", "H" * 400_000 + "TAILMARK")
        e = tap.read()[0]
        check("a pathological event is clipped with head and tail kept",
              len(e["text"]) < tap.EVENT_HEAD + tap.EVENT_TAIL + 100
              and "chars clipped" in e["text"]
              and e["text"].startswith("H") and e["text"].endswith("TAILMARK"))

        # the live slot: what's streaming right now
        tap.stream("big", "c9", "partial out so f", lane="toolsmith", model="qwen")
        lv = tap.live_read("big")
        check("the live slot round-trips",
              lv and lv["id"] == "c9" and lv["text"] == "partial out so f"
              and lv["lane"] == "toolsmith")
        check("the live slot is per-source", tap.live_read("fast") is None)
        tap.stream("big", "c9", "partial out so far, longer now")
        check("the live slot overwrites in place",
              tap.live_read("big")["text"].endswith("longer now"))
        tap.live_clear("big")
        check("live_clear empties the slot", tap.live_read("big") is None)
        # a stale slot (crashed writer) is ignored
        with open(tap.live_path("big"), "w") as f:
            json.dump({"ts": 1000.0, "id": "old", "text": "stale"}, f)
        check("a stale live slot is ignored", tap.live_read("big") is None)
        tap.live_clear("big")

        # find_event returns the FULL stored text by id+kind
        tap.clear()
        tap.write("big", "request", "prompt body", call_id="cf")
        tap.write("big", "response", "the whole answer " * 10, call_id="cf")
        fe = tap.find_event("cf", "response")
        check("find_event returns the full stored event",
              fe and fe["text"].startswith("the whole answer")
              and tap.find_event("cf", "request")["text"] == "prompt body"
              and tap.find_event("nope", "response") is None)

        # a torn/garbage line is skipped, not fatal
        tap.clear()
        tap.write("big", "request", "before garbage")
        with open(tap.feed_path(), "a") as f:
            f.write("{not json at all\n")
        tap.write("fast", "response", "after garbage")
        kinds = [e["kind"] for e in tap.read()]
        check("garbage lines are skipped", kinds == ["request", "response"])

        # the ring trims itself and stays readable
        tap.clear()
        for i in range(400):
            tap.write("big", "response", f"event {i} " + "x" * 60_000)
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
