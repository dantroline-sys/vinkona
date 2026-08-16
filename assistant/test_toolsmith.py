#!/usr/bin/env python
"""Tests for toolsmith.py — the two-phase idle tool-maker.

Phase 1 (identify): spot a missing tool → queue a PLAIN-LANGUAGE spec (status proposed).
Phase 2 (build_next): take one spec → code → self-test → install, or bank the failure
(status failed, with code + error) for a later analyse-and-retry; park when the attempt
budget is spent or the LM judges it unbuildable.  kb_ask guidance is consulted when a
re-analysis asks for it.

The big LM is stubbed (a scripted sequence of JSON replies), so these run without a model.
The self-test gate is real, so they need a sandbox backend; they SKIP loudly without one.

    python test_toolsmith.py
"""
import asyncio
import importlib.util
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tb = _load("toolbox")
ts = _load("toolsmith")

# Prefer the real container default; fall back to bwrap on a bare Linux box.
if tb._ContainerBackend({}).available():
    CFG = {"backend": "container"}
elif tb._BwrapBackend().available():
    CFG = {"backend": "bwrap"}
else:
    CFG = None

PASS = FAIL = SKIP = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


GOOD = ("import sys, json\na = json.load(sys.stdin)\n"
        "print(json.dumps({'double': a.get('n', 0) * 2}))")
BAD = "import sys, json\nthis is not valid python\n"

GOOD_SPEC = {"name": "doubler", "code": GOOD,
             "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
             "test": {"input": {"n": 3}, "expect_keys": ["double"]}}


def _stub(replies):
    """Scripted big-LM: returns the replies in order, then None.  Records the prompts so a
    test can assert what the LM was shown."""
    it = iter(replies)
    seen = []

    async def chat_json(prompt, think=True):
        seen.append(prompt)
        try:
            return next(it)
        except StopIteration:
            return None
    chat_json.seen = seen
    return chat_json


def _box(td, sub="own"):
    return tb.Toolbox(Path(td) / sub, cfg=dict(CFG), seed=False)


async def _main():
    # ── phase 1: identify ─────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([{"deficit": True, "title": "CSV summariser",
                     "purpose": "Read a CSV file and return per-column stats.",
                     "rationale": "user pasted CSVs twice"}])
        r = await ts.identify(box, cj)
        q = box.ideas()
        check("a deficit becomes a queued plain-language spec",
              r["action"] == "proposed" and len(q) == 1
              and q[0]["status"] == "proposed" and "per-column" in q[0]["sketch"])
        cj2 = _stub([{"deficit": False}])
        r2 = await ts.identify(box, cj2)
        check("no deficit → nothing queued", r2["action"] == "none" and len(box.ideas()) == 1)
        cj3 = _stub([{"deficit": True, "title": "csv summariser", "purpose": "again"}])
        r3 = await ts.identify(box, cj3)
        check("a duplicate spec is not queued twice",
              r3["action"] == "none" and len(box.ideas()) == 1)
        r4 = await ts.identify(box, _stub([{"deficit": True, "title": "x", "purpose": "y"}]),
                               max_queue=1)
        check("a full queue pauses the deficit scan",
              r4["action"] == "none" and "full" in r4.get("reason", ""))

    # ── phase 2: build from the queue ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", rationale="r", sketch="given n, return 2n",
                     source="toolsmith")
        cj = _stub([GOOD_SPEC])
        r = await ts.build_next(box, cj)
        check("a queued spec is built and installed",
              r["action"] == "built" and box.has("doubler"))
        check("the built spec leaves the queue", box.ideas() == [])
        check("the built tool actually runs",
              box.call("doubler", {"n": 5}).get("result") == {"double": 10})
        r2 = await ts.build_next(box, _stub([]))
        check("an empty queue is a quiet no-op", r2["action"] == "none")

    # user-jotted specs are picked before toolsmith ones
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("toolsmith spec first in", source="toolsmith", sketch="a")
        box.add_idea("user wish", source="user", sketch="double n")
        r = await ts.build_next(box, _stub([GOOD_SPEC]))
        left = [i["title"] for i in box.ideas()]
        check("a user-jotted spec is built first",
              r["action"] == "built" and left == ["toolsmith spec first in"])

    # failure banks code+error; the NEXT session re-analyses and succeeds with guidance
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="given n, return 2n", source="toolsmith")
        r = await ts.build_next(box, _stub([{**GOOD_SPEC, "code": BAD}]), max_repair=0)
        i = box.ideas()[0]
        check("a failed build is banked with its code and error",
              r["action"] == "failed" and i["status"] == "failed"
              and i["attempts"] == 1 and BAD.strip() in i["last_code"]
              and "self-test failed" in i["last_error"])
        asked = []

        async def guide(q):
            asked.append(q)
            return "Use json.dumps on a dict; test with a small input."
        cj = _stub([
            {"diagnosis": "syntax error", "adjustment": "write valid python",
             "kb_question": "how do I emit JSON from python stdlib?"},
            GOOD_SPEC])
        r2 = await ts.build_next(box, cj, guidance=guide)
        check("a failed spec is re-analysed then built",
              r2["action"] == "built" and r2["attempts"] == 2 and box.has("doubler"))
        check("the analysis consulted kb guidance", asked and "JSON" in asked[0] or asked)
        check("the guidance reached the code prompt",
              any("knowledge base" in p and "json.dumps on a dict" in p for p in cj.seen))
        check("the prior code + error were shown to the analysis",
              any("this is not valid python" in p and "FAILED" in p for p in cj.seen))

    # attempts budget → parked; requeue resets it
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("impossible thing", sketch="cannot work", source="toolsmith")
        for want in ("failed", "parked"):
            cj = _stub([{"diagnosis": "d", "adjustment": "a", "kb_question": ""},
                        {**GOOD_SPEC, "code": BAD},
                        {**GOOD_SPEC, "code": BAD}])
            r = await ts.build_next(box, cj, max_repair=0, max_attempts=2)
            check(f"attempt session ends {want}", r["action"] == want)
        i = box.ideas()[0]
        check("requeue puts a parked spec back with a fresh budget",
              box.requeue_idea(i["id"]).get("ok")
              and box.ideas()[0]["status"] == "proposed"
              and box.ideas()[0]["attempts"] == 0)

    # the LM can declare a spec unbuildable → parked immediately with the reason
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("needs the internet", sketch="fetch a url", source="user")
        r = await ts.build_next(box, _stub([
            {"unbuildable": True, "reason": "needs network access, sandbox has none"}]))
        i = box.ideas()[0]
        check("an unbuildable spec parks immediately with the reason",
              r["action"] == "parked" and i["status"] == "parked"
              and "network" in i["last_error"])

    # a name collision is fed back as an in-session repair, not a crash
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("doubler", GOOD, {"description": "orig"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        box.add_idea("double again", sketch="2n", source="user")
        cj = _stub([GOOD_SPEC, {**GOOD_SPEC, "name": "doubler_two"}])
        r = await ts.build_next(box, cj)
        check("a name collision retries under a new name",
              r["action"] == "built" and r["name"] == "doubler_two"
              and box.read("doubler")["manifest"].find('"orig"') > 0)

    # tool cap: build waits, identify still queues
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("t_seed", GOOD, {"description": "x"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        box.add_idea("one more", sketch="s", source="user")
        r = await ts.build_next(box, _stub([GOOD_SPEC]), max_tools=1)
        check("at the tool cap the build waits",
              r["action"] == "none" and box.ideas()[0]["status"] == "proposed")

    # the run() orchestrator: both phases in one pass
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([
            {"deficit": True, "title": "double a number", "purpose": "given n return 2n",
             "rationale": "came up twice"},
            GOOD_SPEC])
        st = await ts.run(box, cj)
        check("run() queues the spec then builds it in the same pass",
              st["identified"]["action"] == "proposed"
              and st["build"]["action"] == "built" and box.has("doubler")
              and box.ideas() == [])


def main():
    if CFG is None:
        print("SKIP  toolsmith (no sandbox backend — install bubblewrap or pull the image)")
        print("\n0 passed, 0 failed, 1 skipped")
        return
    asyncio.run(_main())
    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
