#!/usr/bin/env python
"""Tests for toolsmith.py — the two-phase idle tool-maker and its build harness.

Phase 1 (identify): spot a missing tool → queue a PLAIN-LANGUAGE spec (status proposed).
Phase 2 (build_next): the harness — code as plain fenced text (chat_text), a free local
syntax gate, metadata as a small JSON ask that DEGRADES to defaults, then the sandbox
self-test; failures bank the WHOLE attempt (code even when faulty, plus the raw model
reply) for the panel's inspector and a later analyse-and-retry with kb_ask guidance.

The big LM is stubbed (scripted replies), so these run without a model.  The self-test
gate is real, so they need a sandbox backend; they SKIP loudly without one.

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
SYNTAX_BAD = "import sys, json\nthis is not valid python\n"
RUNTIME_BAD = "import sys, json\nraise RuntimeError('boom')\n"

def fenced(code, chatter="Here is the tool:"):
    return f"{chatter}\n```python\n{code}\n```\nDone."

META_GOOD = {"name": "doubler",
             "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
             "test": {"input": {"n": 3}, "expect_keys": ["double"]}}


def _stub(replies):
    """Scripted LM: returns the replies in order, then None; records prompts in .seen."""
    it = iter(replies)
    seen = []

    async def call(prompt, think=True):
        seen.append(prompt)
        try:
            return next(it)
        except StopIteration:
            return None
    call.seen = seen
    return call


def _tstub(replies):
    """Scripted chat_text: same, but the callable takes only the prompt."""
    it = iter(replies)
    seen = []

    async def call(prompt):
        seen.append(prompt)
        try:
            return next(it)
        except StopIteration:
            return None
    call.seen = seen
    return call


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
        r2 = await ts.identify(box, _stub([{"deficit": False}]))
        check("no deficit → nothing queued", r2["action"] == "none" and len(box.ideas()) == 1)
        r3 = await ts.identify(box, _stub([{"deficit": True, "title": "csv summariser",
                                            "purpose": "again"}]))
        check("a duplicate spec is not queued twice",
              r3["action"] == "none" and len(box.ideas()) == 1)
        r4 = await ts.identify(box, _stub([{"deficit": True, "title": "x", "purpose": "y"}]),
                               max_queue=1)
        check("a full queue pauses the deficit scan",
              r4["action"] == "none" and "full" in r4.get("reason", ""))

    # ── phase 2: the build harness ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", rationale="r", sketch="given n, return 2n",
                     source="toolsmith")
        r = await ts.build_next(box, _stub([META_GOOD]), _tstub([fenced(GOOD)]))
        check("fenced code + metadata builds and installs",
              r["action"] == "built" and box.has("doubler"))
        check("the built spec leaves the queue", box.ideas() == [])
        check("the built tool actually runs",
              box.call("doubler", {"n": 5}).get("result") == {"double": 10})
        r2 = await ts.build_next(box, _stub([]), _tstub([]))
        check("an empty queue is a quiet no-op", r2["action"] == "none")

    # the syntax gate: a SyntaxError is fed back without spending a sandbox run
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="2n", source="toolsmith")
        ct = _tstub([fenced(SYNTAX_BAD), fenced(GOOD)])
        r = await ts.build_next(box, _stub([META_GOOD]), ct, max_repair=1)
        check("a syntax error is repaired in-session",
              r["action"] == "built" and box.has("doubler"))
        check("the SyntaxError was fed back to the code step",
              len(ct.seen) == 2 and "SyntaxError" in ct.seen[1]
              and "this is not valid python" in ct.seen[1])

    # no reply at all (the 27B timing out): honest error, nothing lost
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="2n", source="toolsmith")
        r = await ts.build_next(box, _stub([]), _tstub([None, None, None]), max_repair=2)
        i = box.ideas()[0]
        check("a silent LM reads as 'no reply', not 'unusable code'",
              r["action"] == "failed" and "no reply" in i["last_error"])

    # unfenced garbage: the RAW reply is banked and visible
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="2n", source="toolsmith")
        prose = "I think the best approach would be to use a loop, then maybe json?"
        r = await ts.build_next(box, _stub([]), _tstub([prose]), max_repair=0)
        i = box.ideas()[0]
        check("an unusable reply banks the RAW model output for the inspector",
              r["action"] == "failed" and i.get("last_raw") == prose
              and prose in i.get("last_code", ""))

    # metadata flake → defaults (slug name, bare test); only code quality gates a build
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="2n", source="toolsmith")
        r = await ts.build_next(box, _stub([None]), _tstub([fenced(GOOD)]))
        check("a metadata flake degrades to defaults instead of failing",
              r["action"] == "built" and box.has("double_a_number"))

    # UNBUILDABLE line → parked with the reason
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("needs the internet", sketch="fetch a url", source="user")
        r = await ts.build_next(box, _stub([]),
                                _tstub(["UNBUILDABLE: needs network access"]))
        i = box.ideas()[0]
        check("an unbuildable spec parks immediately with the reason",
              r["action"] == "parked" and i["status"] == "parked"
              and "network" in i["last_error"])

    # a runtime failure banks the WHOLE attempt (code, manifest, test) for the inspector
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("double a number", sketch="2n", source="toolsmith")
        r = await ts.build_next(box, _stub([META_GOOD]), _tstub([fenced(RUNTIME_BAD)]),
                                max_repair=0)
        i = box.ideas()[0]
        check("a self-test failure is banked with its error",
              r["action"] == "failed" and i["status"] == "failed"
              and "self-test failed" in i["last_error"]
              and "boom" in i["last_error"])
        check("the WHOLE attempt is banked for the panel's inspector",
              i.get("name") == "doubler" and RUNTIME_BAD.strip() in i.get("last_code", "")
              and (i.get("last_manifest") or {}).get("name") == "doubler"
              and (i.get("last_test") or {}).get("input") == {"n": 3})

        # …and the NEXT session re-analyses it, consults kb guidance, and succeeds
        asked = []

        async def guide(q):
            asked.append(q)
            return "Do not raise; return the JSON result."
        cj = _stub([
            {"diagnosis": "it raises instead of returning", "adjustment": "return JSON",
             "kb_question": "how should a tool report results?"},
            META_GOOD])
        ct = _tstub([fenced(GOOD)])
        r2 = await ts.build_next(box, cj, ct, guidance=guide)
        check("a failed spec is re-analysed then built",
              r2["action"] == "built" and r2["attempts"] == 2 and box.has("doubler"))
        check("the analysis consulted kb guidance", bool(asked))
        check("the guidance and prior code reached the code step",
              any("Do not raise" in p for p in ct.seen)
              and any("raise RuntimeError" in p for p in ct.seen))

    # attempts budget → parked; requeue resets it
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("impossible thing", sketch="cannot work", source="toolsmith")
        for want in ("failed", "parked"):
            cj = _stub([{"diagnosis": "d", "adjustment": "a", "kb_question": ""},
                        META_GOOD, META_GOOD])
            r = await ts.build_next(box, cj, _tstub([fenced(RUNTIME_BAD)]),
                                    max_repair=0, max_attempts=2)
            check(f"attempt session ends {want}", r["action"] == want)
        i = box.ideas()[0]
        check("requeue puts a parked spec back with a fresh budget",
              box.requeue_idea(i["id"]).get("ok")
              and box.ideas()[0]["status"] == "proposed"
              and box.ideas()[0]["attempts"] == 0)

    # a name collision is auto-uniquified, never a wasted round
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("doubler", GOOD, {"description": "orig"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        box.add_idea("double again", sketch="2n", source="user")
        r = await ts.build_next(box, _stub([META_GOOD]), _tstub([fenced(GOOD)]))
        check("a name collision installs under a uniquified name",
              r["action"] == "built" and r["name"] == "doubler_2"
              and box.read("doubler")["manifest"].find('"orig"') > 0)

    # user-jotted specs are picked before toolsmith ones
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.add_idea("toolsmith spec first in", source="toolsmith", sketch="a")
        box.add_idea("user wish", source="user", sketch="double n")
        r = await ts.build_next(box, _stub([META_GOOD]), _tstub([fenced(GOOD)]))
        left = [i["title"] for i in box.ideas()]
        check("a user-jotted spec is built first",
              r["action"] == "built" and left == ["toolsmith spec first in"])

    # tool cap: build waits, spec stays queued
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("t_seed", GOOD, {"description": "x"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        box.add_idea("one more", sketch="s", source="user")
        r = await ts.build_next(box, _stub([]), _tstub([]), max_tools=1)
        check("at the tool cap the build waits",
              r["action"] == "none" and box.ideas()[0]["status"] == "proposed")

    # the run() orchestrator: both phases in one pass
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([
            {"deficit": True, "title": "double a number", "purpose": "given n return 2n",
             "rationale": "came up twice"},
            META_GOOD])
        st = await ts.run(box, cj, _tstub([fenced(GOOD)]))
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
