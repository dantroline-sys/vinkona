#!/usr/bin/env python
"""Tests for toolsmith.py — Vinkona's idle tool-maker.

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


def _stub(replies):
    it = iter(replies)

    async def chat_json(prompt, think=True):
        try:
            return next(it)
        except StopIteration:
            return None
    return chat_json


def _box(td, **extra):
    return tb.Toolbox(Path(td) / "own", cfg={**CFG, **extra}, seed=False)


async def _main():
    # 1) a buildable decision → written, self-tested, installed
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([
            {"decision": "build", "name": "doubler", "description": "double a number",
             "plan": "n -> 2n"},
            {"code": GOOD, "parameters": {"type": "object",
             "properties": {"n": {"type": "integer"}}},
             "test": {"input": {"n": 3}, "expect_keys": ["double"]}}])
        r = await ts.run(box, cj)
        check("a buildable tool is built and installed",
              r["action"] == "built" and box.has("doubler"))
        check("the built tool actually runs",
              box.call("doubler", {"n": 5}).get("result") == {"double": 10})

    # 2) an idea she can't build → recorded, nothing installed
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([{"decision": "idea", "title": "OCR scanned PDFs",
                     "rationale": "needs a vision model"}])
        r = await ts.run(box, cj)
        check("an idea is recorded", r["action"] == "idea"
              and [i["title"] for i in box.ideas()] == ["OCR scanned PDFs"])

    # 3) first code is broken → repaired on the second attempt
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([
            {"decision": "build", "name": "doubler", "description": "d", "plan": "p"},
            {"code": BAD, "test": {"input": {"n": 1}}},
            {"code": GOOD, "parameters": {"type": "object", "properties": {}},
             "test": {"input": {"n": 4}, "expect_keys": ["double"]}}])
        r = await ts.run(box, cj, max_repair=2)
        check("a broken tool is repaired then installed",
              r["action"] == "built" and r["attempts"] == 2 and box.has("doubler"))

    # 4) never passes its self-test → downgraded to an idea, not installed
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        cj = _stub([
            {"decision": "build", "name": "broken", "description": "always broken",
             "plan": "p", "rationale": "would help"},
            {"code": BAD, "test": {"input": {}}},
            {"code": BAD, "test": {"input": {}}},
            {"code": BAD, "test": {"input": {}}}])
        r = await ts.run(box, cj, max_repair=2)
        check("an unbuildable tool degrades to an idea",
              r["action"] == "failed" and not box.has("broken") and len(box.ideas()) == 1)

    # 5) at the tool cap → a build request becomes an idea instead
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("t_seed", GOOD, {"description": "x"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        cj = _stub([{"decision": "build", "name": "newone", "description": "blocked by cap",
                     "plan": "p", "rationale": "r"}])
        r = await ts.run(box, cj, max_tools=1)
        check("at the cap, a build becomes an idea",
              r["action"] == "idea" and not box.has("newone"))

    # 6) a name collision with an existing tool → banked as an idea, never clobbers
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("doubler", GOOD, {"description": "orig"},
                    {"input": {"n": 1}, "expect_keys": ["double"]})
        cj = _stub([{"decision": "build", "name": "doubler", "description": "collide",
                     "plan": "p"}])
        r = await ts.run(box, cj)
        check("a name collision is banked, not clobbered",
              r["action"] == "idea"
              and box.read("doubler")["manifest"].find('"orig"') > 0)


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
