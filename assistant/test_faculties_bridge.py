#!/usr/bin/env python
"""End-to-end test of the faculties-RPC bridge in llm_bridge.

A sandboxed tool runs off the event loop (asyncio.to_thread) and calls one of Vinkona's
faculties; the request must marshal back onto the loop, run, and return — through the REAL
LLMBridge._ensure_faculty_dispatch / _faculty_call (only _call_tool is stubbed, so no LM or
tool host is needed).  Also checks the no-re-entrancy guard (a tool can't call an own-tool
as a faculty).

    python test_faculties_bridge.py
"""
import asyncio
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

# llm_bridge imports numpy + aiohttp at module load; neither is needed on the faculty path,
# so stub them for a bare interpreter (the real _call_tool is replaced in the test anyway).
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
if "aiohttp" not in sys.modules:
    _ah = types.ModuleType("aiohttp")
    _ah.ClientSession = object
    _ah.ClientTimeout = lambda **k: None
    _ah.ClientConnectorError = type("ClientConnectorError", (Exception,), {})
    sys.modules["aiohttp"] = _ah

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tb = _load("toolbox")
lb = _load("llm_bridge")

if tb._ContainerBackend({}).available():
    BACKEND = "container"
elif tb._BwrapBackend().available():
    BACKEND = "bwrap"
else:
    BACKEND = None

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


_CALLER = ("import faculties\n"
           "a = faculties.args()\n"
           "try:\n"
           "    r = faculties.call(a['tool'], a.get('args', {}))\n"
           "    faculties.done({'got': r})\n"
           "except Exception as e:\n"
           "    faculties.done({'err': str(e)})\n")


def _bridge(box):
    """A bare LLMBridge with just the attributes the faculty path touches; _call_tool is a
    stub standing in for the real built-in/host dispatch."""
    br = lb.LLMBridge.__new__(lb.LLMBridge)
    br.own_tools = box
    br._faculty_bound = False

    async def _stub_call_tool(name, args):
        if name == "calculate":
            return "42"
        return f"(no tool named {name})"
    br._call_tool = _stub_call_tool
    return br


async def _main():
    with tempfile.TemporaryDirectory() as td:
        cfg = {"backend": BACKEND,
               "faculties": {"enabled": True, "allow": ["calculate"], "max_calls": 4}}
        box = tb.Toolbox(Path(td) / "own", cfg=cfg, seed=False)
        box.install("caller", _CALLER,
                    {"description": "calls a faculty", "uses_faculties": True},
                    {"input": {"tool": "calculate"}, "faculty_stubs": {"calculate": "42"},
                     "expect_keys": ["got"]})
        # also install a trivial own-tool to prove re-entrancy is refused
        box.install("mini", "import sys, json\njson.load(sys.stdin)\nprint(json.dumps({'x': 1}))",
                    {"description": "mini"}, {"input": {}, "expect_keys": ["x"]})

        br = _bridge(box)
        br._ensure_faculty_dispatch()            # captures THIS running loop

        # the tool (in a worker thread) calls calculate, which must run back on the loop
        res = await asyncio.to_thread(box.call, "caller", {"tool": "calculate"})
        check("a tool's faculty call marshals to the loop and back",
              res.get("result") == {"got": "42"})

        # re-entrancy guard: an own-tool may not be called as a faculty (even if the runner
        # were to permit the name, _faculty_call refuses)
        guard = await br._faculty_call("mini", {})
        check("calling an own-tool as a faculty is refused",
              not guard.get("ok") and "own tool" in guard.get("error", ""))

        # a name the stub _call_tool doesn't know still returns cleanly (not a crash)
        unknown = await br._faculty_call("calculate", {})
        check("a known faculty routes through _call_tool", unknown == {"ok": True, "result": "42"})


def main():
    if BACKEND is None:
        print("SKIP  faculties bridge (no sandbox backend)")
        print("\n0 passed, 0 failed, 1 skipped")
        return
    asyncio.run(_main())
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
