#!/usr/bin/env python
"""
Tests for two bridge features:
  • Ephemeral within-conversation working memory — _absorb_facts (parse the big LM's
    `FACT:` lines, rewrite-the-scratchpad) and _working_memory_block (always-inject).
  • The in-process sympy `calculate` tool — guards (empty / too long), the result
    formatting in _sympy_eval (sympy stubbed here; real sympy validates on the box),
    and the CALCULATE_TOOL shape.

aiohttp is stubbed so llm_bridge imports on a bare interpreter.

    python test_working_memory_and_calc.py
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent

# stub aiohttp so llm_bridge imports without the real dependency
_a = types.ModuleType("aiohttp")
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
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def _wm_shim(max_items=12):
    sh = types.SimpleNamespace(working_memory_on=True, working_memory_max=max_items,
                               _working_memory={})
    sh._absorb_facts = lb.LLMBridge._absorb_facts.__get__(sh)
    sh._working_memory_block = lb.LLMBridge._working_memory_block.__get__(sh)
    return sh


def test_absorb_facts_and_strip():
    sh = _wm_shim()
    briefing = ("Keep it warm and ask about the trip.\n"
                "FACT: budget: agreed at £400\n"
                "FACT: meeting spot: the cafe on Mill Road\n")
    clean = sh._absorb_facts(briefing)
    check("FACT lines are absorbed into the blackboard",
          sh._working_memory == {"budget": "agreed at £400",
                                 "meeting spot": "the cafe on Mill Road"})
    check("FACT lines are stripped from the briefing",
          "FACT:" not in clean and clean.startswith("Keep it warm"))


def test_absorb_rewrite_replaces_whole_set():
    sh = _wm_shim()
    sh._absorb_facts("FACT: budget: £400\nFACT: place: cafe")
    # next turn: budget changed, place dropped, new fact added → blackboard replaced
    sh._absorb_facts("FACT: budget: £450\nFACT: deadline: Friday")
    check("an emitted set replaces the blackboard wholesale (supersede + drop)",
          sh._working_memory == {"budget": "£450", "deadline": "Friday"})


def test_absorb_empty_keeps_board():
    sh = _wm_shim()
    sh._absorb_facts("FACT: budget: £400")
    sh._absorb_facts("Just a normal briefing with no facts this turn.")
    check("a turn with no FACT lines does NOT wipe the board",
          sh._working_memory == {"budget": "£400"})


def test_absorb_cap():
    sh = _wm_shim(max_items=2)
    sh._absorb_facts("FACT: a: 1\nFACT: b: 2\nFACT: c: 3")
    check("the blackboard is capped to max_items (most recent kept)",
          sh._working_memory == {"b": "2", "c": "3"})


def test_absorb_drops_placeholder_values():
    sh = _wm_shim()
    sh._absorb_facts("FACT: budget: £400\nFACT: stale: none")
    check("placeholder values ('none', '-') are not stored",
          sh._working_memory == {"budget": "£400"})


def test_working_memory_block():
    sh = _wm_shim()
    check("empty board renders nothing", sh._working_memory_block() == "")
    sh._working_memory = {"budget": "£400", "place": "cafe"}
    block = sh._working_memory_block()
    check("block lists every fact (full, not windowed)",
          "- budget: £400" in block and "- place: cafe" in block)
    check("block tells the model these override assumptions",
          "stay consistent" in block.lower())


def test_disabled_is_inert():
    sh = types.SimpleNamespace(working_memory_on=False, working_memory_max=12, _working_memory={})
    sh._absorb_facts = lb.LLMBridge._absorb_facts.__get__(sh)
    sh._working_memory_block = lb.LLMBridge._working_memory_block.__get__(sh)
    out = sh._absorb_facts("FACT: x: 1")
    check("disabled: FACT lines pass through untouched and nothing is stored",
          out == "FACT: x: 1" and sh._working_memory == {})
    check("disabled: block is empty", sh._working_memory_block() == "")


# ── calculator ──────────────────────────────────────────────────────────────
def _calc_shim():
    sh = types.SimpleNamespace()
    sh._calculate = lb.LLMBridge._calculate.__get__(sh)
    return sh


def test_calc_guards():
    sh = _calc_shim()
    async def run():
        check("empty expression is rejected", (await sh._calculate({})) == "(no expression given)")
        check("over-long expression is rejected",
              "too long" in (await sh._calculate({"expression": "1+" * 200})))
    asyncio.run(run())


def test_calculate_tool_shape():
    fn = lb.CALCULATE_TOOL["function"]
    check("tool is named 'calculate'", fn["name"] == "calculate")
    check("tool requires an 'expression' arg",
          fn["parameters"]["required"] == ["expression"])


def _install_fake_sympy():
    class FakeSym:
        def __init__(self, n): self.name = n
        def __str__(self): return self.name

    class FakeExpr:
        def __init__(self, s, syms=()): self._s = s; self.free_symbols = set(syms)
        def __str__(self): return self._s

    sm = types.ModuleType("sympy")
    sm.N = lambda e, p=12: f"{e}~={p}"
    sm.Eq = lambda a, b: ("Eq", a, b)
    sm.solve = lambda eq, sym: ["2", "3"]
    sm.Symbol = lambda n: FakeSym(n)
    parser = types.ModuleType("sympy.parsing.sympy_parser")
    parser.parse_expr = (lambda s, transformations=None, evaluate=True:
                         FakeExpr(s.strip(), syms=[FakeSym("x")] if any(c.isalpha() for c in s) else []))
    parser.standard_transformations = ()
    parser.implicit_multiplication_application = "imul"
    parser.convert_xor = "xor"
    sys.modules["sympy"] = sm
    sys.modules["sympy.parsing"] = types.ModuleType("sympy.parsing")
    sys.modules["sympy.parsing.sympy_parser"] = parser


def test_sympy_eval_formatting():
    if lb._SYMPY_OK:
        check("real sympy: 2+2 = 4", lb._sympy_eval("2+2") == "4")
        check("real sympy: exact fraction kept", lb._sympy_eval("1/2 + 1/3") == "5/6")
        check("real sympy: ^ is power", lb._sympy_eval("2^10").startswith("1024"))
    else:
        _install_fake_sympy()
        try:
            out = lb._sympy_eval("2+2")
            check("formats exact with a decimal approximation when they differ",
                  out == "2+2  (≈ 2+2~=12)")
            sol = lb._sympy_eval("solve x**2 - 5*x + 6")
            check("the 'solve' convenience returns roots",
                  "→" in sol and "2" in sol and "3" in sol)
        finally:
            for m in ("sympy", "sympy.parsing", "sympy.parsing.sympy_parser"):
                sys.modules.pop(m, None)


def main():
    test_absorb_facts_and_strip()
    test_absorb_rewrite_replaces_whole_set()
    test_absorb_empty_keeps_board()
    test_absorb_cap()
    test_absorb_drops_placeholder_values()
    test_working_memory_block()
    test_disabled_is_inert()
    test_calc_guards()
    test_calculate_tool_shape()
    test_sympy_eval_formatting()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
