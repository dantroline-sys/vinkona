#!/usr/bin/env python
"""
Tests for capture.py — the durable orchestration-trace sink for the future skill-LoRA loop.

    python test_capture.py
"""

import datetime
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).parent

# stub aiohttp so llm_bridge imports (for the _capture_turn integration tests)
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


cap = _load("capture")
lb = _load("llm_bridge")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def _rec(c):
    return c.record(
        input_context={"user": "hi", "system": "SYSTEM PROMPT", "sections": {"recalled_memory": "m"},
                       "tools_offered": ["calculate"], "history_turns": 4},
        model_action={"response_text": "hello", "tool_calls": []},
        outcome={"json_parsed": None})


def test_writes_stamped_record():
    with tempfile.TemporaryDirectory() as d:
        c = cap.TraceCapture(d, format_version="v0-unfrozen", base_model="qwen3.5-9b", enabled=True)
        tid = _rec(c)
        check("record returns a trace_id", isinstance(tid, str) and len(tid) > 0)
        files = list(Path(d).glob("traces-*.jsonl"))
        check("a dated jsonl file is created", len(files) == 1)
        check("file name carries today's date",
              files[0].name == f"traces-{datetime.date.today().isoformat()}.jsonl")
        rec = json.loads(files[0].read_text().strip())
        check("record is stamped with format_version", rec["format_version"] == "v0-unfrozen")
        check("record is stamped with the base model", rec["base_model"] == "qwen3.5-9b")
        check("record carries trace_id + ts", rec["trace_id"] == tid and "ts" in rec)
        check("input_context preserved (the assembled surface form)",
              rec["input_context"]["system"] == "SYSTEM PROMPT")
        check("model_action preserved", rec["model_action"]["response_text"] == "hello")
        check("outcome preserved", "json_parsed" in rec["outcome"])


def test_append_only():
    with tempfile.TemporaryDirectory() as d:
        c = cap.TraceCapture(d, enabled=True)
        _rec(c); _rec(c); _rec(c)
        f = next(Path(d).glob("traces-*.jsonl"))
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        check("three records append to three lines", len(lines) == 3)
        check("every line is valid JSON", all(json.loads(l) for l in lines))


def test_disabled_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        c = cap.TraceCapture(d, enabled=False)
        tid = _rec(c)
        check("disabled record returns None", tid is None)
        check("disabled writes no file", list(Path(d).glob("*.jsonl")) == [])


def test_bad_dir_is_soft():
    # A path whose parent is a FILE can't be made a dir → record returns None, never raises.
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "afile"
        blocker.write_text("x")
        c = cap.TraceCapture(blocker / "sub", enabled=True)
        check("unwritable dir → None, no exception", _rec(c) is None)


# ── bridge integration: _capture_turn assembles the record from the turn variables ──
class _FakeCapture:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = []

    def record(self, **kw):
        self.records.append(kw)
        return "tid"


def _bridge_shim(capture):
    sh = types.SimpleNamespace(capture=capture, _turn_tool_calls=[], _roleplay=False,
                               history=[1, 2, 3, 4])
    sh._capture_turn = lb.LLMBridge._capture_turn.__get__(sh)
    return sh


def test_capture_turn_builds_record():
    c = _FakeCapture(enabled=True)
    sh = _bridge_shim(c)
    sh._turn_tool_calls = [{"name": "calculate", "arguments": '{"expression":"2+2"}'}]
    sh._capture_turn(user_text="what's 2+2", system="SYS", mem="M", working_memory="WM",
                     live_guidance="LG", briefing="BRIEF",
                     tools=[{"function": {"name": "calculate"}}], response_text="four")
    check("a record is written", len(c.records) == 1)
    r = c.records[0]
    check("the assembled surface form is captured", r["input_context"]["system"] == "SYS")
    check("sections are captured", r["input_context"]["sections"]["briefing"] == "BRIEF")
    check("tools_offered captured", r["input_context"]["tools_offered"] == ["calculate"])
    check("tool_calls captured in the action",
          r["model_action"]["tool_calls"][0]["name"] == "calculate")
    check("json_parsed is True for well-formed args", r["outcome"]["json_parsed"] is True)


def test_capture_turn_json_parsed_false():
    c = _FakeCapture(enabled=True)
    sh = _bridge_shim(c)
    sh._turn_tool_calls = [{"name": "calculate", "arguments": "{bad json"}]
    sh._capture_turn(user_text="x", system="S", mem="", working_memory="", live_guidance="",
                     briefing="", tools=[], response_text="r")
    check("json_parsed is False for malformed args",
          c.records[0]["outcome"]["json_parsed"] is False)


def test_capture_turn_gating():
    c = _FakeCapture(enabled=False)
    sh = _bridge_shim(c)
    sh._capture_turn(user_text="x", system="S", mem="", working_memory="", live_guidance="",
                     briefing="", tools=[], response_text="r")
    check("a disabled capture object writes nothing", c.records == [])
    _bridge_shim(None)._capture_turn(user_text="x", system="S", mem="", working_memory="",
                                     live_guidance="", briefing="", tools=[], response_text="r")
    check("capture=None is a silent no-op", True)


def main():
    test_writes_stamped_record()
    test_append_only()
    test_disabled_writes_nothing()
    test_bad_dir_is_soft()
    test_capture_turn_builds_record()
    test_capture_turn_json_parsed_false()
    test_capture_turn_gating()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
