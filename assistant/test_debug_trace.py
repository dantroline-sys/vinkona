#!/usr/bin/env python
"""The copy-pastable turn trace (GET /api/debug) + long-form question detection.

The debug view is the diagnosis tool for "that turn went wrong": it renders the trace
feed as a readable transcript of the fast/big-LM to-and-fro — what each tier was handed,
every tool call and its result, escalation timings, and WHICH stage stalled.  Runs on a
bare interpreter (no server, no model).

    python assistant/test_debug_trace.py
"""
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load("config_server")
bridge = _load("llm_bridge")
H = next(v for v in vars(cs).values() if hasattr(v, "_render_debug"))
B = bridge.LLMBridge

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


T = 1000.0
# A turn that escalated and gave up — the exact shape Dan needs to diagnose.
STALLED = [
    {"ts": T, "kind": "turn", "session": "ab12cd34", "model": "fast-9b",
     "user": "is a beta blocker a good choice?", "system": "SYSTEM PROMPT BODY",
     "recalled": "- Dan is an anaesthetist", "briefing": "Keep it practical.",
     "history_turns": 6, "tools_offered": ["kb_ask", "kb_reason"]},
    {"ts": T + 0.4, "kind": "kb_call", "tool": "kb_ask", "live": True,
     "query": "beta blocker tachycardia", "confidence": 0.22, "abstain": True, "result": ""},
    {"ts": T + 0.9, "kind": "deliberate", "trigger": "tool"},
    {"ts": T + 9.1, "kind": "deliberate_knowledge", "chars": 812},
    {"ts": T + 39.4, "kind": "deliberate_done", "ok": False, "elapsed_s": 38.5,
     "stage": "thinking", "longform": False},
    {"ts": T + 40.0, "kind": "fast_reply", "model": "fast-9b",
     "text": "Sorry — I'm still chewing on that one.", "first_token_ms": 180.0},
]
# A clean turn that used a tool and answered.
CLEAN = [
    {"ts": T + 100, "kind": "turn", "session": "ab12cd34", "model": "fast-9b",
     "user": "what's on tomorrow?", "system": "SYS", "history_turns": 2,
     "tools_offered": ["read_calendar"]},
    {"ts": T + 100.2, "kind": "tool_call", "name": "read_calendar", "arguments": {"day": "+1"}},
    {"ts": T + 100.8, "kind": "tool_result", "name": "read_calendar", "result": "09:00 list"},
    {"ts": T + 101.5, "kind": "fast_reply", "text": "You're on a list at nine.",
     "first_token_ms": 140.0},
]


def main():
    txt = H._render_debug(H, STALLED, 1)

    # ── the transcript answers "what happened on this turn?" ─────────────────
    check("the user's words head the turn", "is a beta blocker a good choice?" in txt)
    check("the fast LM's system prompt is included VERBATIM (fenced)",
          "SYSTEM PROMPT BODY" in txt and "```" in txt)
    check("what was recalled is shown", "Dan is an anaesthetist" in txt)
    check("the planner briefing in play is shown", "Keep it practical." in txt)
    check("the tools actually offered are listed", "kb_ask, kb_reason" in txt)

    # ── the KB leg: the thing Dan suspects first ─────────────────────────────
    check("the kb call shows its query, confidence and abstain",
          "beta blocker tachycardia" in txt and "conf=0.22" in txt and "abstain=True" in txt)

    # ── the escalation: which stage stalled, and how long each leg took ──────
    check("escalation to the big LM is marked with its trigger",
          "ESCALATED to the big LM" in txt and "trigger: tool" in txt)
    check("the KB pull for the think is shown with its size", "812 chars" in txt)
    check("the give-up is unmissable, with elapsed and STAGE",
          "GAVE UP" in txt and "38.5s" in txt and "stage=`thinking`" in txt)
    check("relative timings make the slow leg obvious (+39.4s)", "+ 39.4s" in txt)
    check("what was finally said is included", "still chewing" in txt)

    # ── tool legs render too ─────────────────────────────────────────────────
    txt2 = H._render_debug(H, CLEAN, 1)
    check("a tool call shows its name and arguments",
          "call `read_calendar`" in txt2 and '"day": "+1"' in txt2)
    check("a tool result shows what came back", "09:00 list" in txt2)

    # ── turn splitting + bounds ──────────────────────────────────────────────
    both = H._render_debug(H, STALLED + CLEAN, 2)
    check("consecutive turns are split apart", "## Turn 1" in both and "## Turn 2" in both)
    one = H._render_debug(H, STALLED + CLEAN, 1)
    check("asking for 1 turn returns only the newest",
          "## Turn 1" in one and "## Turn 2" not in one and "what's on tomorrow?" in one)
    check("noise outside the turn kinds is dropped",
          "## Turn" in H._render_debug(H, [{"ts": T, "kind": "scheduler_tick"}] + CLEAN, 1))
    big = H._render_debug(H, [dict(STALLED[0], system="x" * 20000)], 1)
    check("a huge block is truncated with a visible marker, not dumped whole",
          "more chars]" in big and len(big) < 12000)
    check("an empty feed doesn't explode", isinstance(H._render_debug(H, [], 1), str))

    # ── long-form detection: the classifier that picks the deep path ─────────
    for q in ("okay so what drug is the best for this. is a beta blocker a good choice, "
              "or something else?",
              "what are my options?",
              "is this good or bad and what are the reasons and possible actions?",
              "what should I use instead?",
              "why might that not work out?"):
        check(f"long-form: {q[:44]}…", B._is_longform(B, q) is True)
    for q in ("what time is it?", "thanks", "turn the lights off", "play some music",
              "is that ok?", "remind me at five"):
        check(f"NOT long-form: {q}", B._is_longform(B, q) is False)

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"test_debug_trace: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
