#!/usr/bin/env python
"""
Tests for the standalone knowledge-host wiring:

  • knowledge_host.KnowledgeHost — the async client (double-JSON unwrap, ask/search args,
    fail-soft on raised errors / non-200 / ok=false / no-url).
  • _Session._guidance — the cascade's briefing hook: confidence gating + the formatting
    of kb_ask / kb_search results into a compact guidance block.

Like the other suites it runs on a bare interpreter: aiohttp + numpy are stubbed (sys.modules)
and the fake session returns canned knowledge-host responses — no live host, no GPU.

    python test_knowledge_host.py
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent

# ── stub aiohttp (+ numpy, which cascade_server imports) before loading the modules ──
CALLS: list = []                       # (tool_name, arguments) the client POSTed
STATE = {"mode": "ok"}                 # ok | raise | non200 | notok | badjson


def _payload_for(name, args):
    if name == "kb_ask":
        return {"query": args.get("query"), "confidence": 0.72, "abstain": False,
                "grounding": "strong",
                "items": [
                    {"kind": "node", "text": "De-escalate before problem-solving.",
                     "label": "Frustrated user",
                     "steps": ["Acknowledge the feeling", "Confirm the goal",
                               "Offer one concrete next step"]},
                    {"kind": "node", "text": "Surface the deadline proactively.", "label": ""},
                ]}
    if name == "kb_search":
        return {"passages": [{"text": "Offer to set a reminder.", "title": "Assistant playbook",
                              "source_type": "doc", "score": 0.66}],
                "confidence": 0.66, "low_confidence": False, "dense_used": True}
    return {}


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self):
        if STATE["mode"] == "badjson":
            raise ValueError("bad json")
        return self._body


class _FakeSession:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def close(self): pass

    def post(self, url, data=None, headers=None, timeout=None):
        body = json.loads(data.decode()) if data else {}
        CALLS.append((body.get("name"), body.get("arguments")))
        if STATE["mode"] == "raise":
            raise aiohttp_stub.ClientError("unreachable")
        if STATE["mode"] == "non200":
            return _FakeResp(503, {})
        if STATE["mode"] == "notok":
            return _FakeResp(200, {"ok": False, "error": "boom"})
        return _FakeResp(200, {"ok": True, "result": json.dumps(_payload_for(body.get("name"),
                                                                             body.get("arguments") or {}))})


aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientSession = _FakeSession
aiohttp_stub.ClientTimeout = lambda **k: None
aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
aiohttp_stub.web = types.ModuleType("aiohttp.web")   # cascade_server does `from aiohttp import web`
sys.modules["aiohttp"] = aiohttp_stub
sys.modules.setdefault("numpy", types.ModuleType("numpy"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kh_mod = _load("knowledge_host")
cascade = _load("cascade_server")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def test_client_ask_and_search():
    async def run():
        CALLS.clear(); STATE["mode"] = "ok"
        c = kh_mod.KnowledgeHost("http://h", timeout_s=2.0)
        check("client reports enabled when a url is set", c.enabled)
        bundle = await c.ask("user seems annoyed", rigor="high")
        check("ask() unwraps the double-JSON to the tool result",
              bundle is not None and bundle.get("confidence") == 0.72)
        check("ask() forwards rigor=high",
              CALLS[-1] == ("kb_ask", {"query": "user seems annoyed", "rigor": "high"}))
        await c.ask("x")
        check("ask() omits rigor when low", CALLS[-1][1] == {"query": "x"})
        res = await c.search("deadline help", intent="what to do next", k=3)
        check("search() returns passages", res is not None and res["passages"])
        check("search() forwards intent + k",
              CALLS[-1] == ("kb_search",
                            {"query": "deadline help", "k": 3, "intent": "what to do next"}))
    asyncio.run(run())


def test_client_fail_soft():
    async def run():
        c = kh_mod.KnowledgeHost("http://h", timeout_s=1.0)
        STATE["mode"] = "raise"
        check("a raised client error → None (never propagates)", (await c.ask("x")) is None)
        STATE["mode"] = "non200"
        check("non-200 → None", (await c.ask("x")) is None)
        STATE["mode"] = "notok"
        check("ok=false → None", (await c.ask("x")) is None)
        STATE["mode"] = "badjson"
        check("unparseable result → None", (await c.ask("x")) is None)
        STATE["mode"] = "ok"
        check("empty query → None (no call)", (await c.ask("   ")) is None)
        c2 = kh_mod.KnowledgeHost("", timeout_s=1.0)
        check("no url → not enabled", not c2.enabled)
        check("no url → None", (await c2.search("x")) is None)
    asyncio.run(run())


def _shim(*, has_host=True, tool="kb_ask", min_conf=0.30):
    sh = types.SimpleNamespace()
    sh.cfg = {"knowledge_host": {"tool": tool, "rigor": "low", "k": 4, "min_confidence": min_conf}}
    sh._http = _FakeSession()
    sh.s = types.SimpleNamespace(trace=False)
    sh._kh = kh_mod.KnowledgeHost("http://h", timeout_s=2.0) if has_host else None
    sh._last_user_line = cascade._Session._last_user_line
    sh._looks_like_question = cascade._Session._looks_like_question
    sh._query_nucleus = cascade._Session._query_nucleus
    sh._trace_kb = cascade._Session._trace_kb.__get__(sh)   # no-op: sh.s.trace is False
    sh._guidance = cascade._Session._guidance.__get__(sh)
    return sh


def test_guidance_formats_kb_ask():
    async def run():
        CALLS.clear(); STATE["mode"] = "ok"
        sh = _shim(tool="kb_ask")
        block = await sh._guidance("ASSISTANT: hi\nUSER: I'm so done with this\nASSISTANT: ok")
        check("guidance queries the LAST user line",
              CALLS and CALLS[-1][1]["query"] == "I'm so done with this")
        check("guidance renders the item label + text",
              block and "Frustrated user: De-escalate before problem-solving." in block)
        check("guidance renders procedure steps as bullets",
              block and "• Acknowledge the feeling" in block and "• Offer one concrete next step" in block)
    asyncio.run(run())


def test_guidance_formats_kb_search():
    async def run():
        CALLS.clear(); STATE["mode"] = "ok"
        sh = _shim(tool="kb_search")
        block = await sh._guidance("USER: I keep forgetting the deadline")
        check("kb_search guidance uses the intent payload",
              CALLS[-1][0] == "kb_search" and "intent" in CALLS[-1][1])
        check("kb_search guidance renders a cited passage",
              block and "Offer to set a reminder." in block and "[Assistant playbook]" in block)
    asyncio.run(run())


def test_guidance_gate_and_disabled():
    async def run():
        STATE["mode"] = "ok"
        sh = _shim(tool="kb_ask", min_conf=0.95)        # above the fake's 0.72
        check("weak confidence is gated to None", (await sh._guidance("USER: help me")) is None)
        sh2 = _shim(has_host=False)
        check("no knowledge-host → None", (await sh2._guidance("USER: help me")) is None)
    asyncio.run(run())


def test_guidance_live_mode():
    async def run():
        CALLS.clear(); STATE["mode"] = "ok"
        sh = _shim(tool="kb_ask")
        # A question turn → ONE crisp single-line directive (no multi-item block, no bullets)
        block = await sh._guidance("how do I calm them down?", live=True)
        check("live returns a single concise line (label: text → step)",
              block == "Frustrated user: De-escalate before problem-solving. → Acknowledge the feeling")
        check("live block has no bullet list", block and "•" not in block and "\n" not in block)
        check("live block respects the 240-char cap", block and len(block) <= 240)
        # A non-question turn → the fast pull stays silent (no KH call, no latency)
        CALLS.clear()
        none = await sh._guidance("I had a rough day at work", live=True)
        check("live is gated to question turns (statement → None)", none is None)
        check("live gate fires BEFORE any knowledge-host call", CALLS == [])
        # kb_search live → just the top passage text, no citation tag
        shs = _shim(tool="kb_search")
        sblock = await shs._guidance("what should I do about the deadline?", live=True)
        check("live kb_search returns the bare top passage",
              sblock == "Offer to set a reminder.")
    asyncio.run(run())


def main():
    test_client_ask_and_search()
    test_client_fail_soft()
    test_guidance_formats_kb_ask()
    test_guidance_formats_kb_search()
    test_guidance_gate_and_disabled()
    test_guidance_live_mode()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
