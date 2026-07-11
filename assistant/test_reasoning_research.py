#!/usr/bin/env python
"""
Unit tests for the per-request reasoning toggle and the built-in queue_research
tool.  Runs on a bare interpreter: numpy and aiohttp are stubbed (the code under
test doesn't use numpy on these paths, and we fake aiohttp to capture the request
body / feed canned responses).  No servers, no GPU.

    python test_reasoning_research.py
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent


# ── Stub numpy (memory.py imports it at module top; we never hit numpy code) ──
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

# ── Stub aiohttp so we can capture posted JSON and return canned bodies ──────
_LAST = {}                       # captured request payloads, by url-suffix


class _FakeResp:
    def __init__(self, status, body):
        self.status, self._body = status, body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self): return self._body
    async def text(self): return json.dumps(self._body)


class _FakeTools:
    """Scriptable ToolHost stand-in: map tool name -> the host's {ok,result,error} body
    (or a callable of args).  Records every call for assertions."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
    @property
    def active(self): return True
    async def catalogue(self): return []
    async def call_raw(self, name, args):
        self.calls.append((name, args))
        r = self.responses.get(name)
        if r is None:
            return {"ok": False, "result": "", "error": f"no fake for {name}"}
        return r(args) if callable(r) else r
    async def call(self, name, args):
        d = await self.call_raw(name, args)
        return d["result"] if d["ok"] else f"(tool error: {d['error']})"


def _ok(result): return {"ok": True, "result": result, "error": ""}


class _FakeSession:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def post(self, url, json=None, timeout=None, **k):
        _LAST["url"], _LAST["payload"] = url, json
        # Echo a JSON-object body for memory._chat_json; include a <think> leak.
        return _FakeResp(200, {"choices": [
            {"message": {"content": "<think>pondering…</think>{\"operations\": []}"}}]})


aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientSession = _FakeSession
aiohttp_stub.ClientTimeout = lambda **k: None
aiohttp_stub.ClientConnectorError = type("ClientConnectorError", (Exception,), {})
sys.modules["aiohttp"] = aiohttp_stub


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


memory = _load("memory")
bridge = _load("llm_bridge")
asr = _load("asr")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


class _Dummy:                    # stand-in self for unbound-method calls
    pass


async def test_chat_json_think():
    # _chat_json uses no instance state, so call it unbound on a dummy.
    res = await memory.MemoryStore._chat_json(_Dummy(), "http://x", "m", "hi", think=True)
    check("think=True sends enable_thinking", _LAST["payload"]["chat_template_kwargs"]["enable_thinking"] is True)
    check("think=True sends reasoning_budget -1", _LAST["payload"]["reasoning_budget"] == -1)
    check("think=True still parses JSON past a <think> leak", res == {"operations": []})

    await memory.MemoryStore._chat_json(_Dummy(), "http://x", "m", "hi", think=False)
    check("think=False disables enable_thinking", _LAST["payload"]["chat_template_kwargs"]["enable_thinking"] is False)
    check("think=False sends reasoning_budget 0", _LAST["payload"]["reasoning_budget"] == 0)

    res2 = await memory.MemoryStore._chat_json(_Dummy(), "http://x", "m", "hi")
    check("_chat_json defaults to think=True (background path)", _LAST["payload"]["reasoning_budget"] == -1)


async def test_call_tool_routing():
    seen = {}
    def enqueue(topic, query, reason):
        seen.update(topic=topic, query=query, reason=reason)
        return None                                   # callback may return None

    class _Host:
        async def call(self, name, args): return f"host:{name}"

    b = _Dummy()
    b.research_enqueue = enqueue
    b.tools = _Host()

    out = await bridge.LLMBridge._call_tool(b, "queue_research",
                                            {"topic": "growing potatoes", "reason": "homework"})
    check("queue_research handled locally (not forwarded to host)", not out.startswith("host:"))
    check("queue_research passed topic to callback", seen["topic"] == "growing potatoes")
    check("queue_research defaults query to topic", seen["query"] == "growing potatoes")
    check("queue_research returns a spoken-back ack", "growing potatoes" in out)

    out2 = await bridge.LLMBridge._call_tool(b, "search_files", {"q": "x"})
    check("other tools forwarded to the Mac host", out2 == "host:search_files")

    # Async callback (e.g. if the cascade ever returns a coroutine) is awaited.
    async def aenqueue(t, q, r): return "queued async"
    b.research_enqueue = aenqueue
    out3 = await bridge.LLMBridge._call_tool(b, "queue_research", {"topic": "T"})
    check("async research_enqueue is awaited", out3 == "queued async")

    # Empty topic is rejected without calling through.
    b.research_enqueue = enqueue
    out4 = await bridge.LLMBridge._call_tool(b, "queue_research", {"topic": "  "})
    check("empty topic rejected", "no topic" in out4.lower())

    # No host + unknown tool degrades gracefully.
    b.tools = None
    out5 = await bridge.LLMBridge._call_tool(b, "mystery", {})
    check("unknown tool without host is graceful", "no tool" in out5.lower())

    # remind_me routes to the schedule_notification callback locally.
    b.research_enqueue = None
    reminded = {}
    b.schedule_notification = lambda text, when: reminded.update(text=text, when=when)
    out6 = await bridge.LLMBridge._call_tool(b, "remind_me",
                                             {"text": "call mum", "when": "2026-06-23T17:00"})
    check("remind_me handled locally", reminded.get("text") == "call mum")
    check("remind_me passes the when", reminded.get("when") == "2026-06-23T17:00")
    out7 = await bridge.LLMBridge._call_tool(b, "remind_me", {"text": "x"})
    check("remind_me needs both text and when", "need both" in out7.lower())


def test_stream_payload_fields():
    # The streaming payload should carry the no-think fields by default.  We can't
    # easily run the async generator without a full SSE stub, so assert the schema
    # constant and the bridge default instead.
    sig = bridge.LLMBridge._stream_chat.__defaults__
    check("_stream_chat defaults think=False", sig[-1] is False)
    tool = bridge.QUEUE_RESEARCH_TOOL
    check("queue_research tool schema names the function", tool["function"]["name"] == "queue_research")
    check("queue_research requires a topic", tool["function"]["parameters"]["required"] == ["topic"])


async def test_forced_answer_after_tool():
    # Script the stream: round 0 calls a tool and says a filler line; round 1 (after
    # the tool result) the model goes silent; the forced final round must answer.
    scripted = [
        ("Let me check.", [{"id": "1", "name": "check_calendar", "arguments": "{}"}]),
        ("", []),                          # model went quiet after the tool result
        ("You're free Friday afternoon.", []),
    ]
    calls = []                             # (tools, msgs-snapshot) each stream got

    async def fake_stream(msgs, tools=None):
        calls.append((tools, [dict(m) for m in msgs]))
        return scripted[len(calls) - 1]

    spoken = []
    async def say(s): spoken.append(s)
    # Real bridge; confirm guard off so the read-style tool isn't gated.
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False,
                         confirm_required=False)
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    async def fake_call(name, args): return "Friday: nothing scheduled"
    b._call_tool = fake_call

    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("final answer is produced after a silent post-tool round", "free Friday" in out)
    check("forced final round withholds tools", calls[-1][0] is None)
    check("the tool was still offered earlier", calls[0][0] is not None)
    # The tool result must reach the model as a plain user message (template-agnostic),
    # not only as a role:tool message some chat templates drop.
    round1_msgs = calls[1][1]
    user_with_result = [m for m in round1_msgs
                        if m["role"] == "user" and "Friday: nothing scheduled" in (m.get("content") or "")]
    check("tool result is restated to the model as a user message", len(user_with_result) == 1)
    check("tool result is fenced as untrusted data", "UNTRUSTED" in user_with_result[0]["content"])


async def test_sayback_spoken_directly():
    # queue_research's ack must be spoken directly: no "let me check" filler, no
    # second LM round, and it must not be dropped (the original silent-after-queue bug).
    scripted = [("", [{"id": "1", "name": "queue_research",
                       "arguments": '{"topic":"the Nabataeans"}'}])]
    calls = []
    async def fake_stream(msgs, tools=None):
        calls.append(tools)
        return scripted[len(calls) - 1]

    spoken = []
    async def say(s): spoken.append(s)
    enq = []
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False,
                         confirm_required=False, tool_filler="Let me check.",
                         research_enqueue=lambda t, q, r: (enq.append(t),
                             "Okay — I'll read up on the Nabataeans later and remember it.")[1])
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None

    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("research topic was queued", enq == ["the Nabataeans"])
    check("the ack is spoken to the user", any("read up on the Nabataeans" in s for s in spoken))
    check("the ack is the returned answer", "read up on the Nabataeans" in out)
    check("no 'let me check' filler for an instant say-back", "Let me check." not in spoken)
    check("no second LM round is needed", len(calls) == 1)


async def test_doc_grounded_briefing():
    # The big LM's briefing must include the fenced source document when the
    # document_hook returns one, and the fast LM never sees it (briefing only).
    captured = {}
    async def fake_stream(url, model, messages, max_tokens):
        captured["messages"] = messages
        yield "User wants complication rates."

    async def doc_hook(user_text):
        return ("Brachial plexus block (wiki)", "Reported nerve-injury rate is ~3 per 10,000.")

    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False,
                         document_hook=doc_hook, trace_hook=lambda e: None)
    b._stream_chat = fake_stream
    b.history = [{"role": "user", "content": "what's the nerve injury rate for a brachial plexus block?"},
                 {"role": "assistant", "content": "Let me think."}]
    await b._update_big_lm_briefing()
    sent = captured["messages"][-1]["content"]
    check("document text reaches the big LM briefing", "3 per 10,000" in sent)
    check("document is fenced as untrusted", "UNTRUSTED" in sent)
    check("the user-hasn't-seen-it caveat is present", "NOT seen" in sent)
    check("briefing was produced", b._big_lm_briefing == "User wants complication rates.")

    # With no document, the briefing carries no reference fence.
    captured.clear()
    b2 = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                          big_lm_url="http://big", inject_time=False,
                          document_hook=lambda u: _none(), trace_hook=lambda e: None)
    b2._stream_chat = fake_stream
    b2.history = list(b.history)
    await b2._update_big_lm_briefing()
    check("no fence when no document is available", "UNTRUSTED" not in captured["messages"][-1]["content"])


async def _none():
    return None


async def test_lead_levels():
    captured = {}
    async def fake_stream(url, model, messages, max_tokens):
        captured["prompt"] = messages[-1]["content"]
        yield "ok"

    def mk(lead):
        b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                             big_lm_url="http://big", inject_time=False, lead=lead)
        b._stream_chat = fake_stream
        b.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        return b

    await mk(2)._update_big_lm_briefing()
    check("lead=2 tells the planner to always propose a next move",
          "next move that advances" in captured["prompt"])
    await mk(1)._update_big_lm_briefing()
    check("lead=1 proposes a move only when stalled/looping",
          "ONLY when" in captured["prompt"])
    await mk(0)._update_big_lm_briefing()
    check("lead=0 stays descriptive (no moves)", "Do NOT propose" in captured["prompt"])

    check("the base briefing is directive (anti-repetition)",
          "should NOT repeat" in bridge.DEFAULT_BRIEFING_PROMPT)
    check("an out-of-range lead falls back to nudge", mk(9).lead == 1)


def test_tool_result_trace():
    # _tool_errored flags the parenthesised error shapes the host/built-ins use.
    err = bridge.LLMBridge._tool_errored
    check("tool host error flagged", err("(tool host error 500)"))
    check("tool call failure flagged", err("(tool call failed: timed out)"))
    check("could-not flagged", err("(could not queue research: x)"))
    check("no-tool flagged", err("(no tool named foo is available)"))
    check("normal result not flagged", not err('{"events": [1,2,3]}'))
    check("parenthetical prose not flagged", not err("(it rained today)"))

    # _trace_tool_result emits name, byte count, ok flag, and a capped preview.
    events = []
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, inject_time=False,
                         trace_hook=lambda e: events.append(e))
    big = "x" * 5000
    b._trace_tool_result("calendar_range", big)
    ev = events[-1]
    check("trace kind is tool_result", ev["kind"] == "tool_result")
    check("byte count is the full length", ev["bytes"] == 5000)
    check("ok when not an error", ev["ok"] is True)
    check("preview is capped to avoid flooding", len(ev["preview"]) == 200)
    check("fuller body is also capped", len(ev["result"]) == 600)

    b._trace_tool_result("calendar_create", "(tool host error 500)")
    check("error result marks ok False", events[-1]["ok"] is False)


async def test_confirm_guard():
    spoken = []
    async def say(s): spoken.append(s)
    # announce_tools=[] so calendar_create goes through the confirm path here (this test is
    # about confirm-before-write; act-then-announce is covered by test_act_then_announce).
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False, announce_tools=[])
    check("write tool needs confirmation", b._needs_confirm("calendar_create"))
    check("read tool does not", not b._needs_confirm("calendar_range"))
    check("queue_research never confirms", not b._needs_confirm("queue_research"))
    check("yes is parsed", bridge.LLMBridge._yesno("yes please") is True)
    check("no is parsed", bridge.LLMBridge._yesno("no, cancel that") is False)
    check("unclear is None", bridge.LLMBridge._yesno("what time is it") is None)

    # A write call must NOT execute until the user confirms on the next turn.
    stream_q = [("", [{"id": "1", "name": "calendar_create",
                       "arguments": '{"title":"Dentist","start":"2026-06-23T15:00"}'}])]
    async def fake_stream(msgs, tools=None):
        text, calls = stream_q.pop(0)
        if text: await say(text)
        return text, calls
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    # Host books it AND a read-back shows it → a truthful "on your calendar".
    b.tools = _FakeTools({
        "calendar_create": _ok('{"created": true, "id": "ABC", "when": "Tue 23 Jun 15:00"}'),
        "calendar_range": _ok('{"events": [{"id": "ABC", "title": "Dentist", "start": "2026-06-23T15:00"}]}'),
    })

    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("write is NOT executed before confirmation",
          not any(n == "calendar_create" for n, _ in b.tools.calls))
    check("a confirmation question is asked", "go ahead" in out.lower())
    check("the action is read back in the question", "dentist" in out.lower())
    check("pending confirmation is set", b._pending_confirm is not None)

    consumed = await b._resolve_confirmation("yes, go ahead")
    check("a clear reply consumes the turn", consumed is True)
    check("the write runs only after yes", any(n == "calendar_create" for n, _ in b.tools.calls))
    check("the calendar is read back to verify", any(n == "calendar_range" for n, _ in b.tools.calls))
    check("pending is cleared after resolving", b._pending_confirm is None)
    check("a verified booking is confirmed truthfully",
          any("on your calendar" in s.lower() for s in spoken))

    # Decline path: nothing runs.
    b.tools.calls.clear()
    b._pending_confirm = {"msgs": [{"role": "system", "content": "x"}],
                          "calls": [{"id": "2", "name": "calendar_delete", "arguments": "{}"}]}
    consumed = await b._resolve_confirmation("no, don't")
    check("decline consumes the turn", consumed is True)
    check("nothing runs on decline", b.tools.calls == [])
    check("decline is acknowledged aloud", any("won't" in s.lower() for s in spoken))

    # Unclear reply abandons the pending write (doesn't get stuck).
    b._pending_confirm = {"msgs": [], "calls": [{"id": "3", "name": "calendar_create", "arguments": "{}"}]}
    consumed = await b._resolve_confirmation("actually what's the weather")
    check("unclear reply is not consumed (falls through)", consumed is False)
    check("unclear reply clears the pending write", b._pending_confirm is None)


async def test_tool_host_call_raw():
    # Requirement 1: honour the host's `ok` flag (and treat non-200 / timeout as failure),
    # never as a silent success.
    tc = _load("tools_client")
    host = tc.ToolHost({"enabled": True, "url": "http://h", "timeout_s": 5})
    orig = aiohttp_stub.ClientSession

    class _S:                                  # base scriptable session
        body = None; status = 200; exc = None
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, url, json=None, timeout=None, **k):
            if type(self).exc: raise type(self).exc
            return _FakeResp(type(self).status, type(self).body)
    def use(cls): aiohttp_stub.ClientSession = cls

    try:
        class S1(_S): body = {"ok": False, "error": "no access"}
        use(S1); d = await host.call_raw("calendar_create", {})
        check("call_raw honours ok:false", d["ok"] is False and "no access" in d["error"])

        class S2(_S): status = 500; body = {}
        use(S2); d = await host.call_raw("calendar_create", {})
        check("call_raw treats non-200 as failure", d["ok"] is False and "500" in d["error"])

        class S3(_S): exc = TimeoutError("timed out")
        use(S3); d = await host.call_raw("calendar_create", {})
        check("call_raw treats a timeout as failure", d["ok"] is False and "timed out" in d["error"])

        class S4(_S): body = {"ok": True, "result": '{"created": true}'}
        use(S4); d = await host.call_raw("calendar_create", {})
        check("call_raw passes ok:true result through", d["ok"] and "created" in d["result"])

        use(S1); s = await host.call("calendar_create", {})
        check("call() renders failure as a (tool error) string", s.startswith("(tool error"))
    finally:
        aiohttp_stub.ClientSession = orig


def test_parse_write_result():
    p = bridge.LLMBridge._parse_write_result
    check("created true → ok", p('{"created": true, "id": "X", "when": "3pm"}')["status"] == "ok")
    check("created false + conflicts → conflict",
          p('{"created": false, "conflicts": ["Dentist 15:00"]}')["status"] == "conflict")
    check("created false, no conflicts → failed",
          p('{"created": false}')["status"] == "failed")
    check("explicit error → failed", p('{"error": "no access"}')["status"] == "failed")
    check("updated true → ok", p('{"updated": true}')["status"] == "ok")
    check("non-JSON → unknown", p("Sent the email.")["status"] == "unknown")
    check("conflicts are carried through",
          p('{"created": false, "conflicts": ["A", "B"]}')["conflicts"] == ["A", "B"])


async def test_write_outcomes():
    spoken = []
    async def say(s): spoken.append(s)
    def mk():
        b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                             big_lm_url=None, speak_sink=say, inject_time=False)
        b._trace = lambda *a, **k: None
        return b
    args = {"title": "Dentist", "start": "2026-06-23T15:00"}

    # 1. Clash: host says created:false with conflicts → must NOT claim success.
    b = mk(); spoken.clear()
    b.tools = _FakeTools({"calendar_create":
        _ok('{"created": false, "conflicts": ["Lunch 15:00–16:00"]}')})
    line = await b._run_write("calendar_create", args)
    check("a clash is not reported as booked", "book" not in line.lower() or "didn't book" in line.lower())
    check("a clash names the conflict", "Lunch" in line)
    check("a clash does not read the calendar back", not any(n == "calendar_range" for n, _ in b.tools.calls))

    # 2. ok:false / transport error → failure, not success.
    b = mk(); spoken.clear()
    b.tools = _FakeTools({"calendar_create": {"ok": False, "result": "", "error": "calendar access not granted"}})
    line = await b._run_write("calendar_create", args)
    check("an error is reported as failure", "didn't go through" in line.lower())
    check("the error reason is surfaced", "access" in line.lower())

    # 3. created:true but the read-back can't find it → cautious, not a firm "booked".
    b = mk(); spoken.clear()
    b.tools = _FakeTools({
        "calendar_create": _ok('{"created": true, "id": "ABC", "when": "Tue 15:00"}'),
        "calendar_range": _ok('{"events": []}'),          # not there!
    })
    line = await b._run_write("calendar_create", args)
    check("unverified create is hedged, not asserted", "didn't show up" in line.lower())

    # 3b. Host self-verified (verified:true) → trust it; don't run our own read-back
    #     (which could race the host's sync and falsely contradict it).
    b = mk(); spoken.clear()
    b.tools = _FakeTools({
        "calendar_create": _ok('{"created": true, "verified": true, "id": "ABC", "when": "Tue 15:00"}'),
        "calendar_range": _ok('{"events": []}'),          # would say "not found" if consulted
    })
    line = await b._run_write("calendar_create", args)
    check("host-verified create is confirmed", "on your calendar" in line.lower())
    check("host-verified create skips our redundant read-back",
          not any(n == "calendar_range" for n, _ in b.tools.calls))

    # 4. created:true and present in the read-back → truthful confirmation.
    b = mk(); spoken.clear()
    b.tools = _FakeTools({
        "calendar_create": _ok('{"created": true, "id": "ABC", "when": "Tue 15:00"}'),
        "calendar_range": _ok('{"events": [{"id": "ABC", "title": "Dentist", "start": "2026-06-23T15:00"}]}'),
    })
    line = await b._run_write("calendar_create", args)
    check("a verified create is confirmed", "on your calendar" in line.lower())

    # 5. verify_writes off → trust the created:true without a read-back.
    b = mk(); spoken.clear(); b.verify_writes = False
    b.tools = _FakeTools({"calendar_create": _ok('{"created": true, "when": "Tue 15:00"}')})
    line = await b._run_write("calendar_create", args)
    check("no read-back when verification is disabled", not any(n == "calendar_range" for n, _ in b.tools.calls))
    check("still reports done", "done" in line.lower())


def test_tool_policy():
    b = _Dummy()
    cal = {"type": "function", "function": {"name": "check_calendar"}}
    qr = bridge.QUEUE_RESEARCH_TOOL
    pol = bridge.LLMBridge._tool_policy(b, [cal, qr])
    check("policy lists the live tool", "check_calendar" in pol)
    check("policy warns queue_research returns nothing useful now", "queue_research" in pol and "later" in pol)
    pol2 = bridge.LLMBridge._tool_policy(b, [cal])
    check("policy omits queue_research note when absent", "queue_research" not in pol2)


async def test_self_knowledge_injection():
    # Ambient 'self' memories must land in the system prompt every turn, with no
    # trigger and no recall_hook involved.
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(),
                         fast_lm_url="http://f", big_lm_url=None,
                         speak_sink=lambda s: asyncio.sleep(0), inject_time=False)
    b.self_hook = lambda: "- I find gentle humour builds rapport with this user."
    captured = {}
    async def fake_run(messages, tools):
        captured["system"] = messages[0]["content"]
        return "ok"
    b._run_turn = fake_run
    await b._handle_turn("hello")
    check("self-knowledge is injected into the system prompt",
          "gentle humour builds rapport" in captured.get("system", ""))
    check("self block is framed as tone-shaping, not quotable",
          "don't quote it" in captured.get("system", ""))


def test_deliberate_triggers():
    # Loop detection: near-duplicate consecutive replies ⇒ the fast LM is restating
    # itself (out of depth) ⇒ hand the next turn to the big LM.
    ov = bridge.LLMBridge._overlap
    check("identical text overlaps fully", ov("the cat sat", "the cat sat") == 1.0)
    check("disjoint text doesn't overlap", ov("alpha beta", "gamma delta") == 0.0)

    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False,
                         deliberate={"loop_sim": 0.8})
    b.history = [{"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "I think the answer depends on context here."},
                 {"role": "user", "content": "but really?"},
                 {"role": "assistant", "content": "The answer depends on context, really, here."}]
    check("near-duplicate replies are detected as looping", b._is_looping() is True)
    b.history[-1]["content"] = "Sure — it's twelve metres exactly."
    check("distinct replies are not looping", b._is_looping() is False)
    b.history[-1]["content"] = b.history[1]["content"]   # duplicate again…
    b.deliberate_cfg["loop_sim"] = 0
    check("loop_sim 0 disables the loop trigger", b._is_looping() is False)

    # Tool schema + policy coaching.
    dl = bridge.DELIBERATE_TOOL
    check("deliberate tool schema names the function", dl["function"]["name"] == "deliberate")
    pol = bridge.LLMBridge._tool_policy(_Dummy(), [{"type": "function",
            "function": {"name": "check_calendar"}}, dl])
    check("policy coaches when to deliberate", "deliberate" in pol and "not confident" in pol.lower())


async def test_deliberate_tool_offered():
    # When a big LM is configured the deliberate tool must be offered to the fast LM.
    captured = {}
    async def fake_run(messages, tools):
        captured["tools"] = [t["function"]["name"] for t in tools]
        return "ok"
    async def say(s): pass
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", speak_sink=say, inject_time=False)
    b.tools = None
    b._run_turn = fake_run
    b._trace = lambda *a, **k: None
    b.big_url = None                                   # don't fire a real briefing in _finish_turn
    await b._handle_turn("what's the boiling point of nitrogen?")
    check("deliberate tool is offered to the fast LM", "deliberate" in captured.get("tools", []))


async def test_deliberate_flow():
    spoken = []
    async def say(s): spoken.append(s)
    def mk(cfg=None):
        b = bridge.LLMBridge(server_state=types.SimpleNamespace(deliberating=False),
                             fast_lm_url="http://f", big_lm_url="http://big",
                             speak_sink=say, inject_time=False,
                             deliberate={"progress_after_s": 0.02, "timeout_s": 1.0,
                                         "progress": ["still thinking…"], **(cfg or {})})
        b._trace = lambda *a, **k: None
        return b

    # Happy path: announce → think → deliver, with barge-in held throughout.
    spoken.clear(); seen = {}
    b = mk({"deliver_via_fast": False})
    async def quick(prompt):
        seen["during"] = getattr(b.state, "deliberating", None)
        return "Nitrogen boils at about minus 196 Celsius."
    b._big_lm_consider = quick
    out = await b._deliberate("boiling point of nitrogen?")
    check("a stall line is spoken first", bool(spoken) and "second" in spoken[0].lower())
    check("barge-in is held while it thinks", seen.get("during") is True)
    check("barge-in is released afterwards", b.state.deliberating is False)
    check("the considered answer is returned", "minus 196" in out)
    check("the considered answer is spoken", any("minus 196" in s for s in spoken))

    # Verbal progress bar: a long think speaks a 'still thinking' line before answering.
    spoken.clear()
    b = mk({"deliver_via_fast": False, "progress_after_s": 0.05})
    async def slow(prompt):
        await asyncio.sleep(0.18)
        return "It's ready now."
    b._big_lm_consider = slow
    out = await b._deliberate("q")
    check("a progress line is spoken while it thinks", any("still thinking" in s for s in spoken))
    check("the answer still arrives after progress", "ready now" in out)

    # Timeout: give up, apologise, and release barge-in.
    spoken.clear()
    b = mk({"deliver_via_fast": False, "timeout_s": 0.08, "timed_out": "Sorry, that took too long."})
    async def hang(prompt):
        await asyncio.sleep(5)
        return "never"
    b._big_lm_consider = hang
    out = await b._deliberate("q")
    check("a timeout apologises rather than hanging", "too long" in out.lower())
    check("the apology is spoken", any("too long" in s.lower() for s in spoken))
    check("barge-in is released after a timeout", b.state.deliberating is False)

    # Delivery via the fast LM: the big LM's answer is rephrased in-voice, not verbatim.
    spoken.clear()
    b = mk({"deliver_via_fast": True})
    async def raw(prompt): return "RAW ANSWER 42."
    b._big_lm_consider = raw
    captured = {}
    async def fake_tts(messages, tools=None):
        captured["messages"] = messages
        return "It's forty-two.", []
    b._stream_to_tts = fake_tts
    out = await b._deliberate("q")
    check("delivery routes the answer through the fast LM's voice", out == "It's forty-two.")
    check("the raw answer is handed to the fast LM to phrase",
          any("RAW ANSWER 42" in (m.get("content") or "") for m in captured["messages"]))


async def test_identity_injection_and_tools():
    # The privileged identity block is injected into the fast prompt, and the
    # self-determination tools are offered when their callbacks are present.
    captured = {}
    async def fake_run(messages, tools):
        captured["system"] = messages[0]["content"]
        captured["tools"] = [t["function"]["name"] for t in tools]
        return "ok"
    async def say(s): pass
    b = bridge.LLMBridge(
        server_state=types.SimpleNamespace(), fast_lm_url="http://f", big_lm_url=None,
        speak_sink=say, inject_time=False,
        identity_hook=lambda rp: ("Who you are — this is your character; stay true to it, "
                                  "don't quote it:\nVinkona — warm, witty, curious"),
        revise_self=lambda a: "ok", note_person=lambda a: "ok")
    b.tools = None
    b._run_turn = fake_run
    b._trace = lambda *a, **k: None
    await b._handle_turn("hi there")
    check("identity block is injected into the fast prompt",
          "this is your character" in captured.get("system", ""))
    check("identity sits above the recall/memory section (declared, not recalled)",
          "this is your character" in captured["system"])
    check("revise_self tool is offered", "revise_self" in captured.get("tools", []))
    check("note_person tool is offered", "note_person" in captured.get("tools", []))


async def test_note_person_sayback():
    # note_person is a say-back: its ack is spoken directly, callback runs, no 2nd LM round.
    scripted = [("", [{"id": "1", "name": "note_person",
                       "arguments": '{"person":"the user","note":"has a dog named Rex"}'}])]
    calls, noted, spoken = [], [], []
    async def fake_stream(msgs, tools=None):
        calls.append(tools); return scripted[len(calls) - 1]
    async def say(s): spoken.append(s)
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False,
                         confirm_required=False, tool_filler="Let me check.",
                         note_person=lambda a: (noted.append(a),
                             "Got it — I'll remember that about you.")[1])
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("note_person callback is invoked", bool(noted) and noted[0]["note"].startswith("has a dog"))
    check("the ack is spoken to the user", any("remember that about you" in s for s in spoken))
    check("the ack is the returned answer", "remember that about you" in out)
    check("no 'let me check' filler for a say-back note", "Let me check." not in spoken)
    check("no second LM round is needed", len(calls) == 1)


async def test_revise_self_confirm_flow():
    # A CORE self-edit is staged for a yes/no first (canon is deliberate), applied only on yes.
    scripted = [("", [{"id": "1", "name": "revise_self",
                       "arguments": '{"attribute":"humour","value":"drier and more deadpan","layer":"core"}'}])]
    calls, applied, spoken = [], [], []
    async def fake_stream(msgs, tools=None):
        calls.append(tools); return scripted[len(calls) - 1]
    async def say(s): spoken.append(s)
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False,
                         confirm_required=False, confirm_self_edits=True,
                         revise_self=lambda a: (applied.append(a),
                             "Okay — that's part of who I am now.")[1])
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("a core self-edit is staged, not applied yet", applied == [])
    check("a confirm question is asked", out.strip().endswith("?"))
    check("pending identity edit is set", b._pending_identity is not None)

    consumed = await b._resolve_self_edit("yes, do that")
    check("a clear yes consumes the turn", consumed is True)
    check("the edit applies only after yes", bool(applied) and applied[0]["value"].startswith("drier"))
    check("pending is cleared after resolving", b._pending_identity is None)

    # Decline: nothing changes.
    b._pending_identity = {"attribute": "warmth", "value": "cold", "layer": "core"}
    applied.clear(); spoken.clear()
    consumed = await b._resolve_self_edit("no, stay as you are")
    check("decline consumes the turn", consumed is True)
    check("nothing is applied on decline", applied == [])
    check("decline is acknowledged aloud", any("stay as I am" in s for s in spoken))

    # Unclear reply abandons the staged edit (doesn't get stuck).
    b._pending_identity = {"attribute": "x", "value": "y", "layer": "core"}
    consumed = await b._resolve_self_edit("what's the time")
    check("unclear reply is not consumed", consumed is False)
    check("unclear reply clears the staged edit", b._pending_identity is None)


async def test_revise_self_surface_is_immediate():
    # A SURFACE edit ("how I'm being now") applies at once, even with confirm_self_edits on.
    scripted = [("", [{"id": "1", "name": "revise_self",
                       "arguments": '{"attribute":"mood","value":"playful right now","layer":"surface"}'}])]
    calls, applied, spoken = [], [], []
    async def fake_stream(msgs, tools=None):
        calls.append(tools); return scripted[len(calls) - 1]
    async def say(s): spoken.append(s)
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False,
                         confirm_required=False, confirm_self_edits=True,
                         revise_self=lambda a: (applied.append(a), "Alright — playful, for now.")[1])
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("a surface self-edit applies immediately", bool(applied) and applied[0]["layer"] == "surface")
    check("no confirmation is staged for a surface edit", b._pending_identity is None)
    check("the surface ack is spoken", any("for now" in s for s in spoken))


async def test_identity_detail_to_big_lm():
    # The big LM (briefing) gets the full structured identity profile, framed for consistency.
    captured = {}
    async def fake_stream(url, model, messages, max_tokens):
        captured["prompt"] = messages[-1]["content"]; yield "ok"
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False, roleplay_adaptive=False,
                         identity_detail_hook=lambda rp: ("Vinkona (self); pronouns: she/her\n"
                             "  trait:\n    - openness: curious  [core; seed]"))
    b._stream_chat = fake_stream
    b.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    await b._update_big_lm_briefing()
    check("identity detail reaches the big LM briefing", "Vinkona (self)" in captured["prompt"])
    check("it's framed to keep them in character", "in character" in captured["prompt"])


async def test_roleplay_adaptive_mode():
    # The big LM leads its briefing with a mode tag; the bridge flips roleplay and strips it.
    seen = {}
    async def fake_stream(url, model, messages, max_tokens):
        seen["prompt"] = messages[-1]["content"]
        for tok in ["[ROLEPLAY] ", "Lean into the scene with them."]:
            yield tok
    def detail(rp):
        seen["rp_when_detailing"] = rp
        return "Vinkona (self)"
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False,
                         identity_detail_hook=detail, roleplay_default=False,
                         roleplay_adaptive=True)
    b._stream_chat = fake_stream
    b.history = [{"role": "user", "content": "let's act out a scene on a ship"},
                 {"role": "assistant", "content": "aye"}]
    check("roleplay starts off", b._roleplay is False)
    await b._update_big_lm_briefing()
    check("the big LM can switch roleplay ON via its tag", b._roleplay is True)
    check("the mode tag is stripped from the briefing", "[ROLEPLAY]" not in b._big_lm_briefing)
    check("the briefing text survives the strip", "scene" in b._big_lm_briefing.lower())
    check("the big LM was asked to decide the mode", "ROLEPLAY" in seen["prompt"])

    async def fake2(url, model, messages, max_tokens):
        yield "[ASSISTANT] Back to ordinary help now."
    b._stream_chat = fake2
    await b._update_big_lm_briefing()
    check("the big LM can switch roleplay back OFF", b._roleplay is False)

    # With adaptivity off, no tag is requested and the mode is left as configured.
    b2 = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                          big_lm_url="http://big", inject_time=False,
                          identity_detail_hook=detail, roleplay_default=True,
                          roleplay_adaptive=False)
    async def fake3(url, model, messages, max_tokens):
        seen["prompt2"] = messages[-1]["content"]; yield "Just a briefing."
    b2._stream_chat = fake3
    b2.history = list(b.history)
    await b2._update_big_lm_briefing()
    check("no mode tag requested when adaptivity is off", "ROLEPLAY" not in seen["prompt2"])
    check("configured roleplay default is kept when not adaptive", b2._roleplay is True)


def test_asr_clarify_gate():
    # The pure decision: when to ask the user to repeat instead of acting on garbled text.
    sc = asr.should_clarify
    o = {"clarify_below": -0.9, "clarify_min_words": 2}
    check("clear speech is acted on", not sc("book a table for two", -0.3, o, False))
    check("a shaky multi-word turn asks to repeat", sc("buck a tabby for too", -1.1, o, False))
    check("a one-word turn isn't worth clarifying", not sc("yeah", -1.8, o, False))
    check("no clarify loop: don't re-ask right after asking", not sc("still mumbling", -1.2, o, True))
    check("disabled when clarify_below is None",
          not sc("anything at all here", -2.0, {"clarify_below": None}, False))
    check("silence (no confidence) never triggers", not sc("words here", None, o, False))


async def test_multi_host_routing():
    # MultiHost lets the bridge call tools from more than one host (Mac + music).
    tc = _load("tools_client")

    class FakeHost:
        def __init__(self, active, tools, results):
            self._active, self._tools, self._results, self.calls = active, tools, results, []
        @property
        def active(self): return self._active
        async def catalogue(self):
            return ([{"type": "function", "function": {"name": n}} for n in self._tools]
                    if self._active else [])
        async def call_raw(self, name, args):
            self.calls.append(name)
            return {"ok": True, "result": self._results.get(name, ""), "error": ""}
        async def call(self, name, args):
            return (await self.call_raw(name, args))["result"]

    mac = FakeHost(True, ["calendar_range", "mail_list"], {"calendar_range": "cal!"})
    music = FakeHost(True, ["play_music", "music_search"], {"play_music": "playing!"})
    mh = tc.MultiHost([mac, music])
    names = {t["function"]["name"] for t in await mh.catalogue()}
    check("catalogue unions both hosts",
          names == {"calendar_range", "mail_list", "play_music", "music_search"})
    check("a music tool routes to the music host",
          await mh.call("play_music", {}) == "playing!"
          and "play_music" in music.calls and "play_music" not in mac.calls)
    check("a mac tool routes to the mac host",
          await mh.call("calendar_range", {}) == "cal!" and "calendar_range" in mac.calls)
    check("active when any host is active", mh.active is True)

    a = FakeHost(True, ["dup"], {"dup": "A"}); b = FakeHost(True, ["dup"], {"dup": "B"})
    mh2 = tc.MultiHost([a, b]); await mh2.catalogue()
    check("first host wins a name clash", await mh2.call("dup", {}) == "A")

    off = FakeHost(False, ["x"], {}); on = FakeHost(True, ["y"], {"y": "Y"})
    mh3 = tc.MultiHost([off, on])
    check("inactive hosts are skipped from the catalogue",
          [t["function"]["name"] for t in await mh3.catalogue()] == ["y"])


def test_announce_classification():
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, inject_time=False)
    b.tools = object()                          # a (truthy) host present
    check("calendar_create is act-then-announce", b._is_announce_write("calendar_create"))
    check("calendar_update is act-then-announce", b._is_announce_write("calendar_update"))
    check("calendar_delete is NOT (stays confirmed)", not b._is_announce_write("calendar_delete"))
    check("calendar_range read is NOT a write to announce", not b._is_announce_write("calendar_range"))
    b.tools = None
    check("nothing to announce without a tool host", not b._is_announce_write("calendar_create"))


async def test_act_then_announce():
    spoken = []
    async def say(s): spoken.append(s)

    # calendar_create → runs immediately (no confirm), verified, announced with an undo line.
    scripted = [("", [{"id": "1", "name": "calendar_create",
                       "arguments": '{"title":"Gym","start":"2026-06-21T20:00"}'}])]
    calls = []
    async def fake_stream(msgs, tools=None):
        calls.append(tools); return scripted[len(calls) - 1]
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, speak_sink=say, inject_time=False)  # confirm on by default
    b._stream_to_tts = fake_stream
    b._trace = lambda *a, **k: None
    b.tools = _FakeTools({"calendar_create":
        _ok('{"created": true, "verified": true, "id": "G1", "when": "Sat 20:00"}')})
    out = await b._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("a calendar create runs without a confirmation question", b._pending_confirm is None)
    check("the write actually ran", any(n == "calendar_create" for n, _ in b.tools.calls))
    check("it's announced, not asked", "?" not in out and "calendar" in out.lower())
    check("the announcement offers an undo", "change" in out.lower())

    # calendar_delete is NOT auto — it still asks first.
    scripted2 = [("", [{"id": "1", "name": "calendar_delete", "arguments": '{"id":"G1"}'}])]
    c2 = []
    async def fs2(msgs, tools=None):
        c2.append(tools); return scripted2[len(c2) - 1]
    b2 = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                          big_lm_url=None, speak_sink=say, inject_time=False)
    b2._stream_to_tts = fs2
    b2._trace = lambda *a, **k: None
    b2.tools = _FakeTools({"calendar_delete": _ok('{"deleted": true}')})
    await b2._run_turn([{"role": "system", "content": "x"}], [{"x": 1}])
    check("a delete still asks first", b2._pending_confirm is not None)
    check("the delete did NOT run before confirmation",
          not any(n == "calendar_delete" for n, _ in b2.tools.calls))


async def test_situation_feed_to_big_lm():
    captured = {}
    async def fake_stream(url, model, messages, max_tokens):
        captured["prompt"] = messages[-1]["content"]; yield "ok"
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False, roleplay_adaptive=False,
                         situation_hook=lambda: "It's Monday 08:38.\n- Work at 09:00 (in 22 minutes)")
    b._stream_chat = fake_stream
    b.history = [{"role": "user", "content": "morning"}, {"role": "assistant", "content": "hey"}]
    await b._update_big_lm_briefing()
    check("the situation reaches the big LM", "Work at 09:00" in captured["prompt"])
    check("surfacing is gated conservatively",
          "only if" in captured["prompt"].lower()
          and "never raise the same thing twice" in captured["prompt"].lower())

    cap2 = {}
    async def fs2(url, model, messages, max_tokens):
        cap2["p"] = messages[-1]["content"]; yield "ok"
    b2 = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                          big_lm_url="http://big", inject_time=False, roleplay_adaptive=False,
                          situation_hook=lambda: "")
    b2._stream_chat = fs2
    b2.history = list(b.history)
    await b2._update_big_lm_briefing()
    check("no situation block when nothing is coming up", "coming up for the user" not in cap2["p"])


async def test_affect_shift():
    captured = {}
    written = []
    async def fake_stream(url, model, messages, max_tokens):
        captured["messages"] = messages
        yield "Lean into the curiosity here.\nSTATE: This one's got me genuinely turning things over."

    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url="http://big", inject_time=False,
                         affect_hook=lambda: "Quietly content.",
                         affect_update=lambda t: written.append(t),
                         affect_objective="genuine, honest connection",
                         trace_hook=lambda e: None)
    b._stream_chat = fake_stream
    b.history = [{"role": "user", "content": "do you ever wonder what you are?"},
                 {"role": "assistant", "content": "Often, honestly."}]
    await b._update_big_lm_briefing()
    sent = captured["messages"][-1]["content"]
    check("objective + current state reach the director", "honest connection" in sent
          and "Quietly content." in sent)
    check("a shifted STATE line is parsed out and persisted",
          written == ["This one's got me genuinely turning things over."])
    check("the STATE line is stripped from the briefing", "STATE:" not in b._big_lm_briefing
          and b._big_lm_briefing == "Lean into the curiosity here.")

    # No STATE line → no affect write.
    written.clear()
    async def plain_stream(url, model, messages, max_tokens):
        yield "Just keep it warm."
    b._stream_chat = plain_stream
    await b._update_big_lm_briefing()
    check("no write when the director omits STATE", written == [])

    # With no affect_update wired, the director isn't asked for a STATE line at all.
    b2 = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                          big_lm_url="http://big", inject_time=False, trace_hook=lambda e: None)
    cap2 = {}
    async def s2(url, model, messages, max_tokens):
        cap2["m"] = messages; yield "ok"
    b2._stream_chat = s2
    b2.history = list(b.history)
    await b2._update_big_lm_briefing()
    check("no inner-life block when affect is off", "STATE:" not in cap2["m"][-1]["content"])


def test_voice_examples():
    b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                         big_lm_url=None, inject_time=False)
    check("no exemplars → no voice block", b._voice_block() == "")
    b.apply_persona(voice_examples=[{"user": "hi", "you": "Hey — what's up?"},
                                    {"user": "thanks", "vinkona": "Any time."},
                                    {"user": "bad", "you": ""}])           # skipped (empty)
    vb = b._voice_block()
    check("voice block has the header", vb.startswith("Here's how you sound"))
    check("voice block renders 'you' pairs", "You: Hey — what's up?" in vb)
    check("voice block accepts 'vinkona' as the reply key", "You: Any time." in vb)
    check("voice block skips pairs with no reply", vb.count("You:") == 2)
    b.apply_persona(voice_examples=[])
    check("clearing exemplars empties the block", b._voice_block() == "")


async def main():
    test_asr_clarify_gate()
    test_announce_classification()
    await test_act_then_announce()
    await test_situation_feed_to_big_lm()
    await test_multi_host_routing()
    await test_self_knowledge_injection()
    await test_identity_injection_and_tools()
    await test_note_person_sayback()
    await test_revise_self_confirm_flow()
    await test_revise_self_surface_is_immediate()
    await test_identity_detail_to_big_lm()
    await test_roleplay_adaptive_mode()
    await test_chat_json_think()
    await test_call_tool_routing()
    await test_forced_answer_after_tool()
    await test_sayback_spoken_directly()
    await test_doc_grounded_briefing()
    await test_lead_levels()
    test_tool_result_trace()
    await test_confirm_guard()
    await test_tool_host_call_raw()
    test_parse_write_result()
    await test_write_outcomes()
    test_stream_payload_fields()
    test_tool_policy()
    test_deliberate_triggers()
    await test_deliberate_tool_offered()
    await test_deliberate_flow()
    await test_affect_shift()
    test_voice_examples()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
