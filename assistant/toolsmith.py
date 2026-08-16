"""
toolsmith.py — Vinkona's idle tool-maker: a two-phase pipeline over a durable spec queue.

Between conversations she asks herself two separate questions:

  1. **Is a tool missing?**  (``identify``) — look at the current tooling and the recent
     conversations and spot a deficit.  The answer is deliberately NOT code: it's a
     plain-language spec ("what the tool should do"), queued as a *tool idea* with status
     ``proposed``.  Cheap, reviewable, and visible in the Tools panel immediately.

  2. **Is there a queued spec to build?**  (``build_next``) — take one spec from the queue
     and attempt it: write the code, self-test it in the throwaway sandbox, and pass or
     fail it.  A failure is BANKED on the idea (status ``failed``, with the code and the
     traceback), and a LATER idle cycle re-opens it: an analysis step reads the spec, the
     previous code and the error, diagnoses what went wrong — optionally asking the
     knowledge host (kb_ask) for guidance — and tries again with the adjustment.  After
     the attempt budget is spent (or the LM judges it unbuildable) the idea is ``parked``:
     still visible, waiting for you.  A successful build removes the idea — the tool
     itself now stands in the roster.

The user's jotted ideas enter the same queue (status ``proposed``), so "I wish she had a
tool that…" typed into the panel gets attempted on the next idle cycle — user specs are
picked first.

The security story is unchanged: whatever she writes runs in the same sandbox as every
other own-tool — read anywhere (or the shared folders), write only in her store, no
network, killed on timeout — and nothing is offered mid-conversation until it has passed
a self-test.  The big LM only produces text.

This module is deliberately HTTP-agnostic: it takes ``chat_json`` (the big LM) and an
optional ``guidance`` callable (kb_ask), both bound by the caller — so it has no aiohttp
dependency and unit-tests with stubs.
"""
from __future__ import annotations

import re
import typing as tp

# The rules the generated tool must obey — same contract the seed tools follow, stated so
# the big LM writes something that will actually pass the self-test.
_TOOL_CONTRACT = """\
A tool is ONE Python 3 file. Rules it MUST follow:
- Read a single JSON object of arguments from stdin (json.load(sys.stdin)).
- Print EXACTLY ONE JSON object to stdout (json.dumps(...)) — nothing else.
- Standard library only. No pip packages, no network (there is none).
- To READ a file anywhere on the machine, prepend os.environ.get("TOOL_ROOT","") to an
  ABSOLUTE path:  open(os.environ.get("TOOL_ROOT","") + "/etc/hostname").
- You may only WRITE inside the current directory (your private store). Write relative
  paths (e.g. open("note.txt","w")); never an absolute path.
- Catch errors and RETURN them, e.g. print(json.dumps({"error": str(e)})); don't crash.
- Keep it small, single-purpose, and deterministic."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def _s(v: tp.Any, n: int = 4000) -> str:
    return str(v or "")[:n]


def _fac_allow(toolbox) -> list:
    """Faculties a tool may call (empty unless the feature is enabled)."""
    fc = (getattr(toolbox, "cfg", {}) or {}).get("faculties") or {}
    return list(fc.get("allow") or []) if fc.get("enabled") else []


def _faculties_block(fac_allow: list) -> str:
    """The extra contract for a tool that calls Vinkona's OTHER tools, shown only when
    faculties are enabled.  Empty string otherwise."""
    if not fac_allow:
        return ""
    names = ", ".join(fac_allow)
    return f"""
OPTIONAL — calling your other tools.  If (and only if) the tool needs one of these
faculties, it may call them (there is still no network; the host runs them for you):
  {names}
A faculties tool has a DIFFERENT shape — it uses the injected `faculties` helper instead of
reading stdin / printing stdout directly, and it MUST set "uses_faculties": true:
  import faculties
  a = faculties.args()                       # the input arguments (a dict)
  r = faculties.call("kb_search", {{"query": a["q"]}})   # -> that faculty's result
  faculties.done({{"answer": r}})            # emit the final result and exit
For the self-test, add "faculty_stubs" to the test giving a canned result per faculty you
call (the self-test never hits the real tools), e.g.
  "test": {{"input": {{"q":"x"}}, "faculty_stubs": {{"kb_search": {{"hits": []}}}}, "expect_keys": ["answer"]}}
Write the tool to handle an empty/most-any faculty result."""


# ── phase 1: identify a deficit ───────────────────────────────────────────────

def _identify_prompt(*, context: str, roster: list, queue: list, faculties: list,
                     usage: dict, logs: list) -> str:
    have = "\n".join(f"- {t['name']}: {t.get('description','')}" for t in roster) \
        or "(none yet)"
    queued = "\n".join(f"- {i.get('title','')}" for i in queue) or "(none)"
    fac = "\n".join(f"- {f}" for f in faculties[:40]) or "(none)"
    convo = "\n".join(f"{r.get('role','?').upper()}: {_s(r.get('text'), 500)}"
                      for r in logs[-24:]) or "(no recent interactions)"
    unused = [n for t in roster if (n := t["name"]) and usage.get(n, {}).get("calls", 0) == 0]
    unused_note = (f"\nTools you built but have never used (a caution against piling up "
                   f"more like them): {', '.join(unused)}." if unused else "")
    return f"""{context}
You can make small tools for yourself (single-file Python scripts that run in your sandbox:
read files anywhere, write only in your store, no network). Between chats you look for a
GAP: given the recent conversations, is a tool needed that does not exist?

Tools you already have (not a gap if one of these covers it):
{have}

Things you can already do with your other faculties (not a gap either):
{fac}

Specs already in your build queue (do NOT repeat these):
{queued}{unused_note}

Recent interactions:
{convo}

If you see a real deficit, describe the tool IN PLAIN LANGUAGE ONLY — no code, no schema:
what it would be called informally, what problem it solves, what it takes in and what it
should return.  Someone (you, later) will implement it from this description alone.

Reply with ONE JSON object:
  {{"deficit": true, "title": "short informal title",
    "purpose": "2-5 plain sentences: the problem, inputs, and what it returns",
    "rationale": "why the recent conversations show it's needed"}}
or, if the current tooling covers what actually came up:
  {{"deficit": false}}
Prefer "false" over a weak, redundant, or speculative tool."""


async def identify(toolbox, chat_json: tp.Callable, *, logs: list | None = None,
                   faculties: list | None = None, context: str = "",
                   max_queue: int = 10) -> dict:
    """Phase 1: spot a missing tool and queue a plain-language spec for it.
    Returns {action: proposed|none, title?, reason?}."""
    ideas = toolbox.ideas()
    open_specs = [i for i in ideas if i.get("status") in ("proposed", "failed")]
    if len(open_specs) >= int(max_queue):
        return {"action": "none", "reason": f"queue is full ({len(open_specs)})"}
    out = await chat_json(_identify_prompt(
        context=context, roster=toolbox.describe(), queue=ideas,
        faculties=faculties or [], usage=toolbox.usage(), logs=logs or []), think=True)
    if not isinstance(out, dict) or not out.get("deficit"):
        return {"action": "none"}
    title = _s(out.get("title"), 120).strip()
    if not title:
        return {"action": "none", "reason": "deficit reported but no title"}
    r = toolbox.add_idea(title, rationale=_s(out.get("rationale"), 2000),
                         sketch=_s(out.get("purpose"), 4000), source="toolsmith",
                         status="proposed")
    if not r.get("ok"):
        return {"action": "none", "reason": r.get("error")}
    return {"action": "proposed", "title": title}


# ── phase 2: build (or re-analyse and re-test) one queued spec ───────────────

def _analysis_prompt(idea: dict) -> str:
    return f"""You tried to build yourself this tool and the attempt FAILED. Diagnose why and
decide how to fix it (or whether it simply can't be built under the rules).

{_TOOL_CONTRACT}

The spec: {_s(idea.get('title'), 120)}
Purpose: {_s(idea.get('sketch'), 2000)}

The failing code:
{_s(idea.get('last_code'), 5000)}

The error it failed with:
{_s(idea.get('last_error'), 1200)}

Reply with ONE JSON object:
  {{"diagnosis": "what actually went wrong, in a sentence or two",
    "adjustment": "what to do differently this time",
    "kb_question": "a short question for your knowledge base if guidance would help, else \"\"",
    "unbuildable": false}}
Set "unbuildable": true (with the diagnosis saying why) only if the spec cannot be met
under the rules — needs the network, a package, or a capability you don't have."""


def _code_prompt(*, idea: dict, fac_allow: list, adjustment: str = "",
                 guidance: str = "", prev_code: str = "", error: str = "") -> str:
    base = f"""{_TOOL_CONTRACT}
{_faculties_block(fac_allow)}

Implement this tool of yours from its plain-language spec.
Spec: {_s(idea.get('title'), 120)}
Purpose: {_s(idea.get('sketch'), 3000)}
Why it's wanted: {_s(idea.get('rationale'), 1000)}"""
    if adjustment:
        base += f"\nThis is a RETRY. Apply this adjustment: {_s(adjustment, 1000)}"
    if guidance:
        base += f"\n\nGuidance from your knowledge base:\n{_s(guidance, 2500)}"
    base += """

Reply with ONE JSON object:
{"name": "snake_case_name",
  "code": "<the full Python file as a string>",
  "parameters": {"type":"object","properties":{...},"required":[...]},
  "uses_faculties": false,
  "test": {"input":{...},"expect_keys":["..."]},
  "unbuildable": false}
- "name" is the tool's menu name (lower_snake_case, 3-40 chars).
- "parameters" is the JSON-schema of the stdin arguments (for the menu).
- Set "uses_faculties": true ONLY if the code imports the faculties helper (see above).
- "test.input" is a concrete arguments object that exercises the tool; "expect_keys" are
  keys that MUST appear in its output. The test runs in the sandbox and must pass, so pick
  an input that works there (e.g. read a file that exists like /etc/hostname).
- If while writing it you realise it CANNOT be built under the rules, reply
  {"unbuildable": true, "reason": "why"} instead."""
    if error:
        base += (f"\n\nYour previous attempt FAILED its self-test with:\n{_s(error, 1200)}\n\n"
                 f"Previous code:\n{_s(prev_code, 4000)}\n\nFix it and return the full "
                 "corrected JSON object.")
    return base


def _pick(ideas: list, max_attempts: int) -> dict | None:
    """The next spec to work: user-jotted first (they asked), then proposed, then failed
    awaiting re-analysis — oldest first within each band; parked never."""
    def band(i):
        if i.get("status") == "proposed":
            return 0 if i.get("source") == "user" else 1
        return 2
    open_specs = [i for i in ideas
                  if i.get("status") in ("proposed", "failed")
                  and int(i.get("attempts", 0)) < int(max_attempts)]
    return sorted(open_specs, key=lambda i: (band(i), str(i.get("created_at", ""))))[0] \
        if open_specs else None


async def build_next(toolbox, chat_json: tp.Callable, *, guidance: tp.Callable | None = None,
                     max_repair: int = 2, max_tools: int = 24,
                     max_attempts: int = 3) -> dict:
    """Phase 2: take one queued spec and attempt it — code, self-test, pass or fail.  A
    failed spec is banked (code + error) and re-analysed on a later cycle; `guidance` is an
    optional ``async (question) -> str|None`` (kb_ask) consulted when the analysis asks for
    it.  Returns {action: built|failed|parked|none, title?/name?, attempts?, reason?}."""
    if len(toolbox.names()) >= int(max_tools):
        return {"action": "none", "reason": "tool cap reached — build later"}
    idea = _pick(toolbox.ideas(), max_attempts)
    if idea is None:
        return {"action": "none", "reason": "queue empty"}
    session = int(idea.get("attempts", 0)) + 1

    # -- re-analysis of a previously failed attempt (Dan's step 2b) --
    adjustment, kb_text = "", ""
    if idea.get("status") == "failed":
        ana = await chat_json(_analysis_prompt(idea), think=True)
        if isinstance(ana, dict):
            if ana.get("unbuildable"):
                toolbox.update_idea(idea["id"], status="parked",
                                    last_error=_s(ana.get("diagnosis"), 400)
                                    or "judged unbuildable")
                return {"action": "parked", "title": idea.get("title"),
                        "reason": _s(ana.get("diagnosis"), 200)}
            adjustment = _s(ana.get("adjustment"), 1000)
            q = _s(ana.get("kb_question"), 300).strip()
            if q and guidance is not None:
                try:
                    kb_text = _s(await guidance(q), 2500)
                except Exception:
                    kb_text = ""

    # -- code it, with the in-session repair loop --
    fac = _fac_allow(toolbox)
    prev_code = _s(idea.get("last_code"), 4000) if idea.get("status") == "failed" else ""
    last_err = ""
    # The WHOLE last attempt is banked on failure (not just the code) so the panel can open
    # it in the editor for inspection and repair by hand.
    last_name = str(idea.get("name") or "")
    last_manifest = last_test = None
    for _ in range(1, int(max_repair) + 2):              # first try + max_repair retries
        spec = await chat_json(_code_prompt(
            idea=idea, fac_allow=fac, adjustment=adjustment, guidance=kb_text,
            prev_code=prev_code, error=last_err), think=True)
        if not isinstance(spec, dict):
            last_err = "the big LM did not return usable code"
            continue
        if spec.get("unbuildable"):
            toolbox.update_idea(idea["id"], status="parked",
                                last_error=_s(spec.get("reason"), 400) or "judged unbuildable")
            return {"action": "parked", "title": idea.get("title"),
                    "reason": _s(spec.get("reason"), 200)}
        if not isinstance(spec.get("code"), str) or not spec["code"].strip():
            last_err = "the big LM did not return usable code"
            continue
        prev_code = spec["code"]
        name = str(spec.get("name") or idea.get("name") or "").strip().lower()
        last_name = name or last_name
        test = spec.get("test")
        if not isinstance(test, dict) or not isinstance(test.get("input"), dict):
            test = {"input": {}}
        last_test = test
        manifest = {"name": name, "description": _s(idea.get("title"), 300),
                    "author": "toolsmith",
                    "uses_faculties": bool(spec.get("uses_faculties")),
                    "parameters": spec.get("parameters")}
        last_manifest = manifest
        if not _NAME_RE.match(name):
            last_err = f"'{name}' is not a valid tool name (lower_snake_case, 3-40 chars)"
            continue
        if toolbox.has(name):
            last_err = f"the name '{name}' is already taken by another tool — pick a new one"
            continue
        res = toolbox.install(name, prev_code, manifest, test,
                              author="toolsmith", overwrite=False)
        if res.get("ok"):
            toolbox.remove_idea(idea["id"])              # the tool itself is the record now
            return {"action": "built", "name": name, "title": idea.get("title"),
                    "attempts": session}
        last_err = _s(res.get("error"), 1200)

    # -- session over: bank the failure for a later analyse-and-retry, or park --
    exhausted = session >= int(max_attempts)
    patch = {"status": "parked" if exhausted else "failed", "attempts": session,
             "last_error": last_err, "last_code": _s(prev_code, 8000),
             "name": _s(last_name, 60)}
    if last_manifest is not None:                        # keep an older banked one otherwise
        patch["last_manifest"] = last_manifest
    if last_test is not None:
        patch["last_test"] = last_test
    toolbox.update_idea(idea["id"], **patch)
    return {"action": "parked" if exhausted else "failed", "title": idea.get("title"),
            "attempts": session, "reason": last_err}


# ── the idle entry point: both questions, in order ───────────────────────────

async def run(toolbox, chat_json: tp.Callable, *, logs: list | None = None,
              faculties: list | None = None, context: str = "",
              guidance: tp.Callable | None = None, max_repair: int = 2,
              max_tools: int = 24, max_attempts: int = 3, max_queue: int = 10) -> dict:
    """One toolsmith pass = the two questions Dan specified: (1) is a tool missing? →
    queue a plain-language spec; (2) is there a queued spec to build (or a failed one to
    re-analyse and re-test)? → attempt exactly one.  Returns
    {identified: {...}, build: {...}} for the log/trace."""
    identified = await identify(toolbox, chat_json, logs=logs, faculties=faculties,
                                context=context, max_queue=max_queue)
    build = await build_next(toolbox, chat_json, guidance=guidance,
                             max_repair=max_repair, max_tools=max_tools,
                             max_attempts=max_attempts)
    return {"identified": identified, "build": build}
