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
import time
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


# The build harness asks for the code as PLAIN TEXT in a fenced block, not as a string
# escaped inside a JSON object — mid-size models reliably write correct Python but fumble
# JSON-escaping a multiline program.  Metadata (name/schema/test) is a separate, small JSON
# ask afterwards, and degrades to defaults rather than failing the build.

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
    if error:
        base += (f"\n\nYour previous attempt FAILED:\n{_s(error, 1200)}\n\n"
                 f"Previous code:\n```python\n{_s(prev_code, 4000)}\n```\nFix it.")
    base += """

Reply with ONLY the complete Python file in one fenced code block:
```python
<the whole tool>
```
No JSON, no commentary outside the fence.  If the spec CANNOT be built under the rules
(needs the network, a package, a capability you don't have), reply instead with one line:
UNBUILDABLE: <why>"""
    return base


def _metadata_prompt(*, idea: dict, code: str) -> str:
    return f"""This finished tool of yours needs its menu entry.  Read the code and describe it.

Spec: {_s(idea.get('title'), 120)}

```python
{_s(code, 6000)}
```

Reply with ONE JSON object:
{{"name": "snake_case_name",
  "parameters": {{"type":"object","properties":{{...}},"required":[...]}},
  "test": {{"input": {{...}}, "expect_keys": ["..."], "faculty_stubs": {{...}}}}}}
- "name": lower_snake_case, 3-40 chars, matching what the tool does.
- "parameters": the JSON-schema of the stdin arguments the CODE actually reads.
- "test.input": concrete arguments that exercise it; "expect_keys": keys that MUST appear
  in its output.  The test runs in the sandbox, so pick input that works there (e.g. read a
  file that exists, like /etc/hostname).  Include "faculty_stubs" (a canned result per
  faculty the code calls) ONLY if it uses the faculties helper."""


def _slug(title: str) -> str:
    """A usable fallback tool name derived from the spec title (metadata step flaked)."""
    s = re.sub(r"[^a-z0-9]+", "_", str(title or "tool").lower()).strip("_")[:40]
    if not s or not s[0].isalpha():
        s = "t_" + s
    return (s + "_tool")[:40] if len(s) < 3 else s[:40]


def _uniquify(name: str, taken) -> str:
    if not taken(name):
        return name
    for n in range(2, 10):
        cand = f"{name[:37]}_{n}"
        if not taken(cand):
            return cand
    return f"{name[:30]}_{time.monotonic_ns() % 10000}"


def _extract_code(text: str) -> str:
    """The Python file out of a model reply: the LONGEST fenced block (models sometimes emit
    a short example fence before the real one), else the whole reply as-is."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.DOTALL)
    if fences:
        return max(fences, key=len).strip()
    return (text or "").strip()


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


async def build_next(toolbox, chat_json: tp.Callable, chat_text: tp.Callable, *,
                     guidance: tp.Callable | None = None,
                     max_repair: int = 2, max_tools: int = 24,
                     max_attempts: int = 3) -> dict:
    """Phase 2: take one queued spec and attempt it, through the build HARNESS that helps a
    mid-size model iterate until the code runs:

      1. CODE as plain text in a fenced block (``chat_text`` — no JSON-escaping a multiline
         program, the classic mid-model failure).
      2. A free local SYNTAX gate (compile()) — a SyntaxError is fed straight back for
         another go without spending a sandbox run.
      3. METADATA (name/schema/self-test) as a separate small JSON ask (``chat_json``) that
         DEGRADES to defaults (slug name, empty schema, bare test) instead of failing the
         build — only code quality can fail a build, never menu decoration.
      4. The sandbox self-test; its traceback loops back to step 1 (max_repair times).

    Whatever happens, the attempt is banked on the idea — the extracted code even when it
    doesn't parse, and the RAW model reply when nothing usable could be extracted — so the
    panel can always show what actually came back.  A failed spec is re-analysed on a later
    cycle; `guidance` is an optional ``async (question) -> str|None`` (kb_ask) consulted
    when the analysis asks for it.  Returns {action: built|failed|parked|none, …}."""
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

    fac = _fac_allow(toolbox)
    # A retry session starts from the banked attempt: the code AND the error it died with,
    # so the first code prompt of the session carries the full failing context (not just
    # the analysis's adjustment).
    prev_code = _s(idea.get("last_code"), 4000) if idea.get("status") == "failed" else ""
    last_err = _s(idea.get("last_error"), 1200) if idea.get("status") == "failed" else ""
    last_raw = ""
    last_name = str(idea.get("name") or "")
    last_manifest = last_test = None
    for _ in range(1, int(max_repair) + 2):              # first try + max_repair retries
        # 1. the code, as plain fenced text
        txt = await chat_text(_code_prompt(
            idea=idea, fac_allow=fac, adjustment=adjustment, guidance=kb_text,
            prev_code=prev_code, error=last_err))
        if not (isinstance(txt, str) and txt.strip()):
            last_err = ("code step: the big LM returned no reply "
                        "(timed out, errored, or produced an empty message)")
            continue
        last_raw = _s(txt, 6000)
        first = txt.strip().splitlines()[0].strip()
        if first.upper().startswith("UNBUILDABLE"):
            reason = first.split(":", 1)[1].strip() if ":" in first else "judged unbuildable"
            toolbox.update_idea(idea["id"], status="parked", last_error=_s(reason, 400))
            return {"action": "parked", "title": idea.get("title"),
                    "reason": _s(reason, 200)}
        code = _extract_code(txt)
        prev_code = code                                 # always shown to the next round
        # 2. the free syntax gate — no sandbox run spent on a file that can't parse
        try:
            compile(code, "<tool>", "exec")
        except SyntaxError as e:
            last_err = f"code step: SyntaxError: {e.msg} (line {e.lineno})"
            continue
        # 3. metadata — a small ask that degrades to defaults, never fails the build
        meta = await chat_json(_metadata_prompt(idea=idea, code=code), think=False)
        meta = meta if isinstance(meta, dict) else {}
        name = str(meta.get("name") or idea.get("name") or _slug(idea.get("title"))
                   ).strip().lower()
        if not _NAME_RE.match(name):
            name = _slug(idea.get("title"))
        name = _uniquify(name, toolbox.has)
        last_name = name
        test = meta.get("test")
        if not isinstance(test, dict) or not isinstance(test.get("input"), dict):
            test = {"input": {}}
        last_test = test
        params = meta.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            params = {"type": "object", "properties": {}}
        manifest = {"name": name, "description": _s(idea.get("title"), 300),
                    "author": "toolsmith",
                    "uses_faculties": "import faculties" in code,   # detected, not asked
                    "parameters": params}
        last_manifest = manifest
        # 4. the sandbox self-test
        res = toolbox.install(name, code, manifest, test,
                              author="toolsmith", overwrite=False)
        if res.get("ok"):
            toolbox.remove_idea(idea["id"])              # the tool itself is the record now
            return {"action": "built", "name": name, "title": idea.get("title"),
                    "attempts": session}
        last_err = _s(res.get("error"), 1200)

    # -- session over: bank the WHOLE attempt (faulty code, raw reply, error) for the
    #    panel's inspector and a later analyse-and-retry — or park when the budget's spent --
    exhausted = session >= int(max_attempts)
    patch = {"status": "parked" if exhausted else "failed", "attempts": session,
             "last_error": last_err, "last_code": _s(prev_code, 8000),
             "last_raw": last_raw, "name": _s(last_name, 60)}
    if last_manifest is not None:                        # keep an older banked one otherwise
        patch["last_manifest"] = last_manifest
    if last_test is not None:
        patch["last_test"] = last_test
    toolbox.update_idea(idea["id"], **patch)
    return {"action": "parked" if exhausted else "failed", "title": idea.get("title"),
            "attempts": session, "reason": last_err}


# ── the idle entry point: both questions, in order ───────────────────────────

async def run(toolbox, chat_json: tp.Callable, chat_text: tp.Callable, *,
              logs: list | None = None,
              faculties: list | None = None, context: str = "",
              guidance: tp.Callable | None = None, max_repair: int = 2,
              max_tools: int = 24, max_attempts: int = 3, max_queue: int = 10) -> dict:
    """One toolsmith pass = the two questions Dan specified: (1) is a tool missing? →
    queue a plain-language spec; (2) is there a queued spec to build (or a failed one to
    re-analyse and re-test)? → attempt exactly one.  ``chat_json`` serves the structured
    steps (identify/analysis/metadata); ``chat_text`` serves code generation.  Returns
    {identified: {...}, build: {...}} for the log/trace."""
    identified = await identify(toolbox, chat_json, logs=logs, faculties=faculties,
                                context=context, max_queue=max_queue)
    build = await build_next(toolbox, chat_json, chat_text, guidance=guidance,
                             max_repair=max_repair, max_tools=max_tools,
                             max_attempts=max_attempts)
    return {"identified": identified, "build": build}
