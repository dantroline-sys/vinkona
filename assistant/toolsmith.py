"""
toolsmith.py — Vinkona's idle tool-maker.

Between conversations, she asks herself a simple question: *looking at how things have
been going, is there a small tool that would help — or a problem I can see a fix for?*
Two outcomes, both surfaced in the same Tools panel section:

  * **build** — if it's a small, safe Python script, she writes it (``tool.py`` +
    manifest + a self-test), and it goes in through ``Toolbox.install()``, which runs the
    self-test in a throwaway sandbox and only promotes a tool that actually runs.  If the
    self-test fails she gets the traceback back and tries again, up to a small cap — a
    bounded repair loop, not open-ended flailing.
  * **idea** — if it's worth having but she can't code it (needs a capability she doesn't
    have yet, or it's simply too big), she records it as a *tool idea* so you can see it and
    finish it.  A build that never passes its self-test degrades to an idea too, so the
    effort is never lost.

The security story is unchanged: whatever she writes runs in the same sandbox as every
other own-tool — read anywhere (or the shared folders), write only in her store, no
network, killed on timeout.  The big LM only produces *text*; nothing it writes runs
outside that box, and nothing is offered to her mid-conversation until it has passed a
self-test.

This module is deliberately HTTP-agnostic: it takes a ``chat_json`` callable (the caller
binds the big-LM URL + model), so it has no aiohttp dependency and is trivial to unit-test
with a stub LM.  One tool is built (or one idea recorded) per invocation — cheap, bounded,
and easy to review in the panel afterwards.
"""
from __future__ import annotations

import json
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


def _s(v: tp.Any, n: int = 4000) -> str:
    return str(v or "")[:n]


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


def _decide_prompt(*, context: str, roster: list, ideas: list, faculties: list,
                   usage: dict, logs: list, max_reached: bool, fac_allow: list) -> str:
    have = "\n".join(f"- {t['name']}: {t.get('description','')}" for t in roster) \
        or "(none yet)"
    idea_lines = "\n".join(f"- {i.get('title','')}" for i in ideas) or "(none yet)"
    fac = "\n".join(f"- {f}" for f in faculties[:40]) or "(none)"
    convo = "\n".join(f"{r.get('role','?').upper()}: {_s(r.get('text'), 500)}"
                      for r in logs[-24:]) or "(no recent interactions)"
    unused = [n for t in roster if (n := t["name"]) and usage.get(n, {}).get("calls", 0) == 0]
    unused_note = (f"\nTools you built but have never used: {', '.join(unused)}."
                   if unused else "")
    build_note = ("\nYou already have plenty of tools — DO NOT choose \"build\" this time; "
                  "only \"idea\" or \"none\"." if max_reached else "")
    fac_note = (f"\nA tool you build MAY call these faculties of yours: {', '.join(fac_allow)}."
                if fac_allow else "")
    return f"""{context}
You can make small tools for yourself to use during conversations. Between chats you review
how things have gone and decide whether a new one would help.

{_TOOL_CONTRACT}

Tools you have already (do NOT duplicate these):
{have}

Things you can already do with your other faculties (do NOT rebuild these):
{fac}

Tool ideas already noted (do NOT repeat these):
{idea_lines}{unused_note}

Recent interactions:
{convo}

Ask yourself: is there a small, safe, single-purpose tool that would have helped, or a
problem you can see and a concrete fix for?{fac_note}{build_note}

Reply with ONE JSON object:
- To build one now (only if it fits the rules above as a small Python script):
  {{"decision":"build","name":"snake_case_name","description":"one line for the menu",
    "rationale":"why it helps","plan":"inputs, what it does, the JSON it returns"}}
- To note a good idea you can't build yourself (needs a capability you lack, or too big):
  {{"decision":"idea","title":"short title","rationale":"why it would help, and what it needs"}}
- If nothing is worth adding right now:
  {{"decision":"none"}}
Choose exactly one. Prefer "none" over a weak or redundant tool."""


def _code_prompt(*, name: str, description: str, plan: str, fac_allow: list,
                 prev_code: str = "", error: str = "") -> str:
    base = f"""{_TOOL_CONTRACT}
{_faculties_block(fac_allow)}

Write the tool "{name}".
What it should do: {description}
Plan: {plan}

Reply with ONE JSON object:
{{"code":"<the full Python file as a string>",
  "parameters":{{"type":"object","properties":{{...}},"required":[...]}},
  "uses_faculties": false,
  "test":{{"input":{{...}},"expect_keys":["..."]}}}}
- "parameters" is the JSON-schema of the stdin arguments (for the menu).
- Set "uses_faculties": true ONLY if the code imports the faculties helper (see above).
- "test.input" is a concrete arguments object that exercises the tool; "expect_keys" are
  keys that MUST appear in its output. The test runs in the sandbox and must pass, so pick
  an input that works there (e.g. read a file that exists like /etc/hostname)."""
    if error:
        base += (f"\n\nYour previous attempt FAILED its self-test with:\n{_s(error, 1200)}\n\n"
                 f"Previous code:\n{_s(prev_code, 4000)}\n\nFix it and return the full "
                 "corrected JSON object.")
    return base


async def run(toolbox, chat_json: tp.Callable, *, logs: list | None = None,
              faculties: list | None = None, context: str = "",
              max_repair: int = 2, max_tools: int = 24) -> dict:
    """One toolsmith pass.  ``chat_json`` is ``async (prompt, think=True) -> dict|None``.
    Returns a status dict describing what happened (for logging / the trace feed):
    {action: built|idea|failed|none|error, name?/title?, attempts?, reason?}."""
    roster = toolbox.describe()
    ideas = toolbox.ideas()
    usage = toolbox.usage()
    faculties = faculties or []
    logs = logs or []
    at_cap = len(roster) >= int(max_tools)
    # Faculties she may let a tool call (empty unless enabled) — teaches the LM what a tool
    # can reach, so it can write a faculties tool when it helps.
    _fc = (getattr(toolbox, "cfg", {}) or {}).get("faculties") or {}
    fac_allow = list(_fc.get("allow") or []) if _fc.get("enabled") else []

    decision = await chat_json(_decide_prompt(
        context=context, roster=roster, ideas=ideas, faculties=faculties,
        usage=usage, logs=logs, max_reached=at_cap, fac_allow=fac_allow), think=True)
    if not isinstance(decision, dict):
        return {"action": "none", "reason": "no decision from the big LM"}

    kind = str(decision.get("decision") or "none").lower()

    # --- an idea she can't build herself ---
    if kind == "idea" or (kind == "build" and at_cap):
        title = _s(decision.get("title") or decision.get("name")
                   or decision.get("description"), 120)
        if not title:
            return {"action": "none", "reason": "idea had no title"}
        r = toolbox.add_idea(title, rationale=_s(decision.get("rationale"), 2000),
                             sketch=_s(decision.get("plan"), 4000), source="toolsmith")
        return ({"action": "idea", "title": title} if r.get("ok")
                else {"action": "none", "reason": r.get("error")})

    if kind != "build":
        return {"action": "none"}

    name = str(decision.get("name") or "").strip().lower()
    description = _s(decision.get("description") or name, 300)
    plan = _s(decision.get("plan"), 4000)
    if toolbox.has(name):
        # collides with an existing tool — bank it as an idea rather than clobber a
        # working tool (install would refuse the overwrite anyway).
        toolbox.add_idea(name, rationale="(the name collided with an existing tool) "
                         + _s(decision.get("rationale"), 1800), sketch=plan)
        return {"action": "idea", "title": name, "reason": "name collision"}

    # --- build it, with a bounded repair loop ---
    prev_code, last_err = "", ""
    for attempt in range(1, int(max_repair) + 2):        # first try + max_repair retries
        spec = await chat_json(_code_prompt(
            name=name, description=description, plan=plan, fac_allow=fac_allow,
            prev_code=prev_code, error=last_err), think=True)
        if not isinstance(spec, dict) or not isinstance(spec.get("code"), str):
            last_err = "the big LM did not return usable code"
            continue
        prev_code = spec["code"]
        manifest = {"description": description, "author": "toolsmith",
                    "uses_faculties": bool(spec.get("uses_faculties")),
                    "parameters": spec.get("parameters")}
        test = spec.get("test")
        if not isinstance(test, dict) or not isinstance(test.get("input"), dict):
            test = {"input": {}}
        res = toolbox.install(name, prev_code, manifest, test,
                              author="toolsmith", overwrite=False)
        if res.get("ok"):
            return {"action": "built", "name": name, "attempts": attempt}
        last_err = _s(res.get("error"), 1200)

    # exhausted repairs → keep the effort as an idea with the sketch + the last error
    toolbox.add_idea(description or name,
                     rationale=_s(decision.get("rationale"), 1600)
                     + f"  (tried to build '{name}' but it kept failing its self-test: {last_err})",
                     sketch=prev_code, source="toolsmith")
    return {"action": "failed", "name": name, "attempts": int(max_repair) + 1,
            "reason": last_err}
