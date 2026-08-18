#!/usr/bin/env python
"""VIN-TOOL-01 constrained emission (graphsmith.py) — the TG3 stage.

What must stay true:
  * the emission schema is DERIVED from the live registry (a block enum per
    step, that block's real params and ports) and uses only JSON-schema
    keywords llama-server's grammar converter understands — the §11
    "cannot produce out-of-vocabulary calls" criterion, testable offline;
  * the repair loop feeds validator errors back verbatim and gives up with
    evidence (Gap-Report material), never silently;
  * T1 reconfigure can express ONLY the deployed graph's steps and their
    declared params, and re-validates after the merge;
  * the whole compose prompt stays inside the §2.1 context budget.

    python assistant/test_graphsmith.py
"""
import asyncio
import json

import blocks
import palette  # noqa: F401 — populates the registry
import graphsmith as gs
import toolgraph

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


# ── canned graphs ─────────────────────────────────────────────────────────────
def good_graph():
    return {"name": "morning_news", "goal": "A weather digest from my feed.",
            "steps": [
                {"id": "s1", "block": "rss_fetch",
                 "inputs": {"feed": {"url": "http://a/feed"}}},
                {"id": "s2", "block": "sort_rank",
                 "inputs": {"docs": "$s1.docs", "query": {"text": "weather"}}},
                {"id": "s3", "block": "digest_render",
                 "inputs": {"ranked": "$s2.ranked"}}],
            "outputs": {"result": "$s3.digest"}}


def miswired_graph():
    g = good_graph()
    g["steps"][2]["inputs"]["ranked"] = "$s1.docs"   # List into a Ranked port
    return g


class FakeChat:
    """Scripted replies; records every (prompt, schema, name) it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, prompt, schema, name):
        self.calls.append({"prompt": prompt, "schema": schema, "name": name})
        return self.replies.pop(0) if self.replies else None


# ── the schema itself ─────────────────────────────────────────────────────────
_LLAMA_KEYWORDS = {"type", "properties", "required", "additionalProperties",
                   "items", "enum", "const", "oneOf", "pattern",
                   "minItems", "maxItems", "description", "default"}


def _walk_schemas(s, bad):
    """Visit every schema node; collect keys outside the llama-safe subset."""
    if not isinstance(s, dict):
        return
    for k in s:
        if k not in _LLAMA_KEYWORDS:
            bad.add(k)
    for sub in (s.get("properties") or {}).values():
        _walk_schemas(sub, bad)
    if isinstance(s.get("items"), dict):
        _walk_schemas(s["items"], bad)
    if isinstance(s.get("additionalProperties"), dict):
        _walk_schemas(s["additionalProperties"], bad)
    for sub in s.get("oneOf") or []:
        _walk_schemas(sub, bad)


def test_emission_schema():
    s = gs.emission_schema()
    steps = s["properties"]["steps"]["items"]["oneOf"]
    check("one step schema per registered block",
          len(steps) == len(blocks.names()))
    consts = {st["properties"]["block"]["const"] for st in steps}
    check("every step schema pins a real block name",
          consts == set(blocks.names()))
    rss = next(st for st in steps
               if st["properties"]["block"]["const"] == "rss_fetch")
    check("a step admits only its block's declared ports",
          set(rss["properties"]["inputs"]["properties"]) == {"feed"}
          and rss["properties"]["inputs"]["additionalProperties"] is False)
    check("a step's params schema is the block's own derived schema",
          rss["properties"]["params"]
          == blocks.params_json_schema(blocks.get("rss_fetch")))
    fp = next(st for st in steps
              if st["properties"]["block"]["const"] == "filter_predicate")
    check("a block with required params requires the params object",
          "params" in fp["required"])
    check("container inputs are reference-only",
          fp["properties"]["inputs"]["properties"]["docs"]
          == {"type": "string", "pattern": gs._REF_PATTERN})
    check("FeedRef inputs admit an inline literal too",
          "oneOf" in rss["properties"]["inputs"]["properties"]["feed"])
    bad = set()
    _walk_schemas(s, bad)
    check("schema uses only llama-convertible keywords", not bad, str(sorted(bad)))
    check("nothing extra is expressible at the top level",
          s["additionalProperties"] is False
          and s["properties"]["outputs"]["additionalProperties"] is False)


# ── T2 compose ────────────────────────────────────────────────────────────────
def test_emit_graph_happy():
    chat = FakeChat([good_graph()])
    res = asyncio.run(gs.emit_graph(chat, "weather digest"))
    check("a valid emission returns a pinned graph", res["ok"]
          and res["attempts"] == 1
          and all(s["block_version"] == "1.0.0" for s in res["graph"]["steps"]))
    check("the chat was constrained by the emission schema",
          chat.calls[0]["schema"] == gs.emission_schema()
          and chat.calls[0]["name"] == "tool_graph")
    check("capabilities computed for the approval gate",
          res["validation"].capabilities == {"net"})


def test_emit_graph_repair():
    chat = FakeChat([miswired_graph(), good_graph()])
    res = asyncio.run(gs.emit_graph(chat, "weather digest"))
    check("a mis-wired first try is repaired on evidence",
          res["ok"] and res["attempts"] == 2)
    check("the retry prompt carries the validator's words",
          "adapter" in chat.calls[1]["prompt"])


def test_emit_graph_gives_up_with_evidence():
    chat = FakeChat([miswired_graph(), miswired_graph()])
    res = asyncio.run(gs.emit_graph(chat, "weather digest", max_repair=1))
    check("an unrepairable goal fails with evidence, not silently",
          not res["ok"] and res["graph"] is None
          and res["raw"] is not None and "adapter" in (res["feedback"] or ""))
    check("a non-JSON reply is retried, then reported",
          not asyncio.run(gs.emit_graph(FakeChat([None, None]), "x"))["ok"])


# ── T1 configure ──────────────────────────────────────────────────────────────
def _pinned():
    g = good_graph()
    g["steps"].insert(1, {"id": "sf", "block": "filter_predicate",
                          "params": {"terms": ["weather"]},
                          "inputs": {"docs": "$s1.docs"}})
    g["steps"][2]["inputs"]["docs"] = "$sf.docs"
    v = toolgraph.validate(g)
    assert v.ok, v.errors
    return v.pinned


def test_t1_schema():
    pinned = _pinned()
    s = gs.t1_schema(pinned)
    check("every step with params is reconfigurable",
          set(s["properties"]) == {"s1", "sf", "s2", "s3"})
    # a step whose block declares NO params must not appear at all
    g = {"name": "strip", "goal": "g",
         "steps": [{"id": "p1", "block": "parse_html",
                    "inputs": {"doc": {"text": "<b>x</b>"}}}],
         "outputs": {"doc": "$p1.doc"}}
    v = toolgraph.validate(g)
    check("a no-params step is not reconfigurable",
          v.ok and gs.t1_schema(v.pinned)["properties"] == {})
    check("params are all optional for reconfigure",
          all("required" not in ps for ps in s["properties"].values()))
    check("nothing outside the graph is expressible",
          s["additionalProperties"] is False
          and s["properties"]["sf"]["additionalProperties"] is False)
    bad = set()
    _walk_schemas(s, bad)
    check("t1 schema is llama-convertible too", not bad, str(sorted(bad)))


def test_apply_t1():
    pinned = _pinned()
    v = gs.apply_t1(pinned, {"sf": {"terms": ["storm", "flood"]}})
    check("a param change merges and revalidates", v.ok)
    new = next(s for s in v.pinned["steps"] if s["id"] == "sf")
    check("…and the new value landed", new["params"]["terms"] == ["storm", "flood"])
    check("unchanged params survive the merge", new["params"]["mode"] == "any")
    check("an unknown step is rejected",
          not gs.apply_t1(pinned, {"ghost": {"n": 1}}).ok)
    check("a wrong-typed value is caught by revalidation",
          not gs.apply_t1(pinned, {"sf": {"terms": "storm"}}).ok)


def test_emit_reconfigure():
    pinned = _pinned()
    chat = FakeChat([{"sf": {"terms": ["storm"]}}])
    res = asyncio.run(gs.emit_reconfigure(chat, pinned, "storms instead"))
    check("reconfigure emits under the graph's own schema",
          res["ok"] and chat.calls[0]["schema"] == gs.t1_schema(pinned))
    check("the current values are in the prompt (evidence to edit)",
          '"weather"' in chat.calls[0]["prompt"])


def test_prompt_budget():
    p = gs.compose_prompt("a goal")
    check(f"compose prompt fits the §2.1 budget ({len(p)} chars)",
          len(p) < 32000)
    check("the prompt carries no block source",
          "def " not in p and "lambda" not in p)


def main():
    test_emission_schema()
    test_emit_graph_happy()
    test_emit_graph_repair()
    test_emit_graph_gives_up_with_evidence()
    test_t1_schema()
    test_apply_t1()
    test_emit_reconfigure()
    test_prompt_budget()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
