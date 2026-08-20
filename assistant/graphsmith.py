"""
VIN-TOOL-01 TG3: constrained emission — she wires graphs; the server's grammar
keeps her in vocabulary.

The big LM sees the SOURCELESS catalogue (blocks.catalogue()) and a JSON schema
derived from it at call time; llama-server converts that schema to a grammar per
request (§0.3), so a step naming an unknown block, an undeclared param, or a
malformed reference cannot even be tokenised.  The §6.1 validator then judges
what grammar cannot: edge types, cycles, dangling references.  Emission failures
feed back validator errors verbatim (evidence-driven, like §10's repair loop but
for wiring); an unrepairable goal returns its evidence for a Gap Report (§8).

Two pathways here, matching §5:
  T2 compose   emit_graph(chat, goal)            — a whole new graph
  T1 configure emit_reconfigure(chat, pinned, request) — new params for a
               deployed graph, schema'd so ONLY that graph's steps and their
               declared params can be touched

The `chat` seam is any async callable (prompt, schema, schema_name) -> dict|None;
memory._chat_json(..., schema=, schema_name=) fits via a lambda.  Pure stdlib.
"""
from __future__ import annotations

import json

import blocks as _blocks
import toolgraph as _tg

# Emission is deliberately STRICTER than the validator: containers must be
# wired by reference (you don't inline a document list into a graph), and only
# these types may appear as inline literals.
_LITERAL_TYPES = {
    "FeedRef": {"type": "object", "properties": {"url": {"type": "string"}},
                "required": ["url"], "additionalProperties": False},
    "Query": {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False},
}

_ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_REF_PATTERN = r"^\$[a-z][a-z0-9_]{0,31}\.[a-z][a-z0-9_]*$"
_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
MAX_STEPS = 12


def _input_schema(semantic_type: str, resources: dict | None) -> dict:
    ref = {"type": "string", "pattern": _REF_PATTERN}
    lit = _LITERAL_TYPES.get(semantic_type)
    if lit is None:
        return ref
    if semantic_type == "FeedRef" and resources is not None:
        # Grounding: the closed vocabulary extends to INSTANCE data.  With a
        # resources dict present, feed URLs are an enum of the user's
        # configured sources — inventing one is inexpressible at the server;
        # with none configured, FeedRef literals vanish (reference-only).
        feeds = list(resources.get("feeds") or [])
        if not feeds:
            return ref
        lit = {"type": "object",
               "properties": {"url": {"type": "string", "enum": feeds}},
               "required": ["url"], "additionalProperties": False}
    return {"oneOf": [ref, lit]}


def _step_schema(b, resources: dict | None) -> dict:
    props = {"id": {"type": "string", "pattern": _ID_PATTERN},
             "block": {"const": b.name},
             "inputs": {"type": "object",
                        "properties": {p: _input_schema(t, resources)
                                       for p, t in sorted(b.ports_in.items())},
                        "required": sorted(b.ports_in),
                        "additionalProperties": False}}
    required = ["id", "block", "inputs"]
    if b.params:
        props["params"] = _blocks.params_json_schema(b)
        if props["params"].get("required"):
            required.insert(2, "params")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def emission_schema(resources: dict | None = None) -> dict:
    """The whole-graph schema, derived from the live registry at call time —
    nothing stored, nothing to drift (§0.1).  One output, named 'result'.
    `resources` (e.g. {"feeds": [urls]}) grounds instance data — see
    _input_schema."""
    step_schemas = [_step_schema(_blocks.get(n), resources)
                    for n in _blocks.names()]
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": _NAME_PATTERN},
            "goal": {"type": "string"},
            "steps": {"type": "array", "minItems": 1, "maxItems": MAX_STEPS,
                      "items": {"oneOf": step_schemas}},
            "outputs": {"type": "object",
                        "properties": {"result": {"type": "string",
                                                  "pattern": _REF_PATTERN}},
                        "required": ["result"],
                        "additionalProperties": False},
        },
        "required": ["name", "goal", "steps", "outputs"],
        "additionalProperties": False,
    }


def t1_schema(pinned: dict) -> dict:
    """Reconfigure schema for ONE deployed graph: an object keyed by that
    graph's step ids, each admitting only that block's declared params — all
    optional (you change what you change).  Nothing else is expressible."""
    props = {}
    for s in pinned.get("steps", []):
        b = _blocks.get(s["block"], s.get("block_version"))
        if not b.params:
            continue
        ps = _blocks.params_json_schema(b)
        ps.pop("required", None)          # reconfigure touches any subset
        props[s["id"]] = ps
    return {"type": "object", "properties": props, "additionalProperties": False}


# ── Prompts (≤ 8k-token budget, §2.1 — the catalogue is small by design) ─────
def compose_prompt(goal: str, feedback: str | None = None,
                   resources: dict | None = None) -> str:
    cat = json.dumps(_blocks.catalogue(), indent=1)
    feeds = list((resources or {}).get("feeds") or [])
    parts = [
        "You design a small data-pipeline (a 'tool graph') for the assistant "
        "Vinkona.  You may ONLY use blocks from this catalogue — no other "
        "operation exists:",
        cat,
        ("The ONLY external feed URLs that exist are these — any other URL "
         "would be an invention:\n" + "\n".join(f"- {u}" for u in feeds))
        if feeds else
        ("No external feed URLs are configured on this machine — do NOT use "
         "rss_fetch or invent a URL; source news from her own archive with "
         "news_fetch instead."),
        "Rules:\n"
        "- Steps run as a one-way pipeline.  Each step: "
        '{"id", "block", "params", "inputs"}.\n'
        '- Wire data with references "$<step id>.<output port>".  List/Ranked '
        "inputs MUST be references to an earlier step; FeedRef/Query inputs "
        'may be inline ({"url": …} / {"text": …}).\n'
        "- Port types must line up (a Ranked[Document] output may feed a "
        "List[Document] input; nothing else crosses types — use the adapter "
        "blocks to bridge).\n"
        '- "goal" is one plain sentence a non-technical user reads.\n'
        '- "outputs" must be {"result": "$<step>.<port>"} — the graph\'s '
        "final product.",
        f"Task: {goal.strip()}",
    ]
    if feedback:
        parts.append("Your previous graph failed validation:\n" + feedback
                     + "\nEmit a corrected graph.")
    parts.append("Reply with ONLY the JSON graph.")
    return "\n\n".join(parts)


def reconfigure_prompt(pinned: dict, request: str,
                       feedback: str | None = None) -> str:
    current = json.dumps(
        [{"id": s["id"], "block": s["block"], "params": s.get("params", {})}
         for s in pinned.get("steps", [])], indent=1)
    parts = [
        f"The assistant Vinkona has a deployed tool '{pinned.get('name')}' "
        f"whose goal is: {pinned.get('goal')}",
        "Its steps and current parameter values:", current,
        "Change ONLY parameter values to satisfy this request — you cannot "
        "add, remove, or rewire steps:",
        f"Request: {request.strip()}",
        'Reply with ONLY a JSON object of the form '
        '{"<step id>": {"<param>": <new value>}} containing just the '
        "parameters you are changing.",
    ]
    if feedback:
        parts.append("Your previous change failed validation:\n" + feedback
                     + "\nEmit a corrected change.")
    return "\n\n".join(parts)


def apply_t1(pinned: dict, changes: dict) -> _tg.Validation:
    """Merge a T1 param-change object into a pinned graph and re-run the
    static pass.  Unknown steps/params fail — the schema makes them
    inexpressible under constraint, and the validator catches them anyway."""
    v = _tg.Validation()
    if not isinstance(changes, dict):
        v.errors.append("the change is not an object")
        return v
    ids = {s["id"] for s in pinned.get("steps", [])}
    unknown = sorted(set(changes) - ids)
    if unknown:
        v.errors.append(f"no such steps: {unknown}")
        return v
    g = {**pinned, "steps": [dict(s) for s in pinned["steps"]]}
    for s in g["steps"]:
        delta = changes.get(s["id"])
        if isinstance(delta, dict) and delta:
            s["params"] = {**(s.get("params") or {}), **delta}
    return _tg.validate(g)


# ── The emission loops ────────────────────────────────────────────────────────
def _feedback(v: _tg.Validation | None, raw) -> str:
    if raw is None:
        return "- the reply was empty or not JSON"
    if v is None:
        return "- the reply was not a JSON object"
    return "\n".join(f"- {e}" for e in v.errors[:12])


async def emit_graph(chat, goal: str, *, max_repair: int = 1,
                     resources: dict | None = None) -> dict:
    """T2 compose.  Returns {"ok", "graph" (pinned), "validation", "raw",
    "attempts"} — on failure the evidence is Gap-Report material (§8).
    `resources` grounds instance data (configured feeds) in BOTH the schema
    (invention inexpressible) and the validator (belt and braces)."""
    known = list((resources or {}).get("feeds") or []) if resources is not None \
        else None
    fb, raw, v = None, None, None
    attempts = 0
    for _ in range(1 + max(0, max_repair)):
        attempts += 1
        raw = await chat(compose_prompt(goal, fb, resources),
                         emission_schema(resources), "tool_graph")
        v = _tg.validate(raw, known_feeds=known) if isinstance(raw, dict) else None
        if v is not None and v.ok:
            return {"ok": True, "graph": v.pinned, "validation": v,
                    "raw": raw, "attempts": attempts}
        fb = _feedback(v, raw)
    return {"ok": False, "graph": None, "validation": v, "raw": raw,
            "attempts": attempts, "feedback": fb}


async def emit_reconfigure(chat, pinned: dict, request: str, *,
                           max_repair: int = 1) -> dict:
    """T1 configure.  Same result shape; "graph" is the revalidated pinned
    graph with the new params."""
    fb, raw, v = None, None, None
    attempts = 0
    for _ in range(1 + max(0, max_repair)):
        attempts += 1
        raw = await chat(reconfigure_prompt(pinned, request, fb),
                         t1_schema(pinned), "tool_reconfigure")
        v = apply_t1(pinned, raw) if isinstance(raw, dict) else None
        if v is not None and v.ok:
            return {"ok": True, "graph": v.pinned, "validation": v,
                    "raw": raw, "attempts": attempts}
        fb = _feedback(v, raw)
    return {"ok": False, "graph": None, "validation": v, "raw": raw,
            "attempts": attempts, "feedback": fb}


# ── Live harness: one command against the configured big LM ──────────────────
#   vinkona_env/bin/python graphsmith.py "a morning digest of rain news"
#   vinkona_env/bin/python graphsmith.py --reconfigure graph.json "storms too"
def main():
    import argparse
    import asyncio
    import importlib.util
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Emit a tool graph live against the configured big LM.")
    ap.add_argument("goal", help="plain-language goal (or a T1 request with "
                                 "--reconfigure)")
    ap.add_argument("--reconfigure", metavar="GRAPH_JSON",
                    help="path to a pinned graph to T1-reconfigure instead")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--max-repair", type=int, default=1)
    args = ap.parse_args()

    here = Path(__file__).parent

    def load(name):
        spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cfg = load("config").load_config(args.config)
    big = cfg.get("big_lm") or {}
    if not big.get("url"):
        raise SystemExit("big_lm.url is not set — start the big LM first")
    mem = load("memory")
    timeout = ((cfg.get("tools") or {}).get("own_tools") or {}) \
        .get("toolsmith", {}).get("lm_timeout_s", 600)

    class _Self:                       # _chat_json uses no instance state
        pass

    async def chat(prompt, schema, name):
        return await mem.MemoryStore._chat_json(
            _Self(), big["url"], big.get("model") or "", prompt,
            think=True, timeout_s=timeout, tag="toolgraph",
            schema=schema, schema_name=name)

    if args.reconfigure:
        pinned = json.loads(Path(args.reconfigure).read_text())
        res = asyncio.run(emit_reconfigure(chat, pinned, args.goal,
                                           max_repair=args.max_repair))
    else:
        tsc = ((cfg.get("tools") or {}).get("own_tools") or {}) \
            .get("toolsmith", {}) or {}
        res = asyncio.run(emit_graph(
            chat, args.goal, max_repair=args.max_repair,
            resources={"feeds": list(tsc.get("feed_sources") or [])}))

    if res["ok"]:
        print(json.dumps(res["graph"], indent=1))
        print(f"\n{_tg.capability_summary(res['validation'])}")
        print(f"approval required: {_tg.needs_approval(res['validation'])}"
              f"   attempts: {res['attempts']}")
    else:
        print("EMISSION FAILED — Gap-Report evidence:")
        print(res.get("feedback") or "(no feedback)")
        print("\nlast raw reply:")
        print(json.dumps(res.get("raw"), indent=1))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
