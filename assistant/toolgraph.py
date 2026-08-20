"""
VIN-TOOL-01 wiring layer (TG2): the tool-graph format and its validator.

A graph is DATA — the only thing Vinkona emits.  Shape:

    {"name": "morning_news", "goal": "plain-language purpose",
     "steps": [
        {"id": "s1", "block": "rss_fetch", "block_version": "1.0.0",
         "params": {"max_items": 20},
         "inputs": {"feed": {"url": "https://example.org/feed"}}},
        {"id": "s2", "block": "sort_rank",
         "inputs": {"docs": "$s1.docs", "query": {"text": "weather"}}}],
     "outputs": {"ranked": "$s2.ranked"}}

Input bindings: a string "$<step>.<port>" is a slot reference; anything else
is a literal (checked against the port's semantic type).  No loops, no
conditionals, no expressions — anything resembling a language here is a spec
violation (§2.1).

validate() is the §6.1 static pass: every error a person can read, plus the
computed capability set and a pinned copy of the graph.  Behaviour follows
Haystack 2.x's pre-run connection validation semantics (§0.4: build thin,
diff behaviour — typed sockets checked before anything runs); cycle detection
is Kahn's algorithm (§0.2: networkx rejected in writing).

Pure stdlib.
"""
from __future__ import annotations

import re

import blocks as _blocks

_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_REF = re.compile(r"^\$([a-z][a-z0-9_]{0,31})\.([a-z][a-z0-9_]*)$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _is_ref(v) -> bool:
    return isinstance(v, str) and v.startswith("$")


def parse_ref(v: str):
    """'$s1.docs' -> ('s1', 'docs') or None if malformed."""
    m = _REF.match(v or "")
    return (m.group(1), m.group(2)) if m else None


class Validation:
    """The static pass's whole result: errors (empty = valid), the computed
    capability set, resolved blocks per step, and a version-pinned graph."""

    def __init__(self):
        self.errors: list = []
        self.capabilities: set = set()
        self.resolved: dict = {}          # step id -> Block
        self.pinned: dict | None = None   # graph with exact block_version filled

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(graph: dict, known_feeds: list | None = None) -> Validation:
    """`known_feeds` grounds FeedRef literals in the user's CONFIGURED sources:
    pass a list (possibly empty) and any literal feed URL outside it is
    rejected — a model cannot know the user's feeds, so an unlisted URL is an
    invention, the §3 never-invent rule applied to the wiring layer.  None
    skips the check (offline validation of hand-written graphs)."""
    v = Validation()
    err = v.errors.append
    if not isinstance(graph, dict):
        err("the graph is not an object")
        return v
    if not (isinstance(graph.get("name"), str) and _NAME.match(graph["name"])):
        err("graph needs a lowercase_name")
    if not (isinstance(graph.get("goal"), str) and graph["goal"].strip()):
        err("graph needs a plain-language goal (the user reads it)")
    steps = graph.get("steps")
    if not (isinstance(steps, list) and steps):
        err("graph needs a non-empty steps list")
        return v

    # ── steps: ids, blocks, versions ─────────────────────────────────────────
    by_id: dict = {}
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            err(f"step {i} is not an object")
            continue
        sid = s.get("id")
        if not (isinstance(sid, str) and _ID.match(sid)):
            err(f"step {i} needs a short lowercase id")
            continue
        if sid in by_id:
            err(f"duplicate step id {sid!r}")
            continue
        by_id[sid] = s
        try:
            b = _blocks.get(str(s.get("block", "")), s.get("block_version"))
        except _blocks.BlockError as e:
            err(f"step {sid}: {e}")
            continue
        v.resolved[sid] = b
        v.capabilities |= set(b.capabilities)
        perrs = _blocks.check_params(b, s.get("params") or {})
        for p in perrs:
            err(f"step {sid} ({b.name}): {p}")

    # ── edges: every in-port bound, refs resolve, types line up ─────────────
    for sid, s in by_id.items():
        b = v.resolved.get(sid)
        if b is None:
            continue
        inputs = s.get("inputs") or {}
        for port in inputs:
            if port not in b.ports_in:
                err(f"step {sid} ({b.name}): no such input port {port!r}")
        for port, want_type in b.ports_in.items():
            if port not in inputs:
                err(f"step {sid} ({b.name}): input {port!r} is unbound")
                continue
            bound = inputs[port]
            if _is_ref(bound):
                ref = parse_ref(bound)
                if not ref:
                    err(f"step {sid}: malformed reference {bound!r}")
                    continue
                src_id, src_port = ref
                src = v.resolved.get(src_id)
                if src_id not in by_id or src is None:
                    err(f"step {sid}: {bound!r} points at no step")
                    continue
                if src_id == sid:
                    err(f"step {sid}: feeds itself")
                    continue
                if src_port not in src.ports_out:
                    err(f"step {sid}: {src.name} has no output {src_port!r}")
                    continue
                out_type = src.ports_out[src_port]
                if not _blocks.types_compatible(out_type, want_type):
                    err(f"step {sid} ({b.name}): input {port!r} wants {want_type} "
                        f"but {bound} is {out_type} — bridge with an adapter block")
            else:
                bad = _blocks.check_value(want_type, bound)
                if bad:
                    err(f"step {sid} ({b.name}): literal for {port!r}: {bad}")
                elif known_feeds is not None and want_type == "FeedRef" \
                        and bound.get("url") not in known_feeds:
                    err(f"step {sid} ({b.name}): {bound.get('url')!r} is not "
                        "one of the configured feed sources — never invent a "
                        "URL; use the news archive (news_fetch) or a listed feed")

    # ── graph outputs ────────────────────────────────────────────────────────
    outs = graph.get("outputs")
    if not (isinstance(outs, dict) and outs):
        err("graph needs at least one named output")
    else:
        for name, bound in outs.items():
            ref = parse_ref(bound) if _is_ref(bound) else None
            if not ref:
                err(f"output {name!r} must be a $step.port reference")
                continue
            src = v.resolved.get(ref[0])
            if src is None or ref[1] not in src.ports_out:
                err(f"output {name!r}: {bound!r} points at nothing")

    # ── acyclicity + execution order (Kahn) ──────────────────────────────────
    order = _toposort(by_id, v.resolved)
    if order is None:
        err("the graph has a cycle — tool graphs are one-way pipelines")
    elif v.ok:
        pinned = {**graph, "steps": []}
        for sid in order:
            s = dict(by_id[sid])
            s["block_version"] = v.resolved[sid].version
            s["params"] = _blocks.fill_param_defaults(v.resolved[sid],
                                                      s.get("params") or {})
            pinned["steps"].append(s)
        v.pinned = pinned
    return v


def _toposort(by_id: dict, resolved: dict):
    """Kahn's algorithm over slot references; None on a cycle (§0.2)."""
    deps = {}
    for sid, s in by_id.items():
        deps[sid] = set()
        for bound in (s.get("inputs") or {}).values():
            if _is_ref(bound):
                ref = parse_ref(bound)
                if ref and ref[0] in by_id and ref[0] != sid:
                    deps[sid].add(ref[0])
    order, ready = [], sorted(sid for sid, d in deps.items() if not d)
    deps = {sid: set(d) for sid, d in deps.items()}
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        for other in sorted(deps):
            if sid in deps[other]:
                deps[other].discard(sid)
                if not deps[other] and other not in order and other not in ready:
                    ready.append(other)
    return order if len(order) == len(by_id) else None


# ── Plain language for the approval gate (§5 T2, §11) ────────────────────────
_CAP_PHRASES = {
    "net": "reach the internet",
    "fs-read": "read files on this computer",
    "fs-write": "write files (inside her tool store only)",
    "mutate": "change stored data (with an undo snapshot)",
    "process": "run a sandboxed program",
    "sensor": "use a sensor",
    "biometric": "process biometric data",
}


def capability_summary(v: Validation) -> str:
    """One honest sentence a non-technical person can approve or refuse."""
    caps = [c for c in ("net", "fs-read", "fs-write", "mutate", "process",
                        "sensor", "biometric") if c in v.capabilities]
    if not caps:
        return "This tool only transforms data — it touches nothing outside itself."
    phrases = [_CAP_PHRASES[c] for c in caps]
    joined = phrases[0] if len(phrases) == 1 else \
        ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"This tool can {joined}."


def needs_approval(v: Validation) -> bool:
    """§5: T2 approval is required for net+fs-write together, any mutate, or
    any biometric.  Biometric never becomes automatic (§9)."""
    c = v.capabilities
    return ("mutate" in c) or ("biometric" in c) or ({"net", "fs-write"} <= c)
