"""Adjudicate the node-merge queue (spec §9.4 resolution).

``link_to_node`` biases toward NOT merging: any ambiguous-similarity pair is parked
in ``node_merge_candidates`` as a DISTINCT node plus a queue entry, because over-merge
is destructive and under-merge is recoverable.  That queue is the recoverable debt —
this module pays it down with the big LM as judge.

For each pair the LM returns one of:
  * same        → the two nodes are merged (support/edges/cards folded together);
  * a_is_a_b / b_is_a_a → kept separate, linked with a taxonomic is_a edge;
  * distinct    → kept separate, the entry closed.

Merging is destructive, so the prompt is told to choose ``same`` ONLY when confident;
in doubt it must say ``distinct``.  Pairs are judged in batches (one LM call per batch)
to keep the slow LM efficient, and the run is lease-aware and resumable (each resolved
entry leaves the open queue, so a stop-and-restart picks up where it left off).
"""
from __future__ import annotations

import logging
import time

from . import lm_lease
from .kb import _norm_terms

log = logging.getLogger("knowledgehost.adjudicate")


# ── deterministic pre-pass: resolve the lexically-obvious pairs without the LM ──
# Using an LLM to dedupe hundreds of thousands of node pairs is the wrong tool: node
# identity here is mostly a string problem (dog/dogs, "dry eye" ⊃ "evaporative dry
# eye").  This pass clears the clear-same and clear-is_a cases for free, escalates only a
# thin genuinely-ambiguous band (high embedding similarity, no decisive lexical signal —
# e.g. car/automobile) to the LM, and DEFERS the rest.  Deferring is the safe choice: the
# KB biases toward not merging (§9.4) because under-merge is recoverable and over-merge is
# destructive, so a weak pair left unmerged costs nothing it can't undo later.

def _stem(t: str) -> str:
    """Cheap singulariser so plural/singular labels match (dog==dogs, party==parties)."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es") and t[-3] in "sxzo":
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _stems(s: str) -> frozenset:
    return frozenset(_stem(w) for w in _norm_terms(s))


def _lexical_decision(a, b, sim, merge_sim):
    """Return 'same' / 'a_is_a_b' / 'b_is_a_a' from labels alone, or None if undecidable
    lexically.  A strict token-subset is the generalisation (§9.4: the superset of terms
    is the MORE specific concept)."""
    al, bl = _stems(a["label"]), _stems(b["label"])
    if not al or not bl:
        return None
    if al == bl:
        return "same"
    if al < bl:                       # a's terms ⊂ b's → b is the more specific
        return "b_is_a_a"
    if bl < al:
        return "a_is_a_b"
    union = al | bl
    jac = len(al & bl) / len(union) if union else 0.0
    if sim >= merge_sim and jac >= 0.5:   # embedding-strong AND lexically overlapping
        return "same"
    return None


def auto_resolve(kb, cfg, *, limit=None, escalate=True) -> dict:
    """Deterministically drain the merge queue: merge lexically-certain duplicates, add
    is_a for clear generalisations, escalate the thin ambiguous band to the LM (left
    'open'), and defer the rest.  No LM, no network — pure SQL/string work, so it clears
    a six-figure queue in minutes.  Resumable; safe to re-run."""
    merge_sim = float(cfg.get("auto_merge_sim", 0.93))
    esc_sim = float(cfg.get("adjudicate_escalate_sim", 0.90))
    st = {"seen": 0, "merged": 0, "linked": 0, "escalated": 0, "deferred": 0, "stale": 0}
    cursor, page = 0, 1000
    t0 = time.time()
    while True:
        rows = kb.db.execute(
            "SELECT id,node_a,node_b,similarity FROM node_merge_candidates "
            "WHERE status='open' AND id>? ORDER BY id LIMIT ?", (cursor, page)).fetchall()
        if not rows:
            break
        with kb.batch():                    # one fsync per page (bounds the transaction)
            for r in rows:
                cursor = r["id"]
                a = kb._node_brief_full(r["node_a"])
                b = kb._node_brief_full(r["node_b"])
                if not a or not b or a["id"] == b["id"]:
                    kb.resolve_candidate(r["id"], "stale"); st["stale"] += 1
                    continue
                st["seen"] += 1
                sim = r["similarity"] or 0.0
                d = _lexical_decision(a, b, sim, merge_sim)
                if d == "same":
                    surv, los = (a, b) if a["support_n"] >= b["support_n"] else (b, a)
                    if kb.merge_nodes(surv["id"], los["id"]):
                        kb.resolve_candidate(r["id"], "merged"); st["merged"] += 1
                    else:
                        kb.resolve_candidate(r["id"], "stale"); st["stale"] += 1
                elif d in ("a_is_a_b", "b_is_a_a"):
                    spec, gen = (a, b) if d == "a_is_a_b" else (b, a)
                    kb.add_edge(spec["id"], gen["id"], family="taxonomic", type="is_a",
                                regime="empirical")
                    kb.resolve_candidate(r["id"], "is_a"); st["linked"] += 1
                elif escalate and sim >= esc_sim:
                    st["escalated"] += 1            # leave 'open' for the LM pass
                else:
                    kb.resolve_candidate(r["id"], "deferred"); st["deferred"] += 1
                if limit and st["seen"] >= limit:
                    break
        if st["seen"] and st["seen"] % 20000 == 0:
            log.info("auto-resolve: %d seen (%d merged, %d is_a, %d escalated, %d deferred)",
                     st["seen"], st["merged"], st["linked"], st["escalated"], st["deferred"])
        if limit and st["seen"] >= limit:
            break
    st["elapsed_s"] = round(time.time() - t0, 1)
    st["open_remaining"] = kb.counts().get("merge_candidates", 0)
    return st

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "decision": {"type": "string",
                                 "enum": ["same", "distinct", "a_is_a_b", "b_is_a_a"]},
                    "reason": {"type": "string"},
                },
                "required": ["i", "decision"],
            },
        },
    },
    "required": ["verdicts"],
}

JUDGE_SYSTEM = (
    "You de-duplicate a knowledge graph.  For each numbered PAIR of concept entries, "
    "decide whether they denote the SAME thing:\n"
    "- same: the same concept — synonyms, spelling/format variants, or the same entity "
    "(they will be MERGED into one node);\n"
    "- a_is_a_b: A is a more specific case/subtype of B (NOT the same) — kept separate, "
    "linked A is_a B;\n"
    "- b_is_a_a: B is a more specific case/subtype of A;\n"
    "- distinct: different concepts, or you are unsure.\n"
    "MERGING IS DESTRUCTIVE: answer 'same' ONLY when you are confident they are the same "
    "concept; when in doubt answer 'distinct'.  Judge by MEANING, not surface spelling — "
    "look-alikes with different meanings (homonyms) are 'distinct'.  Output JSON only."
)


def _pair_str(i, c):
    a, b = c["a"], c["b"]
    def one(tag, n):
        al = f" | aka {', '.join(n['aliases'][:4])}" if n.get("aliases") else ""
        return f"  {tag}: \"{n['label']}\" ({n['kind']}) — {n['summary'][:240]}{al}"
    return (f"[{i}] (embedding similarity {c['similarity']:.2f})\n"
            + one("A", a) + "\n" + one("B", b))


def _judge(lm, cands, max_tokens):
    body = "\n".join(_pair_str(i, c) for i, c in enumerate(cands))
    user = f"Decide each pair:\n{body}"
    out = lm.chat_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA, max_tokens=max_tokens)
    by_i = {}
    for v in ((out or {}).get("verdicts") or []):
        if isinstance(v, dict) and isinstance(v.get("i"), int):
            by_i[v["i"]] = v
    return by_i


def _apply(kb, cand, verdict, stats):
    """Apply one verdict; survivor on a merge is the better-supported node."""
    cid, a, b = cand["id"], cand["a"], cand["b"]
    decision = (verdict or {}).get("decision", "distinct")
    if decision == "same":
        # keep the better-attested node; tie → b (the pre-existing one).
        survivor, loser = (a, b) if a["support_n"] > b["support_n"] else (b, a)
        if kb.merge_nodes(survivor["id"], loser["id"]):
            kb.resolve_candidate(cid, "merged")
            stats["merged"] += 1
        else:
            kb.resolve_candidate(cid, "stale")      # a node vanished underneath us
    elif decision in ("a_is_a_b", "b_is_a_a"):
        spec, gen = (a, b) if decision == "a_is_a_b" else (b, a)
        kb.add_edge(spec["id"], gen["id"], family="taxonomic", type="is_a",
                    regime="empirical", doc_id=None)
        kb.resolve_candidate(cid, "is_a")
        stats["linked"] += 1
    else:
        kb.resolve_candidate(cid, "distinct")
        stats["distinct"] += 1


def adjudicate_queue(kb, lm, cfg, *, limit=None, batch=8, lease=lm_lease.BIG) -> dict:
    """Drain (up to `limit`) open merge candidates through `lm`, pausing while its GPU is
    leased to Vinkona (`lease` = lm_big for the 3090, lm_fast for the 4090).  Returns a stats
    dict; safe to re-run (resumable)."""
    stats = {"judged": 0, "merged": 0, "linked": 0, "distinct": 0}
    max_tokens = int(cfg.get("verify_max_tokens", 1024))
    remaining = limit
    while True:
        n = min(batch, remaining) if remaining else batch
        cands = kb.open_merge_candidates(limit=n)
        if not cands:
            break
        while lm_lease.is_held(lease, cfg):             # GPU busy — yield, then retry
            log.info("adjudicate paused — %s leased to Vinkona…", lease)
            time.sleep(5)
        verdicts = _judge(lm, cands, max_tokens)
        for i, c in enumerate(cands):
            _apply(kb, c, verdicts.get(i, {}), stats)
            stats["judged"] += 1
        if remaining is not None:
            remaining -= len(cands)
            if remaining <= 0:
                break
        if stats["judged"] % 200 == 0:
            log.info("… adjudicated %d (%d merged, %d is_a, %d distinct); %d still open",
                     stats["judged"], stats["merged"], stats["linked"], stats["distinct"],
                     kb.counts().get("merge_candidates", 0))
    stats["open_remaining"] = kb.counts().get("merge_candidates", 0)
    return stats
