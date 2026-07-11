"""Phase 1 — **graph linkage**: typed structural edges between the concepts you've
distilled, so a card knows what it specialises, requires, is part of, or stands as an
alternative to.  This is the consolidation pass: it reads the KB (not source prose) and
adds the connective tissue that turns a pile of good cards into a navigable graph —
"obstetric hypotension `is_a` hypotension", "post-spinal management `requires` spinal
anaesthesia".

Discipline (mirrors §9.4 — *bias toward not destroying*):
  * It never merges, rewrites, or deletes — it only ADDS typed edges, which are
    recoverable (a wrong link is a row you drop).  It never invents facts: the big LM
    types a relation between two concepts it is SHOWN, choosing `none` when unrelated.
  * Candidates are embedding-neighbours (the ANN arm, à la reconcile_import), so the LM
    only ever judges concepts that are already near each other — it labels a relationship,
    it doesn't hunt for one.  O(card-nodes × top_k), not O(N²).
  * Every edge it writes is provenance-tagged to the `linker:v1` source and
    `mechanism_basis='inferred'` (low trust), so a high-rigor read can tell a structural
    inference from a sourced claim, and weight it accordingly.
  * Resumable: a judged pair is checkpointed in `link_pairs` (even a `none`), so re-runs
    only spend the LM on genuinely new neighbours.  Edge writes are hash-idempotent.

Runs as its own command, big-LM work, yielding the GPU whenever Vinkona holds the lease.
"""
from __future__ import annotations

import json
import logging
import time

from . import lm_lease
from .distill import _first_json
from .reconcile_import import _is_anchor

log = logging.getLogger("knowledgehost.link")

LINKER_DOC = "linker:v1"
_LEASE_POLL_S = 3

# Closed relation set the LM picks from → (src_is, dst_is, family, type, symmetric).
# 'src_is'/'dst_is' say which of the shown pair (a/b) becomes the edge's src/dst, so a
# directional relation always points the canonical way (specific→general, dependent→
# prerequisite, part→whole).
_REL = {
    "a_is_a_b":    ("a", "b", "taxonomic", "is_a", False),
    "b_is_a_a":    ("b", "a", "taxonomic", "is_a", False),
    "a_requires_b": ("a", "b", "functional", "requires", False),
    "b_requires_a": ("b", "a", "functional", "requires", False),
    "a_part_of_b": ("a", "b", "meronymic", "part_of", False),
    "b_part_of_a": ("b", "a", "meronymic", "part_of", False),
    "alternative": ("a", "b", "functional", "alternative_to", True),
    "related":     ("a", "b", "functional", "related_to", True),
}

_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(_REL) + ["none"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["relation", "confidence", "rationale"],
}

_SYSTEM = (
    "You organise a knowledge graph for fast retrieval. You are shown two concepts, A and "
    "B, from the SAME knowledge base. Identify the SINGLE strongest STRUCTURAL relationship "
    "between them, for navigation — not a fact about the world.\n"
    "Choose exactly one `relation`:\n"
    "- a_is_a_b / b_is_a_a: one is a KIND or special case of the other (obstetric "
    "hypotension is_a hypotension). Point from the specific to the general.\n"
    "- a_requires_b / b_requires_a: doing or understanding one needs the other as a "
    "PREREQUISITE or input (managing post-spinal hypotension requires spinal anaesthesia).\n"
    "- a_part_of_b / b_part_of_a: one is a COMPONENT or step of the other, larger one.\n"
    "- alternative: they are competing approaches to the SAME goal.\n"
    "- related: clearly associated and worth cross-linking, but none of the above.\n"
    "- none: not clearly related, or you are unsure.\n"
    "RULES: use ONLY what the two descriptions state — do NOT invent facts or assume domain "
    "knowledge beyond them. Prefer a specific relation (is_a/requires/part_of) over 'related'. "
    "When in doubt, answer 'none'. Keep `rationale` under 20 words."
)


def _node_ctx(kb, nid, cache: dict) -> str | None:
    """A compact description of a node for the LM: label — summary, plus its card's
    title/goal when it has one.  None if the node is gone."""
    if nid in cache:
        return cache[nid]
    r = kb.db.execute("SELECT label, summary, support FROM nodes WHERE id=? AND status='active'",
                      (nid,)).fetchone()
    if not r:
        cache[nid] = None
        return None
    parts = [r["label"] or "(unlabelled)"]
    if r["summary"]:
        parts.append(str(r["summary"])[:400])
    card = kb.db.execute("SELECT title, goal FROM procedure_cards WHERE node_id=? "
                         "AND status='active' LIMIT 1", (nid,)).fetchone()
    if card and (card["title"] or card["goal"]):
        parts.append(f"[card: {card['title']} — {card['goal']}]"[:300])
    txt = " — ".join(p for p in parts if p)
    cache[nid] = (txt, r["support"])
    return cache[nid]


def _pair_key(a, b) -> str:
    return "|".join(sorted((a, b)))


def _await_lease(cfg, log, lease) -> None:
    """Read-only yield: pause while Vinkona holds the LM lease for the tier we're using
    (never write it) — BIG (3090) for the default run, FAST (4090) under --fast."""
    waited = False
    while lm_lease.is_held(lease, cfg):
        if not waited:
            log.info("link: %s leased by Vinkona — yielding…", lease)
            waited = True
        time.sleep(_LEASE_POLL_S)


def link_concepts(kb, lm, cfg, *, limit: int | None = None,
                  top_k: int | None = None, lease: str = lm_lease.BIG) -> dict:
    """Type structural edges between card-bearing concepts and their nearest neighbours.

    `lm` is a live LM client (DistillLM); `lease` is the GPU lease to yield on (BIG by
    default, FAST when driven by the 9B).  Returns a stats dict.  Raises whatever the LM
    transport raises (BackendUnavailable) so the CLI can abort resumably."""
    np = kb._np
    if np is None:
        raise RuntimeError("link needs numpy (the cached node matrix).")
    k = int(top_k or cfg.get("link_top_k", 8))
    floor = float(cfg.get("link_min_sim", 0.5))
    min_conf = float(cfg.get("link_min_conf", 0.6))
    related_min_conf = float(cfg.get("link_related_min_conf", 0.75))
    # generous token budget: a reasoning endpoint may 'think' before the JSON, and a tight
    # cap truncates it to nothing (the unparsed-everything bug).  Mirrors the distill calls.
    mtok = int(cfg.get("link_max_tokens", 1024))

    kb.db.executescript(
        "CREATE TABLE IF NOT EXISTS link_pairs("
        "pair TEXT PRIMARY KEY, relation TEXT, confidence REAL, ts REAL);")
    kb.register_source(LINKER_DOC, "graph linkage pass (inferred)",
                       source_type="inference", trust_weight=0.3, regime="empirical")

    ids, mat = kb._node_matrix()
    if not ids or mat is None:
        return {"anchors": 0, "judged": 0, "linked": 0, "embedded_nodes": len(ids)}
    pos = {nid: i for i, nid in enumerate(ids)}

    # support scan once: who is a real (non-commonsense) node, and each node's regime.
    sup_by_id = {r["id"]: r["support"] for r in kb.db.execute(
        "SELECT id, support FROM nodes WHERE status='active' AND embedding IS NOT NULL")}
    anchor_like = {nid: _is_anchor(s) for nid, s in sup_by_id.items()}

    # anchors: embedded, non-commonsense nodes that carry a card (the user's focus).
    card_nodes = [r["node_id"] for r in kb.db.execute(
        "SELECT DISTINCT node_id FROM procedure_cards WHERE status='active'")]
    anchors = [nid for nid in card_nodes
               if nid in pos and anchor_like.get(nid, True)]
    if limit:
        anchors = anchors[:limit]

    ann = kb._get_ann()
    # over-fetch: pull a deep neighbour pool, THEN keep the nearest k real (non-commonsense)
    # ones.  The ~1M commonsense priors otherwise crowd a shallow top-k and starve the linker
    # of genuine relatives — the same dilution kb_ask fights (filter-after-fetch was the bug).
    pool = max(k * int(cfg.get("link_fetch_mult", 20)), 64)
    st = {"anchors": len(anchors), "embedded_nodes": len(ids), "ann": bool(ann),
          "pool": pool, "candidates": 0, "judged": 0, "skipped_checkpoint": 0,
          "unparsed": 0, "lm_none": 0, "low_conf": 0, "linked": 0,
          "by_relation": {}, "sample": [], "raw_misses": []}

    def _neighbours(a_id):
        """Up to k REAL (non-commonsense) neighbours above the floor, nearest first —
        over-fetched from a deep pool so the commonsense flood can't starve the result."""
        if ann is not None:
            cnt = 0
            for nid, s in ann.query(mat[pos[a_id]], pool):
                if s < floor:
                    break                             # usearch returns highest-sim first
                if nid == a_id or not anchor_like.get(nid, False):
                    continue
                yield nid, s
                cnt += 1
                if cnt >= k:
                    return
            return
        sims = mat @ mat[pos[a_id]]
        sims[pos[a_id]] = -2.0
        cnt = 0
        for j in np.argsort(sims)[::-1]:
            s = float(sims[j])
            if s < floor:
                break
            nid = ids[int(j)]
            if not anchor_like.get(nid, False):
                continue
            yield nid, s
            cnt += 1
            if cnt >= k:
                return

    cache: dict = {}
    t0 = time.time()
    for a_id in anchors:
        actx = _node_ctx(kb, a_id, cache)
        if not actx:
            continue
        for b_id, sim in _neighbours(a_id):
            pk = _pair_key(a_id, b_id)
            if kb.db.execute("SELECT 1 FROM link_pairs WHERE pair=?", (pk,)).fetchone():
                st["skipped_checkpoint"] += 1
                continue
            bctx = _node_ctx(kb, b_id, cache)
            if not bctx:
                continue
            st["candidates"] += 1
            _await_lease(cfg, log, lease)
            user = (f"A: {actx[0]}\nB: {bctx[0]}\n\nThe single strongest structural "
                    "relation between A and B:")
            content = lm._content(_SYSTEM, user, _LINK_SCHEMA, mtok)
            raw = None
            if content is not None:
                try:
                    raw = json.loads(_first_json(content))
                except (ValueError, AttributeError):
                    raw = None
            if raw is None:                           # transient/parse miss — retry next run
                st["unparsed"] += 1
                if len(st["raw_misses"]) < 5:         # keep a few raw outputs to diagnose
                    st["raw_misses"].append((content or "<none>")[:200])
                continue
            rel = raw.get("relation") or "none"
            conf = float(raw.get("confidence") or 0.0)
            rationale = str(raw.get("rationale") or "")[:200]
            st["judged"] += 1
            if len(st["sample"]) < 15:                # representative verdicts for diagnosis
                st["sample"].append(
                    f"{kb._label_of(a_id)} ⇢ {kb._label_of(b_id)} = {rel} {conf:.2f} (sim {sim:.2f})")
            kb.db.execute("INSERT OR REPLACE INTO link_pairs(pair,relation,confidence,ts) "
                          "VALUES(?,?,?,?)", (pk, rel, conf, time.time()))
            spec = _REL.get(rel)
            if not spec:
                st["lm_none"] += 1
                continue
            gate = related_min_conf if rel == "related" else min_conf
            if conf < gate:
                st["low_conf"] += 1
                continue
            src, dst, family, etype, symmetric = spec
            sid = a_id if src == "a" else b_id
            did = b_id if dst == "b" else a_id
            regime = kb._regime_of_support(json.loads(actx[1] or "[]")) or "empirical"
            kb.add_edge(sid, did, family=family, type=etype, mechanism_basis="inferred",
                        regime=regime, doc_id=LINKER_DOC, evidence=rationale)
            if symmetric:                             # alternative_to / related_to both ways
                kb.add_edge(did, sid, family=family, type=etype, mechanism_basis="inferred",
                            regime=regime, doc_id=LINKER_DOC, evidence=rationale)
            st["linked"] += 1
            st["by_relation"][etype] = st["by_relation"].get(etype, 0) + 1
        kb.db.commit()
        if st["judged"] and st["judged"] % 200 == 0:
            log.info("link: judged %d, linked %d (%s)", st["judged"], st["linked"],
                     st["by_relation"])
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st
