"""Reconcile the bulk-imported commonsense node sets against the nodes you already had.

The bulk importers (ConceptNet / ATOMIC / GLUCOSE) write nodes directly, bypassing
``link_to_node`` — so exact ``label+kind`` matches fused for free (INSERT OR IGNORE on
the shared hash) but NEAR-duplicates never got compared: your distilled "dogs" and
ConceptNet's "dog" sit as two unlinked nodes.  This pass closes that gap.

It mirrors the rest of the pipeline's identity discipline (§9.4 — *bias toward not
merging*): it does NOT merge anything itself.  For each **anchor** (a node you already
had — i.e. one with support from a real document, not solely a commonsense import) it
finds the nearest other nodes in the shared embedding space and parks each ambiguous
pair in ``node_merge_candidates``.  The existing ``adjudicate`` command then judges every
pair with the big LM (same → merge, a_is_a_b → taxonomic link, distinct → drop), so the
destructive decision is always the LM's, never a similarity threshold's.

Scope (anchors = your distilled corpus) keeps it tractable: it is O(anchors × nodes),
a blocked matmul over the cached node matrix — fine for thousands of anchors against a
~million-node commonsense backdrop.  Resumable: a pair already in the queue (open or
resolved) is never re-queued, so re-runs only add genuinely new candidates.

Requires embeddings on BOTH sides — run ``embed-nodes`` first so the freshly imported
ATOMIC/GLUCOSE nodes have vectors, or they are invisible to the scan.
"""
from __future__ import annotations

import json
import logging
import time

from . import atomic, causenet, conceptnet, glucose
from .kb import KB

log = logging.getLogger("knowledgehost.reconcile_import")

# the bulk-import sources — a node/item supported ONLY by these is a background prior, not
# something "you already had" (your distilled documents).  Used both to pick reconcile
# anchors and to demote priors in `kb_ask` ranking (query._is_prior).
COMMONSENSE_DOCS = {conceptnet.DOC_ID, atomic.DOC_ID, glucose.DOC_ID, causenet.DOC_ID}


def _is_anchor(support_json: str) -> bool:
    """True if this node predates the commonsense flood: it has at least one support
    entry from a non-commonsense source (a real document), or no support at all."""
    try:
        sup = json.loads(support_json or "[]")
    except (ValueError, TypeError):
        return True
    if not sup:
        return True
    return any((s.get("doc_id") not in COMMONSENSE_DOCS) for s in sup)


def _existing_pairs(kb, a_id, b_id) -> bool:
    """A candidate for this unordered pair already exists (any status)?"""
    return kb.db.execute(
        "SELECT 1 FROM node_merge_candidates WHERE "
        "(node_a=? AND node_b=?) OR (node_a=? AND node_b=?) LIMIT 1",
        (a_id, b_id, b_id, a_id)).fetchone() is not None


def reconcile_imports(kb: KB, cfg, *, anchors: str = "corpus", limit: int | None = None,
                      top_k: int | None = None) -> dict:
    """Queue merge candidates between your existing nodes and their nearest neighbours
    across the whole node set.  Resolve them afterwards with `adjudicate`.

    anchors='corpus' (default): anchor on nodes you already had (non-commonsense
    support).  anchors='all': every embedded node (O(N²) — only for a small KB)."""
    np = kb._np
    if np is None:
        raise RuntimeError("reconcile needs numpy (the vectorised node matrix).")
    k = int(top_k or cfg.get("reconcile_top_k", 3))
    block = max(1, int(cfg.get("reconcile_block", 128)))
    floor = float(cfg.get("reconcile_min_sim", 0.0)) or kb.theta_low

    ids, mat = kb._node_matrix()                 # active, embedded nodes only
    no_vec = kb.db.execute("SELECT COUNT(*) FROM nodes WHERE status='active' "
                           "AND embedding IS NULL").fetchone()[0]
    if not ids or mat is None:
        return {"anchors": 0, "queued": 0, "embedded_nodes": 0, "nodes_without_vectors": no_vec}

    # classify anchors from a single support scan.
    anchor_pos = []
    if anchors == "all":
        anchor_pos = list(range(len(ids)))
    else:
        sup_by_id = {r["id"]: r["support"] for r in kb.db.execute(
            "SELECT id, support FROM nodes WHERE status='active' AND embedding IS NOT NULL")}
        for i, nid in enumerate(ids):
            if _is_anchor(sup_by_id.get(nid, "")):
                anchor_pos.append(i)
    if limit:
        anchor_pos = anchor_pos[:limit]

    ann = kb._get_ann()                           # HNSW: anchor neighbours in ~log(N)
    st = {"anchors": len(anchor_pos), "scanned": 0, "queued": 0, "skipped_existing": 0,
          "embedded_nodes": len(ids), "nodes_without_vectors": no_vec,
          "ann": bool(ann)}
    if ann is None and anchors == "all" and len(anchor_pos) * len(ids) > 5_000_000_000:
        log.warning("anchors=all over %d nodes is O(N²)=%.1e without an ANN index — slow; "
                    "run `build-ann` first.", len(ids), len(anchor_pos) * float(len(ids)))

    def _queue(a_id, b_id, s):
        if _existing_pairs(kb, a_id, b_id):
            st["skipped_existing"] += 1
            return
        kb.db.execute("INSERT INTO node_merge_candidates(node_a,node_b,similarity,reason,"
                      "status) VALUES(?,?,?,?, 'open')", (a_id, b_id, s, "reconcile_import"))
        st["queued"] += 1

    def _progress():
        st["scanned"] += 1
        if st["scanned"] % 2000 == 0:
            log.info("reconcile: scanned %d/%d anchors, queued %d candidates",
                     st["scanned"], len(anchor_pos), st["queued"])

    t0 = time.time()
    with kb.batch():                              # one fsync for the whole candidate flush
        if ann is not None:
            # per-anchor HNSW query (k+1 to drop the anchor's own self-match).
            for gi in anchor_pos:
                a_id = ids[gi]
                for nid, s in ann.query(mat[gi], k + 1):
                    if nid == a_id:
                        continue
                    if s < floor:
                        break                     # usearch returns highest-sim first
                    _queue(a_id, nid, s)
                _progress()
        else:
            # exact brute force: blocked anchors × all-nodes matmul.
            for start in range(0, len(anchor_pos), block):
                blk = anchor_pos[start:start + block]
                sims = mat[blk] @ mat.T            # (b, N) cosine (rows L2-normalised)
                for r, gi in enumerate(blk):
                    row = sims[r]
                    row[gi] = -2.0                 # mask self
                    kk = min(k, row.shape[0] - 1)
                    cand = np.argpartition(row, -kk)[-kk:]
                    for j in cand[np.argsort(row[cand])[::-1]]:
                        s = float(row[j])
                        if s < floor:
                            break                 # sorted desc → rest are lower
                        _queue(ids[gi], ids[int(j)], s)
                    _progress()
    st["elapsed_s"] = round(time.time() - t0, 1)
    st["open_candidates"] = kb.counts().get("merge_candidates", 0)
    return st
