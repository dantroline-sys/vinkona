"""Undo a bulk dataset import (conceptnet / atomic / glucose / causenet).

Why this exists: **threshold tuning**.  Graph quality turns on where you set
min_weight / min_count / trust per dataset, and the only honest way to find
those values is import → inspect → adjust → re-import.  This makes the middle
step cheap instead of "rebuild the KB".

Why it can be done safely: every imported row carries the dataset's single
source entry in its ``support`` JSON (``doc_id`` = "conceptnet:5.7" etc.), and
the importers only ever ``INSERT OR IGNORE`` — they never modify a row that
already existed.  So provenance tells the whole story:

  * a row whose support entries are ALL the dataset's was created by the
    import → delete;
  * a row with MIXED support was enriched later (link/refine merging) → keep
    it, strip the dataset's support entries out;
  * a dataset-created node that other edges still lean on must SURVIVE even
    when solely dataset-supported — fusion works both ways (CauseNet's insert
    of "cancer" was ignored because ConceptNet created it first, so the shared
    node's support names only ConceptNet).  A node is deleted only when, after
    the edge pass, nothing references it.

Known asymmetry (accepted): if a later distillation asserted the same
edge_hash as an imported edge, the INSERT OR IGNORE swallowed the distilled
copy — the row still looks purely dataset-owned and is removed here.
Re-running distill restores it; the tuning loop re-imports anyway.

After an unimport, `build-ann` refreshes the dense node index (deleted nodes
otherwise linger in the ANN file until the next build; lookups of missing rows
are skipped, so it degrades gracefully, not wrongly).
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("knowledgehost.unimport")

# dataset name (CLI/ops) -> the source doc_id its importer registers
DATASETS = {
    "conceptnet": "conceptnet:5.7",
    "atomic":     "atomic:v4",
    "glucose":    "glucose",
    "causenet":   "causenet:precision",
}


def _ownership(support_json: str, doc_id: str):
    """-> ('all'|'some'|'none', stripped_support_json).  'all' = every support
    entry is the dataset's; 'some' = mixed (stripped JSON keeps the others)."""
    try:
        entries = json.loads(support_json or "[]")
    except ValueError:
        return "none", support_json
    if not isinstance(entries, list) or not entries:
        return "none", support_json
    keep = [e for e in entries if not (isinstance(e, dict) and e.get("doc_id") == doc_id)]
    if len(keep) == len(entries):
        return "none", support_json
    if not keep:
        return "all", support_json
    return "some", json.dumps(keep)


def unimport(kb, doc_id: str, *, batch: int = 20_000) -> dict:
    """Remove everything `doc_id` contributed to the KB.  Idempotent."""
    db = kb.db
    like = f"%{doc_id}%"                       # cheap prefilter; JSON decides
    t0 = time.time()
    st = {"doc_id": doc_id, "edges_deleted": 0, "edges_stripped": 0,
          "nodes_deleted": 0, "nodes_kept_referenced": 0, "nodes_stripped": 0,
          "refs_cleaned": 0}

    db.executescript("""
        CREATE TEMP TABLE IF NOT EXISTS _uni_gone(id TEXT PRIMARY KEY);
        CREATE TEMP TABLE IF NOT EXISTS _uni_cand(id TEXT PRIMARY KEY);
        DELETE FROM _uni_gone; DELETE FROM _uni_cand;
    """)

    # ── edges: delete dataset-owned, strip mixed ─────────────────────────────
    del_ids, upd = [], []
    for eid, support in db.execute(
            "SELECT id, support FROM edges WHERE support LIKE ?", (like,)).fetchall():
        own, stripped = _ownership(support, doc_id)
        if own == "all":
            del_ids.append((eid,))
        elif own == "some":
            upd.append((stripped, eid))
    for i in range(0, len(del_ids), batch):
        chunk = del_ids[i:i + batch]
        db.executemany("INSERT OR IGNORE INTO _uni_gone(id) VALUES(?)", chunk)
        db.executemany("DELETE FROM edges WHERE id=?", chunk)
    db.executemany("UPDATE edges SET support=? WHERE id=?", upd)
    st["edges_deleted"], st["edges_stripped"] = len(del_ids), len(upd)

    # ── nodes: candidates are dataset-owned; only unreferenced ones may go ───
    cand, upd = [], []
    for nid, support in db.execute(
            "SELECT id, support FROM nodes WHERE support LIKE ?", (like,)).fetchall():
        own, stripped = _ownership(support, doc_id)
        if own == "all":
            cand.append((nid,))
        elif own == "some":
            upd.append((stripped, nid))
    db.executemany("UPDATE nodes SET support=? WHERE id=?", upd)
    st["nodes_stripped"] = len(upd)
    for i in range(0, len(cand), batch):
        db.executemany("INSERT OR IGNORE INTO _uni_cand(id) VALUES(?)", cand[i:i + batch])
    # Anything still referenced — an edge endpoint or a card's subject — stays,
    # whoever created it: deleting shared structure would orphan other sources.
    db.execute("""
        INSERT OR IGNORE INTO _uni_gone(id)
        SELECT c.id FROM _uni_cand c
        WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.src_id = c.id)
          AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.dst_id = c.id)
          AND NOT EXISTS (SELECT 1 FROM procedure_cards p WHERE p.node_id = c.id)
    """)
    gone_nodes = db.execute(
        "SELECT COUNT(*) FROM _uni_gone WHERE id IN (SELECT id FROM _uni_cand)").fetchall()[0][0]
    db.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM _uni_gone)")
    st["nodes_deleted"] = gone_nodes
    st["nodes_kept_referenced"] = len(cand) - gone_nodes

    # ── referencing side tables: facets, surfaces, merge queue ──────────────
    # target ids are content hashes, so matching by id alone is unambiguous.
    n = 0
    for sql in (
        "DELETE FROM facets WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM surface_questions WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM surface_propositions WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM node_merge_candidates WHERE node_a IN (SELECT id FROM _uni_gone)"
        " OR node_b IN (SELECT id FROM _uni_gone)",
    ):
        n += db.execute(sql).rowcount
    st["refs_cleaned"] = n

    db.execute("DELETE FROM source_registry WHERE doc_id=?", (doc_id,))
    db.executescript("DELETE FROM _uni_gone; DELETE FROM _uni_cand;")
    db.commit()

    # the resident dense-search caches now hold deleted nodes — drop them the
    # same way the importers do, so the next query reloads from disk.
    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None

    st["elapsed_s"] = round(time.time() - t0, 1)
    log.info("unimport %s: %s", doc_id, st)
    return st
