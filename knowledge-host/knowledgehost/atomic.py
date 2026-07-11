"""Bulk importer for the ATOMIC if-then commonsense knowledge graph (Sap et al. 2019).

ATOMIC is crowd-sourced *social* commonsense: for each everyday **event** ("PersonX
pays the bill") it records nine kinds of typed inference — what the agent intended /
needed beforehand, how they are perceived, the after-effects on the agent and on
others, and what each party feels and wants next.  Like ConceptNet it is ungrounded
crowd knowledge, so it imports under the SAME epistemics (see conceptnet.py):

  * regime = ``conventional``                 (firewall-safe social-inference prior)
  * a single low-trust source ``atomic:v4``   (trust default 0.2)
  * ``has_reference = 0`` on every support entry — no citable passage behind it.

Shape mapping: each event becomes an ``event`` node; each non-"none" annotation in a
dimension becomes a ``concept`` node, joined by a typed edge whose ``type`` encodes the
dimension (so the agent/other and before/after roles survive).  The aggregated file
(``v4_atomic_all_agg.csv``, ~24k unique events) is the one to use — annotator agreement
within a dimension is preserved as ``atomic_count`` in the edge modifiers.

Inserts reuse the distiller's hashes (``INSERT OR IGNORE``), so the import is idempotent
and ``concept`` outcomes fuse with same-label ConceptNet/distilled nodes.  Nodes get a
NULL embedding; run ``embed-nodes`` afterwards to make them dense-searchable.
"""
from __future__ import annotations

import csv
import json
import logging
import time

from .conceptnet import _support_json
from .kb import KB, _hash

log = logging.getLogger("knowledgehost.atomic")

DOC_ID = "atomic:v4"
DOC_TITLE = "ATOMIC v4 (if-then social commonsense graph)"

# ATOMIC dimension column -> (edge family, edge type).  The type names keep the
# agent (x*) / other (o*) and before/after distinctions the dimensions encode.
_DIMS = {
    "xIntent": ("causal",      "agent_intends"),     # before: PersonX wanted to…
    "xNeed":   ("causal",      "agent_needs"),       # before: PersonX had to…
    "xAttr":   ("attributive", "agent_attribute"),   # PersonX is seen as…
    "xEffect": ("causal",      "agent_effect"),       # after: effect on PersonX
    "xReact":  ("affective",   "agent_reacts"),       # after: PersonX feels…
    "xWant":   ("affective",   "agent_wants"),        # after: PersonX wants…
    "oEffect": ("causal",      "other_effect"),       # after: effect on others
    "oReact":  ("affective",   "other_reacts"),       # after: others feel…
    "oWant":   ("affective",   "other_wants"),        # after: others want…
}

# unicode placeholder for the blank in event templates ("___" stays as-is; "_" too).
_NONE = {"", "none", "None"}


def import_atomic(kb: KB, path: str, *, trust: float = 0.2,
                  min_count: int = 1, limit: int | None = None,
                  log_every: int = 5_000) -> dict:
    """Stream the ATOMIC csv (`v4_atomic_all_agg.csv`) into the KB.  Returns stats.

    Idempotent (event/outcome nodes and edges are INSERT OR IGNORE).  Skips "none"
    annotations and any whose annotator agreement count is below `min_count`."""
    kb.register_source(DOC_ID, DOC_TITLE, source_type="atomic",
                       trust_weight=trust, regime="conventional")
    support = _support_json(trust, DOC_ID)
    now = time.time()

    node_seen: set = set()
    nbuf: list = []
    ebuf: list = []
    NODE_SQL = ("INSERT OR IGNORE INTO nodes"
                "(id,label,kind,summary,embedding,aliases,support,status)"
                " VALUES(?,?,?,'',NULL,'[]',?,'active')")
    EDGE_SQL = ("INSERT OR IGNORE INTO edges"
                "(id,src_id,dst_id,family,type,mechanism,mechanism_basis,modifiers,"
                "polarity,embedding,edge_hash,support,strength,regime,scope,status,"
                "created_at,updated_at)"
                " VALUES(?,?,?,?,?,'','',?,'',NULL,?,?,NULL,'conventional','{}','active',?,?)")

    def add_node(nid, label, kind):
        if nid not in node_seen:
            node_seen.add(nid)
            nbuf.append((nid, label, kind, support))

    def flush():
        if nbuf:
            kb.db.executemany(NODE_SQL, nbuf); nbuf.clear()
        if ebuf:
            kb.db.executemany(EDGE_SQL, ebuf); ebuf.clear()
        kb.db.commit()

    st = {"events": 0, "imported": 0, "nodes": 0, "skip_none": 0, "skip_count": 0}
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            event = (row.get("event") or "").strip()
            if not event:
                continue
            st["events"] += 1
            if st["events"] % log_every == 0:
                log.info("atomic: %d events, %d edges", st["events"], st["imported"])
            eid = _hash("node", event, "event")
            add_node(eid, event, "event")
            for col, (family, etype) in _DIMS.items():
                cell = row.get(col)
                if not cell:
                    continue
                try:
                    items = json.loads(cell)
                except (ValueError, TypeError):
                    continue
                # collapse annotator repeats -> agreement count (case-insensitive).
                counts: dict = {}
                for it in items:
                    s = (it or "").strip()
                    if s.lower() in _NONE:
                        st["skip_none"] += 1
                        continue
                    key = s.lower()
                    counts[key] = counts.get(key, (s, 0))
                    counts[key] = (counts[key][0], counts[key][1] + 1)
                for _k, (label, cnt) in counts.items():
                    if cnt < min_count:
                        st["skip_count"] += 1
                        continue
                    did = _hash("node", label, "concept")
                    add_node(did, label, "concept")
                    eh = KB._edge_hash(eid, did, family, etype, "", "", "conventional", "", "")
                    mods = '{"atomic_count": %d}' % cnt
                    ebuf.append((eh, eid, did, family, etype, mods, eh, support, now, now))
                    st["imported"] += 1
            if len(ebuf) >= 50_000:
                flush()
            if limit and st["events"] >= limit:
                break
    flush()
    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None
    st["nodes"] = len(node_seen)
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st
