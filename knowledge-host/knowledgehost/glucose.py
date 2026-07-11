"""Bulk importer for GLUCOSE (general causal/explanatory commonsense; Mostafazadeh 2020).

GLUCOSE crowd-sources *causal* commonsense around short stories: for a selected event
it records ten dimensions of explanation — the events, emotions, locations, possessions
and attributes that CAUSE/ENABLE/MOTIVATE it, and those it RESULTS IN.  Crucially each
dimension comes in a *general* form with variable slots (``Someone_A``, ``Somewhere_A``)
— a reusable rule, not a story-specific fact — e.g.

    Someone_A doesn't want to go to sleep  >Causes/Enables>  Someone_A finally falls asleep

That generalised, one-antecedent→one-consequent shape is exactly the reusable causal
convention the distiller aims for, so GLUCOSE makes a strong commonsense backbone.

Same epistemics as the other crowd sources (see conceptnet.py): regime=``conventional``,
one low-trust source ``glucose``, ``has_reference=0`` on every entry.  We import the
GENERAL rules only (the specific, story-grounded ones explode into per-entity nodes and
are not reusable).  Each rule becomes:  antecedent ``proposition`` node —[connective]→
consequent ``proposition`` node, family=causal.  The connective is the edge type
(causes / enables / causes_or_enables / motivates / results_in); identical rules across
workers/stories collapse by hash, their repeat count kept as ``glucose_count``
(annotator/agreement corroboration).

Idempotent (INSERT OR IGNORE on the distiller's hashes).  NULL embeddings — run
``embed-nodes`` afterwards for dense search.
"""
from __future__ import annotations

import csv
import logging
import re
import sys
import time

from .conceptnet import _support_json
from .kb import KB, _hash

log = logging.getLogger("knowledgehost.glucose")

DOC_ID = "glucose"
DOC_TITLE = "GLUCOSE (general causal/explanatory commonsense)"

# the five GLUCOSE connectives -> edge type (all family=causal/explanatory).
_CONN = {
    "Causes/Enables": "causes_or_enables",
    "Enables": "enables",
    "Results in": "results_in",
    "Motivates": "motivates",
    "Causes": "causes",
}
_CONN_RE = re.compile(r">\s*(" + "|".join(re.escape(c) for c in _CONN) + r")\s*>")
_SKIP = {"", "escaped"}


def import_glucose(kb: KB, path: str, *, trust: float = 0.2,
                   min_count: int = 1, limit: int | None = None,
                   log_every: int = 10_000) -> dict:
    """Stream GLUCOSE_training_data_final.csv into the KB.  Returns stats.

    Accumulates rules in memory (the file is small) so identical rules collapse with an
    accurate corroboration count; rules below `min_count` are dropped.  Idempotent."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))   # story cells can be long
    kb.register_source(DOC_ID, DOC_TITLE, source_type="glucose",
                       trust_weight=trust, regime="conventional")
    support = _support_json(trust, DOC_ID)
    now = time.time()

    nodes: dict = {}                 # id -> label
    edges: dict = {}                 # edge_hash -> [eid, sid, did, etype, dim, count]
    st = {"rows": 0, "rules": 0, "skip_empty": 0, "skip_parse": 0}
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            st["rows"] += 1
            if st["rows"] % log_every == 0:
                log.info("glucose: %d rows, %d unique rules", st["rows"], len(edges))
            for d in range(1, 11):
                v = (row.get(f"{d}_generalNL") or "").strip()
                if v in _SKIP:
                    st["skip_empty"] += 1
                    continue
                m = _CONN_RE.search(v)
                if not m:
                    st["skip_parse"] += 1
                    continue
                ante = v[:m.start()].strip()
                cons = v[m.end():].strip()
                if not ante or not cons:
                    st["skip_parse"] += 1
                    continue
                etype = _CONN[m.group(1)]
                sid = _hash("node", ante, "proposition")
                did = _hash("node", cons, "proposition")
                if sid == did:
                    continue
                nodes.setdefault(sid, ante)
                nodes.setdefault(did, cons)
                eh = KB._edge_hash(sid, did, "causal", etype, "", "", "conventional", "", "")
                e = edges.get(eh)
                if e:
                    e[5] += 1
                else:
                    edges[eh] = [eh, sid, did, etype, d, 1]
                    st["rules"] += 1
            if limit and st["rows"] >= limit:
                break

    NODE_SQL = ("INSERT OR IGNORE INTO nodes"
                "(id,label,kind,summary,embedding,aliases,support,status)"
                " VALUES(?,?,'proposition','',NULL,'[]',?,'active')")
    EDGE_SQL = ("INSERT OR IGNORE INTO edges"
                "(id,src_id,dst_id,family,type,mechanism,mechanism_basis,modifiers,"
                "polarity,embedding,edge_hash,support,strength,regime,scope,status,"
                "created_at,updated_at)"
                " VALUES(?,?,?,'causal',?,'','',?,'',NULL,?,?,NULL,'conventional','{}','active',?,?)")

    def _batched(seq, n=50_000):
        buf = []
        for x in seq:
            buf.append(x)
            if len(buf) >= n:
                yield buf; buf = []
        if buf:
            yield buf

    for chunk in _batched((nid, lbl, support) for nid, lbl in nodes.items()):
        kb.db.executemany(NODE_SQL, chunk)
    kb.db.commit()
    kept = 0
    rows_iter = ((eh, sid, did, etype,
                  '{"glucose_dim": %d, "glucose_count": %d}' % (dim, cnt),
                  eh, support, now, now)
                 for (eh, sid, did, etype, dim, cnt) in edges.values() if cnt >= min_count)
    for chunk in _batched(rows_iter):
        kb.db.executemany(EDGE_SQL, chunk)
        kept += len(chunk)
    kb.db.commit()

    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None
    st["imported"] = kept
    st["skip_count"] = st["rules"] - kept
    st["nodes"] = len(nodes)
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st
