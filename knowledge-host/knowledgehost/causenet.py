"""Bulk importer for CauseNet (precision) — a large causal graph extracted from text.

CauseNet mines cause→effect pairs from Wikipedia and ClueWeb with validated linguistic
patterns; the *precision* release is the high-quality subset.  Unlike the crowd-sourced
commonsense graphs (ConceptNet/ATOMIC/GLUCOSE), each CauseNet relation is **grounded** in
real source sentences, so it imports differently:

  * regime = ``conventional`` (configurable) — firewall-safe: web/encyclopedia-extracted
    causal claims stay out of the empirical (peer-reviewed/textbook) tier;
  * ``has_reference = 1`` — it carries citable evidence, so the read path need not
    discount it as an ungrounded prior the way it does the commonsense sets;
  * a representative source sentence (Wikipedia preferred) is kept on the edge, and the
    number of supporting sentences becomes a corroboration signal (``source_count`` /
    ``causenet_sources``) — e.g. cancer→death is attested by thousands of sentences.

Each record → a ``cause`` concept node —[causal/causes, positive]→ an ``effect`` concept
node, joined under one low-trust source ``causenet:precision``.  Concept nodes use
kind ``concept``, so they fuse with same-label ConceptNet/distilled nodes — CauseNet then
adds the *causal* edges over that shared concept backbone.

Streamed (the file is ~1 GB, some records carry thousands of sources), idempotent via the
distiller's hashes.  NULL embeddings — run ``embed-nodes`` afterwards for dense search.
"""
from __future__ import annotations

import json
import logging
import time

from .kb import KB, _hash

log = logging.getLogger("knowledgehost.causenet")

DOC_ID = "causenet:precision"
DOC_TITLE = "CauseNet (precision) — causal graph mined from Wikipedia/ClueWeb"


def _concept(c) -> str:
    return (c or "").replace("_", " ").strip()


def _representative(sources):
    """Best evidence sentence + page for the edge: prefer a Wikipedia sentence; fall
    back to the first source that has a sentence.  Returns (sentence, page)."""
    sent, page = "", ""
    for s in sources:
        p = (s.get("payload") or {}) if isinstance(s, dict) else {}
        snt = (p.get("sentence") or "").strip()
        if not snt:
            continue
        if not sent:
            sent, page = snt, p.get("wikipedia_page_title", "") or ""
        if str(s.get("type", "")).startswith("wikipedia"):
            return snt, p.get("wikipedia_page_title", "") or ""
    return sent, page


def _distinct_sources(sources) -> int:
    """How many DISTINCT places attest this relation — raw len(sources) counts the
    same sentence scraped repeatedly, which inflates corroboration.  Distinct =
    unique (type, page-or-document-or-sentence) across the source list."""
    keys = set()
    for s in sources:
        if not isinstance(s, dict):
            continue
        p = s.get("payload") or {}
        keys.add((str(s.get("type", "")),
                  p.get("wikipedia_page_title") or p.get("clueweb12_page_id")
                  or (p.get("sentence") or "").strip()))
    return len(keys)


def import_causenet(kb: KB, path: str, *, trust: float = 0.4,
                    regime: str = "conventional", min_sources: int = 1,
                    limit: int | None = None,
                    log_every: int = 25_000) -> dict:
    """Stream causenet-precision.jsonl into the KB.  Returns a stats dict.  Idempotent
    (INSERT OR IGNORE on the shared node/edge hashes).

    `min_sources` is the corroboration floor: a relation attested by fewer DISTINCT
    sources (see _distinct_sources) is skipped.  1 keeps everything; raising it
    trades recall for a graph where every causal claim was seen in more places."""
    kb.register_source(DOC_ID, DOC_TITLE, source_type="causenet",
                       trust_weight=trust, regime=regime)
    # constant support for the concept NODES (grounded source, no per-edge count).
    node_support = json.dumps([{
        "doc_id": DOC_ID, "evidence_cluster": "", "date": None,
        "trust_weight": trust, "regime": regime, "origin": regime, "has_reference": 1}])
    now = time.time()

    node_seen: set = set()
    nbuf: list = []
    ebuf: list = []
    NODE_SQL = ("INSERT OR IGNORE INTO nodes"
                "(id,label,kind,summary,embedding,aliases,support,status)"
                " VALUES(?,?,'concept','',NULL,'[]',?,'active')")
    EDGE_SQL = ("INSERT OR IGNORE INTO edges"
                "(id,src_id,dst_id,family,type,mechanism,mechanism_basis,modifiers,"
                "polarity,embedding,edge_hash,support,strength,regime,scope,status,"
                "created_at,updated_at)"
                " VALUES(?,?,?,'causal','causes','','',?,'positive',NULL,?,?,NULL,?,'{}',"
                "'active',?,?)")

    def add_node(nid, label):
        if nid not in node_seen:
            node_seen.add(nid)
            nbuf.append((nid, label, node_support))

    def flush():
        if nbuf:
            kb.db.executemany(NODE_SQL, nbuf); nbuf.clear()
        if ebuf:
            kb.db.executemany(EDGE_SQL, ebuf); ebuf.clear()
        kb.db.commit()

    st = {"records": 0, "imported": 0, "nodes": 0, "skip_empty": 0, "skip_parse": 0,
          "skip_sources": 0}
    t0 = time.time()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            st["records"] += 1
            if st["records"] % log_every == 0:
                log.info("causenet: %d records, %d edges", st["records"], st["imported"])
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cr = rec["causal_relation"]
                cause = _concept(cr["cause"]["concept"])
                effect = _concept(cr["effect"]["concept"])
            except (ValueError, KeyError, TypeError):
                st["skip_parse"] += 1
                continue
            if not cause or not effect or cause == effect:
                st["skip_empty"] += 1
                continue
            sources = rec.get("sources") or []
            n = _distinct_sources(sources)
            if n < min_sources:
                st["skip_sources"] += 1
                continue
            sent, page = _representative(sources)
            sid = _hash("node", cause, "concept")
            did = _hash("node", effect, "concept")
            add_node(sid, cause)
            add_node(did, effect)
            eh = KB._edge_hash(sid, did, "causal", "causes", "positive", "", regime, "", "")
            support = json.dumps([{
                "doc_id": DOC_ID,
                "evidence_cluster": _hash("ev", sent)[:12] if sent else "",
                "date": None, "trust_weight": trust, "regime": regime, "origin": regime,
                "has_reference": 1, "source_count": n}])
            mods = {"causenet_sources": n}
            if sent:
                mods["evidence"] = sent[:300]
            if page:
                mods["wikipedia_page"] = page
            ebuf.append((eh, sid, did, json.dumps(mods), eh, support, regime, now, now))
            st["imported"] += 1
            if len(ebuf) >= 50_000:
                flush()
            if limit and st["records"] >= limit:
                break
    flush()
    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None
    st["nodes"] = len(node_seen)
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st
