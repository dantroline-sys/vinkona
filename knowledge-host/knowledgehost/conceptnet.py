"""Bulk importer for the ConceptNet 5.7 assertions dump.

ConceptNet is crowd-/wiktionary-sourced commonsense: millions of (start, relation,
end) triples.  It is exactly the *conventional* regime (§8) — "what people generally
take to be so" — and it is NOT grounded in a passage we can cite, so every assertion
is imported as:

  * regime = ``conventional``       (firewall-safe: never compares to empirical claims)
  * a single low-trust source       (``conceptnet:5.7``; trust default 0.2)
  * ``has_reference = 0`` on every support entry — it carries no document evidence,
    so the read path can discount or exclude it as an ungrounded prior.

The dump is ~34M lines / 10 GB, so this streams: it never holds the file in memory,
filters to English↔English assertions, maps the relation to a typed KB edge, and bulk-
inserts nodes + edges in batched transactions (INSERT OR IGNORE keyed by the same
deterministic hashes the distiller uses, so the import is idempotent and *fuses* with
distilled nodes that share a label).

Nodes are inserted with a NULL embedding (embedding 3.4M terms via the nomic endpoint
is infeasible during the import itself) — ConceptNet's own canonicalisation is the
identity key.  Out of the box they are reachable by graph walks from embedded nodes and
by label.  Run ``embed-nodes`` (``embed_nodes()`` below) afterwards to backfill vectors
so the commonsense layer also surfaces in dense semantic search; it is resumable, so it
can be left running / re-run as the embed endpoint allows.

The default import drops the low-signal lexical/etymological bulk (RelatedTo, FormOf,
DerivedFrom, HasContext, Etymologically*) — ~2.7M of the 3.4M EN assertions — keeping
the ~750k commonsense core.  Pass ``include_lexical=True`` (CLI ``--all``) to keep them.
"""
from __future__ import annotations

import logging
import re
import time

from .kb import KB, _hash, _pack

log = logging.getLogger("knowledgehost.conceptnet")

DOC_ID = "conceptnet:5.7"
DOC_TITLE = "ConceptNet 5.7 (commonsense knowledge graph)"

# ConceptNet relation -> (edge family, edge type, polarity).  family is free text but
# mirrors the distiller's vocabulary so the two knowledge sources share a graph shape.
# Relations NOT in this table are skipped (counted as 'unknown').
_REL = {
    # taxonomic / definitional
    "IsA":              ("taxonomic",   "is_a",                 ""),
    "InstanceOf":       ("taxonomic",   "instance_of",          ""),
    "MannerOf":         ("taxonomic",   "manner_of",            ""),
    "DefinedAs":        ("taxonomic",   "defined_as",           ""),
    # partonomic / compositional
    "PartOf":           ("partonomic",  "part_of",              ""),
    "HasA":             ("partonomic",  "has_a",                ""),
    "MadeOf":           ("partonomic",  "made_of",              ""),
    # functional / affordance
    "UsedFor":          ("functional",  "used_for",             ""),
    "CapableOf":        ("functional",  "capable_of",           ""),
    "NotCapableOf":     ("functional",  "capable_of",           "negative"),
    "ReceivesAction":   ("functional",  "receives_action",      ""),
    # causal / goal
    "Causes":           ("causal",      "causes",               "positive"),
    "CausesDesire":     ("causal",      "causes_desire",        "positive"),
    "MotivatedByGoal":  ("causal",      "motivated_by_goal",    ""),
    "CreatedBy":        ("causal",      "created_by",           ""),
    "Entails":          ("causal",      "entails",              ""),
    "HasPrerequisite":  ("causal",      "has_prerequisite",     ""),
    # procedural / temporal
    "HasSubevent":      ("procedural",  "has_subevent",         ""),
    "HasFirstSubevent": ("procedural",  "has_first_subevent",   ""),
    "HasLastSubevent":  ("procedural",  "has_last_subevent",    ""),
    # spatial
    "AtLocation":       ("spatial",     "at_location",          ""),
    "LocatedNear":      ("spatial",     "located_near",         ""),
    # attributive
    "HasProperty":      ("attributive", "has_property",         "positive"),
    "NotHasProperty":   ("attributive", "has_property",         "negative"),
    # affective
    "Desires":          ("affective",   "desires",              "positive"),
    "NotDesires":       ("affective",   "desires",              "negative"),
    # interpretive
    "SymbolOf":         ("interpretive", "symbol_of",           ""),
    # lexical / associative (default-skipped — see _LEXICAL)
    "Synonym":          ("lexical",     "synonym",              ""),
    "Antonym":          ("lexical",     "antonym",              "negative"),
    "DistinctFrom":     ("lexical",     "distinct_from",        ""),
    "SimilarTo":        ("lexical",     "similar_to",           ""),
    "RelatedTo":        ("associative", "related_to",           ""),
    "FormOf":           ("lexical",     "form_of",              ""),
    "DerivedFrom":      ("lexical",     "derived_from",         ""),
    "HasContext":       ("lexical",     "has_context",          ""),
    "EtymologicallyRelatedTo":   ("lexical", "etymologically_related_to",  ""),
    "EtymologicallyDerivedFrom": ("lexical", "etymologically_derived_from", ""),
}

# The high-volume, low-reasoning-value relations dropped unless --all is given.
_LEXICAL = {"RelatedTo", "FormOf", "DerivedFrom", "HasContext",
            "EtymologicallyRelatedTo", "EtymologicallyDerivedFrom"}

_WEIGHT_RE = re.compile(rb'"weight":\s*([0-9.]+)')


def _term(uri: str) -> str:
    """`/c/en/united_states/n` -> 'united states'.  Returns '' for a non-/c/en/ uri."""
    # parts: ['', 'c', 'en', '<term>', '<pos?>', ...]
    p = uri.split("/", 4)
    if len(p) < 4 or p[1] != "c" or p[2] != "en":
        return ""
    return p[3].replace("_", " ").strip()


def _support_json(trust: float, doc_id: str = DOC_ID) -> str:
    """A single constant support entry shared by every imported row (same source).
    `has_reference: 0` marks it ungrounded — a commonsense prior, not cited evidence."""
    import json
    return json.dumps([{
        "doc_id": doc_id,
        "evidence_cluster": "",
        "date": None,
        "trust_weight": trust,
        "regime": "conventional",
        "origin": "conventional",
        "has_reference": 0,
    }])


def import_conceptnet(kb: KB, path: str, *, min_weight: float = 1.0,
                      trust: float = 0.2, include_lexical: bool = False,
                      exclude=None,
                      limit: int | None = None, log_every: int = 1_000_000) -> dict:
    """Stream `path` (the assertions.csv dump) into the KB.  Returns a stats dict.

    Idempotent: nodes (hash of label) and edges (edge_hash) are INSERT OR IGNORE, so a
    re-run adds only what is new.  English↔English only; relations not in `_REL` (and,
    unless `include_lexical`, those in `_LEXICAL`) are skipped.

    `exclude` (relation names, e.g. ["FormOf","DerivedFrom"]) is ALWAYS skipped,
    whatever `include_lexical` says — which relations matter is the user's call,
    per relation, not an all-or-nothing lexical toggle.  Names are `_REL` keys."""
    kb.register_source(DOC_ID, DOC_TITLE, source_type="conceptnet",
                       trust_weight=trust, regime="conventional")
    support = _support_json(trust)
    now = time.time()
    skip = set() if include_lexical else set(_LEXICAL)
    user_skip = {str(r).strip() for r in (exclude or []) if str(r).strip()}
    unknown = user_skip - set(_REL)
    if unknown:                     # a typo here would silently exclude nothing
        log.warning("conceptnet: exclude names not in the relation table (typo?): %s",
                    ", ".join(sorted(unknown)))
    skip |= user_skip

    node_seen: set = set()
    nbuf: list = []          # (id, label, support)
    ebuf: list = []          # full edge row
    NODE_SQL = ("INSERT OR IGNORE INTO nodes"
                "(id,label,kind,summary,embedding,aliases,support,status)"
                " VALUES(?,?,'concept','',NULL,'[]',?,'active')")
    EDGE_SQL = ("INSERT OR IGNORE INTO edges"
                "(id,src_id,dst_id,family,type,mechanism,mechanism_basis,modifiers,"
                "polarity,embedding,edge_hash,support,strength,regime,scope,status,"
                "created_at,updated_at)"
                " VALUES(?,?,?,?,?,'','',?,?,NULL,?,?,NULL,'conventional','{}','active',?,?)")

    def flush():
        if nbuf:
            kb.db.executemany(NODE_SQL, nbuf)
            nbuf.clear()
        if ebuf:
            kb.db.executemany(EDGE_SQL, ebuf)
            ebuf.clear()
        kb.db.commit()

    st = {"read": 0, "imported": 0, "nodes": 0,
          "skip_lang": 0, "skip_rel": 0, "skip_weight": 0, "skip_unknown": 0}
    t0 = time.time()
    # binary read: the dump has heavy non-Latin text in non-EN rows we reject before
    # ever decoding, and bytes split/startswith is faster than str over 34M lines.
    with open(path, "rb") as fh:
        for raw in fh:
            st["read"] += 1
            if st["read"] % log_every == 0:
                rate = st["read"] / max(1e-6, time.time() - t0)
                log.info("conceptnet: read %.1fM lines, imported %d edges (%.0fk lines/s)",
                         st["read"] / 1e6, st["imported"], rate / 1000)
            # cols: uri, /r/Rel, /c/.., /c/.., {json}
            cols = raw.split(b"\t")
            if len(cols) < 5:
                continue
            if not (cols[2].startswith(b"/c/en/") and cols[3].startswith(b"/c/en/")):
                st["skip_lang"] += 1
                continue
            rname = cols[1].split(b"/", 3)[2].decode("ascii", "replace")  # /r/IsA -> IsA
            spec = _REL.get(rname)
            if spec is None:
                st["skip_unknown"] += 1
                continue
            if rname in skip:
                st["skip_rel"] += 1
                continue
            m = _WEIGHT_RE.search(cols[4])
            w = float(m.group(1)) if m else 1.0
            if w < min_weight:
                st["skip_weight"] += 1
                continue
            src_label = _term(cols[2].decode("utf-8", "replace"))
            dst_label = _term(cols[3].decode("utf-8", "replace"))
            if not src_label or not dst_label or src_label == dst_label:
                continue
            family, etype, polarity = spec
            sid = _hash("node", src_label, "concept")
            did = _hash("node", dst_label, "concept")
            for nid, lbl in ((sid, src_label), (did, dst_label)):
                if nid not in node_seen:
                    node_seen.add(nid)
                    nbuf.append((nid, lbl, support))
                    st["nodes"] += 1
            eh = KB._edge_hash(sid, did, family, etype, polarity, "", "conventional", "", "")
            mods = '{"cn_weight": %s}' % round(w, 3)
            ebuf.append((eh, sid, did, family, etype, mods, polarity, eh, support, now, now))
            st["imported"] += 1
            if len(ebuf) >= 50_000:
                flush()
            if limit and st["imported"] >= limit:
                break
    flush()
    # the running search caches (if any) are now stale; a fresh KB handle reloads them.
    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st


def embed_nodes(kb: KB, embedder, cfg, *, limit: int | None = None,
                log_every: int = 20_000) -> dict:
    """Backfill embeddings for nodes that have none (the bulk-imported commonsense
    terms), so they surface in dense search alongside distilled concepts.

    Streamed and resumable: it walks the NULL-embedding nodes by an ascending `id`
    cursor (never `LIMIT` off the front), so a row the embed server can't vectorise is
    stepped past rather than re-selected forever, and an interrupted run resumes simply
    by re-running (committed rows already carry a vector).  Stops cleanly if the embed
    endpoint drops — whatever was embedded is durable.  Vectors share the memory store's
    space (search_document: prefix, L2-normalised), so cosine matches the query side."""
    batch = max(1, int(cfg.get("embed_batch", 64)))
    st = {"embedded": 0, "skipped": 0, "remaining": 0}
    t0 = time.time()
    last_id = ""
    marked = 0
    while True:
        rows = kb.db.execute(
            "SELECT id, label, summary FROM nodes WHERE embedding IS NULL "
            "AND status='active' AND id > ? ORDER BY id LIMIT ?",
            (last_id, batch)).fetchall()
        if not rows:
            break
        last_id = rows[-1]["id"]
        texts = [(r["label"] + (". " + r["summary"] if r["summary"] else ""))
                 for r in rows]
        vecs = embedder.embed_many(texts, "document")
        if vecs is None:                                   # endpoint down → stop, resumable
            log.warning("embed endpoint unreachable — stopping (%d embedded; re-run to "
                        "resume).", st["embedded"])
            break
        updates = [(_pack(v), r["id"]) for r, v in zip(rows, vecs) if v]
        st["skipped"] += len(rows) - len(updates)          # un-embeddable; stepped past
        if updates:
            kb.db.executemany("UPDATE nodes SET embedding=? WHERE id=?", updates)
            kb.db.commit()
            st["embedded"] += len(updates)
        if st["embedded"] - marked >= log_every:
            marked = st["embedded"]
            rate = st["embedded"] / max(1e-6, time.time() - t0)
            log.info("embed-nodes: %d embedded (%.0f/s)", st["embedded"], rate)
        if limit and st["embedded"] >= limit:
            break
    st["remaining"] = kb.db.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NULL "
                                    "AND status='active'").fetchone()[0]
    st["elapsed_s"] = round(time.time() - t0, 1)
    kb._nodes_loaded = False                               # invalidate any search cache
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None
    return st
