"""The durable structured knowledge base — the spec's *card/graph tier* source
of truth (§3, §4), kept separate from the raw chunk store.

This is the substrate the metacognitive distiller writes into: canonical concept
**nodes**, **procedure cards** (how), **typed edges** (why/who/where/which …),
and the **surface** questions/propositions retrieval matches against.  It is the
"meaning" layer — what the LM distils *out* of source prose — as opposed to the
raw paraphrase-retrieval tier.

Design rules honoured here (the rest is the distiller's job, next):
  * **Evidence, not verdict** (§9.3): the write path only preserves and structures
    provenance.  ``support`` is a SET keyed by ``doc_id`` so re-ingestion is
    idempotent and copies never stack; ``strength`` is left NULL — it is a
    read-time, rigor-gated ranking signal, not a stored truth.
  * **Never clobber** (§9.2): contradictions become ``meta/disagrees_with`` edges
    between two *active* claims; nothing is overwritten.
  * **Bias toward not merging** (§9.4): over-merge is destructive, under-merge is
    recoverable — ambiguous matches make a DISTINCT node (+ an is_a edge or an
    adjudication-queue entry), never a silent merge.

SQLite + (optional) numpy only; the node set is small (proportional to genuine
concepts), so node-identity search is brute force.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("knowledgehost.kb")
perf = logging.getLogger("knowledgehost.perf")


class _Result:
    """The materialised result of one statement — rows already fetched, so nothing is
    left streaming on the (shared) sqlite cursor after the lock is released."""
    __slots__ = ("_rows", "_i", "rowcount", "lastrowid")

    def __init__(self, rows, rowcount, lastrowid):
        self._rows = rows
        self._i = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self):
        if self._i < len(self._rows):
            r = self._rows[self._i]
            self._i += 1
            return r
        return None

    def fetchall(self):
        r = self._rows[self._i:]
        self._i = len(self._rows)
        return r

    def fetchmany(self, n=1):
        r = self._rows[self._i:self._i + n]
        self._i += len(r)
        return r

    def __iter__(self):
        while self._i < len(self._rows):
            r = self._rows[self._i]
            self._i += 1
            yield r


class _LockedConn:
    """A thread-safe façade over one sqlite connection.  The HTTP server is threaded
    (ThreadingHTTPServer) and shares a single KB/connection; a bare sqlite connection used
    concurrently corrupts cursor state (None fetches) and segfaults in the C layer.  Here
    every statement runs to completion AND its rows are fetched under one re-entrant lock,
    so each `execute(...).fetchone()` is atomic — no two threads ever touch the cursor at
    once.  Writers (distillation, a separate process) are unaffected; same API as before."""

    def __init__(self, raw, lock):
        self._raw = raw
        self._lock = lock

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._raw.execute(sql, params)
            rows = cur.fetchall() if cur.description is not None else []
            return _Result(rows, cur.rowcount, cur.lastrowid)

    def executemany(self, sql, seq):
        with self._lock:
            cur = self._raw.executemany(sql, seq)
            return _Result([], cur.rowcount, cur.lastrowid)

    def executescript(self, sql):
        with self._lock:
            self._raw.executescript(sql)
            return _Result([], -1, None)

    def commit(self):
        with self._lock:
            self._raw.commit()

    def close(self):
        with self._lock:
            self._raw.close()

# Regimes (§8) and the source_type → default regime map (claim-level override later).
REGIMES = ("empirical", "conventional", "fictional", "interpretive", "historical")
TYPE_REGIME = {
    "peer_reviewed": "empirical", "textbook": "empirical", "reference": "empirical",
    "novel": "fictional", "fiction": "fictional", "essay": "interpretive",
    "history": "historical", "web": "empirical", "unknown": "empirical",
    # ingest format-types (no epistemic typing yet) default to empirical; the
    # source_registry is editable so a novel ingested as a PDF can be re-tagged.
    "pdf": "empirical", "epub": "empirical", "html": "empirical",
    "text": "empirical", "wikipedia": "empirical",
    # crowd-sourced commonsense (ConceptNet / ATOMIC / GLUCOSE): "what people hold".
    "conceptnet": "conventional", "atomic": "conventional", "glucose": "conventional",
    # CauseNet: causal claims mined from Wikipedia/ClueWeb — grounded but web-extracted.
    "causenet": "conventional",
}
# Trust priors by source_type (0..1), editable; unknown/web default low (§5).
TYPE_TRUST = {
    "peer_reviewed": 0.9, "textbook": 0.8, "reference": 0.7, "history": 0.6,
    "essay": 0.5, "novel": 0.4, "fiction": 0.4, "web": 0.3, "unknown": 0.25,
    "wikipedia": 0.7, "pdf": 0.5, "epub": 0.5, "html": 0.4, "text": 0.4,
    # ungrounded commonsense priors — low, discountable.
    "conceptnet": 0.2, "atomic": 0.2, "glucose": 0.2,
    "causenet": 0.4,    # grounded in source sentences → above the commonsense priors.
}

_WORD = re.compile(r"[a-z0-9]+")


def _norm_terms(s: str) -> set:
    return set(_WORD.findall((s or "").lower()))


def _pack(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(b: bytes):
    return list(struct.unpack(f"<{len(b)//4}f", b)) if b else []


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


class KB:
    def __init__(self, cfg: dict):
        self.path = cfg["kb_path"]
        self.theta_high = float(cfg.get("node_sim_high", 0.86))
        self.theta_low = float(cfg.get("node_sim_low", 0.72))
        self.min_sim = float(cfg.get("kb_min_sim", 0.35))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One re-entrant lock guards the connection (the HTTP server is threaded and shares
        # this KB).  RLock so a method may hold it across several statements — e.g. the
        # lazy cache loads below — while the per-statement proxy re-acquires it.
        self._lock = threading.RLock()
        # Sensitive working DBs (assembled from an encrypted overlay bundle) are
        # opened via SQLCipher; the base/master is plain sqlite3 (kb_encrypted unset).
        from . import dbcrypt
        raw = dbcrypt.connect(self.path, encrypted=bool(cfg.get("kb_encrypted")),
                              key=dbcrypt.key_for(cfg), check_same_thread=False)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("PRAGMA synchronous=NORMAL")
        raw.execute("PRAGMA busy_timeout=10000")
        # Hydration does many point lookups on a multi-GB table (embeddings stored inline);
        # mmap the DB and give SQLite a big page cache so those reads are RAM-speed instead
        # of faulting from disk.  Safe to set large on 64-bit; raise to ≥ the kb.db size.
        raw.execute(f"PRAGMA mmap_size={int(cfg.get('sqlite_mmap_mb', 4096)) * 1024 * 1024}")
        raw.execute(f"PRAGMA cache_size={-int(cfg.get('sqlite_cache_mb', 128)) * 1024}")
        self._raw = raw
        self.db = _LockedConn(raw, self._lock)
        # ANN index (usearch HNSW) for the read path — lazily memory-mapped on first
        # search if a built index exists; else brute-force.  See ann.py / build-ann.
        self._ann_enabled = bool(cfg.get("ann_search", True))
        ann_path = cfg.get("ann_path") or (self.path + ".ann")
        self._ann_path = str(Path(ann_path).expanduser())
        self._ann_ef = int(cfg.get("ann_expansion_search", 128))
        self._ann_mmap = bool(cfg.get("ann_mmap", False))
        self._ann = None
        self._ann_tried = False
        self._counts_cache = None  # (value, monotonic_ts) — see counts(); spares the viewer
        self._counts_ttl = float(cfg.get("counts_cache_ttl", 30.0))  # full scans every poll
        self._defer = 0            # >0 ⇒ inside a batch(): hold commits until it exits
        self._init_schema()
        try:
            import numpy as np
            self._np = np
        except Exception:
            self._np = None
        # Node-identity cache, grown incrementally (NEVER rebuilt from SQLite per
        # write — that churned an N×dim Python list every new node and leaked RSS).
        self._nodes_loaded = False
        self._node_ids: list = []
        self._node_vecs: list = []         # one (small) vector per node
        self._node_mat = None              # cached numpy stack; rebuilt only when dirty
        # Read-path surface cache (doc2query questions): same incremental scheme as the
        # node cache, so search() does one matmul instead of a per-row SQLite scan.
        self._surf_loaded = False
        self._surf_keys: list = []         # (target_kind, target_id) per row
        self._surf_vecs: list = []
        self._surf_mat = None

    # ── transaction batching ─────────────────────────────────────────────────
    # Each write method calls _maybe_commit() instead of committing outright; inside a
    # `with kb.batch():` block commits are held and flushed ONCE on exit, so distilling
    # a chunk (dozens of inserts) costs one fsync, not dozens.  Same-connection reads
    # still see uncommitted rows, so link_to_node etc. behave identically.
    def _maybe_commit(self):
        if self._defer == 0:
            self.db.commit()

    @contextmanager
    def batch(self):
        self._defer += 1
        try:
            yield
        finally:
            self._defer -= 1
            if self._defer == 0:
                self.db.commit()

    # ── schema (§4) ──────────────────────────────────────────────────────────
    def _init_schema(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS source_registry(
            doc_id TEXT PRIMARY KEY, title TEXT, source_type TEXT,
            trust_weight REAL, regime TEXT, pub_date TEXT,
            status TEXT DEFAULT 'active');

        -- embedding is LAST on purpose: the hot read path (search hydration) selects
        -- id,label,kind,summary,support and must not have to read past the ~3KB embedding
        -- blob's overflow pages to reach the columns after it.  See migrate_node_layout().
        CREATE TABLE IF NOT EXISTS nodes(
            id TEXT PRIMARY KEY, label TEXT, kind TEXT, summary TEXT,
            aliases TEXT, support TEXT, status TEXT DEFAULT 'active', embedding BLOB);

        -- distillation checkpoint (KB-side so it works for either raw backend).
        CREATE TABLE IF NOT EXISTS distilled_chunks(
            chunk_id TEXT PRIMARY KEY, distilled_at REAL);

        CREATE TABLE IF NOT EXISTS node_merge_candidates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_a TEXT, node_b TEXT, similarity REAL, reason TEXT,
            status TEXT DEFAULT 'open');

        CREATE TABLE IF NOT EXISTS procedure_cards(
            id TEXT PRIMARY KEY, node_id TEXT, title TEXT, domain TEXT, goal TEXT,
            preconditions TEXT, tools TEXT, materials TEXT, steps TEXT, tips TEXT,
            mistakes TEXT, safety TEXT, embedding BLOB, card_hash TEXT,
            hit_count INTEGER DEFAULT 0, support TEXT, strength REAL,
            regime TEXT, scope TEXT, status TEXT DEFAULT 'active',
            created_at REAL, updated_at REAL,
            -- §10 read-side payload: red flags to escalate on, the structured
            -- discriminators a query's context_features are scored against, and the
            -- conditional override ("would change to …") when a red flag fires.
            red_flags TEXT, discriminators TEXT, escalation TEXT,
            refined_at REAL);   -- Phase-2 in-place source-grounded refinement timestamp

        CREATE TABLE IF NOT EXISTS edges(
            id TEXT PRIMARY KEY, src_id TEXT, dst_id TEXT,
            family TEXT, type TEXT, mechanism TEXT, mechanism_basis TEXT,
            modifiers TEXT, polarity TEXT, embedding BLOB, edge_hash TEXT,
            support TEXT, strength REAL, regime TEXT, scope TEXT,
            status TEXT DEFAULT 'active', created_at REAL, updated_at REAL);
        CREATE INDEX IF NOT EXISTS edges_src ON edges(src_id);
        CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst_id);
        CREATE INDEX IF NOT EXISTS edges_hash ON edges(edge_hash);

        CREATE TABLE IF NOT EXISTS surface_questions(
            id TEXT PRIMARY KEY, target_kind TEXT, target_id TEXT,
            text TEXT, embedding BLOB);
        CREATE TABLE IF NOT EXISTS surface_propositions(
            id TEXT PRIMARY KEY, target_kind TEXT, target_id TEXT, text TEXT);

        CREATE TABLE IF NOT EXISTS knowledge_gaps(
            id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT, intent TEXT,
            effect_label TEXT, first_seen REAL, count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'open');

        -- multi-axis facet layer (facets.py): (kind,id) tagged on independent axes
        -- (epistemic/time_frame/trust_tier/domain).  Multi-valued per axis.  Only the
        -- epistemic axis gates the firewall; the rest are additive read-filters.
        CREATE TABLE IF NOT EXISTS facets(
            target_kind TEXT, target_id TEXT, axis TEXT, value TEXT,
            PRIMARY KEY (target_kind, target_id, axis, value));
        CREATE INDEX IF NOT EXISTS facets_axis ON facets(axis, value);
        CREATE INDEX IF NOT EXISTS facets_target ON facets(target_kind, target_id);

        -- card-level BM25 (retrieval contract §3.1): the lexical arm over cards that
        -- dense-only recall misses on exact terms.  Standalone FTS5 (not external-content:
        -- cards carry JSON columns), rebuilt lazily from procedure_cards — so a freshly
        -- assembled working DB self-heals on first ask.
        CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(card_id UNINDEXED, text);
        """)
        self._maybe_commit()
        self.migrate_card_layout()                # add §10 card columns to an existing DB
        self.migrate_source_layout()              # add the modular-bundle tag column

    def migrate_source_layout(self) -> list:
        """Add the modular-bundle tag (§16) and the licence columns (§16.4) to a
        source_registry created before they existed.  O(1) ALTERs, idempotent —
        untagged sources read as the 'base' bundle and 'unknown' licence, so this is
        invisible until you tag/license something."""
        cols = {r["name"] for r in self._raw.execute(
            "PRAGMA table_info(source_registry)").fetchall()}
        want = (("bundle", "TEXT"), ("license", "TEXT"), ("license_holder", "TEXT"),
                ("license_url", "TEXT"), ("license_text", "TEXT"),
                ("license_flags", "TEXT"))
        added = []
        with self._lock:
            for c, typ in want:
                if c not in cols:
                    self._raw.execute(f"ALTER TABLE source_registry ADD COLUMN {c} {typ}")
                    added.append(c)
            if added:
                self._raw.commit()
        if added:
            log.info("source_registry: added columns %s", added)
        return added

    def migrate_card_layout(self) -> list:
        """Add the read-side card columns (red_flags / discriminators / escalation) to a
        procedure_cards table created before they existed.  ALTER ADD COLUMN is O(1) in
        SQLite (no table rewrite), so this is a cheap idempotent no-op once applied —
        called at every open so a deploy just needs a restart, not a migration step."""
        cols = {r["name"] for r in self._raw.execute(
            "PRAGMA table_info(procedure_cards)").fetchall()}
        added = []
        with self._lock:
            for c, typ in (("red_flags", "TEXT"), ("discriminators", "TEXT"),
                           ("escalation", "TEXT"), ("refined_at", "REAL"),
                           # polymorphic cards: how-to | criteria | staging | (recommendation
                           # rides as a grade on any card).  criteria/grade are JSON payloads.
                           ("card_type", "TEXT"), ("criteria", "TEXT"), ("grade", "TEXT")):
                if c not in cols:
                    self._raw.execute(f"ALTER TABLE procedure_cards ADD COLUMN {c} {typ}")
                    added.append(c)
            # empirical findings (effect size / study design / population / certainty)
            # attached to a causal edge — the structured form of a paper's claim.
            ecols = {r["name"] for r in self._raw.execute(
                "PRAGMA table_info(edges)").fetchall()}
            if "finding" not in ecols:
                self._raw.execute("ALTER TABLE edges ADD COLUMN finding TEXT")
                added.append("edges.finding")
            if added:
                self._raw.commit()
        if added:
            log.info("procedure_cards/edges: added columns %s", added)
        return added

    # ── source registry (§4.3) ───────────────────────────────────────────────
    def register_source(self, doc_id, title, source_type="unknown", *,
                        trust_weight=None, regime=None, pub_date=None,
                        license=None, license_holder=None, license_url=None,
                        license_text=None, bundle=None):
        source_type = (source_type or "unknown").lower()
        ins_trust = TYPE_TRUST.get(source_type, 0.25) if trust_weight is None \
            else float(trust_weight)
        ins_regime = TYPE_REGIME.get(source_type, "empirical") if regime is None else regime
        # On re-registration only overwrite the epistemic fields the caller
        # EXPLICITLY supplied — otherwise PRESERVE a manual re-tag.  The registry is
        # user-editable (e.g. mark a novel ingested as a PDF 'fictional'); a routine
        # re-distill passes the format default and must not clobber that edit.
        sets = ["title=excluded.title", "source_type=excluded.source_type"]
        if trust_weight is not None:
            sets.append("trust_weight=excluded.trust_weight")
        if regime is not None:
            sets.append("regime=excluded.regime")
        if pub_date is not None:
            sets.append("pub_date=excluded.pub_date")
        # Licence from ingest-time DETECTION is fill-if-empty: it seeds a source that
        # has none, but never overwrites an existing value (a manual `set_source`
        # edit, or an earlier detection, wins).  COALESCE(existing, new).
        lflags = None
        if license is not None:
            from . import licensing
            license = licensing.canonical(license)
            lflags = json.dumps(licensing.flags_for(license))
            sets.append("license=COALESCE(license, excluded.license)")
            sets.append("license_flags=COALESCE(license_flags, excluded.license_flags)")
        if license_holder is not None:
            sets.append("license_holder=COALESCE(license_holder, excluded.license_holder)")
        if license_url is not None:
            sets.append("license_url=COALESCE(license_url, excluded.license_url)")
        if license_text is not None:
            sets.append("license_text=COALESCE(license_text, excluded.license_text)")
        # Bundle assignment (e.g. the research loop's 'vinkona' group): fill-if-empty so a
        # routine re-distill can't clobber a manual re-bundle via set_source.
        if bundle is not None:
            sets.append("bundle=COALESCE(bundle, excluded.bundle)")
        self.db.execute(
            "INSERT INTO source_registry(doc_id,title,source_type,trust_weight,"
            "regime,pub_date,status,license,license_holder,license_url,license_text,"
            "license_flags,bundle) VALUES(?,?,?,?,?,?, 'active', ?,?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET " + ", ".join(sets),
            (doc_id, title, source_type, ins_trust, ins_regime, pub_date,
             license, license_holder, license_url, license_text, lflags,
             (str(bundle).strip() or None) if bundle is not None else None))
        self._maybe_commit()
        # Return the EFFECTIVE values (a preserved re-tag wins over the default).
        row = self.db.execute("SELECT trust_weight, regime, license, license_holder "
                              "FROM source_registry WHERE doc_id=?", (doc_id,)).fetchone()
        return {"doc_id": doc_id, "trust_weight": float(row["trust_weight"]),
                "regime": row["regime"], "license": row["license"],
                "license_holder": row["license_holder"]}

    def get_source(self, doc_id):
        r = self.db.execute("SELECT * FROM source_registry WHERE doc_id=?",
                            (doc_id,)).fetchone()
        return dict(r) if r else None

    def set_source(self, doc_id, *, title=None, bundle=None, license=None,
                   license_holder=None, license_url=None, license_text=None):
        """Curate a source's provenance + rights metadata (§16.4): rename (``title``),
        regroup (``bundle``), and set the licence (``license`` SPDX id, ``license_holder``
        = to whom it belongs, url, verbatim text).  This is the AUTHORITATIVE manual
        edit — it overwrites (unlike ingest detection, which only fills empties).  Only
        supplied fields are written; setting license also recomputes license_flags.
        Returns the updated row (None if unknown)."""
        sets, params = [], []
        if title is not None:
            sets.append("title=?")
            params.append(str(title))
        if bundle is not None:
            b = (str(bundle).strip() or None)       # empty ⇒ back to the base bundle
            sets.append("bundle=?")
            params.append(b)
        if license is not None:
            from . import licensing
            spdx = licensing.canonical(license) if license else None
            sets.append("license=?")
            params.append(spdx)
            sets.append("license_flags=?")
            params.append(json.dumps(licensing.flags_for(spdx)) if spdx else None)
        if license_holder is not None:
            sets.append("license_holder=?")
            params.append(str(license_holder) or None)
        if license_url is not None:
            sets.append("license_url=?")
            params.append(str(license_url) or None)
        if license_text is not None:
            sets.append("license_text=?")
            params.append(str(license_text) or None)
        if not sets:
            return self.get_source(doc_id)
        params.append(doc_id)
        with self._lock:
            cur = self._raw.execute(
                f"UPDATE source_registry SET {', '.join(sets)} WHERE doc_id=?", params)
            self._raw.commit()
        if cur.rowcount == 0:
            return None
        return self.get_source(doc_id)

    # ── licence roll-up & queries (§16.4) ────────────────────────────────────
    def license_of(self, doc_id) -> dict:
        """One source's licence record (canonical id + holder + flags)."""
        r = self.db.execute(
            "SELECT license, license_holder, license_url, license_flags "
            "FROM source_registry WHERE doc_id=?", (doc_id,)).fetchone()
        if not r:
            return {"license": "unknown", "license_holder": None, "flags": None}
        from . import licensing
        flags = None
        if r["license_flags"]:
            try:
                flags = json.loads(r["license_flags"])
            except ValueError:
                flags = None
        return {"license": r["license"] or "unknown",
                "license_holder": r["license_holder"], "license_url": r["license_url"],
                "flags": flags or licensing.flags_for(r["license"])}

    def license_for_support(self, support) -> dict:
        """Shippable-under licence for anything backed by a support set: the
        most-restrictive intersection of its sources' licences (§16.4).  ``support``
        is the JSON string or parsed list stored on a card/edge/node."""
        from . import licensing
        try:
            rows = json.loads(support) if isinstance(support, str) else (support or [])
        except ValueError:
            rows = []
        doc_ids = [e.get("doc_id") for e in rows if isinstance(e, dict) and e.get("doc_id")]
        recs = []
        for d in dict.fromkeys(doc_ids):            # distinct, order-preserving
            lo = self.license_of(d)
            recs.append({"license": lo["license"], "license_holder": lo["license_holder"],
                         "flags": lo["flags"]})
        return licensing.combine(recs)

    def card_license(self, card_id) -> dict:
        r = self.db.execute("SELECT support FROM procedure_cards WHERE id=?",
                            (card_id,)).fetchone()
        return self.license_for_support(r["support"] if r else "[]")

    def sources_by_license(self, *, spdx=None, holder=None, permission=None,
                           limit=500) -> list:
        """Find sources on a licence basis: by SPDX id, by holder (substring), and/or
        by a granted permission (redistribute/derivatives/commercial).  All optional
        and AND-combined."""
        from . import licensing
        rows = self.list_sources(limit)
        out = []
        for s in rows:
            if spdx and licensing.canonical(s.get("license")) != licensing.canonical(spdx):
                continue
            if holder and holder.lower() not in (s.get("license_holder") or "").lower():
                continue
            if permission:
                fl = licensing.flags_for(s.get("license"))
                if fl.get(permission) is not True:
                    continue
            out.append(s)
        return out

    def bundle_summary(self) -> list:
        """Sources grouped by bundle tag, for the control panel: how much knowledge
        each provenance group carries.  ``base`` = untagged."""
        rows = self.db.execute(
            "SELECT COALESCE(NULLIF(bundle,''),'base') AS bundle, "
            "COUNT(*) AS sources, "
            "GROUP_CONCAT(doc_id, '\x1f') AS docs "
            "FROM source_registry WHERE status!='retracted' "
            "GROUP BY 1 ORDER BY 1").fetchall()
        return [{"bundle": r["bundle"], "sources": r["sources"],
                 "docs": (r["docs"] or "").split("\x1f")} for r in rows]

    # ── node identity policy (§9.4) ──────────────────────────────────────────
    def _load_nodes(self):
        """One-time load of node embeddings into the in-memory cache; afterwards
        kept current by incremental appends in _new_node (no re-scan, no churn).  Guarded
        so two server threads can't double-populate the shared lists."""
        if self._nodes_loaded:
            return
        with self._lock:
            if self._nodes_loaded:                # re-check under the lock
                return
            ids, vecs = [], []
            np = self._np
            # Stream through the RAW cursor (not the proxy, which would fetchall ~1M blobs
            # at once) and decode each with np.frombuffer (zero-copy, no per-row struct
            # call) — so the cold load is fast and doesn't spike to several GB at once.
            cur = self._raw.execute("SELECT id, embedding FROM nodes "
                                    "WHERE status='active' AND embedding IS NOT NULL")
            for r in cur:
                ids.append(r["id"])
                vecs.append(np.frombuffer(r["embedding"], dtype="<f4")
                            if np is not None else _unpack(r["embedding"]))
            self._node_ids, self._node_vecs = ids, vecs
            self._node_mat = None
            self._nodes_loaded = True

    def _cache_node(self, nid, embedding):
        if not self._nodes_loaded or embedding is None:
            return                              # picked up by the next _load_nodes if unloaded
        self._node_ids.append(nid)
        self._node_vecs.append(self._np.asarray(embedding, dtype="float32")
                               if self._np is not None else list(embedding))
        self._node_mat = None                   # restack lazily on the next search

    def _node_matrix(self):
        """(ids, numpy matrix) of active node embeddings, restacked only when dirty."""
        self._load_nodes()
        if not self._node_ids or self._np is None:
            return self._node_ids, None
        if self._node_mat is None:
            self._node_mat = self._np.stack(self._node_vecs)
        return self._node_ids, self._node_mat

    def _best_node(self, embedding):
        ids, mat = self._node_matrix()
        if not ids:
            return None, 0.0
        if mat is not None:
            sims = mat @ self._np.asarray(embedding, dtype="float32")  # cosine (both L2-norm)
            i = int(sims.argmax())
            return ids[i], float(sims[i])
        best_i, best_s = 0, -1.0
        for i, v in enumerate(self._node_vecs):
            s = sum(a * b for a, b in zip(v, embedding))
            if s > best_s:
                best_i, best_s = i, s
        return ids[best_i], best_s

    # Read-path surface cache (mirrors the node cache) — see search().
    def _load_surf(self):
        if self._surf_loaded:
            return
        with self._lock:
            if self._surf_loaded:                 # re-check under the lock
                return
            keys, vecs = [], []
            for r in self.db.execute("SELECT target_kind,target_id,embedding FROM "
                                     "surface_questions WHERE embedding IS NOT NULL"):
                v = _unpack(r["embedding"])
                keys.append((r["target_kind"], r["target_id"]))
                vecs.append(self._np.asarray(v, dtype="float32")
                            if self._np is not None else v)
            self._surf_keys, self._surf_vecs = keys, vecs
            self._surf_mat = None
            self._surf_loaded = True

    def _cache_surf(self, key, embedding):
        if not self._surf_loaded or embedding is None:
            return
        self._surf_keys.append(key)
        self._surf_vecs.append(self._np.asarray(embedding, dtype="float32")
                               if self._np is not None else list(embedding))
        self._surf_mat = None

    def _surf_matrix(self):
        self._load_surf()
        if not self._surf_keys or self._np is None:
            return self._surf_keys, None
        if self._surf_mat is None:
            self._surf_mat = self._np.stack(self._surf_vecs)
        return self._surf_keys, self._surf_mat

    def _alias_agreement(self, label, aliases, node_id):
        r = self.db.execute("SELECT label, aliases FROM nodes WHERE id=?",
                            (node_id,)).fetchone()
        if not r:
            return False
        cand = _norm_terms(label)
        for a in (aliases or []):
            cand |= _norm_terms(a)
        existing = _norm_terms(r["label"])
        for a in json.loads(r["aliases"] or "[]"):
            existing |= _norm_terms(a)
        return bool(cand & existing)

    @staticmethod
    def _generalization(a_label, b_label):
        """Cheap is_a heuristic (§9.4 example: 'evaporative dry eye' is_a 'dry eye').
        Returns (specific, general) or None."""
        a, b = _norm_terms(a_label), _norm_terms(b_label)
        if a == b:
            return None
        if b and b < a:
            return (a_label, b_label)            # a is the more specific (superset of terms)
        if a and a < b:
            return (b_label, a_label)
        return None

    def link_to_node(self, label, kind, embedding, *, summary="", aliases=()):
        """Map an extracted concept to a canonical node (§9.4).  Returns
        (node_id, action) where action ∈ {same, distinct_isa, distinct_adjudicate, new}."""
        aliases = list(aliases or [])
        best_id, sim = self._best_node(embedding)
        if best_id and sim >= self.theta_high and self._alias_agreement(label, aliases, best_id):
            self._merge_aliases(best_id, label, aliases)
            return best_id, "same"
        # Anything else above θ_low is ambiguous — including a high-similarity match
        # whose aliases DON'T confirm identity.  Make a distinct node and adjudicate;
        # never silently merge (over-merge is destructive) nor silently split.
        if best_id and sim >= self.theta_low:
            new_id = self._new_node(label, kind, summary, embedding, aliases)
            gen = self._generalization(label, self._label_of(best_id))
            if gen:
                specific = new_id if gen[0] == label else best_id
                general = best_id if specific == new_id else new_id
                self.add_edge(specific, general, family="taxonomic", type="is_a",
                              regime="empirical")
                return new_id, "distinct_isa"
            self.db.execute(
                "INSERT INTO node_merge_candidates(node_a,node_b,similarity,reason,status)"
                " VALUES(?,?,?,?, 'open')", (new_id, best_id, sim, "ambiguous_similarity"))
            self._maybe_commit()
            return new_id, "distinct_adjudicate"
        return self._new_node(label, kind, summary, embedding, aliases), "new"

    def _label_of(self, node_id):
        r = self.db.execute("SELECT label FROM nodes WHERE id=?", (node_id,)).fetchone()
        return r["label"] if r else ""

    def _merge_aliases(self, node_id, label, aliases):
        r = self.db.execute("SELECT label, aliases FROM nodes WHERE id=?",
                            (node_id,)).fetchone()
        have = json.loads(r["aliases"] or "[]")
        seen = {a.lower() for a in have} | {r["label"].lower()}
        for a in [label, *aliases]:
            if a and a.lower() not in seen:
                have.append(a); seen.add(a.lower())
        self.db.execute("UPDATE nodes SET aliases=? WHERE id=?",
                        (json.dumps(have), node_id))
        self._maybe_commit()

    def _new_node(self, label, kind, summary, embedding, aliases):
        nid = _hash("node", label or "", kind or "")
        self.db.execute(
            "INSERT OR IGNORE INTO nodes(id,label,kind,summary,embedding,aliases,support,status)"
            " VALUES(?,?,?,?,?,?,'[]', 'active')",
            (nid, label, kind, summary,
             _pack(embedding) if embedding is not None else None,
             json.dumps(list(aliases or []))))
        self._maybe_commit()
        self._cache_node(nid, embedding)        # incremental — no full rebuild
        return nid

    def add_node_support(self, node_id, doc_id, evidence="", *, summary=None, regime=None):
        """Record a node's provenance (doc_id-keyed set, §9.3) and, if the node has
        no summary yet, set it — existing summaries are never clobbered.  `regime`
        overrides the source default (e.g. a convention or belief distilled FROM a
        novel reads as conventional/interpretive, not fictional)."""
        r = self.db.execute("SELECT summary, support FROM nodes WHERE id=?",
                            (node_id,)).fetchone()
        if not r:
            return
        sup = self._merge_support(r["support"], self._support_entry(doc_id, evidence, regime=regime))
        new_summary = r["summary"] or (summary or "")
        self.db.execute("UPDATE nodes SET support=?, summary=? WHERE id=?",
                        (sup, new_summary, node_id))
        self._maybe_commit()

    # ── support set (§9.3): evidence preserved, no verdict baked in ───────────
    @staticmethod
    def _merge_support(existing_json, entry):
        """SET keyed by doc_id — re-adding the same source is idempotent (copies
        never stack); each entry carries trust/date/regime for read-time strength."""
        rows = json.loads(existing_json or "[]")
        by_doc = {r["doc_id"]: r for r in rows}
        by_doc[entry["doc_id"]] = {**by_doc.get(entry["doc_id"], {}), **entry}
        return json.dumps(list(by_doc.values()))

    def _support_entry(self, doc_id, evidence="", date=None, regime=None, locator=""):
        src = self.get_source(doc_id) or {}
        entry = {"doc_id": doc_id,
                 "evidence_cluster": _hash("ev", evidence)[:12],
                 # human-readable pointer INTO the source (e.g. "S3.2 p.41") so a returned
                 # card can cite the page, not just the document.  Optional; "" when unknown.
                 "locator": locator or "",
                 "date": date or src.get("pub_date"),
                 "trust_weight": src.get("trust_weight"),
                 # `regime` = this claim's epistemic kind (may be overridden per item);
                 # `origin` = the SOURCE's authoritative regime (the fiction/history
                 # folder), never overridden — the axis a strict mode excludes on.
                 "regime": regime or src.get("regime"),
                 "origin": src.get("regime")}
        # Provenance tag so the read path can BAND vinkona-sourced claims below curated
        # ones (research_loop_spec §6 trust posture).  Only stamped when non-curated, so
        # curated support entries stay lean.
        if src.get("source_type") == "vinkona":
            entry["provenance"] = "vinkona"
        return entry

    # ── typed edges (§4.5) — banding by hash (§9.1); contradictions never clobber ─
    @staticmethod
    def _edge_hash(src_id, dst_id, family, type, polarity, mechanism,
                   regime, world="", conditions=""):
        # regime + world are part of claim identity: a fictional/empirical pair must
        # never be hash-identical (else band-1 corroborates across the firewall §8).
        # conditions too, so a refinement misses band-1 and reaches the 5-way (§9.2).
        return _hash("edge", src_id, dst_id, family, type, polarity, mechanism,
                     regime, world or "", conditions or "")

    def add_edge(self, src_id, dst_id, *, family, type, mechanism="",
                 mechanism_basis="stated", modifiers=None, polarity="",
                 regime="empirical", scope=None, doc_id=None, evidence="",
                 embedding=None, finding=None):
        modifiers = modifiers or {}
        eh = self._edge_hash(src_id, dst_id, family, type, polarity, mechanism,
                             regime, (scope or {}).get("world"), modifiers.get("conditions"))
        existing = self.db.execute("SELECT id, support FROM edges WHERE edge_hash=?",
                                   (eh,)).fetchone()
        now = time.time()
        if existing:                                          # auto-corroborate (§9.1 band 2)
            if doc_id:
                sup = self._merge_support(existing["support"],
                                          self._support_entry(doc_id, evidence))
                sets = "support=?, updated_at=?"
                params = [sup, now]
                if finding:                                   # fill an empirical finding if absent
                    sets += ", finding=COALESCE(finding, ?)"
                    params.append(json.dumps(finding))
                params.append(existing["id"])
                self.db.execute(f"UPDATE edges SET {sets} WHERE id=?", params)
                self._maybe_commit()
            return existing["id"], "corroborate"
        eid = eh
        sup = json.dumps([self._support_entry(doc_id, evidence)]) if doc_id else "[]"
        self.db.execute(
            "INSERT OR IGNORE INTO edges(id,src_id,dst_id,family,type,mechanism,"
            "mechanism_basis,modifiers,polarity,embedding,edge_hash,support,strength,"
            "regime,scope,status,created_at,updated_at,finding) VALUES"
            "(?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?, 'active',?,?,?)",
            (eid, src_id, dst_id, family, type, mechanism, mechanism_basis,
             json.dumps(modifiers or {}), polarity,
             _pack(embedding) if embedding is not None else None, eh, sup,
             regime, json.dumps(scope or {}), now, now,
             json.dumps(finding) if finding else None))
        self._maybe_commit()
        return eid, "insert"

    # ── reconciliation primitives (§9.1-9.2) ─────────────────────────────────
    def edge_by_hash(self, src_id, dst_id, family, type, polarity="", mechanism="",
                     regime="empirical", world="", conditions=""):
        eh = self._edge_hash(src_id, dst_id, family, type, polarity, mechanism,
                             regime, world, conditions)
        r = self.db.execute("SELECT * FROM edges WHERE edge_hash=?", (eh,)).fetchone()
        return dict(r) if r else None

    def comparable_edges(self, src_id, dst_id, regime, *, scope_world=None):
        """Edges sharing the (src,dst) pair, **same regime** (the write-time
        firewall §8 — non-empirical claims can never compare to empirical ones),
        optionally same fictional ``world`` scope.  meta edges are not claims."""
        rows = self.db.execute(
            "SELECT * FROM edges WHERE src_id=? AND dst_id=? AND regime=? "
            "AND status='active' AND family!='meta'", (src_id, dst_id, regime)).fetchall()
        out = []
        for r in rows:
            if scope_world is not None:
                if (json.loads(r["scope"] or "{}").get("world")) != scope_world:
                    continue
            out.append(dict(r))
        return out

    def corroborate_edge(self, edge_id, doc_id, evidence=""):
        r = self.db.execute("SELECT support FROM edges WHERE id=?", (edge_id,)).fetchone()
        if not r or not doc_id:
            return
        sup = self._merge_support(r["support"], self._support_entry(doc_id, evidence))
        self.db.execute("UPDATE edges SET support=?, updated_at=? WHERE id=?",
                        (sup, time.time(), edge_id))
        self._maybe_commit()

    def enrich_edge(self, edge_id, modifiers, doc_id=None, evidence=""):
        """Refinement (§9.2): narrow an existing claim with extra conditions/
        discriminators rather than inserting a near-duplicate."""
        r = self.db.execute("SELECT modifiers, support FROM edges WHERE id=?",
                            (edge_id,)).fetchone()
        if not r:
            return
        mods = json.loads(r["modifiers"] or "{}")
        for k, v in (modifiers or {}).items():
            if v:
                mods[k] = v
        sup = (self._merge_support(r["support"], self._support_entry(doc_id, evidence))
               if doc_id else r["support"])
        self.db.execute("UPDATE edges SET modifiers=?, support=?, updated_at=? WHERE id=?",
                        (json.dumps(mods), sup, time.time(), edge_id))
        self._maybe_commit()

    def link_meta(self, edge_a, edge_b, type):
        """A claim-to-claim relation (§4.5 meta family): alternative_to /
        context_variant_of / disagrees_with / refines / supersedes.  Both active —
        contradictions are recorded between live claims, never overwrites."""
        return self.add_edge(edge_a, edge_b, family="meta", type=type, regime="meta")

    # ── procedure cards (§4.4) ───────────────────────────────────────────────
    def add_card(self, node_id, *, title, goal="", domain="", steps=None,
                 preconditions=None, tools=None, materials=None, tips=None,
                 mistakes=None, safety=None, red_flags=None, discriminators=None,
                 escalation=None, regime="empirical", scope=None,
                 card_type="procedure", criteria=None, grade=None,
                 doc_id=None, evidence="", locator="", embedding=None):
        card_type = (card_type or "procedure").strip().lower()
        # card_type + criteria are part of identity: a diagnostic-criteria card and a
        # how-to card for the same concept/title are DIFFERENT cards, not a collision.
        # A plain procedure card keeps the ORIGINAL hash formula (so existing card ids are
        # stable across this upgrade — no churn); only non-procedure/criteria cards extend it.
        if card_type == "procedure" and not criteria:
            ch = _hash("card", node_id or "", title or "", goal or "",
                       json.dumps(steps or []))
        else:
            ch = _hash("card", node_id or "", title or "", goal or "",
                       json.dumps(steps or []), card_type,
                       json.dumps(criteria or {}, sort_keys=True))
        existing = self.db.execute("SELECT id, support FROM procedure_cards WHERE card_hash=?",
                                   (ch,)).fetchone()
        now = time.time()
        if existing:
            if doc_id:
                sup = self._merge_support(existing["support"],
                                          self._support_entry(doc_id, evidence, locator=locator))
                self.db.execute("UPDATE procedure_cards SET support=?, updated_at=? WHERE id=?",
                                (sup, now, existing["id"]))
                self._maybe_commit()
            return existing["id"], "corroborate"
        cid = ch
        sup = (json.dumps([self._support_entry(doc_id, evidence, locator=locator)])
               if doc_id else "[]")
        self.db.execute(
            "INSERT OR IGNORE INTO procedure_cards(id,node_id,title,domain,goal,"
            "preconditions,tools,materials,steps,tips,mistakes,safety,embedding,"
            "card_hash,hit_count,support,strength,regime,scope,status,created_at,updated_at,"
            "red_flags,discriminators,escalation,card_type,criteria,grade)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,NULL,?,?, 'active',?,?,?,?,?,?,?,?)",
            (cid, node_id, title, domain, goal,
             json.dumps(preconditions or []), json.dumps(tools or []),
             json.dumps(materials or []), json.dumps(steps or []),
             json.dumps(tips or []), json.dumps(mistakes or []),
             json.dumps(safety or []),
             _pack(embedding) if embedding is not None else None, ch, sup,
             regime, json.dumps(scope or {}), now, now,
             json.dumps(red_flags or []), json.dumps(discriminators or []),
             json.dumps(escalation or []), card_type,
             json.dumps(criteria) if criteria else None,
             json.dumps(grade) if grade else None))
        self._maybe_commit()
        return cid, "insert"

    _CARD_JSON_FIELDS = ("steps", "preconditions", "tools", "materials", "tips",
                         "mistakes", "safety", "red_flags", "discriminators", "escalation")
    _CARD_TEXT_FIELDS = ("title", "goal", "domain")

    def refresh_card(self, card_id, fields: dict, *, embedding=None,
                     surface=None, surface_vec=None) -> bool:
        """Phase-2 in-place refinement (§ Phase 2): overwrite the supplied card fields with
        improved, source-grounded versions — provenance (`support`) is preserved untouched.
        Recomputes `card_hash` (title/goal/steps may have changed), stamps `refined_at`, and
        optionally re-embeds the card and refreshes its surface question so retrieval tracks
        the new title/goal.  Never inserts a duplicate; returns False if the card is gone."""
        r = self.db.execute("SELECT title, goal, steps, node_id FROM procedure_cards "
                            "WHERE id=?", (card_id,)).fetchone()
        if not r:
            return False
        sets, vals = [], []
        for k, v in fields.items():
            if k in self._CARD_JSON_FIELDS:
                sets.append(f"{k}=?"); vals.append(json.dumps(v or []))
            elif k in self._CARD_TEXT_FIELDS:
                sets.append(f"{k}=?"); vals.append(str(v or "")[:500])
        new_title = fields.get("title", r["title"])
        new_goal = fields.get("goal", r["goal"])
        new_steps = fields["steps"] if "steps" in fields else json.loads(r["steps"] or "[]")
        ch = _hash("card", r["node_id"] or "", new_title or "", new_goal or "",
                   json.dumps(new_steps or []))
        now = time.time()
        sets += ["card_hash=?", "refined_at=?", "updated_at=?"]
        vals += [ch, now, now]
        if embedding is not None:
            sets.append("embedding=?"); vals.append(_pack(embedding))
        vals.append(card_id)
        self.db.execute(f"UPDATE procedure_cards SET {', '.join(sets)} WHERE id=?", vals)
        if surface is not None:                    # retire the stale "how do you {old}?" Q
            self.db.execute("DELETE FROM surface_questions WHERE target_kind='card' "
                            "AND target_id=?", (card_id,))
            self.add_surface_question("card", card_id, surface, surface_vec)
        self._maybe_commit()
        return True

    # ── surface generation (§4.6) ────────────────────────────────────────────
    def add_surface_question(self, target_kind, target_id, text, embedding=None):
        qid = _hash("sq", target_kind, target_id, text)
        cur = self.db.execute(
            "INSERT OR IGNORE INTO surface_questions(id,target_kind,target_id,text,embedding)"
            " VALUES(?,?,?,?,?)",
            (qid, target_kind, target_id, text,
             _pack(embedding) if embedding is not None else None))
        if cur.rowcount:                            # new row → keep the read cache current
            self._cache_surf((target_kind, target_id), embedding)
        self._maybe_commit()
        return qid

    def add_surface_proposition(self, target_kind, target_id, text):
        pid = _hash("sp", target_kind, target_id, text)
        self.db.execute(
            "INSERT OR IGNORE INTO surface_propositions(id,target_kind,target_id,text)"
            " VALUES(?,?,?,?)", (pid, target_kind, target_id, text))
        self._maybe_commit()
        return pid

    # ── gaps (§4.7) ──────────────────────────────────────────────────────────
    def log_gap(self, query_text, intent="", effect_label=""):
        row = self.db.execute("SELECT id FROM knowledge_gaps WHERE query_text=? AND status='open'",
                             (query_text,)).fetchone()
        if row:
            self.db.execute("UPDATE knowledge_gaps SET count=count+1 WHERE id=?", (row["id"],))
        else:
            self.db.execute(
                "INSERT INTO knowledge_gaps(query_text,intent,effect_label,first_seen,count,status)"
                " VALUES(?,?,?,?,1,'open')", (query_text, intent, effect_label, time.time()))
        self._maybe_commit()

    def close_gap(self, query_text, status="acquired") -> int:
        """Close the open knowledge_gap a query opened once a card grounds it
        (research_loop_spec §6.2 — the loop-closer).  Matches case-/space-insensitively
        on the verbatim query.  Returns how many gaps were closed."""
        if not query_text or not str(query_text).strip():
            return 0
        norm = " ".join(str(query_text).lower().split())
        with self._lock:
            cur = self._raw.execute(
                "UPDATE knowledge_gaps SET status=? WHERE status='open' "
                "AND lower(trim(query_text))=? ", (status, norm))
            self._raw.commit()
        if cur.rowcount:
            log.info("closed %d knowledge gap(s) for %r → %s", cur.rowcount, query_text, status)
        return cur.rowcount

    # ── facets (multi-axis classification, facets.py) ────────────────────────
    def set_facets(self, target_kind, target_id, axis, values):
        """Replace all values on one axis for one target (idempotent re-derivation).
        `values` may be a str or iterable; empty clears the axis."""
        if isinstance(values, str):
            values = [values]
        vals = [str(v) for v in (values or []) if v not in (None, "")]
        with self._lock:
            self._raw.execute(
                "DELETE FROM facets WHERE target_kind=? AND target_id=? AND axis=?",
                (target_kind, target_id, axis))
            self._raw.executemany(
                "INSERT OR IGNORE INTO facets(target_kind,target_id,axis,value) "
                "VALUES(?,?,?,?)", [(target_kind, target_id, axis, v) for v in vals])
            self._raw.commit()

    def get_facets(self, target_kind, target_id) -> dict:
        rows = self.db.execute(
            "SELECT axis, value FROM facets WHERE target_kind=? AND target_id=?",
            (target_kind, target_id)).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["axis"], []).append(r["value"])
        return out

    def facets_for(self, targets) -> dict:
        """Batch fetch facets for many (kind,id) pairs → {(kind,id): {axis:set}}.  Used by
        the read-time filter so a whole candidate pool costs one query, not N."""
        ids = list({t[1] for t in targets})
        if not ids:
            return {}
        out: dict = {}
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            qm = ",".join("?" * len(batch))
            for r in self.db.execute(
                    f"SELECT target_kind,target_id,axis,value FROM facets "
                    f"WHERE target_id IN ({qm})", batch):
                key = (r["target_kind"], r["target_id"])
                out.setdefault(key, {}).setdefault(r["axis"], set()).add(r["value"])
        return out

    def facet_counts(self) -> dict:
        """{axis: {value: count}} for the control panel — what the graph actually holds."""
        out: dict = {}
        for r in self.db.execute(
                "SELECT axis, value, COUNT(*) AS n FROM facets GROUP BY axis, value "
                "ORDER BY axis, n DESC"):
            out.setdefault(r["axis"], {})[r["value"]] = r["n"]
        return out

    def facetize(self, *, limit=None, kinds=("node", "card", "edge")) -> dict:
        """Backfill the facet table by DERIVING facets from data already stored (support
        trust/regime/date, source bundle, status/volatility).  Idempotent — re-run after
        imports/distillation to refresh.  Coarse by design; a classifier can enrich `domain`
        later without changing this.  Returns per-kind counts."""
        from . import facets as F
        srcmap = {r["doc_id"]: dict(r) for r in self.db.execute(
            "SELECT doc_id, bundle, regime, trust_weight, pub_date FROM source_registry")}

        def source_lookup(doc_id):
            return srcmap.get(doc_id)

        # nodes carry no regime column (derived from support); neither table has a
        # volatility column yet — NULL keeps derivation robust (just no 'timeless').
        specs = {
            "node": ("nodes", "id, support, status, NULL AS regime, NULL AS volatility"),
            "card": ("procedure_cards", "id, support, regime, status, NULL AS volatility"),
            "edge": ("edges", "id, support, regime, status, NULL AS volatility"),
        }
        done: dict = {}
        for kind in kinds:
            if kind not in specs:
                continue
            table, cols = specs[kind]
            n = 0
            q = f"SELECT {cols} FROM {table}"
            if limit:
                q += f" LIMIT {int(limit)}"
            for r in self.db.execute(q):
                row = dict(r)
                try:
                    support = json.loads(row.get("support") or "[]")
                except (ValueError, TypeError):
                    support = []
                derived = F.derive(kind, row, support, source_lookup)
                for axis in F.AXES:                       # write every axis (clears stale)
                    self.set_facets(kind, row["id"], axis, derived.get(axis, []))
                n += 1
            done[kind] = n
        return done

    # ── distillation checkpoint ──────────────────────────────────────────────
    def is_distilled(self, chunk_id) -> bool:
        return self.db.execute("SELECT 1 FROM distilled_chunks WHERE chunk_id=?",
                              (chunk_id,)).fetchone() is not None

    def mark_distilled(self, chunk_id):
        self.db.execute(
            "INSERT OR IGNORE INTO distilled_chunks(chunk_id,distilled_at) VALUES(?,?)",
            (chunk_id, time.time()))
        self._maybe_commit()

    # ── viewer/eval support ──────────────────────────────────────────────────
    def sample_nodes(self, n: int = 20):
        """A spread of distilled nodes (the 'learnings') with provenance, for the viewer."""
        rows = self.db.execute(
            "SELECT id,label,kind,summary,aliases,support FROM nodes "
            "WHERE status='active' AND summary IS NOT NULL AND summary!='' "
            "ORDER BY RANDOM() LIMIT ?", (int(n),)).fetchall()
        out = []
        for r in rows:
            sup = json.loads(r["support"] or "[]")
            out.append({"id": r["id"], "label": r["label"], "kind": r["kind"],
                        "summary": r["summary"],
                        "aliases": json.loads(r["aliases"] or "[]"),
                        "sources": [s.get("doc_id") for s in sup],
                        "support": sup})
        return out

    # ── read path (§10): retrieve structured items for a query ───────────────
    def _cos(self, qvec, blob):
        v = _unpack(blob)
        if not v:
            return -1.0
        if self._np is not None:
            return float(self._np.dot(self._np.asarray(qvec, dtype="float32"),
                                      self._np.asarray(v, dtype="float32")))
        return sum(a * b for a, b in zip(qvec, v))     # both L2-normalized => cosine

    def _get_ann(self):
        """Lazily memory-map the ANN index if one was built; else None (brute force).
        Loaded once per KB; a failure or absence is cached so we don't retry per query."""
        if not self._ann_enabled or self._ann_tried:
            return self._ann
        self._ann_tried = True
        with self._lock:                          # one loader wins under concurrency
            if self._ann is not None:
                return self._ann
            from . import ann as ann_mod
            if not ann_mod.available() or not ann_mod.index_exists(self._ann_path):
                if not ann_mod.available():
                    log.info("usearch not installed — exact brute-force search.")
                else:
                    log.info("no ANN index at %s — exact brute-force search "
                             "(run `build-ann` for sub-100ms at scale).", self._ann_path)
                return None
            try:
                self._ann = ann_mod.AnnIndex.load(self._ann_path,
                                                  expansion_search=self._ann_ef,
                                                  view=self._ann_mmap)
                log.info("ANN index loaded (%s): %d node vectors",
                         "mmap" if self._ann_mmap else "resident", len(self._ann))
            except Exception as e:
                log.warning("could not load ANN index (%s) — brute force", e)
                self._ann = None
            return self._ann

    def _topk_idx(self, sims, kk):
        """Indices of the `kk` highest scores, sorted descending — via argpartition so the
        cost is O(N) in C with only kk elements touched in Python (NOT a per-row scan)."""
        np = self._np
        n = int(sims.shape[0])
        if kk >= n:
            return np.argsort(-sims)
        part = np.argpartition(-sims, kk)[:kk]      # kk best, unordered (C)
        return part[np.argsort(-sims[part])]        # order just those kk

    # ── card-level BM25 (retrieval contract §3.1) ────────────────────────────
    def _card_fts_text(self, r) -> str:
        """The searchable text for a card: title + goal + steps + criteria feature
        values + the concept label — so an exact-term query finds the card."""
        keys = r.keys()
        parts = [r["title"] or "", r["goal"] or ""]
        for col in ("steps", "red_flags", "escalation"):
            try:
                parts += [str(x) for x in json.loads(r[col] or "[]")]
            except (ValueError, TypeError):
                pass
        raw = (r["criteria"] if "criteria" in keys else None)
        if raw:
            try:
                crit = json.loads(raw)
                for mod in ("required", "supportive", "exclusion"):
                    parts += [d.get("value", "") for d in crit.get(mod, []) if isinstance(d, dict)]
                parts.append(crit.get("threshold") or "")
                parts += [d.get("condition", "") for d in crit.get("differentials", [])
                          if isinstance(d, dict)]
            except (ValueError, TypeError):
                pass
        parts.append(self._label_of(r["node_id"]) or "")
        return " ".join(p for p in parts if p)

    def reindex_cards_fts(self) -> int:
        """(Re)build the card BM25 index from procedure_cards.  Fast at card scale; run
        after distillation, or it self-builds lazily on first search."""
        with self._lock:
            self._raw.execute("DELETE FROM cards_fts")
            rows = self._raw.execute(
                "SELECT * FROM procedure_cards WHERE status='active'").fetchall()
            self._raw.executemany(
                "INSERT INTO cards_fts(card_id, text) VALUES(?,?)",
                [(r["id"], self._card_fts_text(r)) for r in rows])
            self._raw.commit()
        return len(rows)

    def _ensure_cards_fts(self):
        if self._raw.execute("SELECT count(*) FROM cards_fts").fetchone()[0]:
            return
        if self._raw.execute(
                "SELECT count(*) FROM procedure_cards WHERE status='active'").fetchone()[0]:
            self.reindex_cards_fts()

    def search_cards_bm25(self, query: str, k: int = 20) -> list:
        """Lexical (BM25) card recall — the arm dense misses on exact terms.  Returns
        hydrated card items (lowest bm25 = best), tagged so they merge into the ask pool."""
        self._ensure_cards_fts()
        toks = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 1]
        if not toks:
            return []
        match = " OR ".join(toks)                       # OR for recall; bm25 ranks
        try:
            rows = self.db.execute(
                "SELECT card_id, bm25(cards_fts) AS b FROM cards_fts "
                "WHERE cards_fts MATCH ? ORDER BY b LIMIT ?", (match, int(k))).fetchall()
        except sqlite3.OperationalError:                # malformed MATCH — skip the arm
            return []
        out = []
        for rank, r in enumerate(rows):
            it = self._fetch_item("card", r["card_id"])
            if it:
                it["_role"] = "management"
                it["_channel"] = "bm25"
                it["score"] = round(1.0 / (1.0 + rank), 4)   # provisional; reranker equalises
                out.append(it)
        return out

    def search(self, qvec, k=8, *, empirical_only=False, min_sim=None):
        """Dense retrieval over the card/graph tier: node embeddings + the doc2query
        surface, fused to the best score per target item.

        Top-k is taken with ``np.argpartition`` over each cached matrix, so only ~k items
        ever cross into Python — the matmul AND the selection stay in C.  (The old path
        looped over every node in Python to build a dict, which dominated latency at >1M
        nodes; this keeps a query in the tens-of-ms range.)  Items below ``min_sim`` are
        dropped so an off-topic query abstains (§11).

        Memory note: the two cached matrices live in RAM; past a few million vectors this
        wants an ANN index (HNSW) to stay sub-100ms and shed the resident matrix."""
        import time as _t
        ms = self.min_sim if min_sim is None else min_sim
        # oversample beyond k so post-filters (min_sim, empirical_only, node/surface fusion)
        # still leave k results without rescanning.
        kk = max(int(k) * 8, 64)
        cand: dict = {}
        t0 = _t.perf_counter()
        if self._np is not None:
            q = self._np.asarray(qvec, dtype="float32")
            ann = self._get_ann()
            if ann is not None:                     # HNSW node arm: ~log(N), mmap'd
                for nid, s in ann.query(q, kk):
                    if s < ms:
                        break                       # usearch returns highest-sim first
                    cand[("node", nid)] = s
            else:                                   # exact brute-force node arm
                nids, nmat = self._node_matrix()
                if nmat is not None:
                    sims = nmat @ q
                    for i in self._topk_idx(sims, kk):
                        s = float(sims[i])
                        if s < ms:
                            break                   # argpartition slice is sorted desc
                        cand[("node", nids[int(i)])] = s
            skeys, smat = self._surf_matrix()
            if smat is not None:
                sims = smat @ q
                for i in self._topk_idx(sims, kk):
                    s = float(sims[i])
                    if s < ms:
                        break
                    key = skeys[int(i)]
                    if s > cand.get(key, -2):
                        cand[key] = s
        else:                                       # no numpy → original per-row scan
            for r in self.db.execute("SELECT id,embedding FROM nodes "
                                     "WHERE status='active' AND embedding IS NOT NULL"):
                cand[("node", r["id"])] = self._cos(qvec, r["embedding"])
            for r in self.db.execute("SELECT target_kind,target_id,embedding FROM "
                                     "surface_questions WHERE embedding IS NOT NULL"):
                s = self._cos(qvec, r["embedding"])
                key = (r["target_kind"], r["target_id"])
                if s > cand.get(key, -2):
                    cand[key] = s
        t_vec = _t.perf_counter()
        out, n_hyd = [], 0
        for (kind, tid), score in sorted(cand.items(), key=lambda kv: kv[1], reverse=True):
            if score < ms:
                break                          # ranked desc — nothing below the floor left
            n_hyd += 1
            item = self._fetch_item(kind, tid)
            if not item:
                continue
            if empirical_only and (item.get("regime") or "empirical") != "empirical":
                continue
            item["score"] = round(score, 4)
            out.append(item)
            if len(out) >= k:
                break
        if perf.isEnabledFor(logging.INFO):
            t_end = _t.perf_counter()
            perf.info("kb.search vec=%.1fms hydrate=%.1fms cands=%d hydrated=%d hits=%d",
                      (t_vec - t0) * 1e3, (t_end - t_vec) * 1e3, len(cand), n_hyd, len(out))
        return out

    def edges_from(self, node_id, *, families=None, direction="out",
                   empirical_only=False, limit=20):
        """Typed-edge walk from a node (§10): outgoing for forward why/what-if,
        incoming for diagnostic why.  Returns fetched edge items."""
        col = "src_id" if direction == "out" else "dst_id"
        q = (f"SELECT id FROM edges WHERE {col}=? AND status='active' AND family!='meta'")
        params = [node_id]
        if families:
            q += " AND family IN (%s)" % ",".join("?" * len(families))
            params += list(families)
        if empirical_only:
            q += " AND regime='empirical'"
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return [self._fetch_item("edge", r["id"]) for r in self.db.execute(q, params)]

    def neighbours(self, node_id, limit=8):
        """1-hop structural links for a node (the Phase-1 linkage payoff): what it
        specialises (is_a), needs (requires), is part of, or alternates with — each with
        the relation, direction, the other concept's label, and whether that concept has a
        card Vinkona can pull next.  Navigation aid for the read path; meta edges excluded."""
        out, seen = [], set()
        for r in self.db.execute(
                "SELECT src_id,dst_id,type,family FROM edges WHERE status='active' "
                "AND family!='meta' AND (src_id=? OR dst_id=?) "
                "ORDER BY updated_at DESC LIMIT ?", (node_id, node_id, limit * 3)):
            other = r["dst_id"] if r["src_id"] == node_id else r["src_id"]
            if other == node_id or other in seen:
                continue
            seen.add(other)
            has_card = self.db.execute(
                "SELECT 1 FROM procedure_cards WHERE node_id=? AND status='active' LIMIT 1",
                (other,)).fetchone() is not None
            out.append({"relation": r["type"],
                        "direction": "out" if r["src_id"] == node_id else "in",
                        "label": self._label_of(other), "node_id": other,
                        "has_card": has_card})
            if len(out) >= limit:
                break
        return out

    def cards_for(self, node_id, limit=10):
        return [self._fetch_item("card", r["id"]) for r in self.db.execute(
            "SELECT id FROM procedure_cards WHERE node_id=? AND status='active' LIMIT ?",
            (node_id, limit))]

    def alternatives(self, node_id, limit=8):
        """Sideways walk for a comparison ('X or Y?'): the meta `alternative_to` /
        `context_variant_of` siblings of a node (edges_from skips the meta family).
        Returns node items tagged for the answer frame."""
        out, seen = [], set()
        for r in self.db.execute(
                "SELECT src_id, dst_id FROM edges WHERE family='meta' AND status='active' "
                "AND type IN ('alternative_to','context_variant_of') "
                "AND (src_id=? OR dst_id=?) LIMIT ?", (node_id, node_id, limit * 2)):
            other = r["dst_id"] if r["src_id"] == node_id else r["src_id"]
            if other == node_id or other in seen:
                continue
            seen.add(other)
            it = self._fetch_item("node", other)
            if it:
                it["_role"] = "alternative"
                out.append(it)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _regime_of_support(support):
        regimes = [s.get("regime") for s in support if s.get("regime")]
        return max(set(regimes), key=regimes.count) if regimes else None

    def _contradictions(self, edge_id):
        out = []
        for m in self.db.execute(
                "SELECT * FROM edges WHERE family='meta' AND type='disagrees_with' "
                "AND (src_id=? OR dst_id=?)", (edge_id, edge_id)):
            other = m["dst_id"] if m["src_id"] == edge_id else m["src_id"]
            r = self.db.execute("SELECT support FROM edges WHERE id=?", (other,)).fetchone()
            out.append({"claim": self._edge_brief(other),
                        "support": [s.get("doc_id") for s in json.loads((r["support"] if r else "") or "[]")]})
        return out

    def _fetch_item(self, kind, tid):
        if kind == "node":
            r = self.db.execute("SELECT id,label,kind,summary,support FROM nodes "
                               "WHERE id=? AND status='active'", (tid,)).fetchone()
            if not r:
                return None
            sup = json.loads(r["support"] or "[]")
            return {"kind": "node", "id": r["id"], "label": r["label"],
                    "node_kind": r["kind"], "text": r["summary"], "support": sup,
                    "regime": self._regime_of_support(sup), "contradictions": []}
        if kind == "edge":
            r = self.db.execute("SELECT * FROM edges WHERE id=? AND status='active'",
                              (tid,)).fetchone()
            if not r:
                return None
            sup = json.loads(r["support"] or "[]")
            mods = json.loads(r["modifiers"] or "{}")
            ekeys = r.keys()
            finding = r["finding"] if "finding" in ekeys else None
            out = {"kind": "edge", "id": r["id"],
                   "text": self._edge_brief(r["id"]), "type": r["type"],
                   "family": r["family"], "mechanism": r["mechanism"],
                   "mechanism_basis": r["mechanism_basis"] or "stated",
                   "polarity": r["polarity"] or "",
                   "conditions": mods.get("conditions") or "",
                   "discriminators": mods.get("discriminators") or [],
                   "support": sup, "regime": r["regime"],
                   "contradictions": self._contradictions(r["id"])}
            if finding:
                out["finding"] = json.loads(finding)
            return out
        if kind == "card":
            r = self.db.execute("SELECT * FROM procedure_cards WHERE id=? AND status='active'",
                              (tid,)).fetchone()
            if not r:
                return None
            sup = json.loads(r["support"] or "[]")
            keys = r.keys()
            def _j(col):                       # tolerate a pre-migration row missing the col
                return json.loads((r[col] if col in keys else None) or "[]")
            def _obj(col):                     # optional JSON object column (criteria/grade)
                raw = r[col] if col in keys else None
                return json.loads(raw) if raw else None
            out = {"kind": "card", "id": r["id"], "label": r["title"],
                   "node_id": r["node_id"],
                   "card_type": (r["card_type"] if "card_type" in keys else None) or "procedure",
                   "text": r["goal"], "goal": r["goal"],
                   "steps": json.loads(r["steps"] or "[]"),
                   "safety": json.loads(r["safety"] or "[]"),
                   "mistakes": json.loads(r["mistakes"] or "[]"),
                   "red_flags": _j("red_flags"),
                   "discriminators": _j("discriminators"),
                   "escalation": _j("escalation"),
                   "support": sup, "regime": r["regime"], "contradictions": []}
            crit = _obj("criteria")
            if crit:
                out["criteria"] = crit
            grade = _obj("grade")
            if grade:
                out["grade"] = grade
            return out
        return None

    def contra_pressure(self, item):
        """Sum the (independent, copy-discounted) trust mass of the claims that
        disagree with this one — the contradiction term in read-time strength."""
        total = 0.0
        for c in item.get("contradictions") or []:
            clusters: dict = {}
            for doc in c.get("support") or []:
                src = self.get_source(doc) or {}
                clusters[doc] = max(clusters.get(doc, 0.0), float(src.get("trust_weight") or 0.0))
            raw = sum(clusters.values())
            total += 1.0 - math.exp(-raw / 2.0)
        return total

    def _edge_brief(self, edge_id):
        r = self.db.execute("SELECT src_id,dst_id,type,polarity FROM edges WHERE id=?",
                            (edge_id,)).fetchone()
        if not r:
            return edge_id
        pol = f" ({r['polarity']})" if r["polarity"] else ""
        return f"{self._label_of(r['src_id'])} —{r['type']}→ {self._label_of(r['dst_id'])}{pol}"

    def list_edges(self, limit=50, family=None):
        """Claim edges with resolved labels, provenance, and any meta links
        (disagrees_with / alternative_to / context_variant_of) — for debugging
        reconciliation."""
        q = "SELECT * FROM edges WHERE status='active' AND family!='meta'"
        params = []
        if family:
            q += " AND family=?"
            params.append(family)
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        out = []
        for r in self.db.execute(q, params):
            meta = []
            for m in self.db.execute(
                    "SELECT * FROM edges WHERE family='meta' AND (src_id=? OR dst_id=?)",
                    (r["id"], r["id"])):
                other = m["dst_id"] if m["src_id"] == r["id"] else m["src_id"]
                meta.append({"type": m["type"], "other": self._edge_brief(other)})
            out.append({
                "src": self._label_of(r["src_id"]), "dst": self._label_of(r["dst_id"]),
                "family": r["family"], "type": r["type"], "mechanism": r["mechanism"],
                "polarity": r["polarity"], "regime": r["regime"],
                "conditions": json.loads(r["modifiers"] or "{}").get("conditions") or "",
                "support": [s.get("doc_id") for s in json.loads(r["support"] or "[]")],
                "meta": meta})
        return out

    def list_cards(self, limit=50):
        out = []
        for r in self.db.execute(
                "SELECT * FROM procedure_cards WHERE status='active' "
                "ORDER BY updated_at DESC LIMIT ?", (int(limit),)):
            out.append({"title": r["title"], "node": self._label_of(r["node_id"]),
                        "goal": r["goal"], "domain": r["domain"], "regime": r["regime"],
                        "steps": json.loads(r["steps"] or "[]"),
                        "support": [s.get("doc_id") for s in json.loads(r["support"] or "[]")]})
        return out

    def list_sources(self, limit=200):
        rows = [dict(r) for r in self.db.execute(
            "SELECT doc_id,title,source_type,trust_weight,regime,status,"
            "COALESCE(NULLIF(bundle,''),'base') AS bundle,"
            "license,license_holder,license_url "
            "FROM source_registry ORDER BY rowid DESC LIMIT ?", (int(limit),))]
        return rows

    def list_merge_candidates(self, limit=100):
        out = []
        for r in self.db.execute(
                "SELECT * FROM node_merge_candidates ORDER BY id DESC LIMIT ?", (int(limit),)):
            out.append({"node_a": self._label_of(r["node_a"]),
                        "node_b": self._label_of(r["node_b"]),
                        "similarity": round(r["similarity"] or 0, 4),
                        "reason": r["reason"], "status": r["status"]})
        return out

    def list_gaps(self, limit=100):
        return [dict(r) for r in self.db.execute(
            "SELECT query_text,intent,effect_label,count,status FROM knowledge_gaps "
            "ORDER BY count DESC LIMIT ?", (int(limit),))]

    # ── adjudication: resolve the node-merge queue (§9.4) ─────────────────────
    def _node_brief_full(self, node_id):
        r = self.db.execute("SELECT id,label,kind,summary,aliases,support FROM nodes "
                            "WHERE id=? AND status='active'", (node_id,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "label": r["label"], "kind": r["kind"],
                "summary": r["summary"] or "",
                "aliases": json.loads(r["aliases"] or "[]"),
                "support_n": len(json.loads(r["support"] or "[]"))}

    def open_merge_candidates(self, limit=200):
        """Open queue entries with BOTH nodes resolved; entries whose nodes are gone
        (already merged) are auto-closed as stale so the queue self-cleans."""
        out = []
        for r in self.db.execute(
                "SELECT id,node_a,node_b,similarity,reason FROM node_merge_candidates "
                "WHERE status='open' ORDER BY id LIMIT ?", (int(limit),)).fetchall():
            a, b = self._node_brief_full(r["node_a"]), self._node_brief_full(r["node_b"])
            if not a or not b or a["id"] == b["id"]:
                self.resolve_candidate(r["id"], "stale")
                continue
            out.append({"id": r["id"], "similarity": r["similarity"] or 0.0,
                        "reason": r["reason"], "a": a, "b": b})
        return out

    def resolve_candidate(self, cid, status):
        self.db.execute("UPDATE node_merge_candidates SET status=? WHERE id=?", (status, cid))
        self._maybe_commit()

    def merge_nodes(self, survivor, loser) -> bool:
        """Fold `loser` into `survivor` (§9.4 resolution): re-point every edge/card/
        surface row, union the support + aliases, then retire the loser.  Edges that
        collapse onto an identical claim are de-duplicated (support merged)."""
        if survivor == loser:
            return False
        sv = self.db.execute("SELECT * FROM nodes WHERE id=?", (survivor,)).fetchone()
        lo = self.db.execute("SELECT * FROM nodes WHERE id=?", (loser,)).fetchone()
        if not sv or not lo:
            return False
        self._merge_aliases(survivor, lo["label"], json.loads(lo["aliases"] or "[]"))
        sup = sv["support"] or "[]"
        for e in json.loads(lo["support"] or "[]"):
            sup = self._merge_support(sup, e)
        self.db.execute("UPDATE nodes SET support=? WHERE id=?", (sup, survivor))
        # re-point graph references, then re-hash & de-dupe the survivor's edges.
        self.db.execute("UPDATE edges SET src_id=? WHERE src_id=?", (survivor, loser))
        self.db.execute("UPDATE edges SET dst_id=? WHERE dst_id=?", (survivor, loser))
        self.db.execute("UPDATE procedure_cards SET node_id=? WHERE node_id=?", (survivor, loser))
        self.db.execute("UPDATE surface_questions SET target_id=? "
                        "WHERE target_kind='node' AND target_id=?", (survivor, loser))
        self.db.execute("UPDATE surface_propositions SET target_id=? "
                        "WHERE target_kind='node' AND target_id=?", (survivor, loser))
        self._rehash_dedupe(survivor)
        self.db.execute("UPDATE nodes SET status='merged' WHERE id=?", (loser,))
        self.db.execute("UPDATE node_merge_candidates SET status='stale' WHERE status='open' "
                        "AND (node_a=? OR node_b=?)", (loser, loser))
        self._maybe_commit()
        self._nodes_loaded = False            # node set changed → rebuild the search cache
        self._node_ids, self._node_vecs, self._node_mat = [], [], None
        self._surf_loaded = False             # surface target_ids were repointed
        self._surf_keys, self._surf_vecs, self._surf_mat = [], [], None
        return True

    def _rehash_dedupe(self, node_id):
        """After a merge re-pointed edges onto `node_id`, recompute each touched edge's
        hash (the stored one reflects the old endpoint), drop self-loops, and collapse
        any now-identical claims into one (unioning their support).  Caller commits."""
        rows = self.db.execute("SELECT * FROM edges WHERE (src_id=? OR dst_id=?) "
                               "AND status='active' AND family!='meta'",
                               (node_id, node_id)).fetchall()
        seen = {}
        for r in rows:
            if r["src_id"] == r["dst_id"]:                 # a node related to itself
                self.db.execute("DELETE FROM edges WHERE id=?", (r["id"],))
                continue
            mods = json.loads(r["modifiers"] or "{}")
            scope = json.loads(r["scope"] or "{}")
            h = self._edge_hash(r["src_id"], r["dst_id"], r["family"], r["type"],
                                r["polarity"] or "", r["mechanism"] or "", r["regime"],
                                scope.get("world"), mods.get("conditions"))
            if h in seen:
                keep = seen[h]
                merged = keep["support"]
                for e in json.loads(r["support"] or "[]"):
                    merged = self._merge_support(merged, e)
                self.db.execute("UPDATE edges SET support=? WHERE id=?", (merged, keep["id"]))
                self.db.execute("DELETE FROM edges WHERE id=?", (r["id"],))
                keep["support"] = merged
            else:
                self.db.execute("UPDATE edges SET edge_hash=? WHERE id=?", (h, r["id"]))
                seen[h] = {"id": r["id"], "support": r["support"] or "[]"}

    def counts(self, *, fresh=False) -> dict:
        """Table tallies for the viewer/stats.  Each is a full scan of a now-huge table,
        and the live top-bar polls this every few seconds while holding the DB lock — so a
        short TTL cache keeps that polling from stalling concurrent `ask` queries.  Pass
        fresh=True to force a recompute."""
        now = time.monotonic()
        if not fresh and self._counts_cache and (now - self._counts_cache[1]) < self._counts_ttl:
            return self._counts_cache[0]
        c = self.db.execute
        res = {
            "sources": c("SELECT COUNT(*) FROM source_registry").fetchone()[0],
            "nodes": c("SELECT COUNT(*) FROM nodes WHERE status='active'").fetchone()[0],
            "cards": c("SELECT COUNT(*) FROM procedure_cards WHERE status='active'").fetchone()[0],
            "edges": c("SELECT COUNT(*) FROM edges WHERE status='active'").fetchone()[0],
            "merge_candidates": c("SELECT COUNT(*) FROM node_merge_candidates WHERE status='open'").fetchone()[0],
            "gaps": c("SELECT COUNT(*) FROM knowledge_gaps WHERE status='open'").fetchone()[0],
            "distilled_chunks": c("SELECT COUNT(*) FROM distilled_chunks").fetchone()[0],
        }
        self._counts_cache = (res, now)
        return res

    def migrate_node_layout(self) -> bool:
        """Rebuild the `nodes` table so `embedding` is the LAST column.  An existing DB was
        created with embedding mid-row, so every hydration lookup reads past the blob's
        overflow pages (~100ms/row uncached) — this puts it last so the hot columns sit in
        the leaf page and lookups are fast.  One-time, idempotent; copies the table (a few
        minutes at 1M rows).  Code reads/writes columns by name, so nothing else changes."""
        cols = [r["name"] for r in self._raw.execute("PRAGMA table_info(nodes)").fetchall()]
        if not cols or cols[-1] == "embedding":
            return False                          # already last (or empty) — nothing to do
        log.info("migrating nodes layout (embedding → last column); this copies the table…")
        with self._lock:
            self._raw.executescript("""
            CREATE TABLE nodes_new(
                id TEXT PRIMARY KEY, label TEXT, kind TEXT, summary TEXT,
                aliases TEXT, support TEXT, status TEXT DEFAULT 'active', embedding BLOB);
            INSERT INTO nodes_new(id,label,kind,summary,aliases,support,status,embedding)
                SELECT id,label,kind,summary,aliases,support,status,embedding FROM nodes;
            DROP TABLE nodes;
            ALTER TABLE nodes_new RENAME TO nodes;
            """)
            self._raw.commit()
        self._nodes_loaded = False                # caches reference the old table
        self._node_ids, self._node_vecs, self._node_mat = [], [], None
        return True

    def reload(self):
        """Drop the in-memory caches and re-warm, so the live server picks up writes a
        maintenance subprocess made (distill/refine/link/embed/adjudicate) — and recovers a
        wedged cache — WITHOUT a full restart.  The 'Reload KB' button on the panel."""
        with self._lock:
            self._nodes_loaded = False
            self._node_ids, self._node_vecs, self._node_mat = [], [], None
            self._surf_loaded = False
            self._surf_keys, self._surf_vecs, self._surf_mat = [], [], None
            self._ann = None
            self._ann_tried = False
            self._counts_cache = None
        self._get_ann()
        self.warm()
        try:
            self.reindex_cards_fts()          # refresh card BM25 after a write job
        except Exception as e:                # pragma: no cover - defensive
            log.debug("cards_fts reindex skipped (%s)", e)
        return self.counts(fresh=True)

    def warm(self):
        """Pull the node + card tables into the OS page cache with one sequential scan, so
        the first per-query point lookups don't fault from disk (the embeddings are stored
        inline, so reading length(embedding) touches every node page).  Cheap with RAM to
        spare; with mmap on it then stays resident."""
        try:
            with self._lock:
                self._raw.execute(
                    "SELECT count(*), sum(length(embedding)) FROM nodes").fetchone()
                self._raw.execute("SELECT count(*) FROM procedure_cards").fetchone()
        except Exception as e:
            log.debug("table warm skipped (%s)", e)

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
