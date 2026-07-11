"""The store & index — hybrid dense (ANN/brute-force) + sparse (BM25/FTS).

Two interchangeable backends behind one interface (``make_store(cfg)``):

- ``SqliteStore`` (default): SQLite **FTS5** for sparse + a dense matrix scored
  with numpy (or a pure-python fallback).  Ships in the stdlib, no extra
  service.  Brute-force dense is fine to ~1M chunks — perfect for the PDF
  collection and the Phase-1 proof; honestly *not* for a full Wikipedia.
- ``LanceStore``: LanceDB **IVF-PQ** dense + FTS, on-disk and mmap'd.  This is
  the one that scales to a 10-40M-chunk Wikipedia snapshot.  ``pip install
  lancedb``.

Either way the principle holds (KNOWLEDGE.md): ANN/FTS, never re-load-everything
on the hot path the way the small `memories` store does, and **hybrid** dense+
sparse fused at query time — that fusion is the biggest quality lever at scale.

A small **manifest** (path, content_hash, mtime, version, status), always kept
in SQLite, makes every ingest run incremental for both backends.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.store")

# columns every chunk record carries (KNOWLEDGE.md table)
FIELDS = ("id", "source_type", "title", "section", "path_or_url",
          "text", "tokens", "version", "ingested_at")

# Per-document side metadata (keyed by path_or_url), carried alongside the chunk
# store so a document's frame can reach distillation without bloating every chunk
# row.  Used by the research→learning loop to carry the `# Question`, `kb_query`,
# and provenance/bundle/trust from ingest to the distiller (research_loop_spec §6).
# These keys are merged into the chunk dicts iter_chunks yields.
_DOC_META_KEYS = ("question", "kb_query", "provenance", "bundle", "trust", "kind")


def _doc_meta_init(db):
    db.execute("CREATE TABLE IF NOT EXISTS doc_meta("
               "path_or_url TEXT PRIMARY KEY, meta TEXT)")
    db.commit()


def _doc_meta_set(db, path_or_url, meta: dict):
    db.execute("INSERT INTO doc_meta(path_or_url,meta) VALUES(?,?) "
               "ON CONFLICT(path_or_url) DO UPDATE SET meta=excluded.meta",
               (path_or_url, json.dumps(meta or {})))
    db.commit()


def _doc_meta_get(db, path_or_url):
    r = db.execute("SELECT meta FROM doc_meta WHERE path_or_url=?",
                   (path_or_url,)).fetchone()
    if not r:
        return None
    try:
        return json.loads(r["meta"] if isinstance(r, sqlite3.Row) else r[0])
    except (ValueError, TypeError):
        return None


def _doc_meta_all(db) -> dict:
    out = {}
    try:
        for r in db.execute("SELECT path_or_url, meta FROM doc_meta"):
            try:
                out[r[0]] = json.loads(r[1])
            except (ValueError, TypeError):
                pass
    except sqlite3.OperationalError:                    # table not created yet
        pass
    return out


def _merge_doc_meta(chunk: dict, meta_map: dict) -> dict:
    m = meta_map.get(chunk.get("path_or_url"))
    if m:
        for k in _DOC_META_KEYS:
            if k in m and m[k] is not None:
                chunk[k] = m[k]
    return chunk

_FTS_SPECIAL = re.compile(r'["*:^()]')


def _fts_query(text: str, stop: frozenset | None = None) -> str:
    """Turn free text into a safe FTS5 MATCH expression: OR of quoted terms.

    Quoting each token defuses FTS operator characters in the query; OR makes it
    a recall-oriented sparse arm (fusion + rerank handle precision).  `stop` drops
    learned over-reporting terms (see build_stoplist) to keep the OR union small —
    but if the query is ENTIRELY stopwords we keep them, so it never goes empty."""
    toks = [t for t in re.split(r"\W+", _FTS_SPECIAL.sub(" ", text)) if t]
    if not toks:
        return ""
    if stop:
        kept = [t for t in toks if t.lower() not in stop]
        if kept:                       # all-stopword query ⇒ fall back to the originals
            toks = kept
    return " OR ".join(f'"{t}"' for t in toks[:64])


def _pack(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


# ── manifest (shared by both backends) ──────────────────────────────────────
class _Manifest:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        db.execute("""CREATE TABLE IF NOT EXISTS manifest(
            path TEXT PRIMARY KEY, content_hash TEXT, mtime REAL,
            version INTEGER, status TEXT)""")
        db.execute("""CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY, value TEXT)""")
        db.commit()

    def get(self, path: str):
        r = self.db.execute(
            "SELECT content_hash, mtime, version, status FROM manifest WHERE path=?",
            (path,)).fetchone()
        return dict(r) if r else None

    def set(self, path, content_hash, mtime, version, status):
        self.db.execute(
            "INSERT INTO manifest(path,content_hash,mtime,version,status) "
            "VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
            "content_hash=excluded.content_hash, mtime=excluded.mtime, "
            "version=excluded.version, status=excluded.status",
            (path, content_hash, mtime, version, status))
        self.db.commit()

    def meta_get(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def meta_set(self, key, value):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, str(value)))
        self.db.commit()


# ── SQLite backend ──────────────────────────────────────────────────────────
class SqliteStore:
    backend = "sqlite"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cfg["db_path"], check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=10000")   # wait out a concurrent writer
        mmap = int(cfg.get("sqlite_mmap_mb", 0) or 0)
        if mmap > 0:                                    # map the DB into RAM: reads at memory speed
            self.db.execute(f"PRAGMA mmap_size={mmap * (1 << 20)}")
        cache = int(cfg.get("sqlite_cache_mb", 0) or 0)
        if cache > 0:                                   # negative cache_size ⇒ KiB (so MiB×1024)
            self.db.execute(f"PRAGMA cache_size={-cache * 1024}")
        self.db.execute("PRAGMA temp_store=MEMORY")     # FTS merges / ORDER BY in RAM, not /tmp
        self._init_schema()
        self.manifest = _Manifest(self.db)
        self._mat = None          # cached dense matrix (numpy) — invalidated on write
        self._mat_ids: list = []
        self._np = None
        self._stop = None         # cached adaptive stoplist (loaded from meta on first search)
        try:
            import numpy as np
            self._np = np
        except Exception:
            log.info("numpy not present — dense search uses the pure-python fallback")

    def _fts_opts(self) -> str:
        """Trailing FTS5 options — ``detail`` + ``tokenize`` — for the DDL.  Both are sanitised
        to their own fixed vocabularies so a config value can never inject SQL.  detail='column'
        drops within-column positions (we issue no phrase/NEAR queries) for a smaller/faster
        index that still supports column-weighted bm25; empty tokenizer ⇒ default unicode61."""
        detail = re.sub(r"[^a-z]", "", (self.cfg.get("fts_detail") or "").lower())
        if detail not in ("full", "column", "none"):
            detail = "full"                                # unknown value ⇒ the safe superset
        tok = re.sub(r"[^a-z0-9_ ]", "", (self.cfg.get("fts_tokenizer") or "").lower()).strip()
        opts = f", detail='{detail}'"
        if tok:
            opts += f", tokenize='{tok}'"
        return opts

    def _init_schema(self):
        tok = self._fts_opts()
        self.db.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY, source_type TEXT, title TEXT, section TEXT,
            path_or_url TEXT, text TEXT, tokens INTEGER, version INTEGER,
            ingested_at REAL);
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path_or_url);
        CREATE TABLE IF NOT EXISTS vectors(
            id TEXT PRIMARY KEY, dim INTEGER, vec BLOB);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, title, section, content='chunks', content_rowid='rowid'{tok});
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid,text,title,section)
            VALUES(new.rowid,new.text,new.title,new.section); END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts,rowid,text,title,section)
            VALUES('delete',old.rowid,old.text,old.title,old.section);
            DELETE FROM vectors WHERE id=old.id; END;
        """)
        _doc_meta_init(self.db)
        self.db.commit()

    def set_doc_meta(self, path_or_url: str, meta: dict):
        _doc_meta_set(self.db, path_or_url, meta)

    def get_doc_meta(self, path_or_url: str):
        return _doc_meta_get(self.db, path_or_url)

    # ---- writes ----------------------------------------------------------
    def delete_by_path(self, path_or_url: str):
        self.db.execute("DELETE FROM chunks WHERE path_or_url=?", (path_or_url,))
        self.db.commit()
        self._mat = None

    def add_chunks(self, records: list[dict]):
        if not records:
            return
        now = time.time()
        crows = [(r["id"], r.get("source_type"), r.get("title"), r.get("section"),
                  r.get("path_or_url"), r.get("text"), r.get("tokens"),
                  r.get("version", 1), r.get("ingested_at") or now) for r in records]
        self.db.executemany(                               # one bound statement for the batch
            "INSERT OR REPLACE INTO chunks"
            "(id,source_type,title,section,path_or_url,text,tokens,version,ingested_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)", crows)
        vrows = [(r["id"], len(r["vector"]), _pack(r["vector"]))
                 for r in records if r.get("vector") is not None]
        if vrows:
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors(id,dim,vec) VALUES(?,?,?)", vrows)
        self.db.commit()
        self._mat = None

    def optimize_fts(self):
        """Merge the FTS b-tree segments after a bulk load — smaller index, faster MATCH."""
        try:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            self.db.commit()
        except sqlite3.Error as e:                         # pragma: no cover - defensive
            log.debug("fts optimize skipped: %s", e)

    def rebuild_fts(self) -> int:
        """Drop & recreate the FTS with the CONFIGURED tokenizer, then reindex from the stored
        chunk text — NO source re-parse.  This is the migration to run after changing
        fts_tokenizer on an existing DB.  Returns the chunk count reindexed."""
        tok = self._fts_opts()
        self.db.executescript(f"""
            DROP TRIGGER IF EXISTS chunks_ai;
            DROP TRIGGER IF EXISTS chunks_ad;
            DROP TABLE IF EXISTS chunks_fts;
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, title, section, content='chunks', content_rowid='rowid'{tok});
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid,text,title,section)
                VALUES(new.rowid,new.text,new.title,new.section); END;
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts,rowid,text,title,section)
                VALUES('delete',old.rowid,old.text,old.title,old.section);
                DELETE FROM vectors WHERE id=old.id; END;
        """)
        self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.db.commit()
        return self.count()

    # ---- self-adaptive stoplist ------------------------------------------
    def _stopset(self) -> frozenset:
        """The learned high-frequency terms to drop from an OR match, cached from meta."""
        if self._stop is None:
            raw = self.manifest.meta_get("stoplist", "")
            try:
                self._stop = frozenset(json.loads(raw)) if raw else frozenset()
            except (ValueError, TypeError):
                self._stop = frozenset()
        return self._stop

    def build_stoplist(self) -> list:
        """LEARN the corpus's over-reporting terms from fts5vocab and persist them: any term
        present in more than ``stopword_df_ratio`` of all chunks (capped at ``stopword_max``,
        highest document-frequency first), unioned with ``stopwords_extra``.  Recomputed after
        each ingest so it tracks the actual corpus.  BM25's IDF already makes these terms
        score ~0, so dropping them from the match is a latency win, not a recall loss."""
        ratio = float(self.cfg.get("stopword_df_ratio", 0) or 0)
        extra = sorted({str(w).lower().strip() for w in (self.cfg.get("stopwords_extra") or [])
                        if str(w).strip()})
        total = self.count()
        learned: list = []
        if ratio > 0 and total >= int(self.cfg.get("stopword_min_chunks", 5000)):
            try:
                self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab "
                                "USING fts5vocab('chunks_fts','row')")
                cap = int(self.cfg.get("stopword_max", 300))
                rows = self.db.execute(
                    "SELECT term FROM chunks_vocab WHERE doc > ? ORDER BY doc DESC LIMIT ?",
                    (int(total * ratio), cap)).fetchall()
                learned = [r[0] for r in rows]
            except sqlite3.Error as e:                     # fts5vocab missing / odd build
                log.warning("stoplist: could not read fts5vocab (%s) — using extras only", e)
        stop = sorted(set(learned) | set(extra))
        self.manifest.meta_set("stoplist", json.dumps(stop))
        self._stop = frozenset(stop)
        if stop:
            log.info("adaptive stoplist: %d term(s) over %.0f%% doc-freq (e.g. %s)",
                     len(stop), ratio * 100, ", ".join(learned[:8]) or ", ".join(extra[:8]))
        return stop

    # ---- reads -----------------------------------------------------------
    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def has_vectors(self) -> bool:
        return self.db.execute("SELECT 1 FROM vectors LIMIT 1").fetchone() is not None

    def stats_by_source(self) -> dict:
        rows = self.db.execute(
            "SELECT source_type, COUNT(*) c FROM chunks GROUP BY source_type")
        return {(r["source_type"] or "?"): r["c"] for r in rows}

    def sample(self, n: int, source_type: str | None = None) -> list:
        """A random spread of stored chunks — for eyeballing ingestion quality."""
        sql, params = "SELECT * FROM chunks", []
        if source_type:
            sql += " WHERE source_type=?"
            params.append(source_type)
        sql += " ORDER BY RANDOM() LIMIT ?"
        params.append(int(n))
        return [self._row_dict(r) for r in self.db.execute(sql, params)]

    def iter_chunks(self, batch: int = 512):
        """Yield every stored chunk (offline distillation source), paged by rowid.
        Per-document metadata (the research-loop frame: question / kb_query / provenance)
        is merged in from doc_meta, loaded once — it is a tiny table (vinkona docs only)."""
        meta_map = _doc_meta_all(self.db)
        last = 0
        while True:
            rows = self.db.execute(
                "SELECT rowid AS _r, * FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last, batch)).fetchall()
            if not rows:
                return
            for r in rows:
                last = r["_r"]
                yield _merge_doc_meta(self._row_dict(r), meta_map)

    def _row_dict(self, row) -> dict:
        return {k: row[k] for k in FIELDS}

    def chunks_for_path(self, path_or_url: str) -> list:
        """Every chunk of one document, in document order (rowid), each with its section,
        text and (optional) embedding — the source-reconstruction feed for card refinement."""
        rows = self.db.execute(
            "SELECT c.section AS section, c.text AS text, c.tokens AS tokens, v.vec AS vec "
            "FROM chunks c LEFT JOIN vectors v ON v.id = c.id "
            "WHERE c.path_or_url = ? ORDER BY c.rowid", (path_or_url,)).fetchall()
        out = []
        for r in rows:
            b = r["vec"]
            out.append({"section": r["section"] or "", "text": r["text"] or "",
                        "tokens": r["tokens"] or 0,
                        "vec": list(struct.unpack(f"<{len(b)//4}f", b)) if b else None})
        return out

    def search_text(self, query: str, k: int, filters: dict | None = None):
        match = _fts_query(query, self._stopset())
        if not match:
            return []
        w = self.cfg.get("bm25_col_weights")               # (text, title, section) column boosts
        bm = ("bm25(chunks_fts, %s)" % ", ".join("%g" % float(x) for x in w)
              if isinstance(w, (list, tuple)) and len(w) == 3 else "bm25(chunks_fts)")
        sql = (f"SELECT c.*, {bm} AS rank FROM chunks_fts "
               "JOIN chunks c ON c.rowid=chunks_fts.rowid WHERE chunks_fts MATCH ?")
        params: list = [match]
        sql += self._filter_sql(filters, params, prefix="c.")
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        out = []
        for row in self.db.execute(sql, params):
            d = self._row_dict(row)
            d["score"] = -float(row["rank"])      # bm25: lower is better -> negate
            out.append(d)
        return out

    def search_vector(self, qvec, k: int, filters: dict | None = None):
        if qvec is None or not self.has_vectors():
            return []
        ids, sims = self._dense_topk(qvec, k * 4 if filters else k)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = f"SELECT * FROM chunks WHERE id IN ({placeholders})"
        params = list(ids)
        sql += self._filter_sql(filters, params)
        rows = {r["id"]: r for r in self.db.execute(sql, params)}
        out = []
        for cid, sim in zip(ids, sims):
            row = rows.get(cid)
            if row is None:
                continue
            d = self._row_dict(row)
            d["score"] = float(sim)
            out.append(d)
        return out[:k]

    def _filter_sql(self, filters, params, prefix=""):
        if not filters:
            return ""
        clause = ""
        if filters.get("source_type"):
            clause += f" AND {prefix}source_type=?"
            params.append(filters["source_type"])
        if filters.get("title"):
            clause += f" AND {prefix}title LIKE ?"
            params.append(f"%{filters['title']}%")
        return clause

    # ---- dense scoring ---------------------------------------------------
    def _load_matrix(self):
        if self._mat is not None:
            return
        ids, vecs = [], []
        for row in self.db.execute("SELECT id,dim,vec FROM vectors"):
            ids.append(row["id"])
            vecs.append((row["dim"], row["vec"]))
        self._mat_ids = ids
        if self._np is not None and vecs:
            np = self._np
            dim = vecs[0][0]
            arr = np.frombuffer(b"".join(v for _, v in vecs), dtype="<f4")
            self._mat = arr.reshape(len(vecs), dim)
        else:
            self._mat = [struct.unpack(f"<{d}f", v) for d, v in vecs] if vecs else []

    def _dense_topk(self, qvec, k):
        self._load_matrix()
        if not self._mat_ids:
            return [], []
        if self._np is not None:
            np = self._np
            q = np.asarray(qvec, dtype="<f4")
            sims = self._mat @ q                    # both sides L2-normalized
            k = min(k, len(sims))
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            return [self._mat_ids[i] for i in idx], [sims[i] for i in idx]
        # pure-python fallback (no numpy): brute-force dot products
        scored = []
        for cid, row in zip(self._mat_ids, self._mat):
            scored.append((sum(a * b for a, b in zip(row, qvec)), cid))
        scored.sort(reverse=True)
        top = scored[:k]
        return [c for _, c in top], [s for s, _ in top]

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass


# ── LanceDB backend ─────────────────────────────────────────────────────────
class LanceStore:
    """LanceDB-backed store for Wikipedia-scale corpora.  Dense via IVF-PQ, FTS
    via the built-in tantivy index; the manifest stays in a small SQLite db so
    incrementality is identical to the sqlite backend.

    LanceDB's Python API has shifted across versions; the calls here are the
    stable core (connect / add / search / create_fts_index).  Kept lazy-imported
    so the package — and the query server on the sqlite backend — never needs
    lancedb installed.
    """
    backend = "lance"

    def __init__(self, cfg: dict):
        import lancedb
        import pyarrow as pa
        self.cfg = cfg
        self.pa = pa
        Path(cfg["lance_dir"]).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(cfg["lance_dir"])
        self.table_name = cfg["lance_table"]
        self.tbl = self._open_or_none()
        # manifest in sqlite alongside
        Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
        mdb = sqlite3.connect(cfg["db_path"], check_same_thread=False)
        mdb.row_factory = sqlite3.Row
        mdb.execute("PRAGMA busy_timeout=10000")       # tolerate a concurrent writer
        self.manifest = _Manifest(mdb)
        self._mdb = mdb
        _doc_meta_init(mdb)                            # research-loop per-doc frame carrier

    def set_doc_meta(self, path_or_url: str, meta: dict):
        _doc_meta_set(self._mdb, path_or_url, meta)

    def get_doc_meta(self, path_or_url: str):
        return _doc_meta_get(self._mdb, path_or_url)

    def _open_or_none(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            return None

    def _ensure_table(self, dim: int):
        if self.tbl is not None:
            return
        pa = self.pa
        schema = pa.schema([
            ("id", pa.string()), ("source_type", pa.string()),
            ("title", pa.string()), ("section", pa.string()),
            ("path_or_url", pa.string()), ("text", pa.string()),
            ("tokens", pa.int32()), ("version", pa.int32()),
            ("ingested_at", pa.float64()),
            ("vector", pa.list_(pa.float32(), dim))])
        self.tbl = self.db.create_table(self.table_name, schema=schema)
        try:
            self.tbl.create_fts_index("text", replace=True)
        except Exception as e:
            log.warning("FTS index not created (%s) — sparse arm degraded", e)

    def delete_by_path(self, path_or_url: str):
        if self.tbl is None:
            return
        safe = path_or_url.replace("'", "''")
        self.tbl.delete(f"path_or_url = '{safe}'")

    def add_chunks(self, records: list[dict]):
        rows = [r for r in records if r.get("vector") is not None]
        if not rows:
            log.warning("lance backend requires embeddings; skipped %d unembedded "
                        "chunk(s)", len(records))
            return
        self._ensure_table(len(rows[0]["vector"]))
        payload = [{
            "id": r["id"], "source_type": r.get("source_type") or "",
            "title": r.get("title") or "", "section": r.get("section") or "",
            "path_or_url": r.get("path_or_url") or "", "text": r.get("text") or "",
            "tokens": int(r.get("tokens") or 0), "version": int(r.get("version") or 1),
            "ingested_at": float(r.get("ingested_at") or time.time()),
            "vector": list(r["vector"])} for r in rows]
        self.tbl.add(payload)

    def count(self) -> int:
        return self.tbl.count_rows() if self.tbl is not None else 0

    def has_vectors(self) -> bool:
        return self.count() > 0

    def stats_by_source(self) -> dict:
        if self.tbl is None:
            return {}
        out = {}
        for st in ("wikipedia", "pdf", "epub", "html", "text"):
            try:
                c = self.tbl.count_rows(filter=f"source_type = '{st}'")
            except Exception:
                c = 0
            if c:
                out[st] = c
        return out

    def sample(self, n: int, source_type: str | None = None) -> list:
        """A spread of stored chunks (random window when unfiltered)."""
        if self.tbl is None:
            return []
        n = int(n)
        try:
            ds = self.tbl.to_lance()
        except ImportError:
            # No pylance — fall back to the first n rows (head() needs no pylance).
            try:
                return [{k: r.get(k) for k in FIELDS} for r in self.tbl.head(n).to_pylist()]
            except Exception as e:
                log.warning("lance sample failed (%s)", e)
                return []
        try:
            filt = f"source_type = '{source_type}'" if source_type else None
            offset = 0
            if not filt:
                import random
                total = self.count()
                if total == 0:
                    return []
                n = min(n, total)
                offset = random.randint(0, max(0, total - n))
            rows = ds.scanner(columns=list(FIELDS), limit=n, offset=offset,
                              filter=filt).to_table().to_pylist()
        except Exception as e:
            log.warning("lance sample failed (%s)", e)
            return []
        return [{k: r.get(k) for k in FIELDS} for r in rows]

    def iter_chunks(self, batch: int = 512):
        """Yield every stored chunk (offline distillation source), paged.

        Prefers pylance's scanner (memory-efficient offset paging); if pylance
        isn't installed, falls back to a single to_arrow() scan so distillation
        still runs — install pylance for large corpora."""
        if self.tbl is None:
            return
        meta_map = _doc_meta_all(self._mdb)            # research-loop per-doc frame
        try:
            ds = self.tbl.to_lance()
        except ImportError:
            log.warning("pylance not installed — scanning via full to_arrow(); "
                        "install pylance for memory-efficient paging on big corpora")
            cols = [c for c in FIELDS if c in self.tbl.schema.names]
            for rb in self.tbl.to_arrow().select(cols).to_batches(max_chunksize=batch):
                for r in rb.to_pylist():
                    yield _merge_doc_meta({k: r.get(k) for k in FIELDS}, meta_map)
            return
        total, off = self.count(), 0
        while off < total:
            rows = ds.scanner(columns=list(FIELDS), limit=batch,
                              offset=off).to_table().to_pylist()
            if not rows:
                return
            for r in rows:
                yield _merge_doc_meta({k: r.get(k) for k in FIELDS}, meta_map)
            off += len(rows)

    def maybe_build_ann(self):
        """Post-ingest: build the IVF-PQ vector index and (re)build the FTS index
        over the full corpus.  The FTS index created on the empty table at
        creation does not cover rows appended afterwards, so the sparse arm is
        only complete once it is rebuilt here."""
        if self.tbl is None:
            return
        # IVF-PQ has to train a 256-centroid quantizer, so it needs >=256 rows and is
        # pointless on a tiny table anyway — flat (exact) scan is instant at that size.
        # Skip the build (and its alarming error) until the corpus is worth indexing.
        n = self.count()
        min_rows = int(self.cfg.get("ann_min_rows", 4096))
        if n < min_rows:
            log.info("ANN index skipped (%d rows < %d) — exact flat search; "
                     "auto-builds once the corpus grows", n, min_rows)
        else:
            try:
                self.tbl.create_index(metric="cosine", vector_column_name="vector",
                                      replace=True)
            except Exception as e:
                log.info("ANN index not built (%s) — flat search until enough rows", e)
        try:
            self.tbl.create_fts_index("text", replace=True)
        except Exception as e:
            log.warning("FTS index not (re)built (%s) — sparse arm degraded", e)

    def _rows(self, res, k):
        out = []
        for row in res.to_list()[:k]:
            d = {f: row.get(f) for f in FIELDS}
            # lance distance -> similarity-ish score; for fts it's a relevance score
            if "_relevance_score" in row:
                d["score"] = float(row["_relevance_score"])
            elif "_distance" in row:
                d["score"] = 1.0 - float(row["_distance"])   # cosine distance
            else:
                d["score"] = 0.0
            out.append(d)
        return out

    def _where(self, filters):
        if not filters:
            return None
        clauses = []
        if filters.get("source_type"):
            clauses.append(f"source_type = '{filters['source_type']}'")
        if filters.get("title"):
            clauses.append(f"title = '{filters['title']}'")
        return " AND ".join(clauses) if clauses else None

    def search_vector(self, qvec, k, filters=None):
        if self.tbl is None or qvec is None:
            return []
        q = self.tbl.search(list(qvec)).metric("cosine").limit(k)
        where = self._where(filters)
        if where:
            q = q.where(where, prefilter=True)
        return self._rows(q, k)

    def search_text(self, query, k, filters=None):
        if self.tbl is None or not query.strip():
            return []
        try:
            q = self.tbl.search(query, query_type="fts").limit(k)
            where = self._where(filters)
            if where:
                q = q.where(where, prefilter=True)
            return self._rows(q, k)
        except Exception as e:
            log.warning("fts search failed (%s)", e)
            return []

    def chunks_for_path(self, path_or_url: str) -> list:
        """Lance parity for card-refinement source reconstruction: a document's chunks with
        vectors, ordered by ingestion time (best-effort — Lance has no rowid)."""
        if self.tbl is None:
            return []
        safe = path_or_url.replace("'", "''")
        try:
            ds = self.tbl.to_lance()
            rows = ds.scanner(columns=list(FIELDS) + ["vector"],
                              filter=f"path_or_url = '{safe}'").to_table().to_pylist()
        except Exception as e:
            log.warning("lance chunks_for_path failed (%s)", e)
            return []
        rows.sort(key=lambda r: r.get("ingested_at") or 0)
        return [{"section": r.get("section") or "", "text": r.get("text") or "",
                 "tokens": r.get("tokens") or 0, "vec": r.get("vector")} for r in rows]

    def close(self):
        try:
            self._mdb.close()
        except Exception:
            pass


def make_store(cfg: dict):
    if cfg["backend"] == "lance":
        return LanceStore(cfg)
    return SqliteStore(cfg)
