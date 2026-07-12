"""Offline ingestion pipeline — heavy, batch, run on demand / monthly.

A **manifest** (path, content_hash, mtime, version, status) makes every run
**incremental**: only new/changed files are (re)processed.  Per source we
extract ``(section, text)`` blocks, chunk them by heading (~200-400 tokens),
embed the chunks via the nomic endpoint (``search_document:`` prefix), and
upsert into the store under a stable id (idempotent re-ingest).

Security: filenames are treated as **opaque data** — never interpolated into a
shell or an LM prompt (the file-scraper injection surface).  All extracted text
is sanitized before storage.  Parse runs need no network.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time

from . import research, sanitize
from .chunk import chunk_blocks, chunk_id
from .sources import extractor_for, MissingDependency

log = logging.getLogger("knowledgehost.ingest")


class EmbedUnavailable(Exception):
    """The embed endpoint dropped mid-ingest.  On the lance backend (no sparse-only
    fallback) we abort rather than silently drop chunks and mark the doc done."""


def _content_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _embed_and_store(store, embedder, cfg, *, source_type, title,
                     path_or_url, blocks, version):
    """Chunk -> (optionally) embed in batches -> upsert.  Returns chunk count."""
    title = sanitize.clean(title, 300)
    records, batch_text, batch_rec = [], [], []
    n = 0
    # When embedding, the batch is bounded by the embed endpoint / GPU memory.  When NOT
    # embedding (the lexical library), commit in big transactions instead — far fewer fsyncs.
    flush_at = cfg["embed_batch"] if embedder else int(cfg.get("ingest_write_batch", 1000))

    def flush():
        nonlocal batch_text, batch_rec
        if not batch_rec:
            return
        vecs = embedder.embed_many(batch_text, "document") if embedder else None
        # lance stores nothing without a vector; a whole-batch None means the
        # endpoint hiccupped — abort (resumable) so we don't drop chunks AND mark
        # the doc done.  sqlite keeps them sparse-only, so it's fine to proceed.
        if embedder and vecs is None and cfg.get("backend") == "lance":
            raise EmbedUnavailable("embed endpoint dropped mid-ingest")
        for rec, vec in zip(batch_rec, vecs or [None] * len(batch_rec)):
            rec["vector"] = vec
        store.add_chunks(batch_rec)
        batch_text, batch_rec = [], []

    for ch in chunk_blocks(blocks, cfg):
        text = sanitize.clean(ch["text"])
        if not text:
            continue
        section = sanitize.clean(ch["section"], 300)
        rec = {
            "id": chunk_id(path_or_url, section, text),
            "source_type": source_type, "title": title, "section": section,
            "path_or_url": path_or_url, "text": text,
            "tokens": ch["tokens"], "version": version, "ingested_at": time.time(),
        }
        batch_rec.append(rec)
        batch_text.append(text)
        n += 1
        if len(batch_rec) >= flush_at:
            flush()
    flush()
    return n


def ingest_file(store, embedder, cfg, path: str, *, force=False, collection=None) -> int:
    """Ingest one document if new/changed.  Returns chunks added (0 if skipped).
    `collection` (library ingest) tags the chunks' source_type with a topical bucket
    (science/fiction/…) instead of the format, so search can filter by it."""
    ext = os.path.splitext(path)[1].lower()
    vinkona = (collection is None) and research.is_research_doc(path)   # not for library docs
    fn = None if vinkona else extractor_for(path)
    if fn is None and not vinkona:
        return 0
    try:
        st = os.stat(path)
    except OSError:
        return 0

    prev = store.manifest.get(path)
    version = int(store.manifest.meta_get("version", "1"))
    # A file skipped for a missing parser dependency is RETRIED every run — the
    # dependency may have been installed since (./install.sh --pdf) and the file
    # itself hasn't changed, so the unchanged-skips would otherwise bury it
    # forever.  Retrying is free while the dep is still absent (the extractor
    # raises on import, before the file is even opened), so the content hash is
    # deferred until something actually parses.
    retry_dep = bool(prev) and prev["status"] == "missing_dep" and not force
    chash = None
    if not retry_dep:
        if prev and not force and abs(prev["mtime"] - st.st_mtime) < 1e-6:
            return 0                               # unchanged by mtime — cheap skip
        chash = _content_hash(path)
        if prev and not force and prev["content_hash"] == chash:
            store.manifest.set(path, chash, st.st_mtime, version, "ok")
            return 0                               # mtime moved but bytes identical

    if vinkona:
        if chash is None:
            chash = _content_hash(path)
        return _ingest_research_doc(store, embedder, cfg, path, version, chash, st)

    try:
        title, blocks = fn(path, cfg)
    except MissingDependency as e:
        if retry_dep:
            return 0            # still missing — the manifest row already says so
        log.warning("skip %s — %s", os.path.basename(path), e)
        store.manifest.set(path, chash, st.st_mtime, version, "missing_dep")
        return 0
    except Exception as e:
        log.warning("failed to parse %s: %s", os.path.basename(path), e)
        if chash is None:
            chash = _content_hash(path)
        store.manifest.set(path, chash, st.st_mtime, version, "error")
        return 0

    if retry_dep:
        log.info("previously-skipped %s parses now (dependency installed)",
                 os.path.basename(path))
    if chash is None:
        chash = _content_hash(path)
    store.delete_by_path(path)                     # re-ingest cleanly if changed
    source_type = collection or {".pdf": "pdf", ".epub": "epub", ".html": "html",
                                 ".htm": "html"}.get(ext, "text")
    n = _embed_and_store(store, embedder, cfg, source_type=source_type,
                         title=title, path_or_url=path, blocks=blocks,
                         version=version)
    store.manifest.set(path, chash, st.st_mtime, version, "ok")
    return n


def collection_for(cfg, root: str, path: str, default: str | None = None) -> str:
    """Topical collection for a library doc: an explicit `library_collections` folder map
    (bare key matches a path SEGMENT, glob matches the whole path; first match wins) wins;
    else `default` (set when the crawl root is itself a chosen collection folder — the whole
    subtree is one collection); else the doc's top folder under its library root; else 'library'."""
    p = str(path).replace("\\", "/")
    mapping = cfg.get("library_collections") or {}
    if isinstance(mapping, dict):
        pl = p.lower()
        segs = {s for s in pl.split("/") if s}
        import fnmatch as _fn
        for key, coll in mapping.items():
            k = str(key).lower()
            if k in segs or _fn.fnmatch(pl, k):
                return str(coll)
    if default:
        return default
    try:
        rel = os.path.relpath(path, root)
        top = rel.replace("\\", "/").split("/")[0]
        if top and top not in (".", ".."):
            return top.lower()
    except ValueError:
        pass
    return "library"


def _parse_job(path, cfg, collection, prev_chash):
    """Worker-process job: content-hash + parse ONE file to (title, blocks).  Touches NO
    database — the store write happens back on the single main thread — so it's safe to fan
    out across a process pool.  Returns a small status dict the writer applies."""
    try:
        st = os.stat(path)
        chash = _content_hash(path)
    except OSError:
        return {"path": path, "status": "gone"}
    base = {"path": path, "collection": collection, "chash": chash, "mtime": st.st_mtime}
    if prev_chash and chash == prev_chash:
        return {**base, "status": "unchanged"}             # bytes identical → skip parse+write
    fn = extractor_for(path)
    if fn is None:
        return {**base, "status": "unsupported"}
    try:
        title, blocks = fn(path, cfg)
    except MissingDependency as e:
        return {**base, "status": "missing_dep", "err": str(e)}
    except Exception as e:                                  # pragma: no cover - per-file guard
        return {**base, "status": "error", "err": str(e)}
    return {**base, "status": "ok", "title": title, "blocks": blocks}


def _apply_parsed(store, embedder, cfg, res, version) -> int:
    """Main-thread writer for one parsed file: set its manifest status and upsert its chunks.
    The single point that touches the store, so the SQLite single-writer rule is preserved."""
    path, status = res["path"], res["status"]
    if status == "gone":
        return 0
    if status == "unchanged":                              # mtime moved but bytes identical
        store.manifest.set(path, res["chash"], res["mtime"], version, "ok")
        return 0
    if status == "unsupported":
        return 0
    if status in ("missing_dep", "error"):
        log.warning("%s %s: %s",
                    "skip" if status == "missing_dep" else "failed to parse",
                    os.path.basename(path), res.get("err"))
        store.manifest.set(path, res["chash"], res["mtime"], version, status)
        return 0
    store.delete_by_path(path)                             # clean re-ingest if changed
    n = _embed_and_store(store, embedder, cfg, source_type=res["collection"],
                         title=res["title"], path_or_url=path, blocks=res["blocks"],
                         version=version)
    store.manifest.set(path, res["chash"], res["mtime"], version, "ok")
    return n


def crawl_library(store, embedder, cfg, *, force=False) -> dict:
    """Index the search-only library (library_sources) into its OWN store — lexical FTS
    by default (embedder passed only when library_dense), NOT distilled.  Each doc is
    tagged with its topical `collection`.  Parsing fans out across a process pool
    (``ingest_workers``) while the DB is written from this one thread; the FTS index is
    optimised at the end.  The cheap tier that feeds Vinkona's research loop a local 'google'."""
    exts = set(cfg["extensions"])
    every = cfg["ingest_log_every"]
    version = int(store.manifest.meta_get("version", "1"))
    lib_root = os.path.realpath(cfg["library_root"]) if cfg.get("library_root") else ""

    # 1. enumerate work: cheap mtime-skip here; the byte-identical skip is done in the worker
    #    (which is reading the file to hash it anyway).
    jobs = []                                              # (path, collection, prev_content_hash)
    for root in cfg.get("library_sources") or []:
        if not os.path.isdir(root):
            log.info("library root missing, skipping: %s", root)
            continue
        # When the crawl root is a direct subfolder of the trusted library_root (the web
        # Library panel writes exactly these), the whole subtree is ONE collection named for
        # that subfolder — so nested structure doesn't splinter the corpus firewall.
        base_coll = None
        if lib_root and os.path.realpath(os.path.dirname(os.path.normpath(root))) == lib_root:
            base_coll = os.path.basename(os.path.normpath(root)).lower()
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                prev = store.manifest.get(path)
                # missing_dep rows re-parse every run — the dependency may have
                # been installed since (see ingest_file for the reasoning).
                retry_dep = bool(prev) and prev["status"] == "missing_dep"
                if prev and not force and not retry_dep \
                        and abs(prev["mtime"] - st.st_mtime) < 1e-6:
                    continue                               # unchanged by mtime — cheap skip
                coll = collection_for(cfg, root, path, default=base_coll)
                prev_chash = None if (force or retry_dep) \
                    else (prev["content_hash"] if prev else None)
                jobs.append((path, coll, prev_chash))

    docs = chunks = 0
    by_collection: dict = {}

    def _handle(res):
        nonlocal docs, chunks
        added = _apply_parsed(store, embedder, cfg, res, version)
        if added:
            docs += 1
            chunks += added
            c = res.get("collection") or "library"
            by_collection[c] = by_collection.get(c, 0) + 1
        if docs and every and docs % every == 0:
            log.info("library … %d docs / %d chunks", docs, chunks)

    # 2. parse in parallel (I/O + CPU + any OCR), write serially (SQLite is single-writer).
    workers = int(cfg.get("ingest_workers", 0) or 0) or (os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # 'fork' (not 3.14's default forkserver): the ingest CLI is single-threaded here, so
        # fork is safe AND avoids re-importing the whole package per worker.  Workers never
        # touch the DB, so the parent's inherited sqlite connection is harmless.
        try:
            ctx = _mp.get_context("fork")
        except ValueError:                                # non-posix — take the platform default
            ctx = _mp.get_context()
        log.info("library: parsing %d file(s) across %d worker(s)", len(jobs), workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_parse_job, p, cfg, c, h) for p, c, h in jobs]
            for fut in as_completed(futs):
                _handle(fut.result())
    else:
        for p, c, h in jobs:
            _handle(_parse_job(p, cfg, c, h))

    if hasattr(store, "optimize_fts"):
        store.optimize_fts()                              # merge FTS segments after the bulk load
    if hasattr(store, "build_stoplist"):
        store.build_stoplist()                            # LEARN over-reporting terms from the corpus
    return {"docs": docs, "chunks": chunks, "collections": by_collection}


def _ingest_research_doc(store, embedder, cfg, path, version, chash, st) -> int:
    """Ingest one of Vinkona's solved research drops (research_loop_spec §6): the
    ``## Sources`` blocks become the distillable chunks (source_type='vinkona'), and the
    doc's ``# Question`` / ``kb_query`` / provenance are stashed in doc_meta so the
    distiller frames extraction on the question and can close the gap that opened it."""
    try:
        question, blocks, meta = research.parse_research_doc(path)
    except Exception as e:
        log.warning("failed to parse research doc %s: %s", os.path.basename(path), e)
        store.manifest.set(path, chash, st.st_mtime, version, "error")
        return 0
    store.delete_by_path(path)
    title = question or os.path.splitext(os.path.basename(path))[0]
    n = _embed_and_store(store, embedder, cfg, source_type="vinkona",
                         title=title, path_or_url=path, blocks=blocks, version=version)
    # The per-doc frame the distiller reads back (merged into each chunk by iter_chunks).
    store.set_doc_meta(path, {
        "provenance": "vinkona", "bundle": "vinkona",
        "trust": float(cfg.get("vinkona_trust", 0.25)),
        "question": question or None,
        "kb_query": meta.get("kb_query"),
        "kind": meta.get("kind"),
    })
    store.manifest.set(path, chash, st.st_mtime, version, "ok")
    log.info("research drop ingested: %s (%d chunk(s))", os.path.basename(path), n)
    return n


def crawl(store, embedder, cfg, *, force=False) -> dict:
    """Walk every configured source root and ingest supported files.  Vinkona's research
    outbox (research_solved_dir) is crawled too — its .md drops route to the vinkona path."""
    exts = set(cfg["extensions"])
    docs = chunks = 0
    every = cfg["ingest_log_every"]
    roots = list(cfg["sources"])
    solved = cfg.get("research_solved_dir")
    if solved and solved not in roots:
        roots.append(solved)                       # low-trust vinkona bundle (research §6)
    for root in roots:
        if not os.path.isdir(root):
            log.info("source root missing, skipping: %s", root)
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                path = os.path.join(dirpath, name)
                added = ingest_file(store, embedder, cfg, path, force=force)
                if added:
                    docs += 1
                    chunks += added
                if docs and every and docs % every == 0:
                    log.info("… %d docs / %d chunks", docs, chunks)
    return {"docs": docs, "chunks": chunks}


def ingest_wikipedia(store, embedder, cfg, *, limit: int | None = None,
                     force: bool = False) -> dict:
    """Ingest a Kiwix Wikipedia ZIM (pre-rendered HTML articles).

    **Resumable**: each article is checkpointed in the manifest by its
    ``zim://<url>`` key, so a stop-and-restart skips everything already done and
    picks up where it left off — essential for a multi-hour full-Wikipedia run.

    **Duplicate-proof**: the lance backend appends (no upsert), so an article is
    marked ``pending`` before embedding and ``ok`` after; on resume the one
    article interrupted mid-write is the only ``pending`` one, and its partial
    rows are cleared with ``delete_by_path`` before redo.  First-run articles are
    unseen, so no (millions of) no-op deletes are issued on the happy path.
    """
    zim = cfg.get("zim_path")
    if not zim or not os.path.isfile(zim):
        log.info("no zim_path configured/found, skipping Wikipedia")
        return {"articles": 0, "chunks": 0, "skipped": 0}
    from .sources import wikipedia
    version = int(store.manifest.meta_get("version", "1"))
    arts = chunks = skipped = 0
    every = cfg["ingest_log_every"]
    for url, title, blocks in wikipedia.iter_articles(zim):
        key = f"zim://{url}"
        prev = store.manifest.get(key)
        if prev and not force and prev["status"] == "ok" and prev["version"] == version:
            skipped += 1
            continue                                   # already ingested — resume past it
        if prev:                                       # a prior attempt left rows — clear them
            store.delete_by_path(key)
        store.manifest.set(key, "zim", 0.0, version, "pending")
        n = _embed_and_store(store, embedder, cfg, source_type="wikipedia",
                             title=title, path_or_url=key, blocks=blocks, version=version)
        store.manifest.set(key, "zim", 0.0, version, "ok")
        arts += 1
        chunks += n
        if every and arts % every == 0:
            log.info("… %d new articles / %d chunks (%d already done)",
                     arts, chunks, skipped)
        if limit and arts >= limit:
            break
    store.manifest.set(f"zim://{os.path.basename(zim)}",
                       _content_hash(zim) if os.path.getsize(zim) < (1 << 30) else "big",
                       os.path.getmtime(zim), version, "ok")
    return {"articles": arts, "chunks": chunks, "skipped": skipped}
