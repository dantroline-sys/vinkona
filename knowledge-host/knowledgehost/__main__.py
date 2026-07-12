"""Entry point.

  python3 -m knowledgehost [-c config.toml]            # serve the query tool
  python3 -m knowledgehost ingest [--force] [--wikipedia] [--limit N]
  python3 -m knowledgehost stats                       # index stats, then exit
  python3 -m knowledgehost bump-version                # version++ (monthly swap)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

from . import distill as distill_mod
from . import ingest as ingest_mod
from . import server
from .config import load_config
from .distill import BackendUnavailable
from .embed import Embedder
from .kb import KB
from .store import make_store
from .tools import Tools


def _build(cfg):
    store = make_store(cfg)
    embedder = Embedder(cfg)
    return store, embedder


def _remove_path(path: str) -> list:
    """Delete a db file (with its -wal/-shm sidecars) or a directory.  Returns the
    paths actually removed."""
    removed = []
    if os.path.isdir(path):
        shutil.rmtree(path)
        removed.append(path)
    else:
        for p in (path, path + "-wal", path + "-shm"):
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
    return removed


def _cmd_reset(cfg, args, log) -> int:
    """Clear the import so you can start from scratch.  Scopes: default = raw +
    KB; --kb = only the distilled KB; --raw = only the raw chunk store."""
    do_raw = bool(args.raw) or not args.kb
    do_kb = bool(args.kb) or not args.raw

    targets = []
    if do_raw:
        targets.append(("raw store + manifest", cfg["db_path"]))
        targets.append(("raw lance tables", cfg["lance_dir"]))
    if do_kb:
        targets.append(("distilled KB (nodes/edges/cards)", cfg["kb_path"]))
    existing = [(lbl, p) for lbl, p in targets if os.path.exists(p)]
    if not existing:
        log.info("nothing to clear — already empty.")
        return 0

    # Best-effort summary of what's about to be lost.
    try:
        if do_raw and os.path.exists(cfg["db_path"]):
            store = make_store(cfg)
            print(f"  raw chunks: {store.count():,}")
            store.close()
        if do_kb and os.path.exists(cfg["kb_path"]):
            kb = KB(cfg)
            print(f"  kb: {kb.counts()}")
            kb.close()
    except Exception as e:                       # never let reporting block a reset
        log.warning("could not read current contents (%s)", e)

    print("This will PERMANENTLY delete:")
    for lbl, p in existing:
        print(f"  - {lbl}: {p}")
    if not args.yes:
        try:
            ans = input('Type "yes" to confirm: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != "yes":
            print("aborted — nothing deleted.")
            return 1

    removed = []
    for _lbl, p in existing:
        removed += _remove_path(p)
    log.info("cleared %d path(s): %s", len(removed), ", ".join(removed))
    return 0


def _run_distill(cfg, embedder, log, *, limit=None, watch=False, interval=30, bundle=None) -> int:
    """Distil raw chunks into the structured KB (owns its own raw-store handle so
    `--watch` can reopen for a fresh snapshot).  Returns a process exit code.

    In watch mode it loops, distilling new chunks as a concurrent ingest adds them
    (the distilled set is the checkpoint, so each pass only does fresh chunks),
    reopens the store each idle cycle so the lance backend sees new commits, and
    re-probes the big-LM endpoints so a 'sometimes available' duplicate is picked
    up (in parallel) when it appears and dropped when it goes."""
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — distillation needs vectors.")
        return 1
    kb = KB(cfg)
    store = make_store(cfg)
    rc = 0
    prev_up = None
    verify_on = cfg.get("verify", True)
    try:
        while True:
            # Two tiers, re-probed each pass: fast extractors + big verifiers.  Whichever
            # is up shapes the run; a 'sometimes available' endpoint joins/leaves live.
            fast = distill_mod.fast_endpoints(cfg) if verify_on else []
            big = distill_mod.verify_endpoints(cfg)
            if fast and big:
                extractors, verifiers, mode = fast, big, "two-tier (fast→verify)"
            elif big:
                extractors, verifiers, mode = big, None, "single-tier (big LM only)"
            elif fast:                                # fast up but no verifier — unverified
                extractors, verifiers, mode = fast, None, "UNVERIFIED (fast LM, no verifier)"
                log.warning("verifier endpoints down — running fast LM WITHOUT verification")
            else:
                extractors, verifiers, mode = [], None, "none"
            up = (tuple(e.url for e in extractors), tuple(v.url for v in (verifiers or [])))
            if up != prev_up:
                log.info("distill: %s — extract=%s verify=%s", mode,
                         ", ".join(up[0]) or "(none)", ", ".join(up[1]) or "(none)")
                prev_up = up
            if not extractors:
                if watch and not limit:
                    log.warning("no distill endpoint up — waiting %ds…", interval)
                    time.sleep(interval)
                    continue
                log.error("no distill endpoint up (extract=%s verify=%s) — start one first.",
                          ", ".join(cfg.get("extract_urls") or []),
                          ", ".join(cfg.get("verify_urls") or cfg.get("distill_urls") or []))
                rc = 1
                break
            stats = distill_mod.distill_corpus(store, kb, extractors, embedder, cfg,
                                               limit=limit, verifiers=verifiers, bundle=bundle)
            log.info("distilled: %s", stats)
            log.info("kb: %s", kb.counts())
            if not watch or limit:
                break
            log.info("watch: waiting %ds for ingest to add more chunks (Ctrl-C to stop)…",
                     interval)
            time.sleep(interval)
            store.close()
            store = make_store(cfg)            # fresh snapshot (lance sees new versions)
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        rc = 1
    except KeyboardInterrupt:
        log.info("watch stopped")
    finally:
        kb.close()
        try:
            store.close()
        except Exception:
            pass
    return rc


def _run_link(cfg, log, *, limit=None, top_k=None, fast=False) -> int:
    """Phase 1 graph linkage: type structural edges (is_a/requires/part_of/alternative/
    related) between card-bearing concepts and their embedding-neighbours, via an LM.
    KB-only + one LM endpoint; resumable (judged pairs are checkpointed) and lease-aware
    (yields the matching GPU while Vinkona is using it).  --fast routes to the 9B (4090) —
    relation-typing is well within its range and it clears the sweep far quicker than the
    big LM (3090).  Build the ANN index first for speed at scale."""
    from . import distill as distill_mod
    from . import link as link_mod
    from . import lm_lease
    endpoints = distill_mod.fast_endpoints if fast else distill_mod.verify_endpoints
    lease = lm_lease.FAST if fast else lm_lease.BIG
    tier = "fast 9B (4090)" if fast else "big LM (3090)"
    urls_hint = (cfg.get("extract_urls") if fast
                 else (cfg.get("verify_urls") or cfg.get("distill_urls"))) or []
    live = endpoints(cfg, log)
    if not live:
        log.error("no %s endpoint up — start one (%s) first.", tier, ", ".join(urls_hint))
        return 1
    log.info("link: using the %s", tier)
    kb = KB(cfg)
    try:
        stats = link_mod.link_concepts(kb, live[0], cfg, limit=limit, top_k=top_k, lease=lease)
        log.info("link: %s", stats)
        log.info("kb: %s", kb.counts())
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        return 1
    except KeyboardInterrupt:
        log.info("link stopped (resumable)")
    finally:
        kb.close()
    return 0


def _run_refine(cfg, store, embedder, log, *, limit=None, force=False) -> int:
    """Phase 2 card refinement: re-read each card's source document and rewrite the card in
    place into the ideal 'what do I do now' form, grounded in that source.  Big-LM work (needs
    the 64k context); demand-weighted by hit_count, resumable (refined cards are skipped unless
    --force), lease-aware.  Needs the raw chunk store (source) + the big LM endpoint."""
    from . import distill as distill_mod
    from . import refine as refine_mod
    from . import lm_lease
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — refinement re-embeds each card.")
        store.close()
        return 1
    big = distill_mod.verify_endpoints(cfg, log)
    if not big:
        log.error("no big-LM endpoint up — start one (verify_urls=%s) first.",
                  ", ".join(cfg.get("verify_urls") or cfg.get("distill_urls") or []))
        store.close()
        return 1
    lm = big[0]
    # A 46k-token source + a reasoning model is slow; the default distill timeout (~120s) gives
    # up mid-prompt on big documents.  Use a generous refine-specific budget so the client
    # doesn't hang up while llama.cpp is still processing.
    lm.timeout = int(cfg.get("refine_timeout_s", 600))
    log.info("refine: per-call timeout %ds, source budget %d tokens",
             lm.timeout, int(cfg.get("refine_source_tokens", 46000)))
    kb = KB(cfg)
    try:
        stats = refine_mod.refine_cards(kb, store, embedder, lm, cfg,
                                        limit=limit, force=force, lease=lm_lease.BIG)
        log.info("refine: %s", stats)
        log.info("kb: %s", kb.counts())
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        return 1
    except KeyboardInterrupt:
        log.info("refine stopped (resumable)")
    finally:
        kb.close()
        store.close()
    return 0


def _run_adjudicate(cfg, log, *, limit=None, batch=8, watch=False, interval=30,
                    auto=True, auto_only=False, fast=False) -> int:
    """Drain the node-merge queue.  A deterministic pre-pass (auto_resolve) first clears
    the lexically-obvious pairs without any LM — merging plural/exact duplicates, adding
    is_a for token-subsets, and deferring the weak tail — so the LM only judges the thin
    ambiguous band that remains.  --fast routes that residual to the fast 9B (4090) instead
    of the big LM (3090): same/distinct on synonyms is well within its range and it clears
    the queue far quicker.  Resumable and lease-aware (yielding the matching GPU)."""
    from . import adjudicate as adj
    from . import distill as distill_mod
    from . import lm_lease
    endpoints = distill_mod.fast_endpoints if fast else distill_mod.verify_endpoints
    lease = lm_lease.FAST if fast else lm_lease.BIG
    tier = "fast 9B (4090)" if fast else "big LM (3090)"
    urls_hint = (cfg.get("extract_urls") if fast
                 else (cfg.get("verify_urls") or cfg.get("distill_urls"))) or []
    kb = KB(cfg)
    rc = 0
    try:
        if auto:
            astats = adj.auto_resolve(kb, cfg, limit=limit)
            log.info("auto-resolve: %s", astats)
            if auto_only or astats.get("open_remaining", 0) == 0:
                if auto_only:
                    log.info("--auto-only: skipping the LM pass (%d still open for it).",
                             astats.get("open_remaining", 0))
                return 0
        while True:
            lms = endpoints(cfg)
            if not lms:
                if watch and not limit:
                    log.warning("no %s endpoint up for adjudication — waiting %ds…",
                                tier, interval)
                    time.sleep(interval)
                    continue
                log.error("no %s endpoint up (%s) — start it first.",
                          tier, ", ".join(urls_hint))
                rc = 1
                break
            log.info("adjudicating residual with the %s", tier)
            stats = adj.adjudicate_queue(kb, lms[0], cfg, limit=limit, batch=batch, lease=lease)
            log.info("adjudicated: %s", stats)
            if not watch or limit or stats.get("open_remaining", 0) == 0:
                break
            log.info("watch: %ds until the next pass (Ctrl-C to stop)…", interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("adjudication stopped (resumable)")
    finally:
        kb.close()
    return rc


def _run_import_conceptnet(cfg, log, *, path=None, min_weight=None,
                           include_lexical=False, limit=None) -> int:
    """Stream the ConceptNet dump into the KB (regime=conventional, low trust,
    has_reference=0).  Resumable/idempotent — re-run to add only what is new."""
    from . import conceptnet as cn
    src = path or cfg.get("conceptnet_path") or ""
    if not src:
        log.error("no ConceptNet path — pass --path PATH or set conceptnet_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("ConceptNet dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = cn.import_conceptnet(
            kb, src,
            min_weight=cfg["conceptnet_min_weight"] if min_weight is None else min_weight,
            trust=cfg["conceptnet_trust"],
            include_lexical=include_lexical or cfg["conceptnet_include_lexical"],
            limit=limit)
        log.info("conceptnet import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_import_atomic(cfg, log, *, path=None, limit=None) -> int:
    """Stream the ATOMIC if-then graph into the KB (regime=conventional, low trust,
    has_reference=0).  Idempotent — re-run to add only what is new."""
    from . import atomic as at
    src = path or cfg.get("atomic_path") or ""
    if not src:
        log.error("no ATOMIC path — pass --path PATH or set atomic_path in config "
                  "(use the aggregated v4_atomic_all_agg.csv).")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("ATOMIC dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = at.import_atomic(kb, src, trust=cfg["atomic_trust"],
                                 min_count=cfg["atomic_min_count"], limit=limit)
        log.info("atomic import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_import_glucose(cfg, log, *, path=None, limit=None) -> int:
    """Stream GLUCOSE general causal rules into the KB (regime=conventional, low trust,
    has_reference=0).  Idempotent — re-run to add only what is new."""
    from . import glucose as gl
    src = path or cfg.get("glucose_path") or ""
    if not src:
        log.error("no GLUCOSE path — pass --path PATH or set glucose_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("GLUCOSE dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = gl.import_glucose(kb, src, trust=cfg["glucose_trust"],
                                  min_count=cfg["glucose_min_count"], limit=limit)
        log.info("glucose import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_import_causenet(cfg, log, *, path=None, limit=None) -> int:
    """Stream CauseNet-precision into the KB (causal edges, grounded, has_reference=1).
    Idempotent — re-run to add only what is new."""
    from . import causenet as cnet
    src = path or cfg.get("causenet_path") or ""
    if not src:
        log.error("no CauseNet path — pass --path PATH or set causenet_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("CauseNet dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = cnet.import_causenet(kb, src, trust=cfg["causenet_trust"],
                                     regime=cfg["causenet_regime"], limit=limit)
        log.info("causenet import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_optimize(cfg, log, *, vacuum=False) -> int:
    """One-time layout fix: move the `embedding` column last so `ask` hydration doesn't
    read past the inline blob on every row lookup.  --vacuum also reclaims disk after."""
    kb = KB(cfg)
    try:
        import time as _t
        t0 = _t.perf_counter()
        changed = kb.migrate_node_layout()
        if changed:
            log.info("nodes layout migrated in %.1fs", _t.perf_counter() - t0)
            if vacuum:
                log.info("VACUUM (reclaiming disk; needs free space ≈ db size)…")
                kb._raw.execute("VACUUM")
                log.info("VACUUM done")
        else:
            log.info("nodes layout already optimal (embedding is last) — nothing to do.")
    finally:
        kb.close()
    return 0


def _run_bundles(cfg, args, log) -> int:
    """Modular-knowledge CLI (§16), all against the MASTER kb.db:
      bundles                       list provenance groups + source counts
      source <doc_id> [--title T] [--bundle B]   rename / regroup a source
      scenario [name]               show scenarios (or resolve one → sources)
      split [dir] [--force]         export each bundle group to <bundle>.kdb
    """
    import json as _json
    from . import bundles as B
    cmd = args.command
    if cmd == "split":
        out = (args.args[0] if args.args else None) or args.out
        res = B.split(cfg, out, force=args.force, log_fn=log.info)
        print(_json.dumps({k: v.get("counts", v) for k, v in res.items()}, indent=2))
        return 0

    kb = KB(cfg)
    try:
        if cmd == "bundles" and args.args and args.args[0] == "licenses":
            from . import licensing
            for s in kb.list_sources(500):
                fl = licensing.flags_for(s.get("license"))
                perms = "".join(k[0].upper() if fl.get(k) is True else
                                ("·" if fl.get(k) is None else "-")
                                for k in ("redistribute", "derivatives", "commercial"))
                print(f"{perms}  {(s.get('license') or 'unknown'):18s} "
                      f"{(s.get('license_holder') or ''):28s} {s.get('doc_id')}")
            print("\n(RDC = Redistribute/Derivatives/Commercial — CAP=yes ·=unknown -=no)")
            return 0
        if cmd == "bundles":
            summ = kb.bundle_summary()
            for b in summ:
                print(f"{b['bundle']:20s} {b['sources']:4d} source(s)")
            if not summ:
                print("(no sources registered yet)")
            return 0
        if cmd == "source":
            if not args.args:
                # no doc_id → list sources with their bundle tag + licence
                for s in kb.list_sources(500):
                    lic = s.get("license") or "unknown"
                    who = f" © {s['license_holder']}" if s.get("license_holder") else ""
                    print(f"{s.get('doc_id')}\t[{s.get('bundle') or 'base'}]\t{lic}{who}\t{s.get('title')}")
                return 0
            doc_id = args.args[0]
            edits = (args.title, args.bundle, args.license, args.license_holder, args.license_url)
            if all(e is None for e in edits):
                s = kb.get_source(doc_id)
                if s:
                    s["shippable_flags"] = kb.license_of(doc_id)["flags"]
                print(_json.dumps(s, indent=2) if s else f"no such source: {doc_id}")
                return 0 if s else 1
            row = kb.set_source(doc_id, title=args.title, bundle=args.bundle,
                                license=args.license, license_holder=args.license_holder,
                                license_url=args.license_url)
            if not row:
                log.error("no such source: %s", doc_id)
                return 1
            log.info("source %s → title=%r bundle=%r license=%r holder=%r", doc_id,
                     row.get("title"), row.get("bundle") or "base",
                     row.get("license"), row.get("license_holder"))
            return 0
        if cmd == "scenario":
            scenarios = cfg.get("scenarios") or {}
            active = B.active_scenario_name(cfg)
            if not args.args:
                print(f"active: {active}")
                print(f"available: {', '.join(scenarios) or '(none — serving all)'}")
                return 0
            name = args.args[0]
            scen = B.scenario_def(cfg, name)
            srcs = B.list_sources(kb._raw)
            picked = (B.select_sources(srcs, scen) if (scen or name != "all")
                      else {s['doc_id'] for s in srcs})
            print(f"scenario '{name}' → {len(picked)} source(s):")
            for s in srcs:
                if s["doc_id"] in picked:
                    print(f"  {s['doc_id']}  [{s['bundle']}]  {s.get('title')}")
            return 0
    finally:
        kb.close()
    return 0


def _library_store_cfg(cfg) -> dict:
    """The library rides its OWN sqlite index (separate file from the graph corpus)."""
    return {**cfg, "backend": "sqlite", "db_path": cfg["library_db"]}


def _library_store(cfg, log=None):
    """Open the search-only library store if it's been built; else None (tool stays hidden)."""
    db = cfg.get("library_db")
    if not (db and os.path.exists(db)):
        return None
    try:
        from .store import make_store
        return make_store(_library_store_cfg(cfg))
    except Exception as e:                     # pragma: no cover - defensive
        if log:
            log.warning("library store unavailable (%s)", e)
        return None


def _run_rebuild_fts(cfg, log) -> int:
    """Rebuild the full-text index with the configured tokenizer (fts_tokenizer) — reindexes
    from the stored chunk text, NO source re-parse.  Run once after changing the tokenizer.
    Covers the library store and, when the graph runs on sqlite, its chunk store too."""
    from .store import make_store
    targets = []
    lib_db = cfg.get("library_db")
    if lib_db and os.path.exists(lib_db):
        targets.append(("library", _library_store_cfg(cfg)))
    if cfg.get("backend") == "sqlite":                    # lance has no FTS to rebuild
        targets.append(("graph", cfg))
    done = 0
    for name, scfg in targets:
        store = make_store(scfg)
        try:
            if not hasattr(store, "rebuild_fts"):
                continue
            n = store.rebuild_fts()
            store.optimize_fts()
            if hasattr(store, "build_stoplist"):
                store.build_stoplist()                    # re-learn the stoplist post-reindex
            done += 1
            log.info("rebuilt FTS for %s store: %d chunk(s) reindexed", name, n)
        finally:
            store.close()
    if not done:
        log.info("no sqlite FTS store found to rebuild (nothing to do)")
    return 0


def _run_ingest_library(cfg, log, *, force=False) -> int:
    """Index the library folder tree into its own FTS store (lexical by default; embeds too
    only when library_dense).  NOT distilled — this is the cheap search-only tier."""
    from .store import make_store
    from .embed import Embedder
    if not (cfg.get("library_sources")):
        log.error("no library_sources configured (see config: [library_sources]) — nothing to index")
        return 1
    store = make_store(_library_store_cfg(cfg))
    embedder = Embedder(cfg) if cfg.get("library_dense") else None   # lexical-first: no embed
    try:
        stats = ingest_mod.crawl_library(store, embedder, cfg, force=force)
        log.info("library indexed: %s", stats)
        import json as _json
        print(_json.dumps(stats, indent=2))
    finally:
        store.close()
    return 0


def _run_eval(cfg, log, *, gold=None, retriever="current_path", trace=False) -> int:
    """Retrieval eval harness (contract §8): score a retriever over the graded gold set
    and print the abstention-quality report.  Needs the KB + embedder (the current-path
    retriever embeds each query); no LM / raw store."""
    from . import evalharness
    from .embed import Embedder
    gold_path = gold or cfg.get("eval_gold_path") or "eval/gold.jsonl"
    if not os.path.exists(gold_path):
        log.error("gold set not found: %s (see eval/README.md)", gold_path)
        return 1
    embedder = Embedder(cfg)
    kb = KB(cfg)
    try:
        metrics = evalharness.run(kb, embedder, cfg, gold_path=gold_path,
                                  retriever=retriever, trace=trace)
    finally:
        kb.close()
    return 0 if metrics.get("accept", {}).get("passed") else 2


def _run_build_ann(cfg, log) -> int:
    """Build the HNSW ANN index over node embeddings (read-path speed).  Re-run after big
    imports/`embed-nodes` so it reflects the current node set."""
    from . import ann as ann_mod
    if not ann_mod.available():
        log.error("usearch not installed — `pip install usearch` to build the ANN index.")
        return 1
    kb = KB(cfg)
    try:
        path = cfg.get("ann_path") or (cfg["kb_path"] + ".ann")
        path = os.path.expanduser(path)
        stats = ann_mod.build_from_kb(
            kb, path,
            connectivity=cfg["ann_connectivity"],
            expansion_add=cfg["ann_expansion_add"],
            expansion_search=cfg["ann_expansion_search"],
            dtype=cfg["ann_dtype"],
            min_nodes=cfg["ann_min_nodes"])
        log.info("build-ann: %s", stats)
    finally:
        kb.close()
    return 0


def _run_reconcile(cfg, log, *, anchors="corpus", limit=None, top_k=None) -> int:
    """Queue merge candidates between your existing nodes and the imported commonsense
    sets, then point the user at `adjudicate` to resolve them with the big LM."""
    from . import reconcile_import as rec
    kb = KB(cfg)
    try:
        stats = rec.reconcile_imports(kb, cfg, anchors=anchors, limit=limit, top_k=top_k)
        log.info("reconcile: %s", stats)
        if stats.get("nodes_without_vectors"):
            log.warning("%d active nodes have NO embedding and were invisible to the "
                        "scan — run `embed-nodes` first to reconcile them too.",
                        stats["nodes_without_vectors"])
        if stats.get("queued"):
            log.info("next: run `adjudicate --watch` to judge the %d queued pair(s) with "
                     "the big LM.", stats["queued"])
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    finally:
        kb.close()
    return 0


def _run_embed_nodes(cfg, embedder, log, *, limit=None) -> int:
    """Backfill embeddings for NULL-embedding nodes (e.g. bulk-imported ConceptNet
    terms) so they surface in dense search.  Resumable; safe to re-run."""
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — start the nomic embed server first.")
        return 1
    from . import conceptnet as cn
    kb = KB(cfg)
    try:
        stats = cn.embed_nodes(kb, embedder, cfg, limit=limit)
        log.info("embed-nodes: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — embedded nodes are committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="knowledgehost",
                                 description="Vinkona local general-knowledge tool host.")
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "ingest", "distill", "adjudicate", "reconcile",
                             "link", "refine", "import-conceptnet", "import-atomic",
                             "import-glucose", "import-causenet", "embed-nodes", "build-ann",
                             "optimize", "stats", "reset", "bump-version",
                             "bundles", "split", "source", "scenario", "eval", "facetize",
                             "ingest-library", "rebuild-fts"])
    # positional args for the modular-bundle verbs:
    #   source <doc_id> [--title ..] [--bundle ..]   scenario [name]   split [dir]
    ap.add_argument("args", nargs="*", help="positional args for bundles/source/scenario/split")
    ap.add_argument("-c", "--config", help="path to a TOML config file")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--backend", choices=["sqlite", "lance"])
    ap.add_argument("--force", action="store_true",
                    help="ingest: re-process every file, ignoring the manifest")
    ap.add_argument("--wikipedia", action="store_true",
                    help="ingest: also ingest the configured Wikipedia ZIM")
    ap.add_argument("--distill", action="store_true",
                    help="ingest: distil the newly-ingested chunks right after (needs the big LM)")
    ap.add_argument("--watch", action="store_true",
                    help="distill: keep distilling as a concurrent ingest adds chunks")
    ap.add_argument("--interval", type=int, default=30,
                    help="distill --watch: seconds to wait between passes (default 30)")
    ap.add_argument("--limit", type=int,
                    help="ingest/distill/adjudicate: cap items processed (testing)")
    ap.add_argument("--batch", type=int, default=8,
                    help="adjudicate: merge-candidate pairs per big-LM call (default 8)")
    ap.add_argument("--no-auto", action="store_true",
                    help="adjudicate: skip the deterministic pre-pass (LM-judge everything)")
    ap.add_argument("--auto-only", action="store_true",
                    help="adjudicate: run ONLY the deterministic pre-pass, no LM")
    ap.add_argument("--fast", action="store_true",
                    help="adjudicate/link: use the fast 9B (4090) instead of the big LM (3090)")
    ap.add_argument("--path", help="import-conceptnet: path to assertions.csv dump")
    ap.add_argument("--min-weight", type=float, dest="min_weight",
                    help="import-conceptnet: drop assertions below this weight")
    ap.add_argument("--all", action="store_true",
                    help="import-conceptnet: also import the lexical/etymological bulk")
    ap.add_argument("--anchors", choices=["corpus", "all"], default="corpus",
                    help="reconcile: anchor on your distilled corpus (default) or every node")
    ap.add_argument("--top-k", type=int, dest="top_k",
                    help="reconcile: nearest neighbours to queue per anchor")
    ap.add_argument("--vacuum", action="store_true",
                    help="optimize: VACUUM after migrating to reclaim disk space")
    ap.add_argument("--title", help="source: new display title for the source (rename)")
    ap.add_argument("--bundle",
                    help="source: assign the source to this bundle group | "
                         "distill: only chunks from this provenance bundle (e.g. 'vinkona' — "
                         "distil Vinkona's research drops ahead of the big corpus)")
    ap.add_argument("--out", help="split: output directory for bundle files")
    ap.add_argument("--license", help="source: SPDX licence id (e.g. CC-BY-NC-4.0, proprietary)")
    ap.add_argument("--license-holder", dest="license_holder",
                    help="source: to whom the licence/copyright belongs")
    ap.add_argument("--license-url", dest="license_url", help="source: licence terms URL")
    ap.add_argument("--gold", help="eval: path to the JSONL gold set (default eval_gold_path)")
    ap.add_argument("--retriever", default="current_path",
                    help="eval: which registered retriever to score (default current_path)")
    ap.add_argument("--trace", action="store_true",
                    help="eval: per-query debug logging (top card id, tier)")
    ap.add_argument("--kb", action="store_true",
                    help="reset: clear only the distilled KB (keep raw chunks)")
    ap.add_argument("--raw", action="store_true",
                    help="reset: clear only the raw chunk store (keep the KB)")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="reset: skip the confirmation prompt")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # remember the config file so the server's ops-runner launches subprocesses (and the
    # Settings panel writes) against the SAME config the server itself is using.
    cfg["_config_path"] = args.config or os.environ.get("KNOWLEDGEHOST_CONFIG")
    for k in ("host", "port", "backend"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)

    logging.basicConfig(
        level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("knowledgehost")

    if args.command == "reset":            # before _build, so we don't recreate files
        return _cmd_reset(cfg, args, log)

    if args.command == "import-conceptnet":   # KB-only; no raw store / embed endpoint
        return _run_import_conceptnet(cfg, log, path=args.path,
                                      min_weight=args.min_weight,
                                      include_lexical=args.all, limit=args.limit)

    if args.command == "import-atomic":       # KB-only; no raw store / embed endpoint
        return _run_import_atomic(cfg, log, path=args.path, limit=args.limit)

    if args.command == "import-glucose":      # KB-only; no raw store / embed endpoint
        return _run_import_glucose(cfg, log, path=args.path, limit=args.limit)

    if args.command == "import-causenet":     # KB-only; no raw store / embed endpoint
        return _run_import_causenet(cfg, log, path=args.path, limit=args.limit)

    if args.command == "reconcile":           # KB-only; vectorised, no LM/embed endpoint
        return _run_reconcile(cfg, log, anchors=args.anchors, limit=args.limit,
                              top_k=args.top_k)

    if args.command == "link":                # KB + an LM; no raw store / embed endpoint
        return _run_link(cfg, log, limit=args.limit, top_k=args.top_k, fast=args.fast)

    if args.command == "build-ann":           # KB-only; builds the HNSW node index
        return _run_build_ann(cfg, log)

    if args.command == "optimize":            # KB-only; one-time node layout fix
        return _run_optimize(cfg, log, vacuum=args.vacuum)

    if args.command in ("bundles", "split", "source", "scenario"):   # modular §16, KB-only
        return _run_bundles(cfg, args, log)

    if args.command == "eval":                # retrieval eval harness (KB + embedder)
        return _run_eval(cfg, log, gold=args.gold, retriever=args.retriever, trace=args.trace)

    if args.command == "ingest-library":      # index the search-only document library
        return _run_ingest_library(cfg, log, force=args.force)
    if args.command == "rebuild-fts":         # migrate FTS to the configured tokenizer
        return _run_rebuild_fts(cfg, log)

    if args.command == "facetize":            # backfill the multi-axis facet layer (KB-only)
        kb = KB(cfg)
        try:
            import time as _t
            t0 = _t.perf_counter()
            counts = kb.facetize(limit=args.limit)
            log.info("facetized %s in %.1fs", counts, _t.perf_counter() - t0)
            import json as _json
            print(_json.dumps({"facetized": counts, "facet_counts": kb.facet_counts()}, indent=2))
        finally:
            kb.close()
        return 0

    store, embedder = _build(cfg)

    if args.command == "bump-version":
        v = int(store.manifest.meta_get("version", "1")) + 1
        store.manifest.meta_set("version", v)
        print(f"version -> {v}")
        store.close()
        return 0

    if args.command == "stats":
        from . import ann as ann_mod
        kb = KB(cfg)
        ann_path = os.path.expanduser(cfg.get("ann_path") or (cfg["kb_path"] + ".ann"))
        ann_status = ("built" if ann_mod.index_exists(ann_path)
                      else ("absent (run build-ann)" if ann_mod.available()
                            else "absent (usearch not installed)"))
        print({"backend": store.backend, "chunks": store.count(),
               "dense": store.has_vectors(),
               "version": store.manifest.meta_get("version", "1"),
               "kb": kb.counts(), "ann_index": ann_status})
        kb.close(); store.close()
        return 0

    if args.command == "distill":
        store.close()                          # _run_distill owns its own store handle
        return _run_distill(cfg, embedder, log, limit=args.limit,
                            watch=args.watch, interval=args.interval,
                            bundle=getattr(args, "bundle", None))

    if args.command == "refine":               # raw store (source) + KB + big LM
        return _run_refine(cfg, store, embedder, log, limit=args.limit, force=args.force)

    if args.command == "adjudicate":
        store.close()
        return _run_adjudicate(cfg, log, limit=args.limit, batch=args.batch,
                               watch=args.watch, interval=args.interval,
                               auto=not args.no_auto, auto_only=args.auto_only,
                               fast=args.fast)

    if args.command == "embed-nodes":
        store.close()                          # KB + embed endpoint only
        return _run_embed_nodes(cfg, embedder, log, limit=args.limit)

    if args.command == "ingest":
        if not embedder.embed_one("warmup", "document"):
            if cfg["backend"] == "lance":
                log.error("embed endpoint unreachable — the lance backend can't store "
                          "chunks without vectors. Start the embed server and retry.")
                store.close()
                return 1
            log.warning("embed endpoint unreachable — ingesting WITHOUT vectors "
                        "(sparse FTS only; re-run after it's up to add dense).")
        try:
            stats = ingest_mod.crawl(store, embedder, cfg, force=args.force)
            log.info("documents: %s", stats)
            if args.wikipedia:
                wstats = ingest_mod.ingest_wikipedia(store, embedder, cfg,
                                                     limit=args.limit, force=args.force)
                log.info("wikipedia: %s", wstats)
        except ingest_mod.EmbedUnavailable as e:
            log.error("embed endpoint dropped mid-ingest (%s) — aborted; nothing lost, "
                      "re-run to resume.", e)
            store.close()
            return 1
        if cfg["backend"] == "lance" and hasattr(store, "maybe_build_ann"):
            store.maybe_build_ann()
        rc = 0
        if args.distill:                       # opt-in: distil right after ingest
            store.close()                      # _run_distill opens its own handle
            return _run_distill(cfg, embedder, log)
        store.close()
        return rc

    # default: serve.  Modular knowledge (§16): if a non-'all' scenario / bundles
    # are configured, assemble a disposable working DB from the selected sources and
    # serve THAT — the master kb.db stays the untouched authoring source of truth
    # (maintenance verbs still edit the master via the config's kb_path).
    from . import bundles
    cfg["_master_kb_path"] = cfg["kb_path"]
    try:
        work = bundles.assemble_working_db(cfg, log_fn=log.info)
    except Exception as e:                          # never let assembly wedge startup
        log.error("working-DB assembly failed (%s) — serving master kb.db", e)
        work = cfg["kb_path"]
    if work != cfg["kb_path"]:
        cfg["kb_path"] = work
        cfg["ann_path"] = ""                        # derive "<working>.ann", not the master's
        log.info("modular: serving scenario '%s' from %s",
                 bundles.active_scenario_name(cfg), work)
    kb = KB(cfg)
    lib = _library_store(cfg, log)              # search-only document library (if built)
    if lib is not None:
        log.info("document library loaded (%d chunks) — library_search tool enabled",
                 lib.count())
    tools = Tools(store, embedder, cfg, kb, library_store=lib)
    server.serve(cfg, store, tools, kb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
