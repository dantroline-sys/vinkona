"""Digest-first distillation driver — a drop-in for vinur's sequential pass.

Same checkpointing (is_distilled / mark_distilled), same writes — everything
goes through knowledgehost.distill.distill_chunk via its extraction= seam,
exactly as the stock parallel driver does, so the one-card-factory principle
holds.  What changes is the prompting:

- the per-document digest is PREPENDED to the stock user prompt, so the
  extractor sees what the document is before judging the excerpt;
- documents run contiguously and the system prompt is byte-stable per
  (source_type, regime), so a served prefix cache covers skeleton + digest;
- fictional sources fall back to the stock inline path (the fiction second
  pass has its own prompt discipline; enhancement is for reference matter).

Operational rule: do not run this while the host's own distill op is enabled
in autopilot — two writers double-distilling the same pending chunks is a
race, not a speedup.  Disable the stock distill entries (or run this while
the host is idle); the is_distilled checkpoint makes switching drivers safe
in either direction.
"""
import json

from . import digest


def _extract_enhanced(D, lm, chunk, regime, block):
    """lm.extract with the digest-first user prompt; parse mirrors lm.extract
    (including truncation salvage)."""
    system = D._system_for(chunk, regime)
    user = block + D._user_prompt(chunk)
    content = lm._content(system, user, D.DISTILL_SCHEMA, lm.max_tokens)
    if content is None:
        return (None, [], [], []), system, user
    try:
        obj = json.loads(D._first_json(content))
        ext = ((obj.get("concepts") or []), (obj.get("relations") or []),
               (obj.get("procedures") or []), (obj.get("criteria") or []))
    except (ValueError, AttributeError):
        salvaged = D._salvage_concepts(content)
        ext = ((salvaged or None), [], [], [])
    return ext, system, user


def run(store, kb, lm, embedder, cfg, *, bundle=None, limit=None, use_gloss=True,
        log_fn=None):
    """Distil pending chunks with digest-enhanced prompts.  Returns stats; a
    dead backend stops the run cleanly (resume = rerun, checkpoint holds)."""
    from knowledgehost import distill as D
    say = log_fn or (lambda m: None)
    stats = {"chunks": 0, "concepts": 0, "relations": 0, "cards": 0, "docs": 0,
             "digest_cached": 0, "fiction_stock": 0, "skipped": 0, "backend": "ok"}
    skipped = [0]
    cur_doc, block = None, ""
    every = cfg.get("ingest_log_every") or 0

    def lens_for(chunk, doc_id):
        # Mirrors the stock parallel driver: a folder mapping wins; else the
        # source's effective (possibly re-tagged) regime; else None -> format
        # fallback inside _system_for.
        folder = D.regime_for_path(cfg, doc_id)
        if folder:
            return folder, folder
        src = kb.get_source(doc_id)
        return None, (src or {}).get("regime")

    try:
        for chunk in D._pending_chunks(store, kb, skipped, bundle=bundle):
            doc_id = chunk.get("path_or_url") or chunk.get("id")
            if doc_id != cur_doc:
                d, hit = digest.build(store, kb, doc_id, lm=lm, use_gloss=use_gloss)
                block = digest.render(d)
                cur_doc = doc_id
                stats["docs"] += 1
                stats["digest_cached"] += int(hit)
            folder_regime, lens = lens_for(chunk, doc_id)
            if (lens or "").strip() == "fictional":
                stats["fiction_stock"] += 1
                with kb.batch():
                    nc, nr, ncard = D.distill_chunk(kb, lm, embedder, chunk,
                                                    source_regime=folder_regime)
                    kb.mark_distilled(chunk["id"])
            else:
                ext, _sys, _user = _extract_enhanced(D, lm, chunk, lens, block)
                with kb.batch():
                    nc, nr, ncard = D.distill_chunk(kb, lm, embedder, chunk,
                                                    extraction=ext,
                                                    source_regime=folder_regime)
                    kb.mark_distilled(chunk["id"])
            stats["chunks"] += 1
            stats["concepts"] += nc
            stats["relations"] += nr
            stats["cards"] += ncard
            if every and stats["chunks"] % every == 0:
                say(f"… {stats['chunks']} chunks / {stats['concepts']} concepts / "
                    f"{stats['cards']} cards ({stats['docs']} docs)")
            if limit and stats["chunks"] >= limit:
                break
    except D.BackendUnavailable as e:
        stats["backend"] = f"unavailable: {e}"
    stats["skipped"] = skipped[0]
    return stats
