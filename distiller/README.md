# distiller — enhanced distillation add-in for the vinur host

The pipeline-quality upgrades developed for oleum's snippet distillation
(OLEUM-DST-01), applied to text knowledge:

1. **Context digest** — each document gets a one-time digest (deterministic
   outline + registry facts + one small cached LM gloss) prepended to every
   chunk's distill prompt.  No chunk is ever distilled context-free.
2. **Prefix-cache discipline** — documents run contiguously and the digest is
   the first suffix block, so a vLLM-style prefix cache covers the prompt
   skeleton *and* the digest across a document's chunks.
3. **Dedup sweep** — duplicate cards (same node, type, provenance-bundle set,
   normalized content) merge into the oldest card, which accumulates
   `observed_count` / `last_observed` and the support union; duplicates flip
   to `status='merged'` (reversible, invisible to all active-status reads).

## Why it lives here

This repo is PolyForm Noncommercial; vinur is Apache 2.0.  These upgrades are
deliberately **not** donated to the Apache tree.  vinur is consumed strictly
as a library through documented surfaces — notably the same
`distill_chunk(extraction=…)` seam vinur's own parallel driver uses — so all
writes still go through the one card factory, and deleting this directory
restores stock behaviour exactly.

## Running (on the host box, beside a vinur checkout)

```
python3 -m distiller --config /path/to/host/config.toml \
        [--vinur /path/to/vinur] [--bundle vinkona] [--limit 50] \
        [--no-gloss] [--merge] [--merge-only --dry]
```

Rules of engagement:

- **Disable the stock `distill` entries in the host's autopilot ops** while
  this is the driver (two writers racing the same pending chunks is a race,
  not a speedup).  The `is_distilled` checkpoint makes switching drivers safe
  in both directions, any time.
- Fictional sources automatically fall back to the stock inline path (the
  fiction second pass has its own prompt discipline).
- After a sweep that merged anything, run the host's `rebuild-fts`.
- The digest gloss is cached in the store's doc_meta under a content
  signature; unchanged documents are never re-glossed.

## Tests

```
python3 distiller/test_distiller.py          # stdlib + a vinur checkout; stub LM, no services
```
