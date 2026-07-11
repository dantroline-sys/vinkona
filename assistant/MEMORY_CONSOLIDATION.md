# Memory consolidation & sharper semantic recall

**Status:** design spec (not yet built)
**Goal:** two related improvements to the `memories` store — (A) make semantic
recall materially sharper, and (B) periodically *coalesce* the user's scattered
personal facts into consolidated, thematic reads, so Vinkona forms a closer picture
of the user in fewer conversations.

These are independent and can land separately. (A) is mostly cheap correctness
wins on the existing hot path; (B) is a new idle-time layer that reuses the
world-knowledge consolidation machinery, carefully extended to personal facts.

The 128 GB of host RAM is **not** a capacity constraint here — the whole
embedding matrix already lives resident in `_emb_mat`. RAM matters only in §A.4
(global offline clustering), where holding the full similarity graph in memory
lets the consolidation pass pick *good* groups instead of greedy local ones.

---

## A. Sharper semantic recall

Recall today ([memory.py `recall()`](memory.py#L566-L618)) unions trigger hits
(Aho-Corasick) with cosine matches (`_emb_mat @ qv`), scores them by
priority/trigger/semantic/recency/tag weights, and adds two-hop neighbours. It
works, but four things hold it back.

### A.1 Use the embed model's task prefixes (biggest, cheapest win)

`nomic-embed-text-v1.5` is an **asymmetric** retrieval model: it expects
`search_query: …` on the query side and `search_document: …` on the stored side.
We currently embed **raw text on both sides** ([embed()](memory.py#L549-L563),
[upsert()](memory.py#L894-L895) builds `emb_text` with no prefix). Asymmetric
models lose a meaningful chunk of retrieval quality when queried symmetrically.

**Change:**
- `embed(text, *, task="search_document")` — prepend the task prefix.
- Memory writes embed with `search_document:` (the default).
- `recall()` embeds the query with `search_query:`.
- This is a **stored-vector format change** → needs a one-off re-embed of every
  memory (see §C migration). Until re-embedded, mixing prefixed queries against
  unprefixed vectors is *worse*, not better, so the migration must run before the
  query side flips. Gate both behind a `memory.embed_task_prefix` flag and a
  stored `embed_format` version in `worker_state`.

> Verify against the exact GGUF in use — some nomic GGUF conversions bake the
> prefix into the template or use `clustering:`/`classification:` variants. If the
> model turns out to be symmetric, skip A.1 and keep raw text.

### A.2 Calibrate semantic vs trigger scores

Cosine sims from nomic cluster in a narrow, high band (~0.4–0.8 even for unrelated
text), so the flat `w["semantic"] * sim` term either dominates or vanishes
depending on tuning, and weak matches dilute the top-k. Two fixes:

- **Similarity floor:** drop semantic candidates below `min_sim` (≈0.55–0.65,
  configurable) *before* scoring, mirroring what `_neighbours` already does with
  `neighbour_min_sim`. Stops the long tail of lukewarm matches crowding real hits.
- **Rank/centre the semantic term:** instead of raw cosine, score by rank within
  the candidate set or by `(sim - floor)/(1 - floor)`, so the weight knob behaves
  linearly and is tunable. Keep the change behind config so current behaviour is
  the default.

### A.3 Embed a little conversational context, not just the bare turn

`recall(text)` embeds only the current utterance. Short turns ("what about her?")
embed poorly. Cheaply enrich the **query** with the last user turn (and optionally
the prior assistant turn) — concatenated, capped — so pronoun-y / elliptical turns
still land near the right memory. Trigger matching stays on the bare turn (so a
stale earlier word doesn't fire a wrong trigger). No extra LM call; this is just
string assembly on the hot path. Config: `memory.recall_context_turns` (default 0
= current behaviour).

### A.4 Offline global clustering (where the RAM earns its keep)

`_neighbours` and `consolidate()` both do *greedy local* nearest-neighbour walks.
With the full matrix resident we can, during idle, run a real clustering pass
(agglomerative or HDBSCAN over `_emb_mat`) once and cache cluster labels per
memory. Benefits:
- Consolidation (§B) operates on genuine clusters, not whatever the greedy walk
  happened to grab first.
- Recall can surface "others in this cluster" as associative context more coherently
  than per-pair two-hop.

This is optional and additive — a `cluster_id` column refreshed by an idle job,
not a hot-path change. Pure-numpy agglomerative over a few thousand memories is
milliseconds; `scikit-learn`/`hdbscan` only if the store grows large. Keep it a
soft dependency.

---

## B. Personal-fact consolidation ("the read on the user")

### The gap

`consolidate()` ([memory.py:1238](memory.py#L1238)) merges/splits **world
knowledge only** — `_is_world` explicitly protects personal facts, because the
existing pass is *destructive* (it deletes the originals) and personal facts carry
nuance, provenance and perspective we must not lose. So the user's scattered facts
— "you mentioned hill-walking", "you went to the Lakes", "you like being outdoors"
— never get drawn together into a coherent read. The result is that Vinkona's
picture of the user builds slowly and stays fragmentary.

### The design: non-destructive thematic synthesis

Add a separate, **non-destructive** pass — `synthesize_profile()` — that does NOT
merge-delete. It:

1. **Clusters the user's personal memories by theme** (work, family, health,
   preferences, places, projects…) using embeddings (reuse §A.4 clusters if
   present, else a local NN walk over personal entries).
2. For each theme cluster above a size threshold, asks the **big LM** to write one
   coherent, first-person-about-the-user paragraph that integrates the cluster's
   facts — "Here's what I understand about your outdoor life: …".
3. Stores that paragraph as a **new** memory: `category="profile"`,
   `source="synthesis"`, linked to its source ids (`context_tags` carries
   `synth:<theme>` + the member ids), high-ish priority so it's recalled
   preferentially.
4. **Leaves every source memory in place.** On the next pass, an existing
   synthesis note for a theme is *updated* (regenerated from the current cluster)
   rather than duplicated — keyed by the `synth:<theme>` tag.

Why synthesis-on-top rather than merge-destroy:
- No information loss; the atomic facts remain individually recallable and
  editable in the Self/People UI.
- Perspective and provenance are preserved on the originals; only the synthesis is
  derived, and it runs the same `perspective_issue` guard before write.
- A bad synthesis is a single deletable note, not a destroyed set of facts.

### Effect on "closer read in a shorter time"

Recall preferentially surfaces the dense synthesis note (one high-priority recall)
instead of three thin fragments, so even early in a relationship Vinkona speaks from
an integrated picture. As more fragments arrive, the synthesis regenerates and
deepens. This is the acceleration the user asked for.

### Relationship to the People/canon store

Personal-fact synthesis lives in the `memories` table and is recall-scored — it is
**not** the privileged People canon. It must respect that boundary
([people.py:19-22](people.py#L19-L22)):

- Synthesis **may read** People attributes for grounding (who the user is) but
  **must not write canon.** If it surfaces something that belongs in the person
  model, it routes through `people.observe()` (surface layer, never locked, dropped
  if it would shadow a locked core attribute) — never `set_attribute(...core...)`.
- Canon stays writable only from live conversation, exactly as now.

### Trust laundering guard (important)

A consolidation that blends a **crawled** "fact" (untrusted — mail/files) into a
high-priority profile note *launders* its trust: a hostile email's planted claim
would graduate into Vinkona's confident self-narrative about the user. Defences:

- Track source trust per member. Only synthesise from `source in {user,
  reflection}` by default; include crawl-derived facts only when corroborated by a
  user/reflection fact in the same cluster, and never let a crawl-only cluster
  produce a high-priority note.
- Carry the lowest member trust onto the synthesis (a synthesis containing any
  crawl-origin fact is itself marked crawl-tainted and capped in priority).
- Continue to fence nothing here is re-read from raw untrusted text — we synthesise
  from already-extracted memory payloads, not raw mail — but the trust cap still
  applies.

---

## C. Migration & rollout

Phased, each shippable alone, current behaviour preserved by default:

1. **§A.2 + §A.3 (no re-embed):** similarity floor, semantic-term calibration,
   optional query-context. Pure scoring/query changes behind config defaults that
   reproduce today's behaviour. Lowest risk; ship first.
2. **§A.1 (re-embed):** add task prefixes; bump `embed_format` in `worker_state`;
   idle job re-embeds all memories in batches (`embed(payload,
   task="search_document")`), then flips the query side. One-time CPU cost on the
   embed-LM; no data loss (re-embed is idempotent from `payload`+`triggers`).
3. **§A.4 (clustering):** `cluster_id` column + idle refresh job. Additive.
4. **§B (synthesis):** `synthesize_profile()` + idle scheduling + Self-tab
   visibility of synthesis notes (they're `source="synthesis"`, easy to filter and
   edit/delete like any memory).

### Config (additions to `memory` block)

```jsonc
"memory": {
  "embed_task_prefix": false,        // §A.1 — flip after re-embed migration
  "semantic_min_sim": 0.0,           // §A.2 — 0 = off (current behaviour)
  "semantic_calibrate": false,       // §A.2 — rank/centre the semantic term
  "recall_context_turns": 0,         // §A.3 — 0 = bare turn only
  "cluster": { "enabled": false, "min_sim": 0.6, "refresh_idle": true },
  "synthesis": {
    "enabled": false,
    "min_cluster": 3,                // fewest facts before a theme is synthesised
    "max_themes_per_pass": 2,        // bound big-LM calls per idle cycle
    "cooldown_s": 86400,             // per-theme regen cooldown
    "priority": 6,                   // recall weight of a synthesis note
    "allow_crawl_sources": false     // trust-laundering guard (§B)
  }
}
```

### Idle scheduling (research_worker)

Slot `synthesize_profile()` into the existing idle cycle alongside `consolidate()`
/ `garden()` / `audit_perspective()` — same cadence, same big-LM tier, off the hot
path. Reuse the per-theme `cooldown_s` so it doesn't thrash. Gate the whole thing
on `synthesis.enabled`.

---

## D. New / changed surface (summary)

- **memory.py**
  - `embed(text, *, task=…)` — task-prefix support (§A.1).
  - `recall()` — query prefix, `semantic_min_sim` floor, optional calibration,
    optional query-context assembly (§A.1–A.3).
  - `cluster_memories()` + `cluster_id` column (§A.4).
  - `synthesize_profile(big_url, big_model, …)` + `_synthesize_theme()` and a
    `DEFAULT_SYNTHESIS_PROMPT` (§B). Non-destructive; `perspective_issue` guard;
    trust cap; routes person-worthy items via `people.observe()`.
  - migration: `embed_format` version in `worker_state`, batch re-embed job.
- **config.py** — the `memory` additions above.
- **research_worker.py** — schedule `cluster_memories()` and
  `synthesize_profile()` in the idle cycle, behind config flags.
- **config_server / Self tab** — let `source="synthesis"` notes be viewed/edited/
  deleted like any memory (they already would be, via the existing memory list —
  just add a filter/label).

## E. Tests

- Recall: floor drops sub-threshold matches; calibration is monotonic; query
  prefix + context don't change *which* memory wins on existing fixtures when the
  flags are at their defaults (regression guard).
- Re-embed migration: idempotent; `embed_format` flips once; recall quality on a
  small labelled set improves (or at least doesn't regress) after prefixing.
- Synthesis: builds a note from a ≥`min_cluster` theme; **does not delete**
  sources; regenerates (not duplicates) on a second pass; respects `cooldown_s`;
  a crawl-only cluster produces nothing when `allow_crawl_sources=false`;
  perspective guard catches a swapped-voice synthesis; never writes People canon
  (only `observe()` is called).

---

## F. Out of scope / deliberately not done

- **Vector DB / ANN index.** Brute-force cosine over the resident matrix is
  already fast for this scale; an ANN index (FAISS/hnsw) is unnecessary until the
  store is ≫10⁵ memories. Note it as a future swap-in behind the same `recall()`
  API, nothing more.
- **Destructive merge of personal facts.** Explicitly avoided — synthesis is
  additive (§B). The world-knowledge `consolidate()` keeps its destructive merge
  because those notes are fungible; personal facts are not.
- **Embedding model change.** Out of scope; the wins here are from using the
  current model correctly. A stronger embed model is a separate evaluation.
