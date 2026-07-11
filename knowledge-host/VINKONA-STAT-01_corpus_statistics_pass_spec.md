# VINKONA-STAT-01 — Corpus Statistics Pass (Card Salience & Concept Co-occurrence)

**Status:** Draft for implementation · **Doc version:** 1.0 · **Date:** 2026-07-05
**Component:** `vinkona-stat build` — offline global pass · **Algo version string:** `VINKONA-STAT-01/1.0`
**Implementation language:** Python ≥ 3.12. No ML, GPU, or network dependencies. SQLite ≥ 3.45.

RFC 2119 keywords (MUST, MUST NOT, SHOULD, MAY) are normative.

---

## 1. Scope and non-goals

This document specifies the offline global pass that derives two weight sets from the authored corpus and materializes them onto graph edges:

1. **Card→concept salience** — a normalized TF-IDF weight per card-concept reference, consumed by the concept-coverage rank list and as spreading-activation seed mass.
2. **Concept→concept co-occurrence** — a PPMI strength on `co_occurs_with` edges, populating the Associative Fallback retrieval tier and the associative band of spreading activation.

The pass is a pure function of a postings snapshot plus a fixed parameter set. It runs offline (after an ingestion batch, or on a schedule), writes into shadow tables, and swaps atomically. It is **never** in the query hot path.

**Out of scope** (separate specs):

- **Ingestion.** The postings ledger is an input, produced by VINKONA-ING-01. This pass does not create postings.
- **Consumption.** Spreading activation, RRF fusion, ranking, and selection read these weights but are specified elsewhere. This document defines only how the weights are produced and stored.
- **Ontological edges.** `is_a`, `part_of`, `caused_by`, and all other definitional/causal edges are inputs to the graph from other sources. This pass MUST NOT create, modify, or delete them. It writes only `card→concept` salience and `concept→concept` `co_occurs_with` edges.
- **Neural link prediction** (KGE / TransE / ComplEx). Deferred; see Appendix B.
- **Incremental / streaming updates.** v1 is full-recompute only; see Appendix B.

## 2. Definitions

- **Card** — an authored knowledge card. Cards are the **documents** of this corpus. Each has a stable `card_id`.
- **Concept** — a node in the graph vocabulary, referenced by cards. Identified by its global ID (authority prefix included), e.g. `pub:dx.appendicitis`.
- **Posting** — an aggregated `(card_id, concept_id, local_count)` row: `local_count` is the number of times the card references the concept (span count from ingestion), `≥ 1`.
- **N** — the number of distinct cards that have at least one posting.
- **df(c)** (document frequency) — the number of distinct cards with a posting for concept `c`. Used only for IDF.
- **Salient set** `S(card)` — the top-`K_COOC` concepts of a card by salience (§5, stage G3). Co-occurrence is computed over salient sets only.
- **sdf(c)** (salient document frequency) — the number of distinct cards for which `c ∈ S(card)`. Used for co-occurrence marginals. **`sdf(c) ≤ df(c)` in general and the two are NOT interchangeable** (§5.4).
- **Concept-id order** — the total order on concept IDs given by comparing their UTF-8 byte sequences (equivalently, Unicode code-point order for the ASCII-range IDs in use). All "sorted", "`a < b`", and tie-break "lexicographic" clauses in this document mean this order.
- **`ln`** — natural logarithm.

## 3. Input

### 3.1 Postings ledger (produced by VINKONA-ING-01; read-only to this pass)

```sql
CREATE TABLE posting (
  card_id     TEXT    NOT NULL,
  concept_id  TEXT    NOT NULL,                 -- global node ID; MUST exist in node table
  local_count INTEGER NOT NULL CHECK (local_count >= 1),
  PRIMARY KEY (card_id, concept_id)
);
```

The pass also reads the graph's node table solely to validate that every `concept_id` resolves to a live node (stage G0). It reads nothing else.

## 4. Parameters

All parameters are explicit and versioned into `stats_version` (§7). Production defaults:

| Name | Default | Meaning | Constraint |
|---|---|---|---|
| `K_COOC` | 20 | max salient concepts per card entering co-occurrence | integer ≥ 2 |
| `MIN_COOC` | 3 | minimum card co-occurrence count for a pair to survive | integer ≥ 1 |

`K_COOC` bounds the per-card pair blow-up to `C(K_COOC, 2)` and restricts association to concepts that actually characterize the card. `MIN_COOC` is the small-sample floor: pairs observed fewer than `MIN_COOC` times carry too little evidence and MUST be discarded before PMI (a single co-occurrence would otherwise score maximally — the exact failure mode PMI is prone to). These two, plus the PPMI clamp (stage G5), are the complete small-sample and hub-suppression guard; there is no learned weight anywhere in this pass.

## 5. Algorithm STAT-1 (normative)

Stages G0–G6 run in order. Each stage consumes only prior-stage outputs. All arithmetic is IEEE-754 double precision; only the explicitly rounded outputs (weights) are rounded, and all comparisons (`MIN_COOC`, clamp) use unrounded values.

### G0 — Load & validate
Read all postings. Every `concept_id` MUST resolve to a live node (else `E_UNKNOWN_CONCEPT`, listing offending ids). Compute `N` = distinct `card_id`. If `N = 0`, write an empty result set (valid) and a report with `corpus.N = 0`; do not error.

### G1 — Document frequency & IDF
`df(c)` = distinct cards with a posting for `c`.
`IDF(c) = ln((N + 1) / (df(c) + 1)) + 1`.
This smoothed form is always `> 0`; a concept present in every card still receives a small positive IDF (never zero, never negative), so downstream products stay well-defined.

### G2 — Raw & normalized salience
For each posting `(card, c, count)`:
`raw_sal(card, c) = (1 + ln(count)) × IDF(c)`  (sublinear TF × IDF).
Per card, let `M(card) = max over its postings of raw_sal`.
`salience(card, c) = raw_sal(card, c) / M(card)`  → in `(0, 1]`, top concept = 1.0.
The **stored** salience weight is `salience(card, c)` **rounded half-even to 6 decimal places**. Normalization is a strictly-positive per-card monotone scaling, so it does not change within-card concept order (G3 is unaffected by whether raw or normalized values are used).

### G3 — Salient set selection
For each card, order its concepts by `(raw_sal desc, concept_id asc)` and take the first `K_COOC`. This ordered-then-truncated set is `S(card)`. If a card has `≤ K_COOC` concepts, `S(card)` is all of them. (Because hubs carry the lowest IDF, they are the first to fall out of `S` in concept-dense cards — hub self-exclusion is a consequence of the ordering, not a special case.)

### G4 — Co-occurrence accumulation (document-pivoted)
Initialize empty maps `sdf` and `cooc`. For each card:
- for each `c ∈ S(card)`: `sdf[c] += 1`;
- for each unordered pair `{a, b} ⊆ S(card)` with `a < b`: `cooc[(a, b)] += 1`.

`cooc[(a,b)]` is the count of cards in which both are salient — document-level presence, **not** multiplied by `local_count`. The concept×concept matrix is never materialized; only observed pairs are emitted (cost scales with what occurs, not with the vocabulary square). An unobserved pair has no entry and no edge — a true absence of evidence, not a stored zero.

### G5 — PMI, floor, PPMI clamp
For each pair `(a, b)` in `cooc`, in `(concept_a, concept_b)` sorted order:
1. If `cooc[(a,b)] < MIN_COOC` → **discard** (floor).
2. `PMI = ln( (cooc[(a,b)] × N) / (sdf[a] × sdf[b]) )`.
3. If `PMI ≤ 0` → **discard** (PPMI clamp: at-or-below-chance association carries no positive signal, and this is what removes hub pairs, which co-occur with everything at ~chance).
4. Otherwise emit a `co_occurs_with` edge with `ppmi = PMI` **rounded half-even to 6 dp** and `cooc_count = cooc[(a,b)]`.

The marginals in step 2 are `sdf`, **not** `df`: the probability space is "sample a random card; is the concept in that card's salient set." Using `df` here would mismatch the pair counts (which are over salient sets) and yield a malformed PMI.

### G6 — Materialize atomically
Write salience rows and co-occurrence edges into shadow tables (`*_shadow`), compute `stats_version` (§7), verify row counts against the in-memory result, then swap shadow → live inside a single transaction (see §6.2). Runtime readers MUST only ever see a fully-consistent `stats_version`; a half-written state MUST NOT be observable.

## 6. Output

### 6.1 DDL

```sql
CREATE TABLE card_concept_salience (
  card_id     TEXT NOT NULL,
  concept_id  TEXT NOT NULL,
  weight      REAL NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),  -- normalized salience, 6dp
  PRIMARY KEY (card_id, concept_id)
);

CREATE TABLE concept_cooccurrence (
  concept_a   TEXT NOT NULL,      -- concept_a < concept_b in concept-id order (enforced)
  concept_b   TEXT NOT NULL,
  ppmi        REAL NOT NULL CHECK (ppmi > 0.0),   -- 6dp
  cooc_count  INTEGER NOT NULL CHECK (cooc_count >= 1),
  PRIMARY KEY (concept_a, concept_b),
  CHECK (concept_a < concept_b)
);

CREATE TABLE stats_meta (
  k              TEXT PRIMARY KEY,   -- 'stats_version','algo_version','K_COOC','MIN_COOC','N','built_at'
  v              TEXT NOT NULL
);
```

The `co_occurs_with` graph edge type is realized by `concept_cooccurrence`; it is a distinct edge type and MUST NOT be merged into or confused with ontological/causal edges when consumed.

### 6.2 Atomic swap (normative)
Build populates `card_concept_salience_shadow` and `concept_cooccurrence_shadow`. After validation, within one `BEGIN IMMEDIATE … COMMIT`: drop the live tables (or rename aside), rename shadow → live, write `stats_meta`. On any error before `COMMIT`, roll back and leave the prior live tables intact. A reader opening a transaction sees exactly one `stats_version` throughout.

### 6.3 Build report (`<out>/stats_report.json`)
Canonical JSON (§8.2 rules), keys in this order:
`{"stats_version","algo_version","params":{"K_COOC","MIN_COOC"},"corpus":{"N"},"salience":[…],"cooccurrence":[…]}`
where `salience` entries are `{"card_id","concept_id","weight"}` sorted by `(card_id, concept_id)`, and `cooccurrence` entries are `{"concept_a","concept_b","ppmi","cooc_count"}` sorted by `(concept_a, concept_b)`. This report is the byte-exact conformance artifact (§9).

## 7. stats_version

Lowercase hex SHA-256 over the canonical input+params dump:

1. Sort postings by `(card_id, concept_id)` in concept-id order.
2. Serialize each as a compact JSON array `[card_id, concept_id, local_count]` (separators `,`/`:`, UTF-8, no escaping of non-ASCII).
3. `body` = those lines joined by `\n` (U+000A).
4. `canon = algo_version + U+001E + params_json + U+001E + body`, where `params_json = {"K_COOC":<int>,"MIN_COOC":<int>}` (compact, keys in that fixed order).
5. `stats_version = sha256(canon.encode("utf-8"))`.

Any change to postings content, `K_COOC`, `MIN_COOC`, or `algo_version` changes `stats_version`. A change to any G1–G5 formula is a new `algo_version` and REQUIRES full recompute.

## 8. Determinism & serialization

### 8.1 Determinism guarantee
For a fixed postings snapshot and fixed parameters, the build MUST produce a byte-identical `stats_report.json` and identical table contents across runs, platforms, and thread counts. All ordering is total (concept-id order + defined tie-breaks); all rounding is half-even to 6 dp; no floating-point-order-dependent reductions are permitted in emitted values.

### 8.2 Canonical JSON
UTF-8; separators `,` and `:` with no whitespace; object keys in the orders given in §6.3; arrays sorted as specified; numbers emitted by the platform's shortest round-trripping `repr` of the already-6dp-rounded value (e.g. `1.0`, `0.356675`); non-ASCII emitted raw (no `\uXXXX`).

## 9. Errors

Raised as `StatError(code, message)`:

| Code | Condition |
|---|---|
| `E_UNKNOWN_CONCEPT` | a posting references a `concept_id` absent from the node table |
| `E_BAD_PARAM` | `K_COOC < 2` or `MIN_COOC < 1` or non-integer |
| `E_POSTINGS_MALFORMED` | `local_count < 1`, or duplicate `(card_id, concept_id)` |
| `E_SWAP_FAILED` | shadow validation/row-count check failed; live tables left intact |

`N = 0` is NOT an error (§G0).

## 10. Performance

Assumptions: `N ≤ 10,000` cards; postings ≤ 2,000,000 rows; vocabulary ≤ 2,000,000 concepts.

- Full build ≤ 30 s wall on a single performance core (Apple M4 / Zen 4 class); working memory ≤ 2 GiB.
- Per-card pair emission is bounded by `C(K_COOC, 2)`; the concept×concept square is NEVER allocated.
- Zero network, GPU, or model inference. Runtime consumers read only the two output tables; this pass is not in the query path.

## 11. Acceptance test

Conformant iff, given the corpus and parameters below, the build reproduces the byte-exact `stats_report.json` in §11.4 (with `stats_version` matching `^[0-9a-f]{64}$` and equal to the value shown) and the drop ledger in §11.3.

### 11.1 Test corpus (`posting` rows) — parameters **K_COOC = 4, MIN_COOC = 2**

```
card:1  n:appendicitis 3 | n:appendicectomy 2 | n:nausea 1 | n:hub 5
card:2  n:appendicitis 2 | n:appendicectomy 2 | n:nausea 1 | n:hub 4
card:3  n:appendicitis 2 | n:appendicectomy 1 | n:hub 3
card:4  n:analgesia 3 | n:nausea 2 | n:hub 4
card:5  n:analgesia 2 | n:nausea 2 | n:hub 3
card:6  n:appendicitis 1 | n:analgesia 1 | n:hub 2
card:7  n:nausea 1 | n:hub 6
card:8  n:rare 1 | n:appendicitis 1 | n:hub 2
card:9  n:appendicitis 1 | n:appendicectomy 1 | n:nausea 1 | n:analgesia 1 | n:hub 1
card:10 n:appendicitis 5 | n:appendicectomy 4 | n:nausea 3 | n:analgesia 2 | n:rare 1
```

`stats_version` for this input+params = `ff9ecad3acf1d421ccef87b90fba7d8d273e549d05a99d33ca699043439f4af2`.

### 11.2 Key intermediates (for debugging; `N = 10`)

IDF: `n:hub` 1.09531 (df 9) · `n:appendicitis` 1.318454 (df 7) · `n:nausea` 1.318454 (df 7) · `n:appendicectomy` 1.606136 (df 5) · `n:analgesia` 1.606136 (df 5) · `n:rare` 2.299283 (df 2).

**sdf ≠ df** (top-K truncation at work): `n:hub` df 9 → sdf 8 (truncated from card:9's top-4, being lowest-IDF); `n:rare` df 2 → sdf 1 (truncated from card:10's top-4). All other concepts have sdf = df here.

### 11.3 Drop ledger (required)

Floor drops (`cooc_count < MIN_COOC = 2`): `(n:appendicitis, n:rare)` count 1; `(n:hub, n:rare)` count 1.

PPMI-clamp drops (`PMI ≤ 0`): all seven remaining pairs, every one of them a hub pair or an at-/below-chance pair —
`(n:analgesia,n:appendicectomy)` PMI −0.223144 · `(n:analgesia,n:appendicitis)` −0.154151 · `(n:analgesia,n:hub)` −0.287682 · `(n:appendicectomy,n:hub)` −0.287682 · `(n:appendicitis,n:hub)` −0.113329 · `(n:appendicitis,n:nausea)` −0.202941 · `(n:hub,n:nausea)` −0.113329.

**The lesson made concrete:** `n:hub` appears in 9 of 10 cards yet yields **zero** surviving edges — a raw co-occurrence count would have made it the most-connected node in the graph; PMI + clamp correctly discards every hub association as no-better-than-chance. Only three genuine associations survive.

### 11.4 Byte-exact `stats_report.json`

```json
{"stats_version":"ff9ecad3acf1d421ccef87b90fba7d8d273e549d05a99d33ca699043439f4af2","algo_version":"VINKONA-STAT-01/1.0","params":{"K_COOC":4,"MIN_COOC":2},"corpus":{"N":10},"salience":[{"card_id":"card:1","concept_id":"n:appendicectomy","weight":0.951465},{"card_id":"card:1","concept_id":"n:appendicitis","weight":0.968084},{"card_id":"card:1","concept_id":"n:hub","weight":1.0},{"card_id":"card:1","concept_id":"n:nausea","weight":0.461297},{"card_id":"card:10","concept_id":"n:analgesia","weight":0.70953},{"card_id":"card:10","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:10","concept_id":"n:appendicitis","weight":0.897647},{"card_id":"card:10","concept_id":"n:nausea","weight":0.721923},{"card_id":"card:10","concept_id":"n:rare","weight":0.59991},{"card_id":"card:2","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:2","concept_id":"n:appendicitis","weight":0.820886},{"card_id":"card:2","concept_id":"n:hub","weight":0.961134},{"card_id":"card:2","concept_id":"n:nausea","weight":0.484828},{"card_id":"card:3","concept_id":"n:appendicectomy","weight":0.698736},{"card_id":"card:3","concept_id":"n:appendicitis","weight":0.971159},{"card_id":"card:3","concept_id":"n:hub","weight":1.0},{"card_id":"card:4","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:4","concept_id":"n:hub","weight":0.775437},{"card_id":"card:4","concept_id":"n:nausea","weight":0.662285},{"card_id":"card:5","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:5","concept_id":"n:hub","weight":0.845264},{"card_id":"card:5","concept_id":"n:nausea","weight":0.820886},{"card_id":"card:6","concept_id":"n:analgesia","weight":0.866065},{"card_id":"card:6","concept_id":"n:appendicitis","weight":0.71094},{"card_id":"card:6","concept_id":"n:hub","weight":1.0},{"card_id":"card:7","concept_id":"n:hub","weight":1.0},{"card_id":"card:7","concept_id":"n:nausea","weight":0.431171},{"card_id":"card:8","concept_id":"n:appendicitis","weight":0.57342},{"card_id":"card:8","concept_id":"n:hub","weight":0.806565},{"card_id":"card:8","concept_id":"n:rare","weight":1.0},{"card_id":"card:9","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:9","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:9","concept_id":"n:appendicitis","weight":0.820886},{"card_id":"card:9","concept_id":"n:hub","weight":0.681954},{"card_id":"card:9","concept_id":"n:nausea","weight":0.820886}],"cooccurrence":[{"concept_a":"n:analgesia","concept_b":"n:nausea","ppmi":0.133531,"cooc_count":4},{"concept_a":"n:appendicectomy","concept_b":"n:appendicitis","ppmi":0.356675,"cooc_count":5},{"concept_a":"n:appendicectomy","concept_b":"n:nausea","ppmi":0.133531,"cooc_count":4}]}
```

(Three surviving `co_occurs_with` edges: `appendicectomy–appendicitis` PPMI 0.356675 — the strong genuine link; `analgesia–nausea` and `appendicectomy–nausea` each 0.133531.)

## Appendix A — Consumption notes (non-normative)

- **Concept coverage rank list:** weight each query-concept hit on a card by `card_concept_salience.weight`, so a card genuinely about a concept outranks one that merely name-drops it.
- **Spreading activation:** seed with salience mass; allow traversal along `co_occurs_with` in the associative band only, scaled by `ppmi`, kept in a lower confidence tier than ontological/causal edges so it feeds Associative Fallback without contaminating Confident results.

## Appendix B — Deferred to future revisions

- **Incremental update.** All counts (`df`, `sdf`, `cooc`) are additive; adding a card can increment only affected pairs and recompute only touched PMIs. v1 is full-recompute; incrementality is a later optimization, not now.
- **Context-distribution smoothing / PMI^k.** The asymmetric `sdf^α` renormalization (Levy & Goldberg) helps word-context embeddings but breaks PMI symmetry, which an undirected `co_occurs_with` edge needs; deferred unless an eval shows the floor+clamp guard is insufficient.
- **KGE link prediction (VINKONA-KGE-01).** Neural embeddings can propose plausible *unseen* pairs (the one thing a PMI matrix cannot, since its unseen entries are hard zeros). If logs show unseen-pair generalization matters, run KGE offline, surface top-k proposed edges for review, and write accepted ones back with derived provenance — quarantined at write time, invisible at runtime.
