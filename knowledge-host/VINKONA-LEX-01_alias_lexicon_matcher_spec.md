# VINKONA-LEX-01 — Alias Lexicon & Deterministic Span Matcher

**Status:** Draft for implementation · **Doc version:** 1.0 · **Date:** 2026-07-05
**Components:** `vinkona-lex compile` (write time) · `vinkona-lex match` (runtime)
**Implementation language:** Python ≥ 3.12. The runtime component MUST have no ML, GPU, or network dependencies.

RFC 2119 keywords (MUST, MUST NOT, SHOULD, MAY) are normative.

---

## 1. Scope and non-goals

This document specifies Stage 0 (lexicon) and Stage 1 (span detection) of the deterministic utterance→graph matching pipeline: the alias table schema, the normalization and tokenization algorithms, the compiled artifacts, the runtime matcher's exact behavior, its I/O contract, and acceptance tests.

Consumers of the output: Stage 2 (sense resolution) consumes `spans[].candidates`; the gap/failure logger consumes `unmatched_token_indices` and `flags`.

**Out of scope** (separate specs):

- Sense resolution. The matcher emits **unresolved candidate sets**; it MUST NOT choose between nodes sharing a surface form.
- Assertion tagging (NegEx/ConText), frame classification, graph expansion, ranking, selection.
- Numeric/dose/unit extraction. Tokens such as `3.5kg` pass through as opaque tokens.
- Alias generation/curation pipeline (including inflection generation). The compiler **validates** alias rows; it does not create them. Generated inflection rows arrive in the table like any other row, with `derived_from` set.
- Real-word spelling errors (a valid vocabulary token typed in place of another). Correction applies to out-of-vocabulary tokens only in v1.

## 2. Definitions

- **Code point offset** — index into the sequence of Unicode code points of the original input string. All character offsets in this document are code point offsets: never bytes, never UTF-16 units.
- **Token** — output of TOK-1 (§4). **Norm token** — output of NORM-1 (§3) applied to a token.
- **Norm sequence** (`norm_seq`) — norm tokens joined by U+001F (UNIT SEPARATOR).
- **OOV** — a norm token absent from the compiled vocabulary.
- **Span** — half-open token-index interval `[tok_start, tok_end)`.
- **Candidate** — an alias row attached to a span.
- **Covered token** — a token whose index lies within a span.

## 3. NORM-1 — token normalization (normative)

Input: the original code points of one token. Steps, applied in this order:

1. Unicode NFKC normalization.
2. Unicode full case folding (C + F mappings).
3. If the result ends with (U+0027 APOSTROPHE or U+2019 RIGHT SINGLE QUOTATION MARK) followed by `s`: delete those final two code points.
4. Delete every remaining U+0027 and U+2019.

The output MAY be empty (e.g. input `'s`). Empty-norm tokens are retained in the token list with `norm: ""` but MUST NOT participate in matching, correction, or `unmatched_token_indices`.

NORM-1 MUST be applied identically to alias surfaces (per surface token) at compile time and to utterance tokens at runtime. Any semantic change to NORM-1 is a new `norm_version` and REQUIRES full recompilation (§6.6).

## 4. TOK-1 — tokenization (normative)

TOK-1 operates on the **original** string, before any normalization, so recorded offsets are exact.

`WORD(i)` is true iff the code point `c` at offset `i` satisfies any of:

- (a) Unicode general category of `c` ∈ {Lu, Ll, Lt, Lm, Lo, Nd, Nl, No};
- (b) `c` ∈ {U+0027, U+2019};
- (c) `c` ∈ {U+002E FULL STOP, U+002C COMMA} AND offsets `i−1` and `i+1` both exist AND both have general category Nd.

A **token** is a maximal run of offsets where `WORD` is true. Every other code point is a separator. Hyphen (U+002D), all dashes, and slash are separators **by design**: `post-op` ≡ `post op` ≡ `post/op` after tokenization, giving hyphenation-insensitive matching with no special cases.

Each token records `[char_start, char_end)` in original code point offsets. Tokens are numbered `0..n−1` left to right.

Normative consequences:

- `Post-op` → tokens `Post`, `op`
- `0.5` → one token `0.5`; `1,000` → one token `1,000`; `.5` → token `5` (leading `.` has no digit on its left)
- `N2O/O2` → tokens `N2O`, `O2`
- `Parkinson's` → one token (apostrophe is a word character; NORM-1 later strips it)

## 5. Data model

### 5.1 DDL (SQLite ≥ 3.45; lives in, or is attached alongside, the knowledge-graph database)

```sql
CREATE TABLE alias (
  alias_id      INTEGER PRIMARY KEY,
  node_id       TEXT    NOT NULL,   -- global graph ID incl. authority prefix, e.g. 'pub:dx.appendicitis'
  surface       TEXT    NOT NULL,   -- curated form, exactly as entered
  norm_seq      TEXT    NOT NULL,   -- TOK-1 + NORM-1 of surface, tokens joined by U+001F (derived; verified at compile, §6.3 C4)
  n_tokens      INTEGER NOT NULL CHECK (n_tokens BETWEEN 1 AND 8),
  alias_type    TEXT    NOT NULL CHECK (alias_type IN
                  ('preferred','synonym','index_term','variant','abbrev','informal','inflection')),
  weight        REAL    NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),
  case_mode     TEXT    NOT NULL DEFAULT 'fold' CHECK (case_mode IN ('fold','exact','caps')),
  fuzzy_allowed INTEGER NOT NULL DEFAULT 1 CHECK (fuzzy_allowed IN (0, 1)),
  origin        TEXT    NOT NULL,   -- 'pub' | 'addon:<addon_id>' | 'user'
  derived_from  INTEGER REFERENCES alias(alias_id),  -- set for generated inflections, else NULL
  status        TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
  UNIQUE (node_id, norm_seq)
);

CREATE INDEX idx_alias_norm ON alias (norm_seq) WHERE status = 'active';
```

### 5.2 Field semantics

**weight** — set at curation time and stored explicitly; the compiler never infers it. Curation defaults: preferred 1.00, synonym 0.90, index_term 0.85, variant 0.85, abbrev 0.80, informal 0.70, inflection = 0.95 × weight of its `derived_from` row.

**case_mode** — evaluated per covered token against the ORIGINAL utterance text (multi-token aliases: every token must pass):

- `fold` — NORM-1 equality suffices; no further constraint.
- `exact` — NFKC(original utterance token) must equal NFKC(alias surface token) code point for code point, i.e. before case folding.
- `caps` — every cased letter (category Lu/Ll/Lt) in the covered utterance token(s) must be Lu.

**fuzzy_allowed = 0** — declares the alias safety-critical for spelling (drug names, look-alike/sound-alike terms). Effects defined in §6.4 (vocabulary classes) and §7 stages M3/M4. Spelling correction MUST NEVER create a match for such an alias.

Cross-node duplicates of `norm_seq` are legal — that is surface ambiguity, and it surfaces as multiple candidates on one span. Same-node duplicates are illegal (UNIQUE constraint).

### 5.3 Curation constraints (enforced by the compiler, §6.3)

- **C1** — single-token alias with norm length ≤ 2 code points → `case_mode` ∈ {exact, caps} AND `fuzzy_allowed` = 0.
- **C2** — single-token alias whose norm ∈ EN_STOP_V1 (Appendix A) → `case_mode` ∈ {exact, caps}.
- **C3** — `node_id` must exist in the graph's node table.
- **C4** — stored `norm_seq` and `n_tokens` must equal recomputation from `surface` via TOK-1 + NORM-1.

## 6. Compiler — `vinkona-lex compile` (write time)

### 6.1 Invocation

CLI: `vinkona-lex compile --db <kg.sqlite> --out <artifact_dir>`
API: `compile_lexicon(db: Path, out: Path) -> CompileReport`

### 6.2 Input

All `alias` rows with `status = 'active'`.

### 6.3 Validation

Run ALL checks, collect ALL findings, then fail (no artifacts written except the report) if any ERROR exists.

ERROR: violations of C1–C4; any empty norm token inside a `norm_seq`.
WARN: one `norm_seq` mapping to more than 4 distinct `node_id`s; an alias with `origin` `user` or `addon:*` sharing a `norm_seq` with a `pub` alias of a **different** node.

Report: `<out>/compile_report.json` — `{"findings":[{"level","code","alias_id","message"}], "counts":{...}}`.

### 6.4 Artifacts

All artifacts are read-only after build. Binary formats are implementation-defined EXCEPT `lexicon.meta.json`.

1. **Vocabulary** — map: norm token → dense `uint32 tok_id`. Tokens sorted by UTF-8 byte order; ids assigned in that order. Per-token metadata:
   - `freq` — count of active alias rows containing the token at any position.
   - `fuzzy_class` ∈ {`targetable`, `suppressed`, `none`}:
     - `targetable` — appears in ≥ 1 active alias with `fuzzy_allowed=1`, AND norm length ≥ 4, AND contains ≥ 1 category-L code point.
     - `suppressed` — not targetable, AND appears in ≥ 1 active alias with `fuzzy_allowed=0`, AND norm length ≥ 4, AND contains ≥ 1 category-L code point.
     - `none` — otherwise.
2. **Aho–Corasick automaton** over `tok_id` **sequences** of all active aliases; each terminal state stores the `alias_id`s ending there. Matching is token-level; character-level AC MUST NOT be used.
3. **SymSpell deletion index** — max edit distance 2, prefix length 7 — over all vocabulary tokens with `fuzzy_class ≠ none`. Distance metric everywhere in this spec: Damerau–Levenshtein, optimal-string-alignment variant (insert, delete, substitute, adjacent transpose; unit costs).
4. **`lexicon.meta.json`** — `{"lexicon_version","norm_version":"NORM-1","tok_version":"TOK-1","alias_count","vocab_size","built_at"}`.

### 6.5 lexicon_version

Lowercase hex SHA-256 over the canonical dump: active alias rows sorted by `alias_id`, each serialized as a JSON array `[alias_id, node_id, surface, norm_seq, n_tokens, alias_type, weight_as_%.4f_string, case_mode, fuzzy_allowed, origin, derived_from, status]`, rows joined by `\n`, encoded UTF-8.

### 6.6 Version discipline

Any semantic change to NORM-1 or TOK-1 is a new `norm_version`/`tok_version` and REQUIRES recompilation. The matcher MUST refuse artifacts whose version fields differ from its own compiled-in versions (`E_ARTIFACT_MISMATCH`).

## 7. Matcher — `vinkona-lex match` (runtime)

### 7.0 Interface

```python
class Matcher:
    @classmethod
    def load(cls, artifact_dir: Path) -> "Matcher":
        """Loads artifacts; validates versions. Instance is immutable and
        thread-safe after load. match() is reentrant."""
    def match(self, text: str) -> dict:        # schema in §8
    def match_json(self, text: str) -> bytes:  # canonical serialization, §8.3
```

CLI: `vinkona-lex match --artifacts <dir> --text "<utterance>"` → canonical JSON on stdout.

Stages M0–M6 are normative and run in order.

**M0 — Input validation.** `text` MUST be a `str` containing no lone surrogates (else `E_INVALID_INPUT`). Length > 4096 code points → `E_INPUT_TOO_LONG`. Empty or all-separator input → valid result with empty `tokens`/`spans`/`flags` arrays.

**M1 — Tokenize + normalize.** Apply TOK-1, then NORM-1 to each token. Map each non-empty norm to its `tok_id`; absent → OOV sentinel.

**M2 — Exact pass.** Run the AC automaton over the `tok_id` sequence → raw matches `(span, {alias_id...})`. Within each raw match, remove candidate aliases that fail their `case_mode` test (§5.2) against the original covered tokens. Remove matches left with zero candidates. Survivors are the **exact match set**.

**M3 — Spelling correction.** Correction-candidate tokens: every index `i` such that (a) token `i` is OOV, (b) `i` is not covered by any exact-set span, (c) norm length ≥ 4, (d) norm contains ≥ 1 category-L code point. Maximum distance `d_max`: norm length 4–6 → 1; ≥ 7 → 2.

For each candidate token, in ascending `i`: SymSpell lookup → hits `(vocab_token, distance ≤ d_max)`. If no hits, leave unchanged. Else let `D` = minimum distance among hits and `H` = hits at distance `D`:

- If **any** member of `H` has `fuzzy_class = suppressed` → do NOT correct. Emit flag `{type:"fuzzy_suppressed", stage:"token", token_index:i, nearest:<member of H with fuzzy_class=suppressed, chosen by max freq then lexicographic min>, distance:D}`.
- Otherwise correct token `i` to the member of `H` chosen by max `freq`, then lexicographic min; record `corrected_from` (original norm) and `edit_distance = D`.

**M4 — Fuzzy pass.** If M3 corrected ≥ 1 token: rerun AC over the corrected `tok_id` sequence. The working match set becomes the union over both passes of `(span, candidate)` pairs (dedupe on `(tok_start, tok_end, alias_id)`). Apply `case_mode` filtering exactly as in M2 (always against original text). Then remove every candidate whose alias has `fuzzy_allowed = 0` AND whose span covers ≥ 1 corrected token; if a span thereby loses all candidates, emit flag `{type:"fuzzy_suppressed", stage:"span", token_index:<lowest corrected token index in span>, nearest:<space-joined norm_seq of the removed alias with lowest alias_id>, distance:<sum of edit_distance over covered corrected tokens>}` and drop the span.

**M5 — Overlap resolution.** Merge matches with identical spans (union their candidates). Then repeat until none remain: among remaining matches select by (i) smallest `tok_start`; (ii) largest `tok_end`; (iii) highest top-candidate score (§ M6 formula); (iv) lowest top-candidate `alias_id`. Emit the selection; delete every remaining match whose span intersects it. Report emitted spans sorted by `tok_start`. (Effect: leftmost-longest wins; nested and crossing matches are dropped.)

**M6 — Scoring & assembly.**
`candidate_score = weight × ∏ f(edit_distance of each corrected covered token)`, with `f(1) = 0.8`, `f(2) = 0.6`; the product over an empty set is 1 (exact candidates score = weight). Round half-even to 4 decimal places.
Candidates within a span sort by score descending, then `alias_id` ascending.
Span fields: `fuzzy` = (any covered token was corrected); `matched_norm` = norm tokens joined by U+0020; `char_start`/`char_end` = `char_start` of first covered token / `char_end` of last covered token (internal separators such as hyphens are therefore included in `surface_original`); `surface_original` = original substring `[char_start, char_end)`.
`unmatched_token_indices` = every token index with non-empty norm not covered by any emitted span.

## 8. Output contract

### 8.1 Schema

JSON object; keys MUST appear in exactly this order. All `weight`/`score` values are serialized with exactly 4 decimal places (`%.4f`).

```json
{
  "matcher_version": "1.0.0",
  "lexicon_version": "<64 hex chars>",
  "norm_version": "NORM-1",
  "tok_version": "TOK-1",
  "text": "<original input, unmodified>",
  "tokens": [
    {"i": 0, "surface": "Post", "norm": "post",
     "char_start": 0, "char_end": 4,
     "corrected_from": null, "edit_distance": 0}
  ],
  "spans": [
    {"span_id": 0, "tok_start": 0, "tok_end": 3,
     "char_start": 0, "char_end": 14,
     "surface_original": "Post-op nausea",
     "matched_norm": "post op nausea",
     "fuzzy": false,
     "candidates": [
       {"alias_id": 9, "node_id": "pub:dx.ponv", "alias_type": "informal",
        "weight": 0.7000, "score": 0.7000}
     ]}
  ],
  "unmatched_token_indices": [3],
  "flags": []
}
```

Token notes: `surface` is the original substring; a corrected token has `norm` = the corrected vocabulary token, `corrected_from` = its original norm, `edit_distance` > 0. Uncorrected tokens: `corrected_from: null`, `edit_distance: 0`. `span_id` = ordinal position after M5 sorting, from 0.

Flag object: `{"type":"fuzzy_suppressed","stage":"token"|"span","token_index":int,"nearest":str,"distance":int}`.

### 8.2 Determinism guarantee

For fixed `(text, lexicon_version, matcher_version)`, `match_json()` output MUST be byte-identical across runs, platforms, and thread counts.

### 8.3 Canonical serialization

UTF-8; separators `,` and `:` (no whitespace); keys in §8.1 order; arrays ordered: `tokens` by `i`; `spans` by `tok_start`; `candidates` by (score desc, alias_id asc); `flags` by (`token_index` asc, then stage `token` before `span`); non-ASCII emitted as raw UTF-8 (no `\uXXXX` escaping).

## 9. Errors

Raised as `LexError(code, message)`:

| Code | Condition |
|---|---|
| `E_INVALID_INPUT` | not a `str`, or contains lone surrogates |
| `E_INPUT_TOO_LONG` | > 4096 code points |
| `E_ARTIFACT_MISSING` | artifact dir incomplete/unreadable at `load()` |
| `E_ARTIFACT_MISMATCH` | artifact version fields ≠ matcher's compiled-in versions |

Matching MUST NOT raise for any valid input string.

## 10. Performance requirements

Assumptions: ≤ 5,000,000 active alias rows; vocabulary ≤ 2,000,000 tokens.

- `load()` ≤ 10 s cold; process RSS after load ≤ 4 GiB.
- `match()` p99 ≤ 2 ms for utterances ≤ 128 tokens, single performance core (Apple M4 / Zen 4 class).
- Zero network, GPU, or model inference at runtime. Permitted runtime dependencies: Python stdlib plus at most one compiled extension implementing AC/SymSpell.

## 11. Acceptance tests

An implementation is conformant iff it reproduces every vector below exactly (byte-identical canonical JSON), with `lexicon_version` compared against `^[0-9a-f]{64}$` rather than by value.

### 11.1 Test lexicon (all `status=active`, `origin=pub`, `derived_from=NULL`)

| alias_id | node_id | surface | alias_type | weight | case_mode | fuzzy_allowed |
|---|---|---|---|---|---|---|
| 1 | pub:proc.appendicectomy | appendicectomy | preferred | 1.00 | fold | 1 |
| 2 | pub:proc.appendicectomy | appendectomy | variant | 0.85 | fold | 1 |
| 3 | pub:dx.acute_appendicitis | acute appendicitis | preferred | 1.00 | fold | 1 |
| 4 | pub:dx.appendicitis | appendicitis | preferred | 1.00 | fold | 1 |
| 5 | pub:dx.pulm_embolism | PE | abbrev | 0.80 | caps | 0 |
| 6 | pub:dx.pleural_effusion | PE | abbrev | 0.80 | caps | 0 |
| 7 | pub:drug.hydralazine | hydralazine | preferred | 1.00 | fold | 0 |
| 8 | pub:dx.parkinson_disease | Parkinson disease | preferred | 1.00 | fold | 1 |
| 9 | pub:dx.ponv | post op nausea | informal | 0.70 | fold | 1 |

### 11.2 Vectors

**V1** — `text = "Post-op nausea after appendicectomy."` Full expected value:

```json
{"matcher_version":"1.0.0","lexicon_version":"<hex64>","norm_version":"NORM-1","tok_version":"TOK-1",
 "text":"Post-op nausea after appendicectomy.",
 "tokens":[
  {"i":0,"surface":"Post","norm":"post","char_start":0,"char_end":4,"corrected_from":null,"edit_distance":0},
  {"i":1,"surface":"op","norm":"op","char_start":5,"char_end":7,"corrected_from":null,"edit_distance":0},
  {"i":2,"surface":"nausea","norm":"nausea","char_start":8,"char_end":14,"corrected_from":null,"edit_distance":0},
  {"i":3,"surface":"after","norm":"after","char_start":15,"char_end":20,"corrected_from":null,"edit_distance":0},
  {"i":4,"surface":"appendicectomy","norm":"appendicectomy","char_start":21,"char_end":35,"corrected_from":null,"edit_distance":0}],
 "spans":[
  {"span_id":0,"tok_start":0,"tok_end":3,"char_start":0,"char_end":14,"surface_original":"Post-op nausea",
   "matched_norm":"post op nausea","fuzzy":false,
   "candidates":[{"alias_id":9,"node_id":"pub:dx.ponv","alias_type":"informal","weight":0.7000,"score":0.7000}]},
  {"span_id":1,"tok_start":4,"tok_end":5,"char_start":21,"char_end":35,"surface_original":"appendicectomy",
   "matched_norm":"appendicectomy","fuzzy":false,
   "candidates":[{"alias_id":1,"node_id":"pub:proc.appendicectomy","alias_type":"preferred","weight":1.0000,"score":1.0000}]}],
 "unmatched_token_indices":[3],
 "flags":[]}
```

(Pretty-printed here for readability; conformance is against the §8.3 canonical form.)

**V2** — `"Query PE."` → 1 span at tok `[1,2)`, `fuzzy:false`, candidates `[alias 5 (score 0.8000), alias 6 (score 0.8000)]` — equal scores, tie broken by `alias_id` asc. `unmatched_token_indices:[0]`.

**V3** — `"ruled out pe"` → 0 spans (`caps` fails for lowercase `pe`); `unmatched_token_indices:[0,1,2]`; no flags. Case-mode failure is not a flagged event.

**V4** — `"urgent apendicectomy"` → token 1 is OOV, length 13, `d_max=2`; unique minimum-distance hit `appendicectomy` (d=1, targetable) → corrected. 1 span tok `[1,2)`, `fuzzy:true`, candidate alias 1 with `score:0.8000` (1.00 × f(1)). Token 1: `"norm":"appendicectomy","corrected_from":"apendicectomy","edit_distance":1`. `unmatched_token_indices:[0]`.

**V5** — `"started hydralazne"` → token 1 OOV; nearest hit `hydralazine` (d=1) has `fuzzy_class=suppressed` → NO correction, no span. Flag: `{"type":"fuzzy_suppressed","stage":"token","token_index":1,"nearest":"hydralazine","distance":1}`. `unmatched_token_indices:[0,1]`.

**V6** — `"acute appendicitis confirmed"` → exact matches: alias 3 at `[0,2)`, alias 4 at `[1,2)`. M5 selects alias 3 (smaller tok_start, larger tok_end); alias 4's match intersects and is deleted. 1 span; `unmatched_token_indices:[2]`.

**V7** — `"Parkinson's disease"` → tokens `Parkinson's`(norm `parkinson`), `disease`. 1 span tok `[0,2)`, `matched_norm:"parkinson disease"`, candidate alias 8, `score:1.0000`.

**V8** — `""` → `tokens:[]`, `spans:[]`, `unmatched_token_indices:[]`, `flags:[]`; no error.

## Appendix A — EN_STOP_V1 (fixed list, 44 entries)

```
a an and are as at be been but by for from had has have he if in is it
no not of off on or out over she so than that the then they this to
under up was we were with you
```

## Appendix B — Deferred to future revisions

- Real-word error correction (valid token typed for another).
- Locale variants beyond curated `variant` rows; non-English lexicons.
- Alias generation pipeline spec (inflections, UK/US expansion, colloquialism import) — VINKONA-LEX-02.
- Automaton/index binary format standardization (currently implementation-defined).
