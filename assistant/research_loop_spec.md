# Implementation Spec: Research → Learning Loop (Vinkona ⇄ knowledge-host)

**Goal.** Close the research/learning cycle. When a conversation leaves a question inadequately
answered, Vinkona (at idle) poses a well-formed research question, attempts to answer it, and hands
the result to the knowledge-host to be distilled into durable **cards** — so the *next* time the
topic arises the answer is already in `kb_ask`. Genuinely hard questions Vinkona can't crack are
parked for a separate deep-search tool. The two sides never have to be online together.

**Scope.** **Procedural / topical** knowledge only — "how do I X", "what is Y", reference-shaped
things. Personal / relational / mail-derived material is explicitly **out of scope**: it stays in
Vinkona's `memory.db` and feeds the personal graph (see [[personal-graph]]). The privacy boundary is
unchanged — the only thing that crosses to the host is topical source text, never who-knows-whom.

**Transport is the filesystem, not a socket.** Vinkona writes documents into shared folders; the host
mines them on its own cadence. This is deliberate: a folder is a durable, append-only queue, so
neither process has to be up when the other acts. A network share (NFS/SMB) is fine — the contract
is "files in a directory," nothing more.

**Security.** Source text is **untrusted** (hostile-by-default web/file content). It is sanitised on
ingest (the host already does this), stored at **low trust_weight**, and its cards are subordinate
to curated ones (§6). Filenames are opaque data, never interpolated into a shell or an LM prompt.

---

## 0. The core model

```
                    ┌──────────────────────── Vinkona (idle reviewer) ─────────────────────────┐
   chat logs  ─►  reflect (past watermark) ─►  pose better-phrased question  ─►  attempt answer
                                                                                       │
                                             solved (confident + sourced)  ───────────┤
                                                                                       ▼
                                                             solved/<hash>.md  ──►  [host]
                                             unsolved (couldn't) ──► unsolved/<hash>.md   │
                    └──────────────────────────────────────────────────────────────────┘  │
                                                                                            ▼
   [host]   ingest solved/  ─►  chunk+embed  ─►  distill --watch  ─►  cards  ─►  kb_ask ──┐
                                                                                          │
   [your deep-search tool]  drains unsolved/  ─►  solves  ─►  writes solved/ + clears twin┘
```

Three independent actors, coupled only by two directories:

| Actor | Reads | Writes |
|---|---|---|
| **Vinkona idle reviewer** | chat logs | `solved/`, `unsolved/` |
| **knowledge-host** | `solved/` | cards (its own DB) → served via `kb_ask` |
| **deep-search tool** (yours, later) | `unsolved/` | `solved/`, deletes the `unsolved/` twin |

**Read-back is `kb_ask` only.** Vinkona never reads its own dropped files back. The folders are
write-only outboxes; the distilled card is the long-term read path. (This is what lets us drop the
answer-cache and the freshness/TTL machinery entirely.)

---

## 1. The folder contract

Two directories on the shared path (configurable):

- `research/solved/`   — Vinkona's answered questions, the host's ingest input.
- `research/unsolved/` — Vinkona's stuck questions, the deep-search tool's input.

**Filename = `<hash>.md`**, where `hash = sha1(normalize(question))[:16]`.

`normalize()` = lowercase, collapse whitespace, strip trailing punctuation (and optionally trim a
small stop-word set). Normalising **before** hashing collapses "How do I X?" / "how do i x" onto one
file, so trivial paraphrases don't split into two documents. Non-trivial rephrasings *deliberately*
produce different hashes — see §5.

The filename is the **exact-dedup key**. It is opaque; all human-readable metadata lives *inside*
the file (§2).

---

## 2. Document schema

A single markdown file, front-matter + headed sections. Headings matter: the host chunks by
heading, and the `# Question` is lifted as the distillation frame / gap-seed (§6).

```markdown
---
provenance: vinkona
trust: low
kind: procedural            # procedural | topical
question_hash: 9f2a…        # = the filename stem, for integrity
kb_query: "how do I repot a phalaenopsis"   # VERBATIM original kb_ask/kb_search query, if this
                                            # grew out of a kb miss — the gap-join key (§6)
created: 2026-07-02T14:03:00Z
updated: 2026-07-02T14:03:00Z
sources:                    # membership list — lets Vinkona tell "new source?" cheaply (§4)
  - id: sha1:ab12…          # hash of the source text/url
    url: https://…
    fetched: 2026-07-02T14:01:00Z
---

# Question
How do I repot a phalaenopsis orchid without killing it?

## Answer
<Vinkona's synthesised answer — the distilled "what to do", grounded in the sources below.>

## Sources
### [ab12…] https://…
<the relevant extracted source prose, sanitised. This is what actually gets chunked + distilled.>

### [cd34…] https://…
<a second source, if found>
```

The `## Answer` is Vinkona's synthesis; the `## Sources` blocks are the evidence the host distils. Both
are sanitised. `kb_query` is optional but valuable: when the question grew out of a mechanical kb
miss, carrying the *verbatim* original query lets the host close the very gap that spawned it (§6).

---

## 3. Vinkona side — the idle reviewer

A new idle job (big LM, never the voice path). Per cycle:

1. **Reflect past the watermark.** Read chat-log turns newer than `research.watermark` (the only
   persistent state this loop keeps — a transcript position in `memory.db`). Judge, per exchange,
   *was this actually addressed?* This is a **conversation-quality** judgment, richer than the
   host's mechanical `log_gap` (band-'none' only), so it catches the weak near-misses that gap log
   drops (see [[search-failure-gap-detection]]). This reviewer **supersedes** leaning on the host
   gap log for question generation.
2. **Pose the question well.** For each inadequately-addressed exchange, formulate a crisp,
   self-contained research question — rephrasing for clarity is encouraged (§5). Scope-filter here:
   drop anything personal/relational/mail-shaped (that's the personal graph's job).
3. **Attempt it** with Vinkona's existing research tools (web_search/web_fetch, kb_search, docs).
4. **Route by outcome:**
   - **Solved** (a confident answer with ≥1 usable source) → write `solved/<hash>.md` (§4).
   - **Unsolved** (couldn't ground it) → write `unsolved/<hash>.md`, unless that hash already
     exists in `solved/` or `unsolved/`.
5. **Advance the watermark.**

Because output dedups against the folders (§1, §5), re-examining logs is cheap and idempotent — a
re-derived question is a no-op, never a duplicate.

---

## 4. Writing a solved doc (and repeats)

Before writing, check `solved/<hash>.md`:

- **Absent** → write it fresh (front-matter + `# Question` / `## Answer` / `## Sources`).
- **Present, and every source we found is already in its `sources:` list** → **no-op.** Do not
  rewrite; a byte-identical file is a host no-op and rewriting only churns the manifest.
- **Present, but we found a *new* source** → **augment**: append the source block, add its id to
  `sources:`, revise `## Answer` if it shifted, bump `updated`. The content hash changes → the host
  re-ingests and re-distills (its re-ingest is idempotent, upserting chunks under stable ids), and
  the extra grounding makes a *stronger* card.

So a "repeat" is handled the right way by construction: same question is one file; new evidence
accumulates into it; the host reconciles the richer source set. Vinkona tells "new source?" by hashing
each source's text/url and checking membership against the front-matter `sources:` list.

---

## 5. Dedup: filesystem = exact, host = semantic

Two layers, cleanly separated:

- **Exact dedup — the filesystem.** Same normalised question → same filename → the host skips
  byte-identical writes. This is all the filename hash is for.
- **Semantic dedup — the host.** A *non-trivially rephrased* question is a different file and a
  different chunk set, but it distils onto the **same concept nodes** (ids are content hashes) and
  is reconciled by the host's link/refine/dedup machinery. Two good phrasings of one gap converge at
  the **card** level, not the file level.

This is why rephrasing is a feature, not a duplicate to suppress: a better-worded second attempt may
succeed where the first was stuck, and the host merges the results. Vinkona must **not** try to be
clever about paraphrase-matching — that's the host's job.

---

## 6. Host side

Almost no new host code — the ingest pipeline is already directory-based (`os.walk` over a sources
root) and distill already has a `--watch` mode.

1. **Point ingest at `solved/`.** Add the folder to the host's `sources`, tagged into a dedicated
   **`vinkona` bundle** at **low trust_weight**. Ingest chunks by heading (so `## Sources` blocks
   become the retrievable/distillable chunks) and embeds them.
2. **Question-framed distillation.** When distilling a chunk from the `vinkona` bundle, lift the
   doc's `# Question` (and, if present, `kb_query`) and inject it into the extractor prompt as the
   frame — so "how do I X" + source prose yields a **procedure card answering X**, not a generic
   concept. If `kb_query` matches an open `knowledge_gap`, **close that gap** when a card grounds it
   — literally closing the loop that a kb miss opened.
3. **Run `distill --watch`** as a standing daemon (host operator starts it once). Cards appear on
   the host's own cadence — Vinkona submits and walks away. This is the "host auto-cadence" decision.
4. **Encryption at rest.** The `vinkona` bundle is flagged **sensitive** → SQLCipher ciphertext at
   rest (base stays clear). **Prereqs on the host box:** a SQLCipher driver + `$KNOWLEDGEHOST_DB_KEY`
   (or `db_key_file`), or the sensitive bundle **fails loud** — it never silently writes plaintext.
   Note the drop folder itself holds cleartext `.md`s: keep it inside the same protected area.
5. **Read-back needs the bundle in the live scenario (the gotcha).** Cards distilled into the
   `vinkona` bundle are only queryable once that bundle is folded into the active scenario's working
   DB ("ship granular, run consolidated"). So the running scenario must **include the `vinkona`
   bundle**, with a periodic `reload`/scenario-apply after distillation — otherwise Vinkona distils
   into a bundle it never reads back.

**Trust posture (distil-but-subordinate).** vinkona-sourced cards are distilled and surfaced, tagged
`provenance: vinkona`, but **banded below curated cards** — a textbook card always wins a tie. They
enrich coverage without contaminating the high-trust core.

---

## 7. The unsolved queue (your deep-search tool, later)

`unsolved/` is a live work-queue Vinkona only ever writes into. The deep-search tool **owns draining
it**:

- Picks up `unsolved/<hash>.md`, attempts resolution with heavy/deep internet search.
- On success: writes `solved/<hash>.md` (same schema, so the host mines it identically) **and
  deletes the `unsolved/<hash>.md` twin.**

That ownership rule keeps `unsolved/` a true queue and means Vinkona never has to reconcile a
solved/unsolved twin. The tool is out of scope for this spec beyond the folder contract it must
honour (§1, §2).

---

## 8. State, config, and what's *not* here

- **Persistent state:** exactly one thing — `research.watermark` (a transcript position) in
  `memory.db`. No answer-cache, no per-question index (the folders are the ledger), no TTL.
- **Config (Vinkona):** `research.loop.enabled`, `research.loop.solved_dir`, `research.loop.unsolved_dir`,
  `research.loop.confidence_threshold` (solved vs unsolved cut), scope filters.
- **Config (host):** `sources` includes the solved dir; the `vinkona` bundle mapping (low trust,
  sensitive); `distill --watch` interval; scenario includes the `vinkona` bundle.
- **Explicitly out of scope:** personal/relational data (→ personal graph), any read-back of Vinkona's
  own dropped files (→ `kb_ask`), and paraphrase-matching in Vinkona (→ host reconciliation).

---

## Open questions

- **Watermark vs re-reflection.** Strict watermark = each turn reviewed once. Do we ever want to
  *re-*reflect on old logs with a better model/prompt later? If so, the watermark needs a
  "re-open past N days" override — cheap, since output still dedups.
- **Confidence threshold calibration.** What's the bar for "solved"? Too low pollutes the KB with
  weak cards (mitigated by low trust); too high floods `unsolved/`. Start conservative, tune on the
  gap-closure rate.
- **Source retention.** Do we ever prune old `solved/` docs once distilled, or keep them as the
  audit trail / re-distill corpus? (Keeping them is cheap and lets a distiller upgrade re-mine.)
```
