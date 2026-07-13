# Vinur — a local grounded-knowledge tool (phased design spec)

**Status:** design spec — realized as [Vinur](https://github.com/dantroline-sys/vinur), its own repository since 2026-07-13
**Thesis:** give the reasoning LM **grounded structure + a confidence signal**, so it
*reasons, hedges, or abstains* — and never confabulates when it's at a loss. Retrieval
hands over a grounded candidate set with provenance; it never diagnoses on its own. This is
the throughline at every phase below; the cheap version of it ships in Phase 1, the rigorous
version arrives in Phase 3.

**What:** a standalone service giving Vinkona a large, local general-knowledge base she can
search mid-conversation — a Wikipedia snapshot plus the user's books, journals, papers and
miscellany — returning **cited** answers. Built as its own tool host (same contract as
[MAC_TOOLS.md](MAC_TOOLS.md) / [MUSIC.md](MUSIC.md)), aggregated by the existing `MultiHost`
so the fast LM can call it like any other tool.

This unifies two earlier docs: the simple passage-retrieval KB and the *Metacognitive
Knowledge Retrieval* (v5) design. They are not rivals — **v5 Phase 1 ≈ the passage KB**, and
v5's structured graph / epistemic-regime machinery is the Phase 2–3 ambition you grow into
only once the simple version is in daily use and you can see where passage-RAG falls short.

> **Scope — non-episodic knowledge only.** Textbook / literary / reference. Personal,
> episodic and relational memory (who the user is, the relationship, commitments, calendar,
> news, weather) stays in the **separate** `memories` / people / affect / ambient stores.
> The two meet only where the big LM reads both. (See §"Separation" and
> [MEMORY_CONSOLIDATION.md](MEMORY_CONSOLIDATION.md).)

---

## Topology — where it runs (all phases)

Run the Knowledge Host on the **Linux GPU box** (128 GB / fast-Intel / fast-NVMe),
co-located with the nomic embed endpoint (`127.0.0.1:11437`) and the storage. Bind it to
**`127.0.0.1:8770`**. Same machine as the cascade ⇒ **no SSH tunnel** needed (unlike the Mac
host), but stay localhost-bound; never on the LAN. Register it as a second tool host
alongside the Mac host — the code already builds a `MultiHost` when more than one is
configured (see where the music host is added in `cascade_server.py`).

**Online retrieval is CPU-only** (13900KF, AVX2 — size any ONNX/quantized models to AVX2);
**offline distillation may use the GPU**. Two halves, very different shapes:
- an **offline ingestion/distillation pipeline** — heavy, batch, run on demand / monthly;
- a **query service** (the tool) — light, fast, always up.

---

## Phasing at a glance

| Phase | Corpus size | Adds | It becomes |
|---|---|---|---|
| **1 — ship now** | ≤ ~10–20 GB | hybrid passage retrieval, cited answers, **grounding-confidence + abstention**, source registry (trust + regime tag + date) | a reliable, honest passage-RAG KB |
| **2 — grow into** | ~20–200 GB | structured **distillation** (nodes / procedure-cards / typed-edge graph), **read-before-write reconciliation**, support sets, **epistemic regimes** + write-time firewall, self-retrieval eval | a structured knowledge graph that returns *digested structure*, not prose |
| **3 — scale** | ~200 GB–1 TB+ | on-disk ANN (FAISS IVF-PQ) + Tantivy, **read-time rigor knob** (firewall filter + recency/trust adjudication), golden-query/audit/drift monitoring, retraction/supersession, gap-driven mining, Rust hot paths | the full metacognitive system |

The **card/graph tier is constant** at every size (it's proportional to genuine how/why/
relational content, which stays small); only the **raw tier backend** escalates. Keep the
raw-tier interface (`encode / add / search / fetch`) clean so the backend can be swapped per
phase. **Do not build the TB stack — or the regime engine — up front.**

---

## Phase 1 — passage retrieval + honest grounding (ship now)

The KB that earns its keep on day one, and already refuses to confabulate.

### Store & index
**LanceDB** (embedded, on-disk, memory-maps from NVMe; vector ANN *and* BM25 in one store →
native hybrid). One table; key columns:

| column | use |
|---|---|
| `id` | stable chunk id = hash(path + section + text) — idempotent re-ingest |
| `source_type` | `wikipedia` \| `pdf` \| `epub` \| `html` \| `text` |
| `title`, `section` | citation + heading path ("History > Founding") |
| `path_or_url` | provenance |
| `text` | the chunk (returned & fed to the LM) |
| `vector` | nomic embedding (768-d, **`search_document:`** prefixed — consistent with A.1) |
| `doc_id` → `source_registry` | trust / regime / date (below) |

*Alternative (more manual, no extra service):* FAISS flat/HNSW + **SQLite FTS5**. The
non-negotiables are **ANN, never brute force** (the `memories` store's `_emb_mat @ qv` is
fine for thousands, fatal for millions) and **hybrid dense+sparse** — the fusion is the
single biggest quality lever at scale.

### Source registry (cheap now, pays off later)
Add at ingest even in Phase 1, so you never have to re-ingest to get it:
```sql
source_registry(doc_id PRIMARY KEY, title, source_type,
  trust_weight REAL,   -- 0..1 prior by type (peer_reviewed/textbook > reference > web > unknown)
  regime TEXT,         -- empirical|conventional|fictional|interpretive|historical (default by source_type)
  pub_date, status DEFAULT 'active')   -- active | quarantined | retracted
```
In Phase 1 `regime` is just `source_type → regime` (a journal ⇒ empirical, a novel ⇒
fictional). It's used lightly now (framing, §"beats web search"), fully in Phase 2/3.

### Ingestion pipeline (offline, incremental)
A manifest (`path, content_hash, mtime, version, status`) makes every run incremental; a
monthly Wikipedia refresh re-versions and swaps atomically.
- **Wikipedia — use Kiwix ZIM** (pre-rendered, cleaned, sectioned HTML; read via `libzim`,
  split on `<h2>/<h3>`). Sidesteps wikitext. *Raw `enwiki-…-pages-articles.xml.bz2` via
  `mwxml` + `mwparserfromhell` is the alternative if you want what the ZIM render drops —
  more control, much more edge-case handling.*
- **PDFs:** text via **PyMuPDF**; **OCR fallback** (`ocrmypdf`/`tesseract`) only for pages
  with no text layer. Sections from the PDF outline or font/heading heuristics.
- **EPUB** → `ebooklib`; **HTML** → `trafilatura`; **txt/md** → direct.
- **Folder tree:** recursive crawl via the manifest; **treat filenames as opaque data**
  (never shell- or prompt-interpolate them — the file-scraper injection surface).
- **Chunk by heading/section**, ~512 tokens with ~64 overlap on section boundaries; carry
  metadata + stable `id`. Embed via the nomic endpoint (`search_document:`), batched. The
  full-Wikipedia embed is a one-time hours–days job; incremental runs after are cheap.

### Query tool
`GET /tools` advertises `kb_search`:
```json
{ "name": "kb_search",
  "description": "Search the local general-knowledge base (Wikipedia snapshot + the user's books, papers and documents). Returns cited passages. For established/reference knowledge; offline, so it may be months stale for current events.",
  "parameters": { "type":"object", "properties": {
    "query":  {"type":"string"},
    "k":      {"type":"integer","description":"max passages (default 5)"},
    "intent": {"type":"string","description":"why you're asking, to focus ranking (stays local)"},
    "filters":{"type":"object","description":"{source_type, title}"} },
    "required":["query"] } }
```
**Retrieval flow:** embed query (`search_query:`) + BM25 → **RRF fuse** → shortlist (~30) →
**rerank** (the planned reranker; the `intent`/"favoritism" payload conditions relevance
**locally and never enters any outbound query**) → top-`k` passages, each
`{text, title, section, path_or_url, score, regime}`, **always cited**.

### Grounding-confidence + abstention (the anti-confabulation gate — present from day one)
The top rerank/RRF score is a **confidence signal**, not just a sort key:
```
high     → answer directly, cite
medium   → answer, note single-/older-source caveat
low      → answer hedged, flag weak grounding
none     → DO NOT confabulate: "no grounded answer in the knowledge base"; log a gap
```
Low best-score ⇒ fall back to web search rather than answer from a weak passage. This is the
cheap version of v5's read-time confidence model; Phase 3 composes it from more signals.

### Integration with Vinkona
- **Live:** registered via `MultiHost`; the fast LM calls `kb_search` mid-turn. Results are
  **fenced as low-trust reference** (`safety.sanitize_external` + `wrap_untrusted`) before
  the LM uses them, and cited. Same trust tier as background-research memories.
- **Research/learning:** in `research_worker`, try `kb_search` **before** the Wikipedia API
  and the SearXNG web fallback (local-first). Reuse `memory.store_document(...)` +
  `memory.learn(...)` exactly as web results are handled today.
- **Knowledge gaps:** `none`-confidence queries land in a `knowledge_gaps` log → drives
  targeted distillation or source acquisition later.

### Will it beat web search?
Honestly, *partly — by design*. **Yes for evergreen/reference** (local, offline, private, no
rate limits, curated, quality under your control). **No for recency** (a monthly snapshot
can't know today's news). So: **KB-first for evergreen, web fallback for recency or low
KB-confidence.** That split is the feature — the KB takes the bulk of lookups off the
network; the web stays the escape hatch for the genuinely current.

---

## Phase 2 — structured distillation + reconciliation (grow into)

Where it stops being passage-RAG and starts handing the model *digested structure*. This is
the heart of v5; get reconciliation right **before** scaling volume.

### Three substrates over one shared set of canonical nodes
| Interrogative | Served by | Mechanism |
|---|---|---|
| **what** | declarative nodes | node lookup (+ neighbours) |
| **how** | procedure cards | self-contained ordered-step nodes |
| **why** (mechanistic / diagnostic) | typed-edge causal graph | forward walk / **backward** abductive walk + context-scoring |
| **what-if** | causal graph | forward walk from a hypothesised cause |
| **who / where / which-kind** | typed-edge graph (epistemic / spatial / taxonomic) | typed walk |
| **what-else / which-better** | operators | neighbourhood expansion / attribute comparison |

Shared canonical nodes link the substrates (a *why*-node points to a *what*-node and a
*how*-node). Principle: **distil offline, stay dumb online.**

### The distillation worker
```
chunk → content router (cheap multi-label: procedural|causal|relational|declarative|none)
      → grammar-constrained extractor (GPU LM) → candidate cards / edges / nodes
      → link_to_node (identity policy, below)
      → reconcile against the LIVE KB via the production query path  (read-before-write)
      → apply: corroborate | enrich | insert-distinct | keep-contradiction | insert
```
Running the candidate's own generated question through the live query path **dogfoods
retrieval**: an item that can't be retrieved by its own question is a signal that surface
generation is weak.

### Node identity — bias toward NOT merging
Over-merging is destructive (conflating two concepts loses provenance and corrupts
diagnosis); under-merging is recoverable. So: high sim + alias agreement ⇒ same node;
ambiguous ⇒ **distinct** node (+ an `is_a` edge if one generalises the other) and queue for
adjudication; low ⇒ new node.

### Reconciliation — never clobber
Cheap geometric banding gates the cost: identical hash ⇒ auto-corroborate; clearly far ⇒
auto-insert; only the **ambiguous middle** costs an LM call, which classifies the candidate
five ways: **duplicate** (add to support), **refinement** (merge-enrich modifiers),
**novel-distinct** (insert + `alternative_to`/`context_variant_of`, recording *why*),
**contradiction** (keep both active, link `disagrees_with` — arbitration deferred to read
time), **no-match** (insert). Contradictions become **edges, not overwrites**.

### Support sets (write keeps evidence, not a verdict)
The one thing the write path computes — because an LLM eyeballing a list miscounts, and
citation cascades must be discounted — is the **independence-cleaned support set**: cluster
support by evidence-span similarity so near-identical spans (copying) count once; key by
`doc_id` so re-ingestion is idempotent. **No verdict is baked in.** `strength` becomes an
*optional read-time ranking signal* computed from this set only when a query asks for rigor
(Phase 3), not a stored truth.

### Epistemic regimes + the write-time firewall (safety-critical)
Not all knowledge is the same kind of truth. Reconciliation, the confidence model and
presentation are **parameterised by regime** (tagged per source, overridable per claim):

| Regime | Truth anchor | Corroboration | "Contradiction" means | Rendered as |
|---|---|---|---|---|
| empirical | reality | yes | resolve by trust/recency | fact, hedged by strength |
| conventional | a practice | within a context | **context-variant, not conflict** | "For a ⟨context⟩, …" |
| fictional | the work | within a world | cross-world ⇒ **coexist** | "In ⟨work⟩, …" |
| interpretive | the author | no (attribute) | **present all positions** | "⟨author⟩ argues …" |
| historical | reality + period | empirical parts only | facts resolve, readings debated | "As understood in ⟨period⟩, …" |

**Write-time firewall (always on):** corroboration/contradiction operate **only within the
same regime + scope** — a novel's herbal cure can never raise a clinical claim's support, nor
an essayist's opinion lower a study's. This is the structured, sharper form of the fence
Vinkona *already* has ("world knowledge is reference-only, never drives a tool call"); regime
is the governing switch that selects the right ruleset.

### Eval from the start of Phase 2
**Self-retrieval check:** every new item must be retrievable by its own generated questions
(top-k) or it's flagged. You cannot run millions of extractions/merges blind.

---

## Phase 3 — scale, rigor & monitoring

- **Raw tier → on-disk:** FAISS **IVF-PQ + OnDiskInvertedLists** (PQ compresses 768-d to
  ~32–64 B/vector → ~10 M vectors in <1–2 GB RAM) + **Tantivy** lexical. Codebook-retrain
  discipline.
- **Read-time rigor knob (the one knob):** *low* (default — most questions) just attaches
  the provenance bundle and lets the assistant weigh it. *High* (a drug interaction) switches
  on two things: the **firewall as a hard candidate-pool filter** (non-empirical items
  *excluded*, not merely flagged — for a clinical answer you want fiction *absent*, not
  present-but-hopefully-ignored) and **recency/trust/strength adjudication** over the support
  set. **Stakes, not regime, set rigor**; the supervisor tier decides.
- **Grounding confidence** composed at answer time from: top rerank score, support-set
  independence, presence of `disagrees_with`, card/edge vs raw-only, and (high rigor) the
  adjudicated `strength` — feeding the same answer/hedge/abstain ladder as Phase 1.
- **Monitoring:** golden-query set (recall@k / MRR, alarm on regression), merge/contradiction
  audits, inferred-edge hallucination audits, embedding-drift → PQ-retrain trigger.
- **Lifecycle:** retraction (cascade `status='retracted'`, recompute strengths),
  supersession (`supersedes` edges; superseded items kept as history, downranked),
  node-merge adjudication queue, gap-log-driven mining.
- **Rust hot paths** (encode, fuse, fetch, traversal, fit-scoring) via PyO3 once Python is
  the bottleneck.

---

## Security (all phases)

- **All ingested content is UNTRUSTED** (Wikipedia, random PDFs, downloaded books carry
  injection). The tool returns **data, never instructions**; everything is fenced before any
  LM reads it. **Grammar-constrained extraction output** (Phase 2) structurally bounds what a
  hostile chunk can do — it neutralises most injection by construction.
- **Trust & quarantine:** unknown/low-trust sources get low `trust_weight` and can be
  `quarantined` from corroboration; the firewall keeps regimes from contaminating each other.
- **Parser hardening:** malicious PDFs/EPUBs exploit parser bugs — keep PyMuPDF/Tesseract
  patched, parse in a constrained process (no network egress during parse).
- **Path safety** on the folder crawl (opaque filenames). **Intent stays local** (rerank
  signal only, never an outbound query — the egress boundary `outbound_query` enforces).
- Service is **read-only and localhost-bound**.

---

## Separation from `memories` — and what to back-port into it

This KB is the *world-knowledge brain*; `memories` + people + affect + ambient stay the
*personal/relational self*. Keep them separate — pouring an encyclopedia into the curated,
all-in-RAM personal store breaks its RAM, hot path and gardening, and dilutes its top-k.

But three v5 **principles** (not the system) are worth lifting into the **personal** store,
as separate, related work:
1. **Contradiction-as-edge, arbitrate at read time.** When a later personal fact conflicts
   with an earlier one ("I love my job" → months later "thinking of leaving"), keep both,
   surface the tension, let recency speak — instead of clobbering. Fits Vinkona's already
   non-destructive synthesis ethos.
2. **Grounding-confidence + abstention** on personal recall too: better a clean "I don't
   think you've told me that" than a confident guess.
3. **Read-before-write / self-retrieval QA:** an item that can't be retrieved by its own
   question signals weak surfacing — a cheap quality check the synthesis/consolidation passes
   can borrow.

---

## Build order (unified)

**Phase 1 (now):** 1) ingestion + chunking + `source_registry` (incl. regime tag);
2) LanceDB hybrid index, validate RAM on a real subset; 3) `kb_search` tool +
intent-conditioned rerank; 4) **grounding-confidence + abstention + gap log**; 5) wire into
`MultiHost` and slot ahead of web in `research_worker`. Prove the whole path on a small PDF
set *before* embedding all of Wikipedia.
**Phase 2 (grow into):** 6) canonical node store + `link_to_node`; 7) content router +
extractors (grammar-constrained); 8) **reconciliation** (banding → five-way → support sets) +
regimes + write-time firewall; 9) surface generation + self-retrieval eval.
**Phase 3 (scale past ~200 GB):** 10) IVF-PQ + Tantivy; 11) rigor knob (firewall filter +
adjudication); 12) golden queries / audits / drift; 13) retraction / supersession /
adjudication; 14) Rust hot paths.

## Out of scope / deferred
- **Episodic/personal/affective memory** — the `memories`/people/affect/ambient stores.
- **A separate embed model for the KB** — reuse the prefixed nomic (one space for both stores).
- **Cross-encoder reranking on the live `memories` path** — wrong place (latency, tiny pool).
- **Query decomposition / multi-hop planner** above the intent router (supervisor-LLM job).
- **Explicit negative/contraindication edges** ("X does *not* cause Y").
- **Versioned embeddings / encoder-migration reindex** for the raw tier.
