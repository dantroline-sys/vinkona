# Implementation Spec: Metacognitive Knowledge Retrieval System (v7)

**Goal.** Turn a growing, heterogeneous corpus of books/journals/literature/reference into a fast, CPU-served *knowledge* layer that holds **multiple kinds of truth** (factual, conventional, fictional, interpretive, historical) that answers **what / how / why / who / where / which / what-if** questions accurately by retrieving **pre-digested structure**, not by paraphrasing source prose. Retrieval hands the inference model a grounded candidate set *with a confidence signal*; the model reasons, hedges, or abstains. Retrieval never diagnoses on its own.

**Scope.** Non-episodic ("metacognitive") knowledge only — textbook/literary/reference. Episodic data (news, weather, calendar) is a separate system, out of scope.

**Hardware.** Online retrieval is CPU-only (13900KF, AVX2; 128 GB DDR4; fast NVMe). Offline distillation may use a GPU.

**v7 changes:** the **graph tier is now import-scale** (hundreds of millions of nodes from bulk-imported KGs), so storage is respecified (§1, §3): integer-ID/CSR backbone + binary-quantized embeddings + reranker + lexical-first lookup, ~20 GB inside a 64 GB budget. Adds **bulk external-KG import** as an ingestion mode (§5.1) and a `has_reference` flag distinguishing chapter-and-verse-backed claims from dataset-asserted ones (§4.3, §12).

**v6 changes:** **anticipatory reasoning** (§11) — a **divergent** layer *generates* a possibility space unbounded (means-ends decomposition + functional substitution, no self-censoring), a separate **convergent** layer *judges* it (ranks-and-annotates, mines rejects, never deletes), and a belief-state next-best-question loop *navigates* it. The only hard rule is honest labelling of the speculative; the regime firewall keeps the brainstorm out of the empirical lane. Read-time compile, ~no new storage.

**v5 changes:** **evidence/verdict split** — the write path only *preserves and structures* provenance (source, regime, date, trust, independence-cleaned support); the *verdict* (weigh by recency/trust, or not) is deferred to read time and gated by a per-query **rigor level** (§10). `strength` is demoted to an optional read-time ranking signal; the cross-regime **firewall becomes a read-time filter** high-rigor queries switch on. Net: less eager computation, more flexibility.

**v4 changes:** the truth model is no longer mono-epistemic — **epistemic regimes** (§9) make reconciliation, the confidence model, and presentation switch by *truth regime* (empirical / conventional / fictional / interpretive / historical), with a **cross-regime firewall** so non-empirical sources can never corroborate or contradict empirical claims.

**Prior changes retained:** reasoned reconciliation (§9); provenance/source-trust + temporal validity; grounding confidence + abstention (§12); node-identity policy (§9.4); evaluation & monitoring (§13); corpus-as-data hardening (§5, §14).

---

## 0. The core model

Three substrates over **one shared set of canonical concept nodes**, plus **operators** that run across them.

| Interrogative | Served by | Mechanism |
|---|---|---|
| **what** | Declarative nodes | node lookup (+ neighbours) |
| **how** | Procedure cards | self-contained ordered-step nodes |
| **why** — mechanistic | Typed-edge graph (causal) | forward causal walk |
| **why** — diagnostic | Typed-edge graph (causal) | **backward** walk + context-scoring (abduction) |
| **what-if** | Typed-edge graph (causal) | forward walk from hypothesised cause |
| **who / where** | Typed-edge graph (epistemic/spatial) | typed-edge walk |
| **which-kind / how-relates** | Typed-edge graph (taxonomic) | up/down is-a walk |
| **when** | cross-cutting | entity attribute · step order · edge modifier |
| **how much / many** | cross-cutting | parametric attribute in step text + propositions |
| **what else** | **operator** | neighbourhood expansion |
| **which (better)** | **operator** | comparison over discriminating attributes |

Principles:
1. **Retrieve structure, let the model reason** — and tell it how confident the grounding is.
2. **Distil offline, stay dumb online.**
3. **Shared canonical nodes link the substrates** (a *why*-node points to a *what*-node and a *how*-node).
4. **The base is epistemically honest** — it records corroboration, contradiction, source trust, and recency; it never lets a new write silently clobber an old claim.
5. **Truth is regime-relative** (§9). Empirical, conventional, fictional, interpretive, and historical knowledge obey *different* rules for corroboration, contradiction, and recency. The engine applies the right rules per regime and never lets regimes contaminate each other.

---

## 1. Scale constraint & staged rollout

Corpus grows toward a possible ~1 TB ceiling (~200B tokens, hundreds of millions of chunks). **Do not build the TB stack up front.** The card/graph tier (RAM) is constant at every size; only the **raw tier** backend escalates. Keep the raw-tier interface (`encode/add/search/fetch`) clean and swap the backend:

- **Phase 1 — ≤~10–20 GB**: raw embeddings int8 in RAM, flat/HNSW, FTS5. No IVF-PQ, no router. Distil fairly greedily.
- **Phase 2 — ~20–200 GB**: add **content router** + **coverage clustering**; priority-ranked distillation.
- **Phase 3 — ~200 GB–1 TB+**: **FAISS IVF-PQ + OnDiskInvertedLists**, **Tantivy** raw lexical, codebook-retrain discipline, full demand-driven mining.

**Two storage tiers:**
- **Card/graph tier (RAM, ~64 GB budget).** No longer "small": bulk-importing external KGs (ConceptNet, ATOMIC, GLUCOSE, CauseNet, ASER, SemMedDB) pushes this to **hundreds of millions of nodes/edges**, so the old "fp32 in RAM" plan does not hold. Representation instead (see §3): an **integer-ID / CSR graph backbone** (cheap — ~5–10 GB even at 600 M edges), **binary-quantized node/edge embeddings** (~5 GB at 150 M nodes) with a **cross-encoder reranker recovering the precision** quantization costs, and **lexical-first node lookup** so only hub/abstraction nodes need dense vectors. A maximal "import everything" graph lands near **~20 GB**, inside 64 GB, with the LM resident elsewhere.
- **Raw tier (NVMe).** Fallback + mining source (the distilled-from-text path), unchanged.

> 13900KF is **AVX2-only**. Size ONNX/quantization to it.

---

## 2. Data flow

```
INGEST → source_registry (trust weight + date) + raw_chunks (text store, cheap embeddings, lexical)
            │
   CONTENT ROUTER (cheap multi-label): procedural | causal | relational | declarative | none
            │
   EXTRACTORS (GPU LM, grammar-constrained JSON) → candidate cards / edges / nodes
            │
   RECONCILIATION (§9): within-batch → dogfood query against live KB →
        {corroborate | enrich | insert-distinct | keep-contradiction | insert}
        → update support set + strength (source-trust & recency weighted)
            │
   CANONICAL NODES + TYPED EDGES + CARDS  ──► surface generation (questions+propositions)
            │                                        │
   EVAL/MONITOR (self-retrieval, golden queries, audits, gap log)   RETRIEVAL SURFACE (HNSW+FTS5)

QUERY → intent → substrate + traversal/operator → fetch + grounding-confidence → render
        (Tier-2 raw fallback on miss → enqueue passage for distillation; log gap if none)
```

---

## 3. Storage layout

**NVMe:** `raw_text_store` (sha256-addressed, mmap); `raw_embeddings` (Phase-3 FAISS IVF-PQ + OnDisk lists); `raw_lexical` (Phase-3 Tantivy; FTS5 earlier); `kb.sqlite` (durable nodes/cards/edges/support/registry/gaps — source of truth, rebuilt into RAM on boot).

**RAM (resident, ~64 GB graph-tier budget):**
- **Graph backbone — integer-ID / CSR.** Nodes are `u32` ids (4.3 B capacity); edges are compact `(src u32, dst u32, type u16)` in CSR adjacency (offsets + neighbours), bidirectional. ~5–10 GB at 600 M edges. This is exact symbolic traversal; it carries *no meaning* (synonyms get unrelated ids), so it cannot do fuzzy entry or cross-domain leaps on its own.
- **Semantic layer — embeddings, quantized.** Node/edge embeddings give fuzzy entry, similarity, and the cross-domain analogy that §11 brainstorming depends on (the "reactor → engine block" leap is an embedding neighbour, not a stored edge). Store **binary (~256-bit, ~32 B/node)** → ~5 GB at 150 M nodes; Hamming-ANN for recall, **cross-encoder reranker** restores the precision the quantization drops. The reranker is what *makes* aggressive quantization viable — it is load-bearing, not optional.
- **Selective / tiered dense embedding.** Do **not** densely embed every tail node. The **lexical index (FTS5/Tantivy over labels + aliases)** handles the bulk of node lookup (most phrasings overlap a label/alias); reserve dense vectors for the **query** (embedded at runtime) and **hub/abstraction nodes** that power functional substitution and analogy. Cuts embedding RAM by ~an order of magnitude.
- Plus: cards + card embeddings; HNSW/FTS5 over cards/questions; content router + intent classifier + query encoder + reranker; feature vocabulary; source-trust table.

> Node *count* drives only the cheap integer backbone; the **embedding strategy**, not node count, decides whether you fit in 64 GB.

---

## 4. Schemas

### 4.1 Canonical concept node
```sql
nodes(id, label, kind, summary, embedding BLOB, aliases JSON,
      status TEXT DEFAULT 'active')        -- active | merged_into:<id> | retired
node_merge_candidates(node_a, node_b, similarity, reason, status)  -- adjudication queue
```

### 4.2 Raw chunk
```sql
raw_chunks(id, doc_id, section, text_hash, token_count,
           route_labels JSON, cluster_id, distilled INTEGER DEFAULT 0, offset)
```

### 4.3 Source registry (NEW — trust & time)
```sql
source_registry(
  doc_id TEXT PRIMARY KEY, title, source_type,   -- peer_reviewed|textbook|reference|web|unknown
  trust_weight REAL,        -- 0..1 prior on reliability (by type, editable)
  regime TEXT,              -- §8: empirical|conventional|fictional|interpretive|historical (default for the source)
  pub_date,                 -- for recency / supersession
  has_reference INTEGER,    -- 1 = chapter-and-verse retrievable (protocols, papers); 0 = dataset-asserted only (imported KGs)
  status TEXT DEFAULT 'active')   -- active | quarantined | retracted
```

### 4.4 Procedure card (how)
```sql
procedure_cards(id, node_id, title, domain, goal,
  preconditions JSON, tools JSON, materials JSON,
  steps JSON, tips JSON, mistakes JSON, safety JSON,
  embedding BLOB, card_hash TEXT, hit_count INTEGER DEFAULT 0,
  support JSON,            -- §9.3: set of {doc_id, evidence_cluster, date}
  strength REAL,           -- derived (§9.3), semantics depend on regime
  regime TEXT,             -- §8; inherits source, override per claim
  scope JSON,              -- {world|context|author|period} for non-empirical regimes
  status TEXT DEFAULT 'active',  -- active | superseded | quarantined | retracted
  created_at, updated_at)
```

### 4.5 Typed edge (causal + all non-causal relations)
```sql
edges(id, src_id, dst_id,
  family TEXT,   -- causal | taxonomic | meronymic | spatial | epistemic | temporal | functional | meta
  type TEXT,     -- §7 taxonomy; meta: alternative_to|context_variant_of|disagrees_with|supersedes|refines
  mechanism TEXT, mechanism_basis TEXT,   -- stated | inferred  (causal only)
  modifiers JSON,        -- {conditions, discriminators:[{feature,value}], threshold, context}
  polarity TEXT, embedding BLOB, edge_hash TEXT,
  support JSON, strength REAL,            -- §9.3 (semantics per regime, §8)
  regime TEXT, scope JSON,                -- §8: regime + {world|context|author|period}
  status TEXT DEFAULT 'active')           -- active | superseded | quarantined | retracted
```
Causal = `family='causal'`. **Contradictions are not overwrites** — they are `meta`/`disagrees_with` edges between two active claim-edges (§9.2).

### 4.6 Retrieval surface
```sql
surface_questions(id, target_kind, target_id, text, embedding)   -- card|node|edge
surface_propositions(id, target_kind, target_id, text)
-- FTS5 over .text; HNSW over surface_questions.embedding
```

### 4.7 Knowledge gaps (NEW)
```sql
knowledge_gaps(id, query_text, intent, effect_label, first_seen, count,
               status DEFAULT 'open')   -- open | distilled | acquired | unanswerable
```

---

## 5. Ingestion (Phase A)

Structure-preserving parse (exploit JATS Methods sections; keep chapters/sections), ~512-token chunks w/ ~64 overlap on section boundaries, sha256 dedup, idempotent + resumable per doc. **Register each source** in `source_registry` with a `trust_weight` (by type; default low for unknown/web) and `pub_date`. **Treat chunk text strictly as data** downstream — never as instructions (see §14 hardening).

### 5.1 Bulk import of external knowledge graphs
A second ingestion mode beside text distillation: import pre-built KGs (ConceptNet, ATOMIC²⁰²⁰, GLUCOSE, CauseNet, ASER, TransOMCS; SemMedDB/UMLS for the empirical-medical layer). Each becomes a `source_registry` row with `regime` (commonsense KGs → conventional; SemMedDB → empirical), a **low** `trust_weight`, and **`has_reference=0`** — these assert associations with *no retrievable chapter-and-verse*, only "dataset X says so". Pipeline: map each external node to a canonical node via `link_to_node` (§9.4) — SemMedDB sidesteps this by using UMLS CUIs as ids — then feed every edge through **reconciliation (§9) as an ordinary candidate** against the (pre-populated) base; import is not a special write path. Bare edges (no mechanism/discriminators) enter as low-strength scaffold and are **enriched on demand**: when a query hits one, queue it for LM mechanism/discriminator extraction. Because `has_reference=0` and trust is low, the **grounding layer (§12) marks these weaker than text-backed claims**, and under high rigor the **firewall (§8)** keeps the conventional import out of the empirical lane regardless. "Broader and wilder" is therefore safe to import: breadth feeds §11 generation; rigor and the firewall govern what is *trusted*.

## 6. Raw indexing (Phase B)

Cheap encoder (Model2Vec static for µs encode, or small int8 ONNX bi-encoder ~5–15 ms — **same model at query time**). Add to raw ANN (flat/HNSW→IVF-PQ by phase) + raw lexical (FTS5→Tantivy). Validate embedding RAM on a real subset before committing a quantization scheme.

---

## 7. Routing, extraction & relation taxonomy (Phase C)

**Content router** (bootstrap: few-shot big LM labels a sample → train tiny multi-label head on embeddings → run cheaply over all). Routes each chunk to procedure / causal-edge / typed-edge extractor, or nothing (declarative-only rides the raw tier). **Extractors emit grammar-constrained JSON** (see companion prompts file); they propose canonical `label`+`aliases`+`kind` for node linking and a short `evidence` span (≤25 words) for provenance/independence.

**Relation taxonomy (one graph, several families):**

| Family | Types | Traversal |
|---|---|---|
| causal | causes, prevents, exacerbates, reduces | forward (mech) / backward+score (diagnostic) |
| taxonomic | is_a, instance_of, subtype_of | up/down generalise–specialise |
| meronymic | part_of, has_part | containment |
| spatial | located_in, adjacent_to, contains | topology ("where") |
| epistemic | described_by, authored_by, taught_by, cites | attribution ("who") |
| temporal | precedes, follows | ordering (non-procedural "when") |
| functional | treats, used_for, requires, produces | links *why* ↔ *how* |
| meta | alternative_to, context_variant_of, disagrees_with, supersedes, refines | claim-to-claim relations |

---

## 8. Epistemic regimes (truth regimes)

Not all knowledge is the same *kind* of truth. The reconciliation engine (§9), the confidence model (§9.3), and presentation (§12) are **parameterised by regime**. Regime is tagged per source at ingest (a novel → fictional; a journal → empirical; an essay → interpretive) and may be **overridden per claim** by the extractor when a source mixes registers (e.g. a history book stating facts *and* offering interpretation).

| Regime | Truth anchor | Scope key | Corroboration | "Contradiction" means | Recency / supersession | Adjudicate? | Rendered as |
|---|---|---|---|---|---|---|---|
| **empirical** | reality | — | yes | disagreement → resolve by trust/recency | yes | yes | fact, hedged by strength |
| **conventional** | a practice/tradition | context / tradition | yes (within context) | **context-variant, not conflict** | rarely | no | "For a ⟨context⟩, …" (+ alternatives) |
| **fictional** | the work / world | world | within a world | cross-world → **coexist**; in-work → continuity note | no | no | "In ⟨work⟩, …" |
| **interpretive** | the author | author | no (attribute) | debate → **present all positions** | for "current view" only | no | "⟨author⟩ argues …" |
| **historical** | reality + period | period | empirical parts yes | split: facts resolve, readings debated | "as-of ⟨period⟩" vs current | partial | "As understood in ⟨period⟩, …" |

**The cross-regime firewall (safety-critical) — two faces.**
- *Write-time (always on):* corroboration and contradiction operate **only within the same regime** and scope, so a novel's herbal cure can never raise a clinical claim's support, nor an essayist's opinion lower a study's. This is structural and never optional.
- *Read-time (rigor-gated):* high-rigor queries (§10) additionally apply the firewall as a **hard candidate-pool filter** — non-empirical items are *excluded*, not merely flagged for the model to discount in prose. For a clinical answer you want fiction and opinion **absent**, not present-but-hopefully-ignored. Versatility and clinical safety are the same mechanism.

**Most of the machinery already exists** — regime is the *governing switch* that selects it:
- conventional cross-context cases route to `context_variant_of` (§9.2), **not** `disagrees_with`;
- fictional claims carry a `world` scope and never merge or adjudicate across worlds;
- interpretive claims become `described_by`/attribution edges — positions, not facts;
- historical claims decompose into an empirical part (corroborable) and an interpretive part (attributed, period-scoped).

Regime also drives **routing**: the content router gains a light regime classifier (simplest: source_type → regime, with claim-level override). Fictional causal edges ("the spell drained his strength") are stored but regime-tagged so they stay walled off from empirical reasoning.

---

## 9. Reconciliation (read-before-write) — replaces mechanical merge

Every candidate is reconciled against the **live KB via the production query path** before it is written. This dogfoods retrieval and QA-tests surface generation (an item that can't be retrieved by its own question signals weak question generation).

### 9.0 Regime gate (first)
Read the candidate's `regime` (§8). It selects the ruleset for everything below:
- **empirical** → full five-way with trust/recency adjudication.
- **conventional** → contradictions across `context` become `context_variant_of`, never `disagrees_with`; recency off; no adjudication.
- **fictional** → reconcile only within the same `world`; differences across worlds coexist as separate edges; trust-weight ignored.
- **interpretive** → never a "duplicate/contradiction" to resolve; emit attribution (`described_by`) and keep all positions.
- **historical** → split the claim: route empirical part through the empirical ruleset, interpretive part through the interpretive ruleset, both `period`-scoped.

**Firewall:** comparables are retrieved only from the *same regime* (and same scope where applicable). Cross-regime items never enter corroboration or contradiction.

### 9.1 Geometric banding (cheap gate — only the ambiguous middle costs an LM call)
1. **Within-batch first.** Dedup/reconcile the batch's own candidates so repeated new edges corroborate each other instead of multi-inserting.
2. `edge_hash`/`card_hash` equal to an existing item → **auto-corroborate**, no LM.
3. Embedding clearly far **and** no shared (src,dst) node-pair / procedure node → **auto-insert**, no LM.
4. **Else** (similar-but-not-identical, or shares the cause→effect pair / procedure goal) → run the **reasoned reconciliation LM** on {candidate, top-k retrieved comparables}.

### 9.2 Five-way decision (the reasoning step)
The LM (or supervisor tier) classifies the candidate against each comparable:

- **Duplicate** — same claim, mechanism, conditions → *don't insert*; add `doc_id` to the existing `support` set, recompute strength.
- **Refinement** — same direction/mechanism, adds or narrows a condition/discriminator/population → *merge-enrich* the existing item's `modifiers`; add support.
- **Novel-distinct** — same (cause,effect) or goal but **different mechanism or different condition regime** → *insert separately* and link `meta`/`alternative_to` (or `context_variant_of`), recording **why** it is distinct in the new edge's `context`. This is the "make the entry handle the distinction" case.
- **Contradiction** — opposite polarity or incompatible claim → *keep both active*, link `meta`/`disagrees_with`. Never clobber. (Arbitration by source-trust/recency happens at query time, surfaced to the user.)
- **No match** — plain insert.

### 9.3 Support set (write time) + optional strength (read time)
**Write time keeps the *evidence*, not a verdict.** The one thing the write path must compute (because an LLM eyeballing a list will miscount it, and copies/citation-cascades must be discounted) is the **independence-cleaned support set**:
```
support(item) = set of { doc_id, evidence_cluster, date, trust_weight, regime }   # keyed by doc_id (SET)
# cluster by evidence-span similarity → near-identical spans (copying/citation cascade) count once.
```
That set is preserved verbatim and attached to the item. **No verdict is baked in.** Nothing is clobbered; contradictions stay as `disagrees_with` edges.

**`strength` is now an *optional read-time ranking signal*, not a stored truth.** When a query asks for rigor (§10), it is computed on demand from the support set — repetition counts **only when sources are independent**:
```
support(item) = set of { doc_id, evidence_cluster, date, trust_weight }   # keyed by doc_id (a SET)
# 1. Independence: cluster support by evidence-span similarity (near-identical spans =
#    copying / citation cascade) → each cluster counts once, taking its max trust_weight.
raw       = Σ_over_clusters  trust_weight_cluster          # independent, copy-discounted
base      = 1 - exp(-raw / k)                              # saturating: diminishing returns
# 2. Recency / validity:
recency   = 1.0 if any source flagged current-consensus
            else decay(now - max(date))                    # older-only → lower
            ; if status touched by a `supersedes` edge → heavy penalty
# 3. Contradiction pressure:
contra    = Σ base(e') for edges e' linked disagrees_with, weighted by their trust
strength  = clamp01( base * recency - λ * contra )
```
**Regime-relative semantics of `strength`** (the formula above is the *empirical* case):
- **conventional** → how well-attested the convention is *within its context*; drop `recency`, drop contradiction (cross-context = variant).
- **fictional** → canonicity within the `world` (primary text > derivative); `recency`/`contra` off.
- **interpretive** → prominence/influence of the position + attribution clarity; never raised by agreement nor cut by disagreement.
- **historical** → empirical part as above; interpretive part by prominence; both reported "as-of" the period unless a current-consensus source updates it.

These regime-relative readings are **interpretations applied at read time**, not stored scalars — low-rigor queries skip them entirely and just pass the raw support set up. Keying `support` by `doc_id` keeps re-ingestion **idempotent**; copies don't stack; one strong recent peer-reviewed source can outweigh many old echoes; live contradictions are visible rather than averaged away.

### 9.4 Node-identity policy (link_to_node)
Over-merging is **destructive** (splitting a conflated node loses provenance mapping and corrupts diagnosis); under-merging is **recoverable** (merge later). **Bias toward not merging.**
```
match by embedding sim s and alias/lexical agreement:
  s ≥ θ_high AND alias agreement → SAME node
  θ_low ≤ s < θ_high (ambiguous)  → create DISTINCT node; if one label generalises the
       other, add is_a/subtype_of edge (e.g. "evaporative dry eye" is_a "dry eye");
       else enqueue node_merge_candidates for adjudication
  s < θ_low → new node
periodic adjudication pass (LM/human) resolves the candidate queue
```

### 9.5 Worker loop
```
for chunk in batch:
    cands = grammar_constrained_extract(chunk)          # cards / edges
    cands = reconcile_within_batch(cands)
    for c in cands:
        c.src/dst/node = link_to_node(c)                # §9.4
        register_features(c.modifiers.discriminators)
        comparables = query_path(surface_question_of(c))  # dogfood live KB
        decision = reconcile(c, comparables)              # §9.1–8.2 (bands → maybe LM)
        apply(decision)                                   # corroborate|enrich|insert|link
        recompute_strength(affected)                      # §9.3
    chunk.distilled = 1
checkpoint()
```

---

## 10. Online query path

```
def answer(q):
    intent = classify_intent(q)                 # cheap classifier; supervisor LLM as fallback
    rigor  = set_rigor(q)                        # supervisor tier: low (default) | high (stakes)
    q_emb  = encoder.encode(q)
    route:
      what → node lookup;  how → card path;
      why_mech → causal forward walk;  why_diag → diagnostic_why(q);  what_if → forward walk;
      who/where → epistemic/spatial walk;  taxonomy → is_a walk;
      which → compare();  what_else → expand()
    pool = fetch(...)                            # cards/nodes/edges (status filter)
    if rigor == high:                           # READ-TIME verdict, only when it's needed
        pool = firewall_filter(pool, empirical_only=True)   # exclude non-empirical from the pool
        pool = adjudicate(pool, by=[recency, trust, strength_from_support])  # compute strength now
    bundle = provenance_bundle(pool)            # source, regime, date, trust, support set, contradictions
    if pool empty: log_gap(q); return abstain(q)
    return render(pool, bundle, rigor)          # low: attach bundle, let the model weigh; high: pre-judged
```

**Hybrid primitive (every path):** `dense=hnsw(q_emb,40); lex=fts5(q,40); hits=rrf_fuse(...); top=rerank(q,hits[:20])[:K]`. ~1–3 ms without rerank; ~30–150 ms with. Tier-2 raw fallback on miss enqueues passages for distillation.

**diagnostic_why** (abduction): match effect node + extract query context features (shared vocabulary) → gather **incoming** causal edges to the effect → `score_context_fit(edge.discriminators, query_features)` → ranked differential (each: cause + mechanism + discriminators + fit). Retrieval produces the differential; the model weighs it.

**Rigor is the one knob.** *Low* (default, most questions — the knight on his steed) just attaches the provenance bundle and lets the assistant weigh it; nothing is filtered or scored. *High* (a drug interaction) switches on the firewall filter and the recency/trust/strength adjudication. Stakes, not regime, decide rigor; the supervisor tier sets it.

**Operators:** `expand` = alternative_to + sibling causes (shared dst) + nearest-neighbour + has_part. `compare` = align candidates' discriminating attributes across a shared dimension set. `plan`/`navigate` = build & traverse a possibility space (§11).

---

## 11. Anticipatory reasoning: building & navigating the possibility space

A shift from **reactive** (cue → fetch → reply) to **anticipatory** (situation → lay out the space of options and branch points → use it to lead). It needs almost no new storage: a decision tree is a *read-time compile* over substrates you already have. Two engines compose, and they are kept **strictly separate**: a **divergent** layer that *generates* the possibility space unbounded (§11.1), and a **convergent** layer that *judges* it (§11.4). Navigation (§11.2) traverses whatever set exists. Fusing generation and judgment degrades both — the screen censors the brainstorm, the brainstorm leaks unvetted; separating them is strictly better on both axes.

### 11.1 Generating the space — means-ends decomposition + functional substitution
For a goal with no pre-stored solution ("boil an egg with a camping mug and tea lights"), the candidate set must be *constructed*, not narrowed:
1. **Decompose** the goal into functional requirements — from the procedure card's `preconditions`/`requires` edges. *boil egg → {vessel holding water, heat source raising ~250 ml to ~100 °C, sustained N min, the egg}.*
2. **Abstract** each requirement to a role and **substitute**: query the graph for role-fillers via functional + taxonomic edges — climb `is_a` to the abstract role ("heat source"), enumerate descendants/`produces`/`capable_of`/`used_for`, then keep those matching on-hand constraints. *heat source → {flame, concentrated sunlight, exotherm, …} ∩ {tea lights, solar reflector} → candidate fillers.* This is a generalisation of the `which`/`what_else` operators: **find X that fills role R under constraints C.**
3. **Assemble** candidate plans (combinations of fillers) and emit *everything*, each tagged with the **principle it embodies** and a `basis` marker (`grounded` | `speculative`). No feasibility gate, no safety filter here. "Put the mug on a reactor-core heat grille" is a **legal, useful** output — it names the functional principle ("large sustained low-grade heat surface") more vividly than the safe-but-boring candidate, and that principle is what the next layer mines for accessible neighbours. The graph supplies candidates + properties; generation just casts wide.

**This layer maximises coverage of the possibility space. It must not also judge** — the two objectives oppose each other, and a generator that screens itself suppresses exactly the wild candidates that carry the strongest abstractions. The graph never stored "boil egg with tea lights"; functional abstraction surfaces *tea light = heat source*, *mug = vessel*, and cross-domain analogues (a buddy burner, an engine block at temperature) arrive via the shared embedding space. Its only obligation to the world: **label the speculative as speculative**, so nothing wild is ever mistaken for vetted.

### 11.2 Navigating the space — tree compile + belief-state loop
When a candidate set exists (a differential, a stuck procedure, the plans from §11.1):
- `compile_tree(candidates, discriminators)` — internal nodes are discriminating tests, leaves are causes/outcomes. It builds itself from the structured `{feature,value}` discriminators you already store; **a differential and a decision tree are the same data in two shapes.**
- **Belief state:** the live candidate list with normalised fit scores.
- **Next-best-question:** pick the unobserved feature that best improves the state.
  - *low rigor* → information gain: `argmax_f [H(C) − E[H(C|f)]]`.
  - *high rigor* → **expected utility of information**, weighting branches by outcome stakes: a low-probability, high-cost leaf (a red flag) is checked regardless of how little it narrows the space. Ties straight into the rigor knob.
- Fold the user's answer back into the belief state; repeat.
- **Determinism:** scoring/selection is arithmetic done *outside* the LM (an LM judging "most discriminating question" miscounts as badly as it miscounts independent sources). The LM phrases the question naturally, interprets free-text answers back into feature values, and decides flow.
- **Budget (anti-interrogation):** ask the one or two highest-value questions, then help; stop when the belief state concentrates or the next question's expected value is small. Peeking one branch ahead ("if X we go here, if Y there") is what lets the assistant *lead* — "the thing that would most change the picture is whether…" — rather than wait for cues.

### 11.3 One pattern, four faces
- **symptomology** → compile over the causal differential; leaves = diagnoses; recovery via `treats` edges.
- **stuck mid-procedure** → effect node = the stuck-state; candidates = the card's `mistakes`/failure modes + causal edges into that failure; leaves = recovery procedures.
- **social** → branches from the conventional/interpretive substrate (the ATOMIC/GLUCOSE if-then folk psychology): plausible readings of the other person and where each response leads. **Suggestive, not predictive** — "ways this could go", never a flowchart of what *will* happen. Regime stays conventional; the firewall keeps social speculation out of the empirical lane.
- **novel goal / improvised means** → §11.1 generates the plans unbounded; §11.4 ranks-and-annotates them.

### 11.4 The convergent layer — judgment (separate from generation)
A **distinct** pass, with the opposite objective: rank the candidates by feasibility, safety, availability, and effort, pulling in the measuring stick — physics, constraints, and (when the task is medical) the retrieved protocols and the rigor knob. Three rules make it a judge, not a censor:

- **Rank and annotate; do not delete.** Every candidate survives, scored and explained. *"Reactor grille — sound principle (large sustained heat sink), inaccessible, lethal. Same principle, accessible: a cluster of tea lights (marginal, ~15–20 min), an engine block at temperature, a dark mug in a solar concentrator (ample)."* The absurd entry stays visible because it earns its place as the principle's clearest statement.
- **Mine the rejects.** An infeasible candidate is a *search seed*, not waste: it points at a feasible neighbour through the principle it shares (reactor → "big warm surface" → engine block, compost heap, sun-baked rock). A generator that never proposed the reactor never finds the engine block. Bad ideas are navigational — they say which direction the good ones are in.
- **Labelling, not suppression, is the only hard rule.** Suggesting the absurd is fine; smuggling it through as *feasible* is the sole real failure. Speculative stays marked speculative. Under high rigor the empirical protocols are pulled as the standard and the firewall keeps the brainstorm (conventional/inferential, speculative-tagged) from ever corroborating or contaminating them — so "unbounded creativity" and "clinically safe" are just layer 1 and layer 2 with the firewall between.

Generate like there are no rules; judge like everything matters. (Carries over: utility-weighting + red-flag rule and the question budget from §11.2; social trees stay suggestive, never predictive.)

### 11.5 Design defaults (decided; flip per deployment)
- **Question objective:** expected-utility under high rigor (medical default), information-gain under low.
- **Derivation:** on-demand by default; cache hot/frequent presentations as compiled trees during overnight consolidation.
- **Explicit algorithms:** when a source contains an authored clinical pathway, troubleshooting flowchart, or dichotomous key, capture its branch structure as a **first-class object** (extractor in prompts §9) and use it directly instead of re-deriving — fused with derived trees via shared `feature` vocabulary.

---

## 12. Grounding confidence, presentation & abstention

Grounding is read from the **provenance bundle** at answer time (not a stored field). Compose it from: top rerank/RRF score; independence of the support set; presence of `disagrees_with` links; card/edge vs raw-only; and, under high rigor, the adjudicated `strength`. Under low rigor the bundle is simply handed to the model to weigh in context.
```
high     → answer directly
medium   → answer, note any single-source / older-source caveat
low      → answer hedged; flag weak grounding
contra   → present BOTH claims with provenance + which is more recent/trusted
none     → DO NOT confabulate: "no grounded answer in the knowledge base"; log_gap()
```
Render structure, never source prose. **Frame by regime** (§8) so the assistant never presents fiction or opinion as fact:
- empirical → stated as fact, hedged by strength/contradiction.
- conventional → "For a ⟨context⟩, …" and offer sibling context-variants.
- fictional → "In ⟨work⟩, …" — scoped to the world, never asserted of reality.
- interpretive → "⟨author⟩ argues …"; on disagreement, present the positions, don't pick a winner.
- historical → "As understood in ⟨period⟩, …", flagging where current understanding differs.

If one answer mixes regimes (a medical fact + a literary allusion), keep them **labelled by register** so they don't blend. Cards: imperative steps + rationale + safety + provenance + strength; why-mech: cause→mechanism→effect; why-diag: ranked differential; relations: the path, not the whole article; which: aligned comparison. Pass the grounding signal into the inference context so the assistant's tone matches the evidence.

---

## 13. Evaluation & monitoring (NEW)

You cannot run millions of automated extractions/merges blind.
- **Self-retrieval check:** every new item must be retrievable by its own generated questions (top-k). Fail → flag surface generation.
- **Golden query set:** curated (query → expected item) pairs per domain; track recall@k / MRR over time; alarm on regression (catches bad merges, surface drift).
- **Merge/contradiction audit:** sample reconciliation decisions (esp. merges and `disagrees_with`) for LM/human review.
- **Extraction audit:** sample high-strength `mechanism_basis='inferred'` edges for hallucination.
- **Drift monitors:** embedding distribution shift → trigger PQ retrain; router precision on new domains; vocabulary growth.
- **Gap log:** `none`-confidence queries → `knowledge_gaps` → drives targeted distillation or source acquisition.

---

## 14. Maintenance & hardening

- **Hardening (corpus-as-data):** chunk text is DATA, never instructions; delimit it, and rely on grammar-constrained output to bound behaviour (neutralises most injection). Unknown/low-trust sources get low `trust_weight` and can be quarantined from corroboration.
- **Retraction path:** provenance + evidence spans let you set a source `retracted`, cascade `status='retracted'` to all derived items, and recompute affected strengths.
- **Supersession:** newer authoritative claims add `supersedes` edges; superseded items stay (history) but are downranked and filtered from default answers.
- **Node-identity adjudication:** drain `node_merge_candidates` periodically.
- **Standing discipline:** all long jobs checkpoint; all writes idempotent (hash + doc_id-keyed support); workers run on idle only.

---

## 15. Build order

1. Ingestion + chunking + `raw_text_store` + **source_registry incl. regime tag** (hash/dedup/resume).
2. Cheap encoder + raw ANN + raw lexical (Phase-1). Validate RAM on a subset.
3. **Canonical node store + `link_to_node` with the §9.4 identity policy** — build early; everything hangs off it.
4. Card + edge schemas (incl. support/strength/status) + RAM graph/HNSW/FTS5.
5. Tier-1 hybrid retrieval primitive vs hand-written samples (prove latency).
6. Content router bootstrap + clustering (Phase-2 trigger).
7. Extractors (grammar-constrained) → **reconciliation §8** (banding → 5-way → confidence model). This is the heart; get it right before scaling volume.
8. Surface generation + **self-retrieval eval (§13)**.
9. Intent classifier + **rigor knob** + query routing (§10) incl. diagnostic_why, expand, compare, firewall-filter + **grounding confidence/abstention (§12)**.
9b. **Anticipatory reasoning (§11):** functional-substitution operator, `compile_tree`, belief-state + next-best-question loop, feasibility/safety screen.
10. Tier-2 fallback + demand queue + gap log.
11. Golden-query harness + audits + drift monitors (§13).
12. Maintenance/retraction/supersession/adjudication (§14).
13. Port hot paths (encode, fuse, fetch, traversal, fit-scoring) to Rust via PyO3.
14. Escalate raw-tier to Phase-3 (IVF-PQ, Tantivy) past ~200 GB.

---

## Appendix: choices deliberately not taken
- **Eager verdicts at write time** — the write path now only preserves + structures provenance (incl. the independence-cleaned support set); weighing by recency/trust is a deferred, rigor-gated read-time step.
- **Mono-epistemic truth model** — replaced by regime-relative truth (§8); empirical adjudication is no longer mis-applied to conventions, fiction, or argument, and regimes are firewalled from each other.
- **Overwrite-on-merge** — replaced by reasoned reconciliation; contradictions are recorded, not lost.
- **Flat corroboration counter** — replaced by independent, copy-discounted, trust/recency-weighted support sets.
- **Aggressive node merging** — biased toward under-merge (recoverable) + taxonomic edges + adjudication.
- **Separate what-else/which/causal stores** — operators and one typed-edge graph instead.
- **GraphRAG generic triples** — typed families + per-type traversal + causal modifiers carry more signal.
- **SPLADE / RAPTOR / ColBERT** — doc2query questions, the card hierarchy + taxonomic edges, and the reranker respectively cover these at lower CPU/storage cost.

## Deferred (flagged, not yet specified)
- **Query decomposition / planner** above the intent router for multi-hop questions (supervisor-LLM job).
- **Negative/contraindication knowledge** ("X does not cause Y", "contraindicated") as explicit negative polarity.
- **Embedding-model migration / versioned embeddings** for raw-tier reindex on encoder upgrades.
