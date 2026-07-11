"""Configuration: defaults < TOML file < environment variables.

Every key in DEFAULTS can be overridden by an env var named
``KNOWLEDGEHOST_<KEY>`` (e.g. ``KNOWLEDGEHOST_PORT``, ``KNOWLEDGEHOST_DB_PATH``).
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULTS = {
    # ── store ──────────────────────────────────────────────────────────────
    # "sqlite"  -> FTS5 (sparse) + numpy/brute-force dense.  Zero extra service,
    #              ships in the stdlib, great for the PDF collection / Phase 1.
    # "lance"   -> LanceDB IVF-PQ (dense) + FTS.  On-disk, mmap'd, scales to a
    #              full Wikipedia snapshot (10-40M chunks).  Needs `pip install
    #              lancedb`.  Use this once you bulk-ingest Wikipedia.
    "backend": "sqlite",

    # sqlite backend: the index database.  lance backend: see lance_dir.
    "db_path": "var/index.db",
    "lance_dir": "var/lance",
    "lance_table": "kb",
    # lance only: don't build the IVF-PQ ANN index below this many rows (PQ needs >=256
    # to train, and flat/exact scan is instant on a small table) — avoids a spurious
    # "not enough rows to train PQ" error on small/test corpora.  It auto-builds later.
    "ann_min_rows": 4096,

    # ── service ────────────────────────────────────────────────────────────
    "host": "127.0.0.1",   # bind localhost; it is NEVER on the LAN (read-only).
    # KNOWLEDGE.md suggests 8770, but the music host already claims that port;
    # default to 8771 so the two hosts can co-locate.  Override freely.
    "port": 8771,
    "auth_token": "",      # if set, /call requires  Authorization: Bearer <it>

    # ── embeddings (the nomic endpoint, shared with Vinkona's memory store) ────
    # Co-located on the Linux GPU box at 127.0.0.1:11437 (llama.cpp --embedding,
    # OpenAI /v1/embeddings).  Asymmetric model => task prefixes (search_query: /
    # search_document:).  If unreachable, the host degrades to sparse-only FTS.
    "embed_url": "http://127.0.0.1:11437",
    "embed_model": "nomic-embed-text-v1.5.f16.gguf",
    "embed_task_prefix": True,
    "embed_query_prefix": "search_query: ",
    "embed_document_prefix": "search_document: ",
    "embed_batch": 64,        # ingestion embed batch size
    "embed_timeout_s": 30,
    # The embed server processes each sequence in ONE physical batch; an input
    # over its n_ubatch is rejected with HTTP 500.  Chunks are clipped to this
    # many (estimated) tokens before embedding so a stray oversized chunk is
    # embedded from its head rather than nulling the whole batch to sparse-only.
    "embed_max_tokens": 512,

    # ── metacognitive distillation (spec §3,§4,§7-9) ─────────────────────────
    # The durable structured KB (canonical nodes / cards / typed edges) — the
    # "card/graph tier" source of truth, kept separate from the raw chunk store.
    "kb_path": "var/kb.db",
    # The offline distiller's LM (the big reasoning model; GPU-backed, grammar-
    # constrained JSON).  Defaults to Vinkona's big_lm tier.
    "distill_url": "http://127.0.0.1:11438",
    # All candidate big-LM endpoints; the live ones are auto-detected at startup.
    # A "sometimes available" endpoint is re-probed every --watch pass.
    "distill_urls": ["http://127.0.0.1:11438"],
    "distill_model": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    "distill_timeout_s": 120,
    # ── two-tier distillation: fast EXTRACTOR (4090) + big VERIFIER (3090) ────
    # A fast small LM does the bulk extraction; the big LM then VETS every
    # submission (accept / reject / adjust).  The two stages run as decoupled,
    # queue-fed pipelines so each GPU works at its own max rate (bounded queues give
    # backpressure).  If no fast endpoint is up the system falls back to the big LM
    # doing extraction directly (the previous single-tier behaviour).
    "verify": True,                              # gate fast-LM output through the verifier
    "extract_urls": ["http://127.0.0.1:11435"],  # the fast primary distiller(s) — 4090
    "extract_model": "Qwen3.5-9B-Instruct-Q4_K_M.gguf",
    "extract_timeout_s": 90,
    "extract_max_tokens": 3072,
    "verify_urls": [],                           # the verifier(s); empty => use distill_urls (11438)
    "verify_model": "",                          # empty => distill_model (the 32B)
    "verify_timeout_s": 120,
    "verify_max_tokens": 1024,                   # verdicts are short -> keep the slow LM fast
    # Verify several chunks per big-LM call (amortise its fixed per-call cost).  Sources
    # are truncated to verify_source_chars each so a batch fits the big LM's context;
    # set verify_batch=1 to disable batching.  Raise if your big LM has a larger window.
    "verify_batch": 6,
    "verify_source_chars": 800,
    # ── cooperative GPU leases (Vinkona yields the GPUs to the live assistant) ──
    # Vinkona publishes lease files in her repo's logs/control/.  Before using a GPU we
    # check the lease and pause that stage if held — distil yields the 4090 (lm_fast)
    # during a live chat; verify yields the 3090 (lm_big) during Vinkona's research.
    # Held = file exists AND float(contents) > now; missing/expired/unparseable = free.
    # Resolved as: $VINKONA_CONTROL_DIR > control_dir > ../assistant/logs/control (monorepo sibling).
    "control_dir": "",
    # Output budget per chunk.  Multi-concept JSON overruns a small cap and gets
    # truncated mid-array (=> unparseable); 3072 comfortably fits a dense chunk
    # and still leaves room in an 8k-ctx model.
    "distill_max_tokens": 3072,
    # link_to_node identity policy (§9.4): bias toward NOT merging (under-merge is
    # recoverable, over-merge is destructive).
    "node_sim_high": 0.86,    # ≥ this + alias agreement => same node
    "node_sim_low": 0.72,     # [low,high) => distinct node (+ is_a or adjudication)
    # read path: a query whose best structured match is below this cosine has no
    # grounded answer => abstain (§11), rather than return a weak item.
    "kb_min_sim": 0.35,

    # ── ingestion ──────────────────────────────────────────────────────────
    # Roots to crawl for documents (PDF/EPUB/HTML/txt/md).  Add your collection.
    "sources": ["~/Documents", "~/Books"],
    # Classify whole folders by epistemic regime so the distiller mines each with the
    # right text-type lens (fiction → interpretive/narrative, essays → argument, …).
    # A bare key matches any path SEGMENT; a glob key matches the whole path; first
    # match wins.  TOML:  [source_regimes]\n  fiction = "fictional"
    # Unmapped sources fall back to the format default (empirical).  Example:
    #   {"fiction": "fictional", "science": "empirical", "essays": "interpretive",
    #    "history": "historical"}
    "source_regimes": {},
    "extensions": [".pdf", ".epub", ".html", ".htm", ".txt", ".md"],
    # Wikipedia: a Kiwix ZIM (pre-rendered, cleaned HTML).  Keep it in its OWN folder
    # (e.g. ~/dev/knowledge-host/wikipedia/enwiki.zim) — deliberately NOT under `sources`,
    # so the document crawl never touches it.  Tier-0 search queries the ZIM's built-in
    # Xapian full-text index live (no ingestion, auto-current on swap); `ingest --wikipedia`
    # is the separate, optional path that chunks+embeds it.  Empty => Wikipedia disabled.
    "zim_path": "",
    # Wikipedia arm slicing+ranking: Xapian retrieves the top `wiki_articles`, each sliced
    # into `wiki_chunk_chars` sections (capped at wiki_max_chunks total), then the slices
    # are embedded and re-ranked by cosine to the query so the relevant SECTION surfaces,
    # not the lead.  wiki_semantic=false falls back to plain Xapian order (no per-query
    # embedding).  More articles/chunks = better recall, more embed latency per search.
    "wiki_articles": 8,
    "wiki_chunk_chars": 1000,
    "wiki_max_chunks": 120,
    "wiki_semantic": True,
    # ── the LIBRARY: a search-only document corpus (NOT distilled into the graph) ──
    # A second, cheap tier: point at a big folder tree of source texts (the ones you DON'T
    # curate into the graph) and get a fast local "google" over them for Vinkona's research
    # loop — lexical FTS5 (no bulk GPU embedding), with the top slices reranked per query.
    # Its own index file, never distilled, kept apart from the curated graph corpus.
    # Topical folders → a `collection` tag (science/fiction/history…) filterable at search,
    # so fiction can't leak into a factual pull.  Empty library_sources ⇒ feature off.
    "library_sources": [],                # folder roots of the search-only library
    # ONE trusted parent folder, set here in the file (never over HTTP).  When set, the web
    # Library panel can toggle WHICH immediate subfolders under it are indexed — each write is
    # containment-validated (must resolve under this root), so the web can pick blessed subtrees
    # but never point the crawler outside them.  Leave empty to manage library_sources by hand.
    "library_root": "",
    "library_db": "var/library.db",   # its own FTS index (separate file)
    # folder → collection map (a bare key matches a path SEGMENT, a glob the whole path;
    # first match wins).  Unmapped docs take their top folder name under the root.
    #   TOML:  [library_collections]\n  fiction = "fiction"\n  papers = "science"
    "library_collections": {},
    "library_dense": False,               # lexical-first; True also embeds (costly at GB scale)
    # ── ConceptNet bulk import (commonsense, regime=conventional, ungrounded) ──
    # Path to the ConceptNet 5.7 `assertions.csv` dump (10 GB, streamed).  Imported as
    # a single low-trust source with has_reference=0 on every entry.  By default the
    # noisy lexical/etymological relations (RelatedTo/FormOf/DerivedFrom/HasContext/
    # Etymologically*) are skipped — set conceptnet_include_lexical=true to keep them.
    "conceptnet_path": "",
    "conceptnet_trust": 0.2,        # source trust_weight (low, discountable prior)
    "conceptnet_min_weight": 1.0,   # drop assertions below this ConceptNet weight
    "conceptnet_include_lexical": False,
    # ── ATOMIC if-then commonsense import (same regime/trust as ConceptNet) ──
    # Use the aggregated dump `v4_atomic_all_agg.csv` (one row per event).  min_count is
    # the annotator-agreement floor for an inference (1 = keep all non-"none").
    "atomic_path": "",
    "atomic_trust": 0.2,
    "atomic_min_count": 1,
    # ── GLUCOSE general causal-rule import (same regime/trust as the above) ──
    # Imports the GENERAL (variable-slot) rules only; min_count is the agreement floor.
    "glucose_path": "",
    "glucose_trust": 0.2,
    "glucose_min_count": 1,
    # ── CauseNet causal-graph import (grounded; has_reference=1) ──
    # The precision JSONL (cause→effect mined from Wikipedia/ClueWeb).  regime is
    # conventional (firewall-safe) by default; set causenet_regime="empirical" to let it
    # corroborate the empirical tier.  Source count rides along as corroboration.
    "causenet_path": "",
    "causenet_trust": 0.4,
    "causenet_regime": "conventional",
    # ── reconcile imported nodes against your existing ones (node identity §9.4) ──
    # For each anchor (a node you already had) queue its top-K nearest neighbours as
    # node_merge_candidates for `adjudicate` to judge.  min_sim=0 => use node_sim_low.
    "reconcile_top_k": 3,
    "reconcile_block": 128,        # anchors per matmul block (RAM vs speed)
    "reconcile_min_sim": 0.0,
    # ── link (Phase 1 graph linkage): the big LM types structural edges between
    # card-bearing concepts and their nearest neighbours (is_a/requires/part_of/
    # alternative/related), so cards know what they specialise, need, or alternate with.
    # Additive and provenance-tagged ('linker:v1', inferred, low trust) — never destructive.
    "link_top_k": 8,               # REAL neighbours kept per card-bearing anchor
    "link_fetch_mult": 20,         # over-fetch depth (k×this) before filtering out commonsense
    "link_min_sim": 0.5,           # ignore neighbours below this cosine (ANN floor)
    "link_min_conf": 0.6,          # write an is_a/requires/part_of/alternative edge above this
    "link_related_min_conf": 0.75, # the vaguer 'related_to' needs more confidence
    "link_max_tokens": 1024,       # LM budget per judgement (room for a 'thinking' endpoint)
    # ── refine (Phase 2 card refinement): re-read each card's SOURCE document and rewrite
    # the card in place into the ideal 'what do I do now' form, grounded in that source.
    # Whole-doc when it fits refine_source_tokens, else a window around the best-matching
    # chunk.  Big-LM work; demand-weighted by hit_count; resumable (refined cards skipped).
    "refine_source_tokens": 46000, # source budget fed to the LM (of its ~64k context)
    "refine_max_tokens": 4096,     # output budget for the improved card (room to 'think')
    "refine_timeout_s": 600,       # per-call client timeout — a 46k-token reasoning call is
                                   # slow; the default ~120s gives up mid-prompt on big docs
    # ── ask: prefer grounded documents over the bulk commonsense priors ──
    # The commonsense imports (ConceptNet/ATOMIC/GLUCOSE/CauseNet) add ~1M nodes that
    # otherwise drown the few thousand items distilled from YOUR documents.  In kb_ask an
    # item sourced ONLY from those bulk imports has its ranking score multiplied by
    # ask_prior_penalty (0.0 ≈ documents-only; 1.0 = no preference); ask_pool widens the
    # candidate shortlist so grounded items make it in before that re-ranking.
    "ask_prior_penalty": 0.5,
    "ask_pool": 24,
    # The relevance gate (§11): when a kb_ask carries context_features (or the query holds
    # vocabulary values), answers are scored on how well they MATCH that context, and a
    # candidate whose discriminators CLASH with it abstains rather than presenting a
    # confident near-miss ("orange ≈ orange" but the recipe doesn't fit "how do I cut one").
    # Off → plain embedding-ranked answers.  Plain look-ups (no features) are unaffected.
    "ask_fit_gate": True,
    # ── query understanding: optional spaCy structure extraction (understand.py) ──
    # A dependency parse isolates the HEAD concept from context modifiers + gives entity
    # spans, sharpening recall — ~1-5ms/query on CPU (fine on an M4), between the regex
    # classifier (µs) and an LLM (seconds).  OPTIONAL: absent the library the regex path
    # stands alone.  Adds NO tuning knobs — its output only feeds extra pool candidates.
    #   pip install spacy && python -m spacy download en_core_web_sm
    "use_spacy": False,       # enable the parse (no-op + one log line if spaCy not installed)
    "spacy_model": "en_core_web_sm",
    # ── modular knowledge bundles (spec §16) ─────────────────────────────────
    # The master kb.db stays the authoring source of truth.  A *scenario* selects
    # sources by provenance and the server assembles a disposable working DB from
    # just those (ids are content hashes, so the merge dedups + relinks for free).
    # Leave scenarios empty and active_scenario 'all' ⇒ the master is served as-is,
    # byte-for-byte identical to before (feature dormant).
    "active_scenario": "all",      # which scenario this session serves ('all' = master)
    "default_scenario": "all",     # fallback when active_scenario is unset
    # Where split bundle files live (optional).  If present + a scenario's sources are
    # tagged into bundles that have files here, the working DB is merged from THOSE;
    # else it's extracted straight from the master.  Empty => always from master.
    "bundle_dir": "",
    # Where assembled working DBs are cached (empty => "<kb_path dir>/work").  Cached by
    # (scenario, selected sources, master mtime); an unchanged selection reuses the build.
    "bundle_work_dir": "",
    # Scenarios: TOML  [scenarios.clinical]\n  include = ["obstetric","pharmacology"]
    #                  [scenarios.casual]\n   exclude = ["obstetric"]
    # tokens match a source's bundle tag, doc_id, or title; "*" = everything.
    "scenarios": {},
    # ── encryption at rest (overlay/sensitive bundles; spec §16.6) ────────────
    # SQLCipher (via pysqlcipher3/sqlcipher3) encrypts flagged bundle files at rest.
    # Key comes from $KNOWLEDGEHOST_DB_KEY (or the OS keystore) — never stored here.
    # Bundles named below are opened/written encrypted; the big base stays clear for
    # speed.  Requires the sqlcipher lib on the run box; inert (clear) without it.
    "encrypted_bundles": [],       # e.g. ["overlay"] — bundle names to encrypt at rest
    "db_key_file": "",             # optional mode-600 keyfile (else $KNOWLEDGEHOST_DB_KEY)
    # ── research → learning loop (Vinkona's solved/*.md drops; research_loop_spec §6) ──
    # Vinkona writes answered research questions as markdown (front-matter provenance:vinkona)
    # into a shared folder; point this at it and crawl mines it into the 'vinkona' bundle at
    # low trust.  Each doc's `# Question` frames distillation into a card answering it, and
    # a matching `kb_query` closes the knowledge_gap that first opened it.  Empty ⇒ off.
    # Flag it sensitive with encrypted_bundles=["vinkona"], and include it in the live
    # scenario so cards distilled into it are actually read back by kb_ask.
    "research_solved_dir": "",     # the solved/ outbox to ingest (its own low-trust bundle)
    "vinkona_trust": 0.25,           # trust_weight for vinkona-sourced content (low, subordinate)
    # vinkona cards are distilled + surfaced but BANDED below curated: an item sourced only
    # from the vinkona bundle has its ask ranking score multiplied by this (curated wins ties).
    "ask_vinkona_penalty": 0.85,
    # ── retrieval eval harness (retrieval contract §8) ───────────────────────
    "eval_gold_path": "eval/gold.jsonl",           # graded gold fixtures (JSONL)
    "eval_out_dir": "",                            # run artifacts (empty => <control_dir>/eval-runs)
    # ── Tier-3 "unsure" fallback (contract §6.1 tier 3) — wired in a later step ──
    # When retrieval can't confidently answer but linked/related cards exist, the caller
    # says THIS instead of confabulating, optionally volunteering a few related card titles.
    # All adjustable so Vinkona's voice + willingness-to-guess can be tuned without code.
    "unsure_message": "I'm not sure how to answer that confidently.",
    "unsure_alternatives": 3,                      # max related card titles to volunteer (0 = none)
    "unsure_min_sim": 0.30,                        # only volunteer relatives at/above this similarity
    "unsure_style": "titles",                      # how to volunteer: titles | none
    # ── ANN index over node embeddings (usearch HNSW; read-path speed) ──
    # Brute-force cosine is O(N) per query (~90ms at 1M).  Build an HNSW index with
    # `build-ann` and search/reconcile use it (~single-digit ms, mmap'd).  Absent index or
    # usearch => exact brute-force fallback (identical results).  Built explicitly so it is
    # never stale-by-surprise; re-run after big imports/embeds.  Used on READ paths only —
    # the distiller's link_to_node stays exact.
    "ann_search": True,            # use the ANN index in search() when one is present
    "ann_path": "",                # index path stem; empty => "<kb_path>.ann"
    "ann_connectivity": 32,        # HNSW M (graph degree) — higher = better recall, more RAM
    "ann_expansion_add": 128,      # ef_construction — build-time accuracy
    "ann_expansion_search": 128,   # ef_search — query-time accuracy/recall (vs speed)
    "ann_min_nodes": 50000,        # below this, build-ann skips (brute force is exact+instant)
    # Stored index precision: f16 ≈ half the RAM of f32 with negligible cosine error — the
    # right default on a CPU box where keeping the index resident drives latency.  f32 for
    # max fidelity, i8 to quarter the RAM (small recall cost) at very large scale.
    "ann_dtype": "f16",
    # Load the index RESIDENT in RAM (fast, ~index-size RSS) vs memory-map it (low RSS but
    # every NEW query pages its graph traversal from disk — seconds/query on a tight box).
    # Resident is right for a latency-critical server; only set True if RAM can't hold it.
    "ann_mmap": False,
    # counts() does full scans of the (now huge) node/edge tables; the live viewer polls
    # them every ~2.5s holding the DB lock, which stalls concurrent asks.  The tallies
    # barely move, so cache them well past the poll interval (the streaming ops LOG, not
    # these numbers, is what needs to feel live).  Lower it if you want fresher counts.
    "counts_cache_ttl": 30.0,
    # SQLite memory: the kb.db is multi-GB (embeddings stored inline), and `ask` hydrates
    # many candidate rows per query.  mmap the DB + a big page cache so those lookups are
    # RAM-speed not disk faults.  Raise mmap_mb to ≥ your kb.db size if you have the RAM.
    "sqlite_mmap_mb": 4096,
    "sqlite_cache_mb": 128,
    # adjudicate: a deterministic pre-pass clears the lexically-obvious pairs without the
    # LM (merge plural/exact dupes, is_a for token-subsets), so the big LM only judges the
    # thin ambiguous band.  auto_merge_sim: embedding sim above which a lexically-
    # overlapping pair is auto-merged.  escalate_sim: high-sim-but-undecidable pairs at/above
    # this go to the LM; weaker ones are deferred (recoverable under-merge, §9.4).
    "auto_merge_sim": 0.93,
    "adjudicate_escalate_sim": 0.90,
    "ocr": True,              # OCR scanned PDF pages that yield no text layer
    "ocr_min_chars": 32,      # a page under this many chars is treated as scanned
    "ingest_log_every": 200,  # progress line every N documents
    # bulk-ingest throughput (matters at 10k+ docs): parse files across a process pool
    # (0 = auto = os.cpu_count()), and commit chunks in big transactions instead of one per
    # embed-batch.  The write batch only bites the lexical (non-embedding) path; the embed
    # path stays bounded by embed_batch so GPU memory isn't blown.
    "ingest_workers": 0,          # library parse parallelism (0 = auto/cpu_count, 1 = serial)
    "ingest_write_batch": 1000,   # chunks per commit when NOT embedding (lexical library)
    # full-text search quality.  Porter stemming (running↔run) + diacritic folding (café↔cafe)
    # lifts recall on a large diverse corpus; empty string = SQLite's default unicode61.
    # A tokenizer change needs a one-time `rebuild-fts` (cheap: reindexes from stored text,
    # NO re-parse).  bm25_col_weights order = (text, title, section): boost title/heading hits.
    "fts_tokenizer": "porter unicode61 remove_diacritics 2",
    "bm25_col_weights": [1.0, 4.0, 2.0],
    # FTS5 index detail.  We only ever issue OR-of-terms matches (no phrase/NEAR queries, no
    # highlight/snippet), so the within-column POSITIONS that 'full' stores are dead weight:
    # 'column' drops them for a smaller, faster index while keeping the column info that
    # title-boost bm25 needs — provably identical results.  Change needs a one-time rebuild-fts.
    # ('full' if you ever add phrase search; 'none' is smallest but disables column weighting.)
    "fts_detail": "column",
    # SELF-ADAPTIVE stoplist: after each ingest the store learns which terms "over-report"
    # (appear in more than stopword_df_ratio of all chunks) straight from the corpus via
    # fts5vocab, and drops them from the OR match — so a query word like "the" doesn't force a
    # scan of millions of doclists.  BM25's IDF already makes such terms score ~0, so recall is
    # unaffected; this is purely a latency guard that TUNES ITSELF to the actual corpus
    # (incl. non-English books, which a static English list would miss).  df_ratio<=0 disables.
    "stopword_df_ratio": 0.5,     # term in >50% of chunks ⇒ auto-stoplisted (IDF≈0, safe)
    "stopword_max": 300,          # cap the learned list (top-N by document frequency)
    "stopword_min_chunks": 5000,  # don't bother below this corpus size (small = already fast)
    "stopwords_extra": [],        # manual always-drop terms, unioned with the learned list

    # ── chunking ───────────────────────────────────────────────────────────
    "chunk_target_tokens": 320,   # ~200-400 token chunks, split by heading
    "chunk_max_tokens": 512,      # hard cap before a forced split (== embed window)
    "chunk_overlap_tokens": 48,   # small overlap across a forced split

    # ── retrieval ──────────────────────────────────────────────────────────
    "default_k": 5,           # passages returned to the LM by default
    # Candidates per arm before fusion/rerank.  A cross-encoder reranks the whole
    # shortlist, so a deeper shortlist directly buys recall at encyclopedia scale.
    "shortlist": 64,
    "rrf_k": 60,              # Reciprocal Rank Fusion constant
    "min_confidence": 0.0,    # below this top rerank score => "KB lacks this"
    # heuristic | none | cross-encoder.  cross-encoder calls a local reranker
    # endpoint (llama.cpp --reranking / Jina-style /rerank); if it is unreachable
    # the search degrades to the transparent lexical `heuristic` reranker.
    "rerank": "heuristic",
    "rerank_pool": 40,        # card_hybrid: pool depth the cross-encoder reranks (contract §5)
    "rerank_url": "http://127.0.0.1:11439",   # ./run-reranker.sh (llama.cpp --reranking)
    "rerank_model": "bge-reranker-v2-m3-Q8_0.gguf",
    "rerank_timeout_s": 30,
    "snippet_max_len": 1200,  # truncate a returned passage to this many chars

    # ── read-time knowledge modes (regime filter) ───────────────────────────
    # A mode restricts answers to certain epistemic regimes, so e.g. 'science' mode
    # excludes anything from the fiction/essay folders altogether.  `general` keeps
    # everything; per-query override with ?mode=.  Extend/retune freely; an unknown
    # mode keeps everything.  A regime not in any list is kept (never over-excludes).
    "mode": "general",
    # strict=false (default): a mode filters by the CLAIM's regime, so a real technique
    # distilled from a novel survives science mode.  strict=true: filter by source
    # ORIGIN, excluding everything from the fiction/essay folder wholesale.
    "strict": False,
    "modes": {
        "science":    ["empirical", "historical"],
        "scholarly":  ["empirical", "historical", "interpretive", "conventional"],
        "humanities": ["interpretive", "conventional", "historical", "empirical"],
        "fiction":    ["fictional", "interpretive", "conventional"],
    },

    "log_level": "INFO",
}

_PATH_KEYS = {"db_path", "lance_dir", "kb_path"}
_INT_KEYS = {"port", "embed_batch", "embed_timeout_s", "embed_max_tokens",
             "chunk_target_tokens",
             "chunk_max_tokens", "chunk_overlap_tokens", "default_k",
             "shortlist", "rrf_k", "rerank_timeout_s", "distill_timeout_s",
             "distill_max_tokens", "ocr_min_chars", "ingest_log_every",
             "ingest_workers", "ingest_write_batch", "stopword_max", "stopword_min_chunks",
             "extract_timeout_s", "extract_max_tokens",
             "verify_timeout_s", "verify_max_tokens", "ann_min_rows",
             "verify_batch", "verify_source_chars", "atomic_min_count",
             "glucose_min_count", "reconcile_top_k", "reconcile_block",
             "ann_connectivity", "ann_expansion_add", "ann_expansion_search",
             "ann_min_nodes", "ask_pool", "sqlite_mmap_mb", "sqlite_cache_mb",
             "wiki_articles", "wiki_chunk_chars", "wiki_max_chunks", "link_top_k",
             "link_fetch_mult", "link_max_tokens", "refine_source_tokens", "refine_max_tokens",
             "refine_timeout_s", "unsure_alternatives", "rerank_pool"}
_FLOAT_KEYS = {"min_confidence", "node_sim_high", "node_sim_low", "kb_min_sim",
               "conceptnet_trust", "conceptnet_min_weight", "atomic_trust", "counts_cache_ttl",
               "glucose_trust", "reconcile_min_sim", "causenet_trust", "ask_prior_penalty",
               "auto_merge_sim", "adjudicate_escalate_sim",
               "link_min_sim", "link_min_conf", "link_related_min_conf",
               "vinkona_trust", "ask_vinkona_penalty", "unsure_min_sim", "stopword_df_ratio"}
_BOOL_KEYS = {"embed_task_prefix", "ocr", "verify", "strict",
              "conceptnet_include_lexical", "ann_search", "ann_mmap", "wiki_semantic",
              "ask_fit_gate", "use_spacy", "library_dense"}
_LIST_KEYS = {"sources", "extensions", "distill_urls", "extract_urls", "verify_urls",
              "encrypted_bundles", "library_sources", "stopwords_extra"}


def _coerce(key: str, value):
    if key in _INT_KEYS:
        return int(value)
    if key in _FLOAT_KEYS:
        return float(value)
    if key in _BOOL_KEYS:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if key in _LIST_KEYS and isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return value


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


# The ALLOWLIST of settings the web panel may read/write.  This is fail-closed on
# purpose: it is an explicit set of harmless numeric/bool/enum tuning knobs, so
# security-relevant keys can NEVER be edited over HTTP even by a token holder.
# Deliberately EXCLUDED (file/env-only): the network bind (host/port), every endpoint
# URL and model name, every filesystem path, the encryption key file, the auth token,
# the backend, and the scenario selection (use the Bundles tab / config for those).
# A new tuning knob must be added here to appear in the panel — if it isn't listed it
# stays invisible, which is the safe direction.  See test_settings_allowlist.
EDITABLE_SETTINGS = frozenset({
    # embeddings / chunking (sizes & timeouts, not the endpoint)
    "embed_task_prefix", "embed_batch", "embed_timeout_s", "embed_max_tokens",
    "chunk_target_tokens", "chunk_max_tokens", "chunk_overlap_tokens",
    # distillation budgets & the verify pipeline (not URLs/models)
    "verify", "verify_timeout_s", "verify_max_tokens", "verify_batch", "verify_source_chars",
    "extract_timeout_s", "extract_max_tokens", "distill_timeout_s", "distill_max_tokens",
    # node identity / read-abstain thresholds
    "node_sim_high", "node_sim_low", "kb_min_sim",
    # wikipedia arm
    "wiki_articles", "wiki_chunk_chars", "wiki_max_chunks", "wiki_semantic",
    # bulk-import trust/floors (not the file paths)
    "conceptnet_trust", "conceptnet_min_weight", "conceptnet_include_lexical",
    "atomic_trust", "atomic_min_count", "glucose_trust", "glucose_min_count",
    "causenet_trust", "causenet_regime",
    # reconcile / link / refine tuning
    "reconcile_top_k", "reconcile_block", "reconcile_min_sim",
    "link_top_k", "link_fetch_mult", "link_min_sim", "link_min_conf",
    "link_related_min_conf", "link_max_tokens",
    "refine_source_tokens", "refine_max_tokens", "refine_timeout_s",
    # ask ranking / relevance gate
    "ask_prior_penalty", "ask_pool", "ask_fit_gate", "ask_vinkona_penalty", "vinkona_trust",
    "use_spacy",             # the one spaCy knob — on/off, nothing else to tune

    # Tier-3 "unsure" fallback voice/threshold (contract §6.1)
    "unsure_message", "unsure_alternatives", "unsure_min_sim", "unsure_style",
    # ANN read-index tuning (not ann_path)
    "ann_search", "ann_connectivity", "ann_expansion_add", "ann_expansion_search",
    "ann_min_nodes", "ann_dtype", "ann_mmap", "ann_min_rows",
    # SQLite memory / polling
    "counts_cache_ttl", "sqlite_mmap_mb", "sqlite_cache_mb",
    # adjudication similarity gates
    "auto_merge_sim", "adjudicate_escalate_sim",
    # ingestion knobs (not the source paths)
    "ocr", "ocr_min_chars", "ingest_log_every",
    # retrieval / rerank behaviour (not rerank_url/model)
    "default_k", "shortlist", "rrf_k", "min_confidence", "rerank", "rerank_timeout_s",
    "rerank_pool", "snippet_max_len",
    # read-time knowledge mode
    "mode", "strict",
})

# Belt-and-braces: even if a sensitive key were mistakenly added to the allowlist,
# these patterns force it out (fail-closed on the categories).  Suffix/exact matching,
# not substring — so it catches `auth_token`/`embed_url`/`db_key_file` but NOT the
# many legitimate `*_max_tokens` / `*_source_tokens` tuning knobs.
_SENSITIVE_SUFFIX = ("_url", "_urls", "_path", "_dir", "_file", "_key", "_token",
                     "_secret", "_password", "_passwd")
_SENSITIVE_EXACT = frozenset({"host", "port", "auth_token", "backend", "password"})


def _is_sensitive_key(k: str) -> bool:
    return k in _SENSITIVE_EXACT or k.endswith(_SENSITIVE_SUFFIX)


def settings_schema() -> dict:
    """The editable scalar tunables for the Settings panel: key -> {type, default}.
    Restricted to the EDITABLE_SETTINGS allowlist and re-filtered against sensitive
    name patterns, so no endpoint / path / secret / network-bind key is ever
    reachable over HTTP (nested tables like source_regimes stay file-edited)."""
    out = {}
    for k in EDITABLE_SETTINGS:
        if k not in DEFAULTS or _is_sensitive_key(k):   # never surface a sensitive key
            continue
        v = DEFAULTS[k]
        t = ("bool" if isinstance(v, bool) else "int" if isinstance(v, int)
             else "float" if isinstance(v, float) else "str" if isinstance(v, str) else None)
        if t is not None:
            out[k] = {"type": t, "default": v}
    return out


def update_config_file(path: str, updates: dict) -> dict:
    """Update top-level scalar keys in a config.toml IN PLACE, preserving comments and any
    other lines, validating each key/type against the schema.  Keys inside [tables] are left
    untouched; new keys are inserted before the first table header (so they stay top-level).
    Returns the applied {key: value}.  Raises ValueError on an unknown/non-scalar key."""
    import re as _re
    schema = settings_schema()
    coerced: dict = {}
    for k, raw in (updates or {}).items():
        if k not in schema:
            raise ValueError(f"not an editable scalar setting: {k}")
        t = schema[k]["type"]
        if t == "bool":
            coerced[k] = raw if isinstance(raw, bool) else \
                str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif t == "int":
            coerced[k] = int(raw)
        elif t == "float":
            coerced[k] = float(raw)
        else:
            coerced[k] = str(raw)

    p = Path(path).expanduser()
    lines = p.read_text().splitlines() if p.exists() else []
    remaining = dict(coerced)
    applied: dict = {}
    out, in_table, first_table_at = [], False, None
    key_re = _re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*).*$")
    for line in lines:
        if line.lstrip().startswith("["):
            in_table = True
            if first_table_at is None:
                first_table_at = len(out)
        m = key_re.match(line)
        if m and not in_table and m.group(2) in remaining:
            k = m.group(2)
            out.append(f"{m.group(1)}{k} = {_toml_scalar(remaining.pop(k))}")
            applied[k] = coerced[k]
        else:
            out.append(line)
    if remaining:                                       # brand-new keys → keep them top-level
        block = [f"{k} = {_toml_scalar(v)}" for k, v in remaining.items()]
        for k, v in remaining.items():
            applied[k] = v
        at = first_table_at if first_table_at is not None else len(out)
        out[at:at] = (["", "# --- set via the Settings panel ---", *block, ""]
                      if first_table_at is not None else
                      ["", "# --- set via the Settings panel ---", *block])
    p.write_text("\n".join(out) + "\n")
    return applied


# ── web-managed library folders (sandboxed under a file-set library_root) ──────────
# The Settings firewall keeps every path key out of HTTP.  The Library panel is the ONE
# deliberate, narrow exception: it can toggle which immediate subfolders of a trusted,
# file-set `library_root` are indexed.  The web only ever sends bare subfolder NAMES; the
# server maps each to `library_root/<name>` and rejects anything whose realpath isn't
# strictly contained under the root — so a token holder can pick blessed subtrees but can
# never repoint the crawler outside them (no '..', no absolute paths, no symlink escape).

def _library_subdirs(root: str) -> list:
    """Immediate, non-hidden, real (non-symlink) subdirectories of the trusted root,
    sorted by name.  Symlinked subfolders are omitted — the crawler doesn't follow them
    and they could point anywhere, so they're not offered as selectable collections."""
    try:
        with os.scandir(root) as it:
            return sorted(e.name for e in it
                          if e.is_dir(follow_symlinks=False) and not e.name.startswith("."))
    except OSError:
        return []


def library_status(cfg: dict) -> dict:
    """Read-only state for the web Library panel: the trusted root, its immediate
    subfolders, and which are currently indexed (present in library_sources)."""
    root = cfg.get("library_root") or ""
    root_ok = bool(root) and os.path.isdir(root)
    root_real = os.path.realpath(root) if root_ok else ""
    active_real = {os.path.realpath(s) for s in (cfg.get("library_sources") or [])}
    subs = [{"name": n, "active": os.path.realpath(os.path.join(root_real, n)) in active_real}
            for n in (_library_subdirs(root) if root_ok else [])]
    return {"root": root, "root_exists": root_ok, "subdirs": subs,
            "dense": bool(cfg.get("library_dense")),
            "config_path": cfg.get("_config_path")}


def resolve_library_selection(cfg: dict, names) -> list:
    """Map web-supplied subfolder NAMES to absolute paths under the trusted library_root,
    rejecting anything that isn't a real directory strictly contained under it (no path
    separators, no '..', no absolute paths, no symlink escape).  FAIL-CLOSED: raises
    ValueError on the first bad name rather than silently dropping it."""
    root = cfg.get("library_root") or ""
    if not root:
        raise ValueError("library_root is not set — add it to config.toml on the server first")
    root_real = os.path.realpath(root)
    if not os.path.isdir(root_real):
        raise ValueError(f"library_root is not a directory: {root}")
    out, seen = [], set()
    for raw in (names or []):
        name = str(raw).strip()
        if (not name or name in (".", "..") or os.path.isabs(name)
                or "/" in name or "\\" in name or os.path.basename(name) != name):
            raise ValueError(f"not a plain subfolder name: {raw!r}")
        cand = os.path.realpath(os.path.join(root_real, name))
        if cand == root_real or not cand.startswith(root_real + os.sep):
            raise ValueError(f"folder escapes the library root: {name!r}")
        if not os.path.isdir(cand):
            raise ValueError(f"not a directory under the library root: {name!r}")
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def write_library_sources(config_path: str, paths: list) -> list:
    """Persist library_sources as a TOML array in config.toml IN PLACE, preserving comments
    and every other line.  Replaces an existing top-level library_sources of ANY shape
    (bare string, single- or multi-line array); else inserts it before the first [table].
    Only the containment-validated output of resolve_library_selection should reach here."""
    import re as _re
    p = Path(config_path).expanduser()
    arr = "[" + ", ".join(_toml_scalar(x) for x in paths) + "]"
    newline = f"library_sources = {arr}"
    lines = p.read_text().splitlines() if p.exists() else []
    out, in_table, first_table_at, done = [], False, None, False
    key_re = _re.compile(r"^\s*library_sources\s*=")
    it = iter(lines)
    for line in it:
        if line.lstrip().startswith("["):
            in_table = True
            if first_table_at is None:
                first_table_at = len(out)
        if not in_table and not done and key_re.match(line):
            out.append(newline)
            done = True
            depth = line.count("[") - line.count("]")   # swallow a multi-line array value
            while depth > 0:
                try:
                    cont = next(it)
                except StopIteration:
                    break
                depth += cont.count("[") - cont.count("]")
            continue
        out.append(line)
    if not done:
        block = ["", "# --- library folders (set via the Library panel) ---", newline]
        at = first_table_at if first_table_at is not None else len(out)
        out[at:at] = block
    p.write_text("\n".join(out) + "\n")
    return list(paths)


def load_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULTS)

    cfg_path = path or os.environ.get("KNOWLEDGEHOST_CONFIG")
    if cfg_path:
        p = Path(cfg_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with open(p, "rb") as fh:
            file_cfg = tomllib.load(fh)
        for k, v in file_cfg.items():
            if k in cfg:
                cfg[k] = _coerce(k, v)

    for k in list(cfg):
        env = os.environ.get("KNOWLEDGEHOST_" + k.upper())
        if env is not None:
            cfg[k] = _coerce(k, env)

    cfg["extensions"] = [e.lower() if e.startswith(".") else "." + e.lower()
                         for e in cfg["extensions"]]

    # Path resolution: ~ expands; RELATIVE paths anchor to the knowledge-host
    # root (this package's parent), never the caller's cwd — so the defaults
    # ("var/…") land inside this tree no matter where the process starts, and
    # an absolute path in config.toml is honoured verbatim.
    root = Path(__file__).resolve().parent.parent
    def _abspath(v: str) -> str:
        return str(root / Path(v).expanduser())   # Path join: absolute rhs wins

    for k in _PATH_KEYS:
        cfg[k] = _abspath(cfg[k])
    if cfg["zim_path"]:
        cfg["zim_path"] = _abspath(cfg["zim_path"])
    if cfg.get("research_solved_dir"):
        cfg["research_solved_dir"] = _abspath(cfg["research_solved_dir"])
    if cfg.get("library_db"):
        cfg["library_db"] = _abspath(cfg["library_db"])
    if cfg.get("library_root"):
        cfg["library_root"] = _abspath(cfg["library_root"])
    if cfg.get("control_dir"):
        cfg["control_dir"] = _abspath(cfg["control_dir"])
    cfg["library_sources"] = [_abspath(s) for s in cfg.get("library_sources", [])]
    cfg["sources"] = [_abspath(s) for s in cfg["sources"]]
    return cfg
