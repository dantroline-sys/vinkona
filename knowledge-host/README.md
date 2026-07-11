# Vinkona Knowledge Host — a local general-knowledge tool host

A standalone service that gives Vinkona a large, **local, offline** knowledge base
she can search mid-conversation — a **Wikipedia snapshot** plus the user's own
**PDFs, books, journals and documents** — returning **cited passages**. It speaks
the standard Vinkona tool-host contract (`GET /tools` + `POST /call`, see
[`../assistant/MAC_TOOLS.md`](../assistant/MAC_TOOLS.md)), so it is just another host in
Vinkona's `MultiHost`: point `knowledge.tool_url` at it and the fast LM can call
`kb_search` like any other tool. This is the build of
[`../assistant/KNOWLEDGE.md`](../assistant/KNOWLEDGE.md).

It is a **separate store from Vinkona's `memories`**: bulk, low-trust, reference-
only, with its own ANN/FTS index. It returns **data, never instructions** —
every passage is sanitized and cited before any LM reads it.

## Two halves

- **Query service** (`serve`) — light, fast, always up. The tool Vinkona calls.
- **Ingestion pipeline** (`ingest`) — heavy, batch, run on demand / monthly.

## Two store backends (one interface)

| backend  | sparse | dense | needs | use for |
|----------|--------|-------|-------|---------|
| `sqlite` *(default)* | FTS5 | brute-force (numpy or pure-python) | **nothing** beyond the stdlib | the PDF collection / Phase 1; fine to ~1M chunks |
| `lance`  | LanceDB FTS | IVF-PQ, on-disk/mmap | `pip install lancedb pyarrow` | a **full Wikipedia** snapshot (10–40M chunks) |

Retrieval is **hybrid** on either: a dense (embedding) arm and a sparse (BM25/
FTS) arm, fused by **Reciprocal Rank Fusion**, then reranked. FTS carries exact
terms (proper names, IDs) where embeddings are weakest; fusion is the biggest
quality lever at encyclopedia scale.

## Quick start

```bash
cp config.example.toml config.toml      # set sources, backend, embed_url
# (optional, for fast dense + scale)
pip install numpy                        # or: pip install lancedb pyarrow pymupdf ...

# 1) ingest your documents (incremental; only new/changed files are processed)
./ingest.sh                              # crawls config's `sources`
./ingest.sh --wikipedia                  # also a configured Kiwix ZIM
./ingest.sh --wikipedia --limit 500      # cap ZIM articles (smoke)

# 2) serve the query tool
./run.sh                                 # http://127.0.0.1:8771
```

The embed endpoint (nomic at `127.0.0.1:11437`, shared with Vinkona's memory
store) is **optional**: if it's down, ingestion and search run **sparse-only**
(FTS) and log it; re-run `ingest` once it's up to add the dense vectors.

## Verify it (zero installs)

```bash
bash tests/make_fixtures.sh /tmp/kb-fixtures   # tiny txt/md/html corpus
python3 tests/smoke.py /tmp/kb-fixtures        # ingest -> kb_search -> cited passages
```

```bash
curl -s localhost:8771/health
curl -s localhost:8771/tools
curl -s -X POST localhost:8771/call -H 'Content-Type: application/json' \
     -d '{"name":"kb_search","arguments":{"query":"who discovered the Krebs cycle","k":3}}'
```

`kb_search` returns a JSON object (string-encoded per the contract):

```json
{ "passages": [ {"text","title","section","path_or_url","source_type","score"}, … ],
  "confidence": 0.62, "low_confidence": false, "dense_used": true }
```

`confidence` is the top rerank score; **`low_confidence`** is the signal for
Vinkona to fall back to web search instead of answering from a weak passage.

## Ingestion: what's supported

| format | extractor | dependency (lazy) |
|--------|-----------|-------------------|
| `.txt` / `.md` | section split on Markdown headings | stdlib |
| `.html` / `.htm` | `trafilatura` else a stdlib `html.parser` sectioner | `trafilatura` (optional) |
| `.pdf`  | PyMuPDF text layer + TOC sections; **OCR fallback** for scanned pages | `pymupdf`; `ocrmypdf`/`tesseract` on PATH |
| `.epub` | `ebooklib`, chapters through the HTML sectioner | `ebooklib` |
| Wikipedia | Kiwix **ZIM** (pre-rendered HTML) via `libzim`, split on `<h2>/<h3>` | `libzim` |

A **manifest** (path, content_hash, mtime, version) makes every run incremental;
chunk ids are `sha1(path+section+text)` so re-ingest is idempotent. A monthly
Wikipedia refresh: drop in the new ZIM, `bump-version`, re-ingest.

## Wiring into Vinkona

Already wired (see `../assistant/config.py` `knowledge` block and the `MultiHost`
build in `cascade_server.py`):

```toml
knowledge = { enabled = true, tool_url = "http://127.0.0.1:8771" }
```

On the research path, prefer `kb_search` **before** the web (local-first); use
web for recency or when `low_confidence` is set.

## Security

- All ingested content is **UNTRUSTED**; the tool returns data, fenced as low-
  trust by Vinkona before any LM reads it. Passages can colour an answer, never
  issue commands.
- Filenames are treated as **opaque data** — never shelled or prompt-interpolated.
- Service is **read-only and localhost-bound**; optional Bearer token on `/call`.
- Keep parsers (PyMuPDF/Tesseract) patched; parsing needs no network.

## Layout

```
knowledgehost/
  config.py     defaults < TOML < env (KNOWLEDGEHOST_*)
  embed.py      nomic /v1/embeddings client (stdlib urllib; search_query/document prefixes)
  chunk.py      section-aware chunking + stable idempotent ids
  store.py      SqliteStore + LanceStore behind make_store(); shared SQLite manifest
  rerank.py     RRF fusion + intent-conditioned heuristic reranker (cross-encoder = drop-in)
  ingest.py     incremental crawl + Wikipedia ZIM; sanitize -> chunk -> embed -> upsert
  tools.py      the kb_search tool (embed+FTS -> fuse -> rerank -> cited passages + confidence)
  server.py     stdlib HTTP: /health /tools /call
  sources/      pdf, epub, html, text, wikipedia extractors (heavy deps lazy)
tests/          make_fixtures.sh + smoke.py (zero-install end-to-end)
```
