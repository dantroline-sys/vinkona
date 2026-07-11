# Vinkona

A local, private voice assistant that learns its user — and the knowledge engine
behind it. Everything runs on your own hardware: speech recognition, language
models, speech synthesis, memory, and research. Nothing about you is sent to a
cloud service.

Vinkona is built around a deliberately **small** language model (~9B). The bet is
that *understanding isn't about model size — it's about explicit, persistent,
personalized context*. A small model with an explicit user model, a reflective
research loop, calibrated retrieval confidence, and a curated personal knowledge
base behaves as though it "gets" you — and it fits on hardware most people
already own, alongside a lower-tier TTS model.

## The two components

| Directory | What it is |
|---|---|
| [`assistant/`](assistant/) | The voice assistant: a real-time local cascade (denoise → VAD → ASR → fast LM → TTS, with a big LM reasoning in the background), persistent memory, personas, a Flutter client, a config web UI, and an autonomous research worker. |
| [`knowledge-host/`](knowledge-host/) | A standalone knowledge-base service: ingests Wikipedia snapshots, PDFs, books and the assistant's own research; distills them into typed, cited knowledge cards; answers `kb_search`/`kb_ask` over HTTP with trust tiers, facets, and conflict checking. |

They are separate services that speak a tiny, shared **tool-host contract**
(`GET /tools` + `POST /call`, documented in
[`assistant/MAC_TOOLS.md`](assistant/MAC_TOOLS.md)). Anything that implements the
contract plugs into the assistant as a tool the fast LM can call mid-conversation.

## Architecture

```
                   ┌──────────────────────────────────────────────────────────┐
                   │                     assistant/                           │
 Flutter client    │                                                          │
 ┌─────────────┐   │  mic → RNNoise → VAD → faster-whisper ASR                │
 │ mic/speaker │◄──┼──► cascade_server.py (WebSocket, :8998)                  │
 └─────────────┘   │        │                                                 │
                   │        ▼                                                 │
                   │  llm_bridge.py ── fast LM (llama.cpp, :11435) ── TTS     │
                   │        │              sentence-by-sentence      (:11436) │
                   │        ▼                                                 │
                   │  big LM (:11438) — background briefing, reflection,      │
                   │  deep reasoning, memory consolidation                    │
                   │        │                                                 │
                   │  memory.db (SQLite) — memories, people, calendar, news,  │
                   │  user model, research queue                              │
                   │        │                                                 │
                   │  research_worker.py — idle research → solved/*.md drops  │
                   └────────┼─────────────────────────────────────────────────┘
                            │ tool-host contract (GET /tools + POST /call)
          ┌─────────────────┼───────────────────────┬─────────────────────┐
          ▼                 ▼                       ▼                     ▼
   knowledge-host/    Mac tool host*          music host*         (your own hosts)
   kb_search/kb_ask   calendar, mail,         local library
   cards + citations  files, keyless          search & queue
   (:8771)            research sources
                      (* not included — see "External tool hosts")
```

The research loop closes on itself: conversations raise questions → the research
worker investigates them during idle time → findings are exported to the
knowledge-host, which distills them into cited cards → the assistant retrieves
those cards (with confidence scores) in later conversations → periodic
reflection reviews what was learned and updates the user model.

## What makes it feel like it "gets" you

Three composable systems (see
[`assistant/CONSCIOUS_REASONING.md`](assistant/CONSCIOUS_REASONING.md)):

- **User model** (`assistant/user_model.py`) — explicit, persistent tracking of
  your domain fluency, communication preferences, corrections, and whether you
  acted on advice. Injected into the big LM's prompts so responses are tailored
  to *you*, not a generic user.
- **Research reflection** (`assistant/research_reflection.py`) — the assistant
  periodically reviews its own recent research, synthesizes what it learned,
  and adjusts its model of you and its future research direction.
- **Retrieval confidence** (`assistant/retrieval_confidence.py`) — retrieved
  knowledge is scored for recency, source convergence, domain fit, and
  base-rate usefulness, so the assistant can say "I'm fairly confident" or
  "verify this elsewhere" and mean it.

## Design principles

- **Local and private.** All models run locally (llama.cpp, faster-whisper,
  vLLM). The only outbound traffic is explicit, keyless research fetches
  (Wikipedia, OpenAlex, PubMed, …) — and outbound queries pass a privacy filter
  that masks emails, phone numbers, and known private names first.
- **Personal content is firewalled.** Your memory database, live config,
  personas, and research drops are user data, never source — they are
  git-ignored and stay on your machine.
- **Data, never instructions.** Everything retrieved from outside (web pages,
  documents, the knowledge base) is sanitized and fenced as untrusted before
  any LM reads it.
- **Small-model scaffolding.** Explicit context (user model, cited cards,
  confidence bounds) instead of raw parameter count.
- **Separate stores, separate trust.** The assistant's personal memory and the
  knowledge host's reference knowledge are different databases with different
  trust tiers; bulk knowledge can never overwrite personal fact.

## External tool hosts (referenced, not included)

The assistant talks to tool hosts over the contract in
[`assistant/MAC_TOOLS.md`](assistant/MAC_TOOLS.md). Two hosts used in
development are **not** part of this repository:

- **Mac tool host** — calendar, reminders, mail, file search, and keyless
  research sources (OpenAlex, Europe PMC, StackExchange, GDELT, Wikidata,
  Internet Archive, …). Any implementation of the contract works; back it with
  whatever your platform offers (e.g. MCP servers).
- **Music host** — local music library search and playback queue, per
  [`assistant/MUSIC.md`](assistant/MUSIC.md) (Surface 1).

The knowledge-host in this repository implements the same contract and is the
reference example of writing a host.

## Getting started

Each component has its own README and install scripts:

- **Assistant:** [`assistant/README.md`](assistant/README.md) — Python venvs
  ([`assistant/ENVIRONMENTS.md`](assistant/ENVIRONMENTS.md)), model downloads,
  the `vinkona.sh` tmux orchestrator, and the Flutter client.
- **Knowledge host:** [`knowledge-host/README.md`](knowledge-host/README.md) —
  install, ingest your documents (and optionally a Wikipedia ZIM), and serve.
  Large third-party datasets used by optional importers are documented in
  [`knowledge-host/external/README.md`](knowledge-host/external/README.md) and
  are downloaded separately.

Rough hardware guide: the live voice path (fast LM + embeddings + TTS) fits on
one consumer GPU; the big LM prefers a second GPU but is off the latency path,
so slower/CPU setups degrade gracefully. The knowledge host's query service is
CPU-friendly; heavy ingestion borrows the LMs when the voice path is idle.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
