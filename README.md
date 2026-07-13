# Vinkona

A local, private voice assistant that learns its user. Everything runs on your
own hardware: speech recognition, language models, speech synthesis, memory,
and research. Nothing about you is sent to a cloud service.

> **Vinkona and Vinur.** Until 2026-07-13 this repository also contained the
> knowledge host. It now lives in its own repository,
> [**Vinur**](https://github.com/dantroline-sys/vinur) (*vinur* and *vinkona*
> are Icelandic for a friend), and the pair are licensed separately: Vinur — a
> more or less headless knowledge API — stays **Apache 2.0**; Vinkona, the
> user-facing front-end, is **PolyForm Noncommercial 1.0.0** from the split
> onward (all earlier commits remain Apache 2.0 — see [LICENSE](LICENSE)).
> Vinkona works without Vinur, and talks to it only over the tool-host HTTP
> contract when it's there.

Vinkona is built around a deliberately **small** language model (~9B). The bet is
that *understanding isn't about model size — it's about explicit, persistent,
personalized context*. A small model with an explicit user model, a reflective
research loop, calibrated retrieval confidence, and a curated personal knowledge
base behaves as though it "gets" you — and it fits on hardware most people
already own, alongside a lower-tier TTS model.

## The two components

| Where | What it is |
|---|---|
| [`assistant/`](assistant/) (this repo) | The voice assistant: a real-time local cascade (denoise → VAD → ASR → fast LM → TTS, with a big LM reasoning in the background), persistent memory, personas, a Flutter client, a config web UI, and an autonomous research worker. |
| [Vinur](https://github.com/dantroline-sys/vinur) (its own repo, Apache 2.0) | A standalone knowledge-base service: ingests Wikipedia snapshots, PDFs, books and the assistant's own research; distills them into typed, cited knowledge cards; answers `kb_search`/`kb_ask` over HTTP with trust tiers, facets, and conflict checking. |

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
   Vinur (own repo)   Mac tool host*          music host*         (your own hosts)
   kb_search/kb_ask   calendar, mail,         local library
   cards + citations  files, keyless          search & queue
   (:8771)            research sources
                      (* not included — see "External tool hosts")
```

The research loop closes on itself: conversations raise questions → the research
worker investigates them during idle time → findings are exported to Vinur,
which distills them into cited cards → the assistant retrieves
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

- **Local and private.** All models run locally (llama.cpp, faster-whisper).
  The only outbound traffic is explicit, keyless research fetches
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

[Vinur](https://github.com/dantroline-sys/vinur) implements the same contract
and is the reference example of writing a host.

## Getting started

```bash
./install.sh        # interactive checklist — pick off tasks until everything is green
./vinkona.sh start  # then start the system (asks once what THIS machine runs:
                    #   everything / assistant only / knowledge host only —
                    #   so the knowledge host can live on a separate device)
./vinkona.sh status # what's up;  stop / restart / attach / services also available
./install.sh status # installer state;  ./install.sh uninstall  to undo
```

To run the knowledge host too, clone [Vinur](https://github.com/dantroline-sys/vinur)
**next to** this repository (`../vinur`) — the installer and orchestrator find
it there (or wherever `$VINUR_DIR` points) and manage it alongside the
assistant. Vinur is also entirely usable on its own.

The top-level installer drives one installer per component (each also usable
directly, with its own uninstall):

- **Assistant** ([`assistant/README.md`](assistant/README.md),
  [`assistant/ENVIRONMENTS.md`](assistant/ENVIRONMENTS.md)) — venvs,
  dependencies, models, an in-tree llama.cpp build. Run the stack with the
  `vinkona.sh` tmux orchestrator.
- **Knowledge host** ([Vinur](https://github.com/dantroline-sys/vinur), cloned
  alongside) — venv + config, then its `./ingest.sh` and `./run.sh`. Large
  third-party datasets used by optional importers are documented in its
  `external/README.md` and are downloaded separately.

**Filesystem guarantee:** everything Vinkona writes — config, memory, models,
indexes, caches, logs, temp files — stays inside this folder tree; nothing
lands in `~/.cache`, `/usr/local`, or anywhere else (see each component's
`env.sh`). Reads can come from wherever you point them. `./install.sh
uninstall` in each component removes what was installed; deleting the folder
removes every trace.

Rough hardware guide: the live voice path (fast LM + embeddings + TTS) fits on
one consumer GPU; the big LM prefers a second GPU but is off the latency path,
so slower/CPU setups degrade gracefully. The knowledge host's query service is
CPU-friendly; heavy ingestion borrows the LMs when the voice path is idle.

## Platforms

Linux is the reference platform. On Linux+NVIDIA setups the Python services can
run inside a distrobox container that carries the CUDA userland; without one,
everything runs directly on the host — `vinkona.sh` detects the missing
container and places services accordingly.

**macOS** works host-only: llama.cpp uses Metal automatically (`./install.sh
llama` builds it in-tree), uv downloads any Python it needs, and the CUDA/
container logic simply never engages. You'll want Homebrew for the handful of
system tools (tmux, cmake, autotools for the optional rnnoise build) — the
installers name the exact packages and offer to install them. One caveat:
faster-whisper runs CPU-only on macOS (CTranslate2 has no Metal backend), which
is fine for the small ASR models the cascade uses.

**Windows** is planned: the Python layer is already portable (uv + one
lockfile), but service orchestration is bash+tmux and will move to a small
cross-platform supervisor first.

## Disclaimer

This software is provided as-is, for research and reference purposes, without
warranty, and is not validated or intended for production or safety-critical
use.

## License

PolyForm Noncommercial License 1.0.0 — see [LICENSE](LICENSE). Free for
personal, research, educational, and other noncommercial use; the source stays
open to read and audit. Commercial licensing: contact the author.

History: everything up to and including commit `1ec8d93` (2026-07-12) was
published under the Apache License 2.0 and remains available under it. The
knowledge host continues under Apache 2.0 in its own repository,
[Vinur](https://github.com/dantroline-sys/vinur).
