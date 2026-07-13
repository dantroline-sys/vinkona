# Vinkona — the assistant

A real-time, full-duplex voice assistant running entirely on local hardware: a
classic **cascade** (denoise → VAD → ASR → LLM → TTS) wrapped around persistent
memory, personas, tool calling, and an autonomous background research loop.

This directory is the assistant half of the Vinkona monorepo; the knowledge-base
service lives in [`../knowledge-host/`](../knowledge-host/), and the system-level
overview is in the [top-level README](../README.md).

---

## Architecture

```
┌─────────────────────┐      WSS (WebSocket TLS, :8998)     ┌──────────────────────────────────────┐
│  Flutter client     │ ◄──────────────────────────────────►│  cascade_server.py                   │
│  (vinkona_client/)  │  0x03 float32 PCM both directions   │                                      │
│  • mic capture      │  0x02 JSON chat bubbles             │  mic → RNNoise denoise → VAD →       │
│  • audio playback   │  0x04 typed text (?mode=text)       │  faster-whisper ASR → llm_bridge     │
│  • chat transcript  │  first frame: 0x01 + access token   │  → sentence-by-sentence → TTS        │
└─────────────────────┘                                     └──────┬───────────────────┬───────────┘
                                                                   │                   │
                                            ┌──────────────────────▼───┐   ┌───────────▼──────────┐
                                            │ llm_bridge.py            │   │ tts_server.py :11436 │
                                            │ • fast LM: streamed      │   │ Orpheus (llama.cpp   │
                                            │   real-time replies      │   │ :11439 + SNAC) or    │
                                            │ • big LM: background     │   │ NeuTTS —             │
                                            │   briefing, reflection,  │   │ 24 kHz PCM           │
                                            │   deep reasoning         │   └──────────────────────┘
                                            └───┬──────────────┬───────┘
                                                │              │
                                   ┌────────────▼──┐  ┌────────▼───────────┐
                                   │ fast LM :11435│  │ big LM :11438      │
                                   │ llama.cpp     │  │ llama.cpp          │
                                   │ (llm_server)  │  │ embed LM :11437    │
                                   └───────────────┘  └────────────────────┘

  memory.py — SQLite (WAL): memories + trigger recall (Aho-Corasick) + embeddings,
              people, calendar, news, user model, research queue
  research_worker.py — separate process: idle deep-research, exports solved/*.md
              to the knowledge-host, reflects on what was learned
  config_server.py :8090 — web UI: Personas, Models, Live trace, Memory, Settings
  tools_client.py — tool-host contract (MAC_TOOLS.md): knowledge-host, Mac tools,
              music host … whatever implements GET /tools + POST /call
```

Every stage is a separate process with its own environment (see
[`ENVIRONMENTS.md`](ENVIRONMENTS.md)); [`vinkona.sh`](vinkona.sh) orchestrates
them all in one tmux session.

### Why a cascade?

An earlier iteration ran on a full-duplex speech-to-speech model; it fought
injected LLM tokens, so the project pivoted to a cascade: less exotic, fully
controllable, and every tier is swappable from config. Latency is engineered
instead: streamed ASR, sub-150 ms-TTFT fast LM, sentence-by-sentence TTS, and
barge-in via VAD.

### The two-tier LM design

- **Fast LM** answers in real time and can call tools (knowledge base, calendar,
  news, music, …) mid-turn.
- **Big LM** (the ~9B "mind") never blocks the voice path: it writes context
  briefings for the next turn, distills memories at session end, does deep
  research reasoning, and runs the reflective loops described in
  [`CONSCIOUS_REASONING.md`](CONSCIOUS_REASONING.md).

## What's in here

| Area | Files | Docs |
|---|---|---|
| Voice cascade | `cascade_server.py`, `asr.py`, `rnnoise_frontend.py`, `llm_bridge.py`, `tts_server.py`, `tts_orpheus_gguf.py`, `tts_orpheus.py`, `tts_neutts.py` | — |
| LM serving (llama.cpp) | `llm_server.py`, `serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_embed.sh` | [`ENVIRONMENTS.md`](ENVIRONMENTS.md) |
| Memory & people | `memory.py`, `people.py`, `news_store.py`, `calendar_sync.py`, `calendar_resolve.py` | [`MEMORY_CONSOLIDATION.md`](MEMORY_CONSOLIDATION.md) |
| Conscious reasoning | `user_model.py`, `research_reflection.py`, `retrieval_confidence.py` | [`CONSCIOUS_REASONING.md`](CONSCIOUS_REASONING.md), [`USER_MODEL_INTEGRATION.md`](USER_MODEL_INTEGRATION.md) |
| Research loop | `research_worker.py`, `research_export.py`, `capture.py` | [`research_loop_spec.md`](research_loop_spec.md) |
| Tools & hosts | `tools_client.py`, `knowledge_host.py`, `safety.py`, `wsauth.py` | [`MAC_TOOLS.md`](MAC_TOOLS.md), [`KNOWLEDGE.md`](KNOWLEDGE.md), [`MUSIC.md`](MUSIC.md), [`WS_AUTH.md`](WS_AUTH.md) |
| Awareness | `timesense.py`, `spoken_time.py`, `ambient.py` | [`NOTIFICATIONS.md`](NOTIFICATIONS.md) |
| Config & UI | `config.py`, `config_server.py`, `config_ui.html`, `chat_ui.html` | — |
| Client | `vinkona_client/` (Flutter) | — |
| Tests | `test_*.py` (stdlib-only, no pip installs) | — |

## Setup

### 1. Install

```bash
./install.sh               # core: vinkona_env + cascade/ASR/memory deps + rnnoise (in-tree)
./install.sh tts           # TTS: orpheus_gguf (llama.cpp + SNAC — default, no extra venv)
                           #   or: tts neutts (cloned voice, own venv)
./install.sh models        # download the default GGUFs into Models/
./install.sh llama         # build llama.cpp's llama-server into ./bin (if not on PATH)
# or all four at once:  ./install.sh all
# later:                ./install.sh status | uninstall [--with-models] [--purge]
```

Each step is also a standalone script (`install_orpheus_gguf.sh`,
`install_rnnoise.sh`, `fetch_models.sh`, …) — `install.sh` orchestrates them.

**TTS engines.** The default `orpheus_gguf` runs the Orpheus 3B backbone as a
GGUF on a plain llama-server (the `tts_lm` tier, port 11439) and vocodes with
SNAC via onnxruntime on the CPU — preset voices and `<laugh>`/`<sigh>` tags, no
separate venv, no Python-version pin, and it's the path that can port to macOS
(Metal). The alternative `neutts` engine clones a voice from a reference clip;
it needs torch, so it lives in its own venv — see
[`ENVIRONMENTS.md`](ENVIRONMENTS.md). Switch engines with `tts.engine` in the
config UI; `vinkona.sh` starts the right services for whichever is set.

**Filesystem guarantee:** everything the assistant writes stays inside this
folder — live config, personas and memory in `config/`, weights in `Models/`,
logs in `logs/`, caches/builds/temp in `var/`, binaries in `bin/`, venvs in
`*_env/`. That includes third-party stacks: HuggingFace downloads, torch
compile caches and temp files are pinned in-tree by [`env.sh`](env.sh), which
every service sources. Reads can come from anywhere (e.g. symlink `Models/` to
your weight store). The only exceptions are system packages you install
yourself (espeak-ng, C toolchain) and tmux's own socket.
`./install.sh uninstall` removes what was installed and keeps your data;
`--purge` removes the data too; deleting the folder removes every trace.

### 2. Models

GGUF weights live in `Models/` (git-ignored; symlink to wherever you store
them). Pick your tiers in the config UI's Models tab or `config.py` DEFAULTS —
each tier (fast/big/embed) has its own model, GPU, and context settings.

### 3. Configuration

`./install.sh` seeds `config/config.json` + `config/personas.json` from the
examples (they're user data, never committed). Edit them directly or in the
web UI (`./serve_config.sh`, then http://localhost:8090).

### 4. Run everything

```bash
./vinkona.sh start        # whole stack in tmux ("vinkona" session)
./vinkona.sh status       # what's running
./vinkona.sh restart tts  # bounce one service
```

Services can also be started individually: `serve_cascade.sh`,
`serve_fast_lm.sh`, `serve_big_lm.sh`, `serve_embed.sh`, `serve_tts.sh`,
`serve_config.sh`, `serve_research.sh`.

### 5. Client

```bash
cd vinkona_client
flutter pub get
flutter run                # or build an APK and sideload
```

On first server start a human-typable access token is generated and printed
(and written to `config/ws_token.txt`) — enter it once in the client
([`WS_AUTH.md`](WS_AUTH.md)). There's also a browser text client at
`chat_ui.html` (`?mode=text`).

## Security & privacy posture

- Everything runs locally; the only outbound traffic is explicit research
  fetches from keyless sources, and those queries pass `safety.query_privacy()`
  first (masks emails, phone numbers, long numbers, known private names).
- All external content (web, documents, KB passages) is wrapped as untrusted
  data — fenced, sanitized of chat-template control tokens, and never treated
  as instructions (`safety.py`).
- The WebSocket requires a pre-shared token (`wsauth.py`); the config UI binds
  to localhost only.
- `config/` (live config, personas, memory.db, trace, token) is git-ignored
  user data.

## Tests

Every subsystem has a stdlib-only self-test — no pip installs, no GPUs needed:

```bash
python3 test_safety.py
python3 test_people.py
python3 test_idle_learning.py
# … see test_*.py
```

