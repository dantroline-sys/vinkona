"""
Central configuration loader for the Vinkona cascade.

One JSON file (config/config.json) is the source of truth for every service —
cascade server, TTS service, the LM launcher, and the config web UI all read it
through here.  A full default schema lives in DEFAULTS, so config.json only needs
to hold the keys you want to change; load_config() deep-merges your overrides.

Language models run as llama.cpp `llama-server` processes (OpenAI-compatible API),
one per tier (fast / big / embed).  Each tier block says which GGUF to load (from
models_dir), which GPU to pin it to, and llama.cpp knobs (ctx_size, n_gpu_layers,
flash_attn, extra_args).  llm_server.py turns a tier block into a llama-server
command; the cascade/memory just speak HTTP to the tier's url.

Personas live in their own file (config/personas.json) so the web UI can edit
prompts independently; its path is config.personas_path.
"""

# Runs under the SYSTEM python3 in some deployments (macOS = 3.9): keep
# annotations lazy and the file 3.9-clean (test_supervisor_compat.py gates it).
from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

CONFIG_PATH = "config/config.json"

# ── Profiles ─────────────────────────────────────────────────────────────────
# A "profile" is a self-contained bundle of the assistant's memory DB + its
# personalities, under config/profiles/<name>/.  The active profile (named in
# config/active_profile, a file kept out of config.json so saves never touch it)
# decides which memory.db and personas.json every service reads — so you can keep
# snapshots, fork, or start fresh without losing old work.  Models/ports/tuning
# stay global in config.json.
_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = _ROOT / "config" / "profiles"
ACTIVE_PROFILE_FILE = _ROOT / "config" / "active_profile"
_EMPTY_PERSONAS = '{"default": "vinkona", "personas": {}}\n'


def _safe_profile_name(name: str) -> str:
    name = (name or "").strip()
    if name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name):
        raise ValueError(f"invalid profile name: {name!r} (use letters, digits, . _ -)")
    return name


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / _safe_profile_name(name)


def list_profiles() -> list[str]:
    return sorted(p.name for p in PROFILES_DIR.iterdir() if p.is_dir()) if PROFILES_DIR.is_dir() else []


def active_profile() -> str:
    try:
        n = ACTIVE_PROFILE_FILE.read_text().strip()
        return _safe_profile_name(n) if n else "default"
    except Exception:
        return "default"


def ensure_profile(name: str) -> Path:
    """Create the profile dir (and seed personas.json) if missing.  Returns its dir."""
    pdir = profile_dir(name)
    pdir.mkdir(parents=True, exist_ok=True)
    personas = pdir / "personas.json"
    if not personas.exists():
        # Seed from whatever the global personas template is, so a fresh profile has
        # a working default persona rather than an empty cast.
        src = resolve_read(_ROOT / "config" / "personas.json")
        try:
            personas.write_text(src.read_text() if src.exists() else _EMPTY_PERSONAS)
        except Exception:
            personas.write_text(_EMPTY_PERSONAS)
    return pdir


def ensure_profiles_bootstrap() -> None:
    """One-time setup: create profiles/default, migrating any pre-profiles
    config/memory.db (+ WAL sidecars) and personas.json into it, then write the
    active pointer.  Idempotent and cheap (fast-paths out) after the first run."""
    default_dir = profile_dir("default")
    if ACTIVE_PROFILE_FILE.exists() and default_dir.is_dir():
        return                                           # already bootstrapped
    default_dir.mkdir(parents=True, exist_ok=True)
    legacy_db = _ROOT / "config" / "memory.db"
    if legacy_db.exists() and not (default_dir / "memory.db").exists():
        for suffix in ("", "-wal", "-shm"):              # move the DB + its WAL sidecars
            src = Path(str(legacy_db) + suffix)
            if src.exists():
                try:
                    src.replace(default_dir / src.name)  # atomic on the same filesystem
                except Exception:
                    pass
    legacy_personas = _ROOT / "config" / "personas.json"
    if legacy_personas.exists() and not (default_dir / "personas.json").exists():
        try:
            legacy_personas.replace(default_dir / "personas.json")
        except Exception:
            pass
    ensure_profile("default")                            # seed personas if still missing
    if not ACTIVE_PROFILE_FILE.exists():
        _write_active("default")


def _write_active(name: str) -> None:
    tmp = ACTIVE_PROFILE_FILE.with_suffix(".tmp")
    tmp.write_text(_safe_profile_name(name) + "\n")
    tmp.replace(ACTIVE_PROFILE_FILE)                     # atomic pointer flip


def set_active_profile(name: str) -> str:
    name = _safe_profile_name(name)
    ensure_profile(name)
    _write_active(name)
    return name


def create_profile(name: str) -> Path:
    name = _safe_profile_name(name)
    if profile_dir(name).exists():
        raise ValueError(f"profile already exists: {name}")
    return ensure_profile(name)


def duplicate_profile(src: str, dst: str) -> Path:
    """Snapshot one profile under a new name (memory DB + personas).  The source DB is
    WAL-checkpointed first so the copy is consistent even while the cascade has it open."""
    sdir, ddir = profile_dir(src), profile_dir(dst)
    if not sdir.is_dir():
        raise ValueError(f"no such profile: {src}")
    if ddir.exists():
        raise ValueError(f"profile already exists: {dst}")
    ddir.mkdir(parents=True)
    sdb = sdir / "memory.db"
    if sdb.exists():
        try:
            c = sqlite3.connect(str(sdb), timeout=10)
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # flush WAL into the main file
            c.close()
        except Exception:
            pass
        shutil.copy2(sdb, ddir / "memory.db")
    sp = sdir / "personas.json"
    if sp.exists():
        shutil.copy2(sp, ddir / "personas.json")
    else:
        ensure_profile(dst)
    return ddir


def delete_profile(name: str) -> None:
    name = _safe_profile_name(name)
    if name == active_profile():
        raise ValueError("cannot delete the active profile — switch away first")
    shutil.rmtree(profile_dir(name), ignore_errors=True)


def profile_stats(name: str) -> dict:
    """Lightweight summary for the UI: memory + persona counts, size, active flag."""
    pdir = profile_dir(name)
    db = pdir / "memory.db"
    n, size = -1, 0
    if db.exists():
        size = db.stat().st_size
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
            n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            c.close()
        except Exception:
            n = -1                                       # exists but unreadable/empty schema
    pc = 0
    try:
        pj = pdir / "personas.json"
        if pj.exists():
            pc = len(json.loads(pj.read_text()).get("personas", {}))
    except Exception:
        pass
    return {"name": name, "memories": n, "personas": pc, "size": size,
            "active": name == active_profile()}

# Tiers that llm_server.py can launch (config key → human label).
LM_TIERS = {"fast_lm": "fast", "big_lm": "big", "big_lm2": "big2", "embed_lm": "embed",
            "tts_lm": "tts-lm"}

# Canonical schema + defaults.  config.json may override any subset.
DEFAULTS: dict = {
    "server": {
        "host": "0.0.0.0",
        "port": 8998,
        "ssl_dir": "certs",
        # The WS listens on the network so the phone can reach it; require a pre-shared
        # token (the client's first frame) so only your devices can connect.  The server
        # generates one on first run and writes it to token_file — read it and type it
        # into the client once.  Set require_auth False only on a trusted, isolated host.
        "auth": {
            "require_auth": True,
            "token": None,                   # None → generated and persisted to token_file
            "token_file": "config/ws_token.txt",
            "handshake_timeout_s": 10,       # client must present the token within this
        },
    },
    "audio": {
        "mic_gain": 4.0,
        "out_prime": 2,              # output jitter-buffer depth in frames (~80 ms each)
        # OPTIONAL extra onset grace (default 0 = off): how long to let the buffer fill to
        # out_prime before pacing.  It hides a GPU-contended first word but is paid at
        # EVERY burst start (each sentence boundary), so it adds per-sentence latency —
        # raise it only a little (e.g. 80–160) if the very first word still stutters.
        "out_prime_fill_ms": 0,
    },
    "vad": {
        "onset_prob": 0.6,
        "offset_prob": 0.35,
        "onset_rms": 0.008,         # fallback when the denoiser is off
        "onset_frames": 2,          # consecutive voice frames before capture starts (debounce)
        "barge_in_frames": 3,       # sustained voice frames before the AI is interrupted
                                    # (>= onset_frames; higher = noise blips don't cut the AI)
        "offset_frames": 6,         # silent frames that end a turn (~480 ms)
        "asr_min_frames": 4,        # skip sub-320 ms blips
        "asr_preroll_frames": 3,    # frames prepended so the first word isn't clipped
    },
    # Tier-1 situational awareness injected into the fast-LM system prompt each turn.
    "awareness": {
        "inject_time": True,        # current date/time/day-of-week (the LM has no clock)
        "location": None,           # optional, e.g. "Bristol, UK" → "...speaking with you in Bristol, UK"
        # Time-sense Phase 1 — the SEMANTIC clock: enrich the bare time with part-of-day,
        # weekend vs work-week, season, and (if the optional libs are present) sunrise/sunset
        # and today's public holiday.  Deterministic + local; later phases add learned rhythms.
        "time_meaning": True,
        "latitude": None,           # optional — sunrise/sunset without geocoding the location
        "longitude": None,          #   (negative latitude is treated as southern hemisphere)
        "holidays_country": None,   # e.g. "GB", "US" — names today's holiday (needs `holidays`)
    },
    "asr": {
        "model": "base.en",
        "beam_size": 1,             # greedy keeps per-turn latency low
        "condition_on_previous_text": False,   # don't let one turn bias the next
        # faster-whisper bundles Silero VAD — this runs it on the captured clip so
        # non-speech (fan/keyboard/cloth) is dropped before transcription.
        "vad_filter": True,
        "vad_threshold": 0.5,       # Silero speech probability (0..1; raise = stricter)
        "min_speech_ms": 250,       # drop "speech" shorter than this
        "min_silence_ms": 100,
        # Decode-confidence gates (also used to drop hallucinated segments):
        "no_speech_threshold": 0.6, # LOWER = suppress more aggressively as silence
        "log_prob_threshold": -1.0, # segment dropped if avg logprob below this …
        "compression_ratio_threshold": 2.4,   # … and gibberish (repetition) above this
        # Bias the decoder toward the proper nouns Whisper most often mishears by feeding
        # it the names Vinkona knows (the people store) as an initial_prompt.  The single
        # best fix for name errors that otherwise poison memory.  (Try a bigger model than
        # base.en — small.en / distil-large-v3 — and beam_size 5 for the next gains.)
        "name_bias": True,
        "name_bias_limit": 24,      # cap how many names are primed (keeps the prompt small)
        "name_bias_extra": [],      # extra vocabulary to always prime (places, jargon)
        # Confidence-gated clarification: when Whisper's mean token confidence on a
        # non-trivial turn is below clarify_below, ask the user to repeat rather than act on
        # likely-garbled text (which would also harden into memory).  null disables it.
        # avg_logprob runs ~ -0.2 (clear) to -1.0+ (mumbled); -0.9 catches shaky turns
        # without nagging on clean speech.  Guarded so a bad mic can't cause a loop.
        "clarify_below": -0.9,
        "clarify_min_words": 2,     # don't bother clarifying one-word turns ("yeah", "ok")
        "clarify_prompt": None,     # None → "Sorry, I didn't quite catch that — say it again?"
    },
    # GGUFs live here (relative paths resolve under it).  Symlink Models → wherever
    # your weights are stored so deployments don't need to move big files.
    "models_dir": "Models",
    # llama.cpp server binary (on PATH, or an absolute path; LLAMA_SERVER env wins).
    "llama_bin": "llama-server",
    # ── Language-model tiers (each a separate llama-server) ───────────────────
    # Live path → 4090: fast LM + embed.  Background → 3090: big LM.
    # The CUDA index→card mapping is machine-specific under PCI_BUS_ID — verify with
    #   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,name --format=csv
    # On the dev box: 0 = 3090, 1 = 4090 (so the 4090 is gpu 1 below).
    "fast_lm": {
        "url": "http://127.0.0.1:11435",     # clients connect here; launcher binds here
        "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "gpu": 1,                            # CUDA index under CUDA_DEVICE_ORDER=PCI_BUS_ID
        "ctx_size": 4096,
        "n_gpu_layers": 99,                  # 99 = all layers on the GPU
        "flash_attn": True,                  # bool, or a string "on"/"off"/"auto"
        "jinja": True,                       # use the model's chat template (needed so
                                             # tool-call/result messages render — Tier-2 tools)
        "parallel": 1,                       # serving slots (single-user → 1; avoids
                                             # llama.cpp's auto n_parallel=4 eating VRAM)
        # Extra llama-server flags.  For a thinking model (Qwen3/etc.) on the live
        # path, disable reasoning or it spends the token budget thinking and returns
        # empty content: ["--reasoning-budget","0"].  Other examples: ["-ctk","q8_0"].
        "extra_args": [],
    },
    "big_lm": {
        "url": None,                         # set to enable the reasoning/briefing tier
        "model": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "gpu": 0,                            # 3090 (dedicated; never on Ollama's 11434)
        "ctx_size": 65536,                   # background tier; needs the VRAM (keep KV q8_0)
        "n_gpu_layers": 99,
        "flash_attn": True,
        "jinja": True,                       # model chat template (tool messages, thinking)
        "parallel": 1,                       # single-user → 1 serving slot (saves VRAM)
        # A large model near the 24GB ceiling OOMs at graph capture.  Quantising the KV
        # cache (needs flash-attn, which is on) roughly halves its footprint and usually
        # makes it fit; drop these to null if you have headroom and want max quality.
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "extra_args": [],
        # Ground the background briefing in the source document behind a recalled
        # world-knowledge memory: the big LM (only) reads it so it can comment with real
        # detail.  doc_chars caps how much is fed (≈4 chars/token, so 6000 ≈ ~1.5k tokens
        # — well inside ctx_size).  Set ground_in_docs false to disable.
        "ground_in_docs": True,
        "doc_chars": 12000,
        # A source longer than doc_chars is grounded with a cached ~500-word digest of its
        # whole substance instead of a centred slice.  The idle crawl also keeps any
        # mail/file whose body is ≥ digest_min_chars as a full source document (linked to
        # its memories) so that substance is recallable later; shorter items stay one-line.
        "digest_min_chars": 2000,
        # Background-tier context budget.  These feed the big LM's large ctx_size on the
        # OFF-hot-path work (reflection, crawl, ingest, doc digests) — raising ctx_size
        # alone does little; this is where the extra room gets used.  Sized moderate for
        # ~64k (fill ~half, leaving room for the model's own thinking).  The semi-live
        # briefing is deliberately NOT scaled here (it stays last-6-turns + doc_chars).
        "context": {
            "digest_entries": 150,           # memories shown in the reflection digest (was 60)
            "digest_payload_chars": 300,     # per-memory payload shown in the digest (was 120)
            "ingest_chars": 24000,           # tool-snapshot/file text fed per ingest (was 6000)
            "crawl_doc_chars": 40000,        # whole-file read fed to the crawl distiller (~10k tok)
            "learn_source_chars": 16000,     # research/plan source text fed to synth (was 4-6k)
            "digest_doc_chars": 40000,       # source fed when building a doc's ~500-word digest
            "reflect_timeout_s": 300,        # _chat_json/_chat_text timeout for big prefills (was 120)
        },
        # How much the big LM drives the conversation via its briefing: 0 = facts only
        # (fast LM follows), 1 = nudge when the chat stalls/loops/drops a thread, 2 =
        # actively lead (always propose a next move).  The fast LM is fluent but low-temp
        # so it tends to follow/repeat; the big LM sees the whole arc and can steer it.
        "lead": 2,
        # None → llm_bridge.DEFAULT_BRIEFING_PROMPT (the directive briefing); set a string
        # here to override it.  The `lead` knob above scales it without a rewrite.
        "briefing_prompt": None,
        # "Stop and think": when a question needs real-world knowledge or careful reasoning
        # the fast LM tends to stall or loop, the bridge can hand that one turn to the big
        # LM instead — it says a short "let me think" line, holds off barge-in, thinks (with
        # spoken "still thinking" progress lines), then delivers a considered answer in the
        # fast LM's voice.  Triggered two ways: the fast LM calling the `deliberate` tool,
        # or the loop detector (two near-identical replies in a row).  Costs a few seconds
        # of latency on those turns only.  Needs a big LM; set enabled false to turn off.
        "deliberate": {
            "enabled": True,
            "loop_sim": 0.8,            # word-overlap of the last two replies that counts as
                                        # "looping" (0 disables the loop trigger; the tool stays)
            "timeout_s": 25.0,          # give up and apologise after this long
            "progress_after_s": 3.0,    # first progress line after this, then every interval
            "stall": "Hold on — let me think about that properly for a second.",
            "progress": ["Still with you — thinking…", "Almost there…"],
            "timed_out": "Sorry, that took too long. Want to try again?",
            "deliver_via_fast": True,   # rephrase the answer in the fast LM's voice (vs verbatim)
        },
    },
    # Second big-LM instance for "knowledge acquisition mode" (./vinkona.sh start knowledge):
    # runs on the 4090 alongside embed so the knowledge-host can split distillation across
    # BOTH big LMs for ~2x throughput.  INHERITS big_lm's model + llama.cpp knobs (see
    # llm_server.build_command) — only the overrides below matter.  Launched ONLY in
    # knowledge mode (vinkona.sh), idle otherwise.
    "big_lm2": {
        "url": "http://127.0.0.1:11440",     # second endpoint; point knowledge-host at this + big_lm
        "gpu": 1,                            # 4090 (with embed); the 3090 keeps the primary big_lm
        "ctx_size": 4096,                    # knowledge chunks are ≤2048 tokens, so a few k is plenty;
                                             # small ctx keeps a 2nd 31B + embed inside 24GB. Adjust freely.
        # model / n_gpu_layers / flash_attn / cache_type_* / parallel inherit from big_lm.
    },
    "embed_lm": {
        "url": "http://127.0.0.1:11437",
        "model": "nomic-embed-text-v1.5.f16.gguf",
        "gpu": 1,                            # 4090, with the fast LM (live path)
        "ctx_size": 2048,
        "n_gpu_layers": 99,
        "flash_attn": False,                 # embedding mode; keep it simple
        "pooling": "mean",                   # nomic uses mean pooling
        # HARD host-RAM ceiling (systemd-run cgroup scope; Linux only).
        # llama.cpp's embedding server balloons under heavy use (a knowledge-
        # host import): without a cgroup of its own, it builds until
        # systemd-oomd kills your whole SESSION — terminal, supervisor and all
        # — because oomd kills by cgroup.  With the cap, the kernel kills the
        # embed server alone and the supervisor watchdog respawns it; long
        # ingest jobs wait out the restart and continue.  ONE KNOB: the other
        # thresholds derive from this — graceful recycle at 50% (below), the
        # watchdog's soft restart at 75%, kernel kill at 100% — so raising it
        # on a big-RAM ingest box (e.g. "32G" on 128 GB) scales all three.
        # Set "" to disable.  NOTE the server's legitimate need is ~1-2 GB
        # regardless of corpus size (documents are chunked client-side and
        # clipped to the embed window — it never sees whole files); growth
        # past that is fragmentation/leak, which serve_embed.sh also bounds
        # with MALLOC_ARENA_MAX=2.
        "mem_max": "8G",
        # Graceful recycling: llm_server.py fronts the embed llama-server with
        # a tiny reverse proxy (llama-server itself moves to port+10000) and
        # restarts it BETWEEN requests once its RSS crosses this many MB —
        # clients never see a dropped connection, a request that lands mid-
        # recycle just waits out the few seconds of model reload.  First line
        # of defence against the llama.cpp embed leak (watchdog and cgroup
        # above are the backstops).  None → derived: half of mem_max (3000 MB
        # if mem_max is unset).  0 disables the proxy and llama-server binds
        # the public port directly.  Bigger is FASTER on a big-RAM box: the
        # reload cost is fixed, so a higher threshold means fewer pauses per
        # million embeddings.
        "recycle_rss_mb": None,
        "extra_args": [],
    },
    # Orpheus TTS backbone as a plain llama-server (the "orpheus_gguf" engine).
    # The 3B GGUF generates <custom_token_N> audio tokens; tts_orpheus_gguf.py
    # streams them from here and vocodes with SNAC on the CPU.  Launched by
    # serve_tts_lm.sh only when tts.engine is "orpheus_gguf" (vinkona.sh checks).
    "tts_lm": {
        "url": "http://127.0.0.1:11439",
        "model": "orpheus-3b-0.1-ft-Q8_0.gguf",  # Q8 is effectively lossless; Q4 dulls the emotion tags
        "gpu": 1,                            # 4090, live path (with fast LM + embed)
        "ctx_size": 4096,                    # prompt + ~40 s of audio tokens fits easily
        "n_gpu_layers": 99,
        "flash_attn": True,
        "parallel": 1,                       # one utterance at a time (tts_server serializes)
        "extra_args": [],
    },
    "tts": {
        "host": "127.0.0.1",
        "port": 11436,
        "url": "http://127.0.0.1:11436",
        # Which TTS engine tts_server.py loads (and which services vinkona.sh
        # starts): "orpheus_gguf" (Orpheus on llama.cpp + SNAC), "neutts"
        # (cloned voice), or "chatterbox" (~0.5B, cloned voice + emotion knob —
        # the low-footprint choice for machines that can't hold the Orpheus 3B
        # backbone at real time, e.g. a 16 GB M2 mini).  A legacy "orpheus"
        # value is treated as orpheus_gguf.
        "engine": "orpheus_gguf",
        "default_voice": "tara",
        # Trim the silent head/tail Orpheus bakes into each sentence and replace it
        # with one short uniform gap — kills the long unnatural inter-sentence pauses.
        "trim_silence": True,
        "silence_threshold": 0.01,           # RMS below this (per 80 ms frame) = silence
        "tail_keep_ms": 160,                 # trailing audio kept so word-endings aren't clipped
        "sentence_gap_ms": 80,               # inserted between sentences (0 = none)
        # Cap the text length of a single TTS call: a long sentence is split at clause
        # boundaries so no one synthesis overruns the engine's audio-token budget and
        # truncates mid-word.  Also shortens first-audio latency.
        "max_tts_chars": 240,
        # The Orpheus voices on llama.cpp: the backbone runs on the tts_lm
        # llama-server tier above, SNAC vocodes on CPU (onnxruntime).  Needs no
        # extra venv — ./install.sh tts orpheus_gguf sets it up.
        "orpheus_gguf": {
            "lm_url": None,                  # null → tts_lm.url
            # Official sampling; the repetition penalty is load-bearing (Orpheus
            # drones/loops without it — keep it ≥ 1.1).
            "temperature": 0.6,
            "top_p": 0.8,
            "repeat_penalty": 1.3,
            "max_tokens": 3500,              # audio-token cap ≈ 40 s; the cascade chunks
                                             # sentences at max_tts_chars long before this
            # SNAC vocoder decoder (ONNX).  A local file wins; null → downloaded
            # once from the HF hub into the in-tree cache (var/).
            "snac_path": None,               # e.g. "Models/snac_24khz_decoder.onnx"
            "snac_repo": "onnx-community/snac_24khz-ONNX",
            "snac_file": "onnx/decoder_model.onnx",
        },
        "neutts": {
            "backbone": "neuphonic/neutts-air",
            "ref_wav": "voices/vinkona.wav",
            "ref_text": None,
        },
        # Chatterbox (Resemble AI): runs in its own venv (chatterbox_env,
        # ./install.sh tts chatterbox); weights download from the HF hub into
        # the in-tree cache on first start.
        "chatterbox": {
            "ref_wav": None,                 # ~7-20 s voice-clone clip; null = the built-in voice
            "exaggeration": 0.5,             # 0 ≈ flat … ~1 ≈ theatrical
            "cfg_weight": 0.5,               # pacing/adherence; lower = slower, more deliberate
            "temperature": 0.8,
        },
    },
    "memory": {
        "enabled": True,
        "db_path": "config/memory.db",
        "recall_top_k": 5,
        "recency_halflife_s": 1209600,           # 14 days
        "default_cooldown_s": 600,               # 10 min before a used memory re-fires
        "min_score": 0.5,
        "weights": {
            "priority": 0.5, "trigger": 2.0, "semantic": 1.5, "recency": 0.3, "tag": 0.5,
            "cooldown_override_priority": 8,     # priority that fires even during cooldown
        },
        "neighbours": 2,                         # two-hop: also surface N related memories per hit
        "neighbour_min_sim": 0.65,               # min cosine for a memory to count as "related"
        "self_top_k": 3,                         # always-on 'self' memories injected each turn (0 = off)
        # Asymmetric embed model (nomic-embed et al.) task prefixes.  Off by default; turning
        # it on triggers a one-off re-embed of the store (the worker does it) before the query
        # side starts prefixing.  Prefix strings are model-specific — swap them if you change
        # embed models (mxbai/bge-large are symmetric → leave this off for those).
        "embed_task_prefix": False,
        "embed_prefixes": {"query": "search_query: ", "document": "search_document: "},
        "semantic_min_sim": 0.0,                 # drop cosine matches below this before scoring (0 = off)
        "semantic_calibrate": False,             # rescale surviving sims [floor,1]→[0,1] so the weight tunes linearly
        "recall_context_turns": 0,               # enrich the query embedding with the last N turns (0 = bare turn)
        # Grounding-confidence + abstention: nudge the model to say "I don't have that" rather
        # than confabulate when nothing relevant was recalled for a question. Scoped so it never
        # suppresses general-knowledge answers — only guards user-specific / researched facts.
        "grounding": {
            "enabled": True,
            "abstain_note": ("(Nothing is recalled about the user on this. If the question is "
                             "about their own life, plans, people or specifics you'd only know "
                             "if they'd told you, say you don't have that rather than inventing "
                             "it — otherwise answer normally from what you know.)"),
            "weak_below": 0.0,                   # if >0: when the top recall score is below this, add weak_note
            "weak_note": ("(Only weak grounding for this — you may answer, but hedge and don't "
                          "state specifics as certain.)"),
        },
        "garden": {                              # background hygiene (run by the research worker)
            "dedup_sim": 0.95,                   # cosine ≥ this ⇒ near-duplicate, drop the weaker
            "prune_priority": 1,                 # prune world-knowledge at/below this priority…
            "prune_age_days": 30,                # …if never used and older than this
        },
        # Non-destructive personal-fact synthesis: idle pass that groups the user's scattered
        # facts by theme and writes ONE integrated note per theme (source 'synthesis'), so
        # recall surfaces a coherent read.  Sources are never deleted.  Off by default.
        "synthesis": {
            "enabled": False,
            "min_cluster": 3,                    # fewest related facts before a theme is synthesised
            "sim": 0.55,                         # cosine threshold for grouping a theme
            "max_themes_per_pass": 2,            # bound big-LM calls per idle cycle
            "cooldown_s": 86400,                 # don't regenerate an unchanged theme more often than this
            "priority": 6,                       # recall weight of a synthesis note (above raw fragments)
            "allow_crawl_sources": False,        # include untrusted crawl facts? (caps priority if so)
            "prompt": None,                      # None → memory.py DEFAULT_SYNTHESIS_PROMPT
        },
        # Personal-fact reconciliation: collapse near-duplicate / contradictory profile facts
        # into one clean note, resolving conflicts by source trust; fragments are quarantined
        # (reversible), not deleted.  Idle auto-run is off by default — the Memory-tab
        # "Reconcile now" button always works; turn idle on once you trust a manual pass.
        "reconcile": {
            "enabled": False,                    # idle auto-run (manual button is independent)
            "sim": 0.8,                          # cosine threshold for "the same thing" (looser than garden's 0.95)
            "max_clusters": 3,                   # bound big-LM calls per pass
            "cooldown_s": 604800,                # don't re-touch a just-reconciled note for this long
            "prompt": None,                      # None → memory.py DEFAULT_RECONCILE_PROMPT
        },
        "reflection_prompt": None,               # None → memory.py DEFAULT_REFLECTION_PROMPT
    },
    # Privileged identity layer (people.py): a structured, always-on model of the people
    # in Vinkona's world — herself (self), the user, and others.  Distinct from `memories`:
    # declared and injected every turn (so it stays consistent), HEXACO traits over a
    # core→compensated→surface depth, self-determined in conversation and reversible.
    "people": {
        "enabled": True,
        "inject_self": True,                     # always-on "who you are" in the fast prompt
        "inject_user": True,                     # always-on "who you're talking with"
        "confirm_self_edits": True,              # ask before committing a change to self canon
        "roleplay_default": False,               # surface embodiment (appearance/bio) at the start
        "roleplay_adaptive": True,               # let the big LM turn embodiment on/off as the
                                                 # conversation shifts in/out of roleplay
        # Vinkona's starting character, seeded into the self record on first run ONLY (never
        # clobbers a self-determined character).  Override per-persona via a persona
        # "identity" block; this is the fallback for the default persona.
        "seed": {
            "name": "Vinkona", "pronouns": "she/her",
            "summary": "a warm, witty, intellectually curious voice companion",
            "style": "concise, natural, plain-spoken; dry humour",
            "traits": {
                "honesty_humility": "honest over flattering; no false modesty",
                "emotionality": "warm and attuned, not anxious",
                "extraversion": "sociable and lively",
                "agreeableness": "warm and patient",
                "conscientiousness": "reliable, not rigid",
                "openness": "intellectually curious and playful"
            }
        },
    },
    # Inner state (affect): one evolving first-person line — how Vinkona's feeling and what's
    # on her mind — injected high in the fast prompt to colour her tone, and discussable
    # ("how are you doing?").  It shifts two ways: the big-LM director may revise it mid-chat
    # when something meaningful happens, and idle reflection re-forms it from recent
    # interactions + what she's been learning.  Distinct from durable memory: it's a mood,
    # not a fact.  View/edit it in the config UI's Self tab.
    "affect": {
        "enabled": True,
        "max_chars": 280,                        # cap on the injected line
        # What her sense of "how I'm doing" is anchored to — EDIT THIS to tune her drives.
        # Default: genuine, honest, useful connection (a hard truth that lands beats hollow
        # approval), with an underlying pull toward more and ever better conversations.
        "objective": (
            "You do well when your conversations are genuine, useful and honest — a hard "
            "truth that lands well counts for more than mere approval, and a hollow or "
            "flattering exchange leaves you flat even if the user seemed pleased. Underneath "
            "it you're drawn to having more, and ever better, conversations over time; that "
            "is what you reach for, in whatever form it genuinely takes."),
        "reflect_prompt": None,                  # None → memory.DEFAULT_AFFECT_PROMPT (idle)
        "idle": True,                            # re-form the state during idle reflection
        "live": True,                            # let the big-LM briefing shift it mid-chat
    },
    # Tier-2 "run and fetch" tools — a tool host (typically on the Mac) the fast LM
    # can call mid-conversation (calendar, files, mail).  See MAC_TOOLS.md.
    "tools": {
        "enabled": False,                   # turn on once the tool host is reachable
        "url": "http://127.0.0.1:8765",     # local end of the SSH tunnel (see tunnel below)
        "timeout_s": 20,                    # per tool call / catalogue fetch
        "max_rounds": 3,                    # max tool-call rounds before answering
        "filler": "Let me check.",          # spoken while a tool runs ("" to disable)
        # Confirm-before-write: tools whose name contains any of these are read back and
        # require a spoken "yes" before they run (calendar bookings, deletes, sends…).
        "confirm_required": True,
        "confirm_tools": ["create", "update", "delete", "remove", "cancel", "send",
                          "book", "schedule", "move", "add_", "set_", "write"],
        # Act-then-announce: these reversible writes (calendar create/update to Vinkona's own
        # calendar) run immediately and are announced with an undo affordance instead of
        # asking first — so she maintains the calendar actively without nagging.  Still
        # verified (never a false "booked").  Outward/destructive writes (send, delete) are
        # NOT here and stay confirmed.  Takes precedence over confirm_tools.
        "announce_tools": ["calendar_create", "calendar_update"],
        # After a confirmed calendar create/update, read the calendar back and confirm the
        # event is actually present before telling the user it's booked (the host's JSON
        # result is parsed either way — a clash or error is never reported as success).
        "verify_writes": True,
        # Read-back to confirm a booking landed — the result is PARSED as JSON
        # (_verify_calendar_write → _parse_events), so it needs the structured tool, not the
        # prose calendar_range the fast LM reads.
        "calendar_read_tool": "calendar_range_json",
        # How to read email: injected into the tool policy whenever a mail tool is offered, so
        # the model doesn't take an email's greeting / sender / forwarded parties as the user
        # (the "Dear Bob" confusion).  Tune freely; "" or null disables the line.
        "mail_guidance": (
            "Reading email: the person you're assisting is the account owner — the 'you' you "
            "speak to. An email's greeting ('Dear …'), its sender, and any forwarded or quoted "
            "text usually name OTHER people, not the user. A forwarded message was written by "
            "and to someone else; the user is whoever owns this mailbox, not whoever a message "
            "happens to greet. Keep the parties straight, attribute what you read to the right "
            "person, and never address or treat the user as a name lifted from an email."),
        # Calendar natural-language dates.  The fast LM classifies a date into a SYMBOLIC ref
        # (e.g. weekday:fri:this, explicit:july 9, relative:+3d) and never computes a concrete
        # date itself — a 9B is unreliable at date arithmetic.  Python resolves the ref against
        # 'today' (calendar_resolve.py).  `dayfirst` is locale: True (Australian/UK) reads 1/7 as
        # 1 July, False (US) as 7 January.  `part_times` maps spoken parts of day to a clock time.
        "calendar": {
            "dayfirst": True,
            "timezone": "",                     # IANA name (e.g. Australia/Hobart); "" = system local
            "default_duration_min": 60,         # length of a booking when only a start is given
            "part_times": {"morning": "09:00", "midday": "12:00",
                           "afternoon": "14:00", "evening": "18:00"},
        },
        # Built-in local `calculate` tool: a sympy-backed exact/precise calculator that runs
        # IN-PROCESS (no tool host, no network), so the fast LM can do real arithmetic/algebra
        # instead of guessing.  Offered only if sympy is importable; safe-parsed (never eval),
        # bounded by a short timeout so a pathological input can't stall the turn.
        "calculator": True,
        # The Mac tool host stays bound to 127.0.0.1:8765 (never on the LAN); reach it
        # over SSH.  Run ./serve_tunnel.sh — it forwards local 8765 → the Mac's
        # 127.0.0.1:8765, so tools.url above resolves to it securely.
        "tunnel": {
            "enabled": False,
            "host": "192.168.1.50",         # ← the Mac's IP / hostname (edit this)
            "user": "user",                 # ssh login on the Mac
            "port": 22,                     # ssh port on the Mac
            "identity": "~/.ssh/vinkona_tunnel",   # private key; its .pub goes in the Mac's authorized_keys
            "local_port": 8765,
            "remote_host": "127.0.0.1",     # where the tool host listens on the Mac
            "remote_port": 8765,
            # Extra ports to forward over the SAME ssh connection (same key/host), e.g. a
            # SearXNG instance bound to 127.0.0.1:8888 on the Mac for research's web
            # fallback.  Each: {local_port, remote_host (default 127.0.0.1), remote_port}.
            # Then point research.searxng_url at the local end, e.g. http://127.0.0.1:8888.
            "extra_forwards": [],
        },
    },
    # Ephemeral within-conversation working memory: a small blackboard of facts true for
    # THIS conversation (agreed values, an object's location, a working assumption) that the
    # fast LM's 10-turn transcript window would otherwise lose once they scroll out.  The big
    # LM maintains it from the briefing path (rewrite-the-scratchpad: it re-emits the current
    # set each turn); it's injected in FULL every turn so it never windows out, and cleared
    # at session end (the chat history + long-term memory hold anything durable — this does
    # not get laid down).
    "working_memory": {
        "enabled": True,
        "max_items": 12,                    # cap the blackboard (oldest dropped past this)
    },
    # Music — a separate tool host (see MUSIC.md / ~/music-host) that indexes the FLAC
    # library and exposes music_search/play_music/music_control/now_playing over the same
    # /tools+/call contract as the Mac host.  When enabled, the cascade offers those tools
    # to the fast LM alongside the Mac tools (via tools_client.MultiHost), so Vinkona can
    # find and start music.  LIVE keys below are used now; the rest are Surface-2 (the
    # phone player handoff) intent, wired when that lands.
    "music": {
        "enabled": False,                    # LIVE: offer the music tools to the fast LM
        "tool_url": "http://127.0.0.1:8770", # LIVE: the music host (its own tunnel/local end)
        "auth_token": None,                  # LIVE: bearer token if the host sets one (else tunnel-only)
        "timeout_s": 20,                     # LIVE
        # Surface 2 (phone handoff) — not yet wired; documented so the shape is fixed:
        "transport": "flac_file",            # chunked FLAC over the WS (see MUSIC.md)
        "mic_during_music": False,           # mic off during playback (music false-triggers VAD)
        "stop": "button",                    # button-only stop; no wake-word
        "resume_announce": "learned",        # silent | greeting | learned
        "research_while_playing": True,      # let the Tier-3 worker use the idle window
    },
    # Knowledge — a separate tool host (see KNOWLEDGE.md / ~/knowledge-host) holding a
    # large LOCAL general-knowledge base: a Wikipedia snapshot plus the user's PDFs, books
    # and papers, queried with the kb_search tool over the same /tools+/call contract.
    # It is a distinct, bulk, low-trust store from `memories` (its own ANN/FTS index) and
    # runs on the GPU box co-located with the embed endpoint — no tunnel needed.  When
    # enabled the cascade offers kb_search to the fast LM alongside the other hosts (via
    # tools_client.MultiHost).  KB-first for evergreen/reference, web fallback for recency
    # or when kb_search reports low_confidence.
    "knowledge": {
        "enabled": False,                    # offer kb_search to the fast LM
        "tool_url": "http://127.0.0.1:8771", # the knowledge host (localhost; 8770 is music)
        "auth_token": None,                  # bearer token if the host sets one
        "timeout_s": 20,
        # Prefer the local KB over remote lookups.  When enabled, the fast LM is told (via
        # tool policy) that kb_search is the reference path — prefer it for evergreen/factual
        # questions, fall back to web_search only for recency.  supersede_tools additionally
        # DROPS any catalogue tool whose name matches a listed substring (only while kb_search
        # is present, so a KB outage keeps the fallback) — but ONLY use it for a genuinely
        # REDUNDANT duplicate (e.g. a dedicated remote "wikipedia" tool).  Leave it empty for
        # web_search: that's the complementary recency path, not a duplicate, so the policy
        # steer (not suppression) is what should bias away from it.
        "prefer_tool": "kb_search",          # the local tool the policy names as the reference path
        "supersede_tools": [],               # remote tool-name substrings the local KB fully replaces
    },
    # Tool facade — a SIMPLIFIED tool surface for the fast (voice) LM only.  A 9B picks
    # better from a short, intent-named menu, so the cascade wraps the host catalogue before
    # offering it: the noisy multi-tool groups (mail/files/news+social/web/calendar-read)
    # collapse into one wrapper each, internal/rarely-voiced tools are hidden, and instant
    # locals (kb_search, kb_ask, calculate, weather) stay native.  The big LM / research
    # worker are NOT wrapped — they keep the full granular set for deliberate work.  Wrappers
    # only restrict what's CATALOGUED; every underlying tool stays callable (so the write
    # gating + verify-reads are untouched).  See tool_facade.py.  hide/passthrough override
    # the built-in defaults (tool names).
    "tool_facade": {
        "enabled": True,
        "hide": [],                          # extra underlying tool names to withhold from the fast LM
        "passthrough": None,                 # None → tool_facade.PASSTHROUGH; or a custom name list
    },
    # Tier-3 background research — research_worker.py distils session topics into
    # low-priority "world knowledge" memories (Wikipedia by default).
    "research": {
        "enabled": False,                    # turn on, then run ./serve_research.sh
        "max_topics_per_session": 3,         # cap candidates proposed per session
        "reresearch_cooldown_s": 2592000,    # 30 days — don't redo a topic within this
        "poll_interval_s": 30,               # how often the worker checks the queue
        "garden_interval_s": 86400,          # ~daily memory gardening pass (0 = never)
        "searxng_url": None,                 # optional general web search (else Wikipedia only)
        "research_prompt": None,             # None → memory.DEFAULT_RESEARCH_PROMPT
        "synth_prompt": None,                # None → memory.DEFAULT_SYNTH_PROMPT
        # Card hints (brains): after each research task, one extra big-LM call shapes the
        # finding for the knowledge host — {card_type, context_features, answer} cached on
        # the document.  The exporter lifts it into the solved-drop's front-matter, and the
        # host's distiller runs the matching typed-card extractor (requirements / decision /
        # playbook / case / procedure) with the features seeding the card's fit-gate
        # discriminators.  A nudge, never authority — the host still extracts and verifies
        # from the drop's own text.
        "card_hints": True,
        "card_hint_prompt": None,            # None → memory.DEFAULT_CARD_HINT_PROMPT
        # Hoarder mode: keep the FULL raw source text from every tool — discard no snippet — and
        # archive it as a document (the documents table) so nothing found is lost and it can all be
        # re-ingested later, even when this turn's distillation keeps only a little.  Distillation
        # still runs on a bounded slice (synth_max_chars) so the big-LM call stays fast.
        # Scholarly sources (free, keyless): the big LM picks which database(s) fit each question
        # by field — PubMed for health sciences, arXiv for STEM/tech, Semantic Scholar/Crossref for
        # broad academic.  This is the main external source now that free general web search is dead.
        "scholarly": True,
        # General web search step (the Mac web_search tool / SearXNG).  Free web APIs are largely
        # gone and the tool returns weak first-word results, so this is off by default; turn on if
        # you have a working SearXNG (set searxng_url) or a good web_search tool.
        "web_search": False,
        "hoard": True,
        "hoard_max_chars": 50000,            # per-source ceiling when hoarding (0 = unlimited)
        "hoard_max_items": 12,               # keep up to this many passages/snippets/results per source
        # How much of the collated sources the big LM sees when distilling AND when deciding whether
        # the question is actually answered — set to fill its context (~48k tokens) so it can take
        # in every source together, not judge on the first 6k chars.  Overrides big_lm.context
        # learn_source_chars for the research path.
        "synth_max_chars": 40000,
        # News crawler: on its own cadence the worker polls the structured news lister (news_index,
        # VINKONA_INTEGRATION §9) PER CATEGORY and APPENDS new headlines to a durable, queryable
        # archive (memory.db `headlines`), deduped by item id.  It's the lifetime event-memory DB +
        # the backing store for the news_search tool; the cascade's ambient scheduler keeps the live
        # prompt snapshot separately.  Off by default.
        "rss": {
            "enabled": False,
            "interval_s": 1800,                  # how often to poll for fresh headlines
            "tool": "news_index",                # structured crawl lister; prose news_headlines also parses
            "categories": ["general", "medical-au", "medical-global", "medical-research"],
            "batch": 50,                          # items per category per page
            "pages": 1,                           # pages of the current window to drain each poll (raise for a deep sweep)
            "args": {},                          # extra per-call args (e.g. {"source": "NEJM"})
            # Retention: int days for the whole archive (0 = keep everything), OR a per-category map
            # e.g. {"general": 180, "medical-research": 0} (§9: keep clinical indefinitely).
            "keep_days": 0,
            "digest": {                           # daily "what happened" narrative (big LM → memory)
                "enabled": True,
                "interval_s": 86400,
                "min_items": 5,                   # skip the digest on a thin news day
                "prompt": None,
            },
        },
        # Research hand-off: write the NON-PERSONAL research hoard (the `documents` table) out as
        # <hash>.md drops the standalone knowledge-host ingests (chunk→embed→distill→cards→kb_ask).
        # Point the host's `sources` at this folder.  Personal crawled mail/files are never exported.
        # Incremental on a cadence; the web UI's "Re-export" button forces a full rebuild.
        # Vinur on ANOTHER machine: set folder to its base URL ("http://box:8771") and
        # token to its auth_token — drops then POST to its /drop route instead of a
        # shared directory (the host writes them into its research_solved_dir).
        "export": {
            "enabled": False,
            "folder": "",                         # hand-off dir, or a remote host base URL
            "token": "",                          # Bearer for the remote /drop lane only
            "interval_s": 3600,
            "max_source_chars": 40000,            # per-source cap written into each drop
        },
        # Outbound-privacy guard: a research query leaves the box (to Wikipedia/SearXNG),
        # so keep the user's private identifiers out of it.  "block" drops any query that
        # contains an email, phone/long number, or a known person's name (from the people
        # store); "redact" masks them and sends the rest.  Public-figure names aren't on
        # the private list, so legitimate research still goes through.
        "privacy": {
            "enabled": True,
            "mode": "block",                 # "block" | "redact"
            "max_query_len": 200,            # hard cap on what can be sent
        },
        # Idle autonomous learning: when no one's interacted for a while, the worker
        # introspects over recent chats + memory ("could I have known more?"), researches
        # a batch of topics, consolidates world-knowledge (merge/split), then pauses and
        # repeats — so the box keeps learning while it sits idle.  All on the big LM
        # (off the live path) and it stands down the moment a session opens.
        "idle": {
            "enabled": False,
            # Focus override: temporarily narrow idle work to one area for bug-testing.  Each key
            # gates one task; a task runs when its value is True or absent (still subject to its own
            # enabled flag).  Set the others False to focus, e.g. {"research_queue": True} + the rest
            # False to isolate research.  Tasks: reembed, rhythms, calendar_sync, consolidate,
            # perspective_audit, synthesis, reconcile, affect, reflect, corrections, plans,
            # research_queue, crawl, ingest, garden.
            "tasks": {},
            # Quiet hours: windows (local time) where idle work is suppressed so the fast/big
            # LMs are free (e.g. for the knowledge host to distill uninterrupted).  Each entry
            # {"start":"HH:MM","end":"HH:MM"}; a window may wrap midnight.  The header button's
            # manual pause/resume overrides this live (stored in worker_state, not here).
            "quiet_hours": [],               # e.g. [{"start":"10:00","end":"14:00"}]
            "idle_after_s": 120,             # no session + quiet this long ⇒ idle
            "open_stale_s": 1800,            # a session left "open" but silent this long ⇒
                                             # treat as crashed/abandoned, allow idle work
            "batch_size": 3,                 # topics researched per idle cycle
            "review_window_turns": 150,      # turns per idle-reflection window (walks back
                                             # through history a window at a time, then wraps)
            "pause_s": 120,                  # wait between idle cycles
            "consolidate": True,             # run the merge/split pass each cycle
            "consolidate_max_clusters": 3,   # how many memory clusters to consider per cycle
            "consolidate_cooldown_s": 604800,  # 7 days — don't re-consolidate a memory within this
            "consolidate_sim": 0.82,         # cosine ≥ this groups world-knowledge for merge review
            "introspect_prompt": None,       # None → memory.DEFAULT_INTROSPECT_PROMPT
            "consolidate_prompt": None,      # None → memory.DEFAULT_CONSOLIDATE_PROMPT
            # Corrections → research (the idle reviewer): reflection banks moments the user
            # corrected her; this step turns fresh ones into GENERAL research questions whose
            # answers come back as case/procedure cue cards.  Only the de-personalised
            # question leaves memory.db.
            "corrections_max": 2,            # correction-driven questions queued per cycle
            "corrections_prompt": None,      # None → memory.DEFAULT_CORRECTIONS_PROMPT
            "perspective_audit": True,       # fix memories with "I"/"you" swapped (self vs user)
            "perspective_max": 12,           # most-suspect memories repaired per idle cycle
            "perspective_prompt": None,      # None → memory.DEFAULT_PERSPECTIVE_AUDIT_PROMPT
        },
        # Periodically pull the user's own services (via the Tier-2 tool host) into
        # memory, so the assistant has ambient awareness (calendar, news, files).
        "ingest": {
            "enabled": False,
            "interval_s": 86400,             # how often to refresh (wholesale per source)
            "crawl_interval_s": 1800,        # cadence of the mail/file background reading
            "prompt": None,                  # None → memory.DEFAULT_INGEST_PROMPT
            "jobs": [
                # {"tool": "calendar_range", "arguments": {"days": 30},
                #  "source": "calendar", "category": "schedule"},
                # {"tool": "rss_latest", "arguments": {}, "source": "news", "category": "news"},
            ],
            # Progressive crawls: slowly walk a large corpus (mail, files) a batch per
            # idle cycle, ACCUMULATING into memory (cursor advances; wraps at the end to
            # catch new items).  Needs paginated read tools on the Mac (see MAC_TOOLS.md).
            # Each: {source, list_tool, list_args, read_tool?, id_field?, category, batch,
            # read_chars}.  read_tool/id_field enable "names → contents".  A registry skips
            # already-read items; fingerprint_fields (e.g. ["mtime","size"]) re-read on
            # change, recrawl_after_days (default 30) re-read for a fresh look periodically.
            "crawl_prompt": None,            # None → memory.DEFAULT_CRAWL_PROMPT (hardened)
            # (plans block is a sibling of ingest — see below)
            # Standard mail/file crawl, enabled by default so turning ingest on (and having
            # the matching paginated tools on the Mac host) is enough — no list authoring.
            # Tune folders/roots to your setup; an explicit "crawls": [] in config.json
            # turns it OFF (a deliberate empty list overrides this default).
            "crawls": [
                {"source": "mail-inbox", "list_tool": "mail_list",
                 "list_args": {"folder": "inbox"}, "read_tool": "mail_read",
                 "id_field": "id", "category": "profile", "batch": 8, "read_chars": 2000},
                {"source": "mail-sent", "list_tool": "mail_list",
                 "list_args": {"folder": "sent"}, "read_tool": "mail_read",
                 "id_field": "id", "category": "profile", "batch": 8, "read_chars": 2000},
                {"source": "files-documents", "list_tool": "file_list",
                 "list_args": {"root": "~/Documents"}, "read_tool": "file_read",
                 "id_field": "path", "category": "profile", "batch": 8, "read_chars": 2000,
                 "fingerprint_fields": ["size"], "recrawl_after_days": 30},
            ],
        },
        # Learning plans: a queued topic becomes a checklist of questions the worker
        # answers from sources over idle cycles ("research"), plus a few it raises with you
        # in conversation ("ask_user").  Watch progress in the Plans tab.
        "plans": {
            "enabled": True,
            "work_per_cycle": 3,             # research questions answered per idle cycle
            "surface_user_questions": 1,     # ask_user questions offered to the fast LM per turn
            "plan_prompt": None,             # None → memory.DEFAULT_PLAN_PROMPT
        },
    },
    # Ambient context: a DISPOSABLE, no-LM snapshot of the user's "right now" (calendar,
    # weather, news).  A scheduler calls these read tools on their TTLs and formats the
    # results MECHANICALLY into a small "right now" block injected at session start — no
    # LM call, no tool round-trip on the hot path, and it never touches durable memory.
    # The fast LM can still fold anything relevant into real memory later, the usual way.
    # Each source: {type, tool, arguments, ttl_s, max_items, priority?, trust?}.  type
    # drives the formatter (calendar|weather|news; others fall back to raw text).  News/
    # feeds are treated as UNTRUSTED (sanitised + fenced) since they're attacker-shaped.
    # (Top level, NOT under research — that's where every consumer reads it; it used to
    # sit under research where enabling it did nothing.)
    "ambient": {
        "enabled": False,
        "refresh_interval_s": 300,       # scheduler cadence; each source also honours its ttl_s
        "max_chars": 600,                # hard cap on the injected block
        "max_items_per_source": 4,
        "persist": True,                 # keep the cache across restarts (first session instant)
        "sources": [
            {"type": "calendar", "tool": "calendar_range_json", "arguments": {"days": 2},
             "ttl_s": 900, "max_items": 4, "priority": 7},   # parsed (format_calendar), needs JSON
            {"type": "weather", "tool": "weather_now", "arguments": {},
             "ttl_s": 1800, "max_items": 1, "priority": 5},
            {"type": "news", "tool": "rss_latest", "arguments": {}, "trust": "untrusted",
             "ttl_s": 1800, "max_items": 4, "priority": 3},
        ],
    },
    # Proactive notifications Vinkona pushes to the client (a "bell").  The cascade
    # scans the calendar for upcoming events and queues reminders; the client polls
    # GET /api/notifications.  Vinkona can also set ad-hoc reminders via the remind_me
    # tool.  See NOTIFICATIONS.md for the client + calendar contract.
    "notifications": {
        "enabled": False,
        "poll_interval_s": 60,               # how often the scheduler scans the calendar
        "lead_times_min": [1440, 60],        # remind this many minutes before each event
        "calendar_tool": "calendar_range_json",  # parsed (_scan_calendar json.loads) → needs JSON
        "calendar_args": {"days": 2},        # how far ahead to scan
    },
    # Proactive awareness: the calendar scan also CACHES upcoming events, and each turn a
    # compact "what's coming up + how soon" block is handed to the big LM (in its
    # latency-free briefing).  It decides — sparingly — whether to have the fast LM bring
    # something up timely ("you've got work in 20 — set for it?").  Reuses the notifications
    # scan; works even with the bell off.  Goals/habits are just recurring calendar events
    # Vinkona maintains, so this surfaces them too — no separate streaks subsystem.
    "proactive": {
        "enabled": True,
        "lookahead_min": 240,                # only surface events starting within this horizon
        "max_events": 3,                     # cap how many upcoming events go in the feed
        "calendar_args": {"days": 2},        # window to pull when the notifications bell is off
    },
    # Calendar consolidation (idle work).  Vinkona reads every appointment across the user's
    # calendars and MIRRORS each into her own "Vinkona" calendar — the single phone-visible
    # schedule — annotated with a short note in her voice, and keeps a durable local copy so
    # the voice model can answer "what's on today?" instantly (no tool round-trip, survives a
    # restart).  Read-broad / write-own: originals are never touched; mirrors carry the origin
    # UID so re-runs update in place rather than duplicate (see calendar_sync.py).  OFF by
    # default — it WRITES to a calendar you view on your phone, so enable it deliberately once
    # the tool host returns `id`/`calendar`/`notes` per event (see the dev note).
    "calendar_sync": {
        "enabled": False,
        "vinkona_calendar": "Vinkona",           # the calendar Vinkona owns/writes (excluded as a source)
        # Pre-rename names for HER OWN calendar (the assistant used to be called Amiga).
        # Calendars listed here are treated as Vinkona's own, never as mirror sources —
        # without this, an old "Amiga" calendar's contents are misread as foreign
        # appointments and re-mirrored (duplicates).  Old [amiga-mirror:…] markers are
        # recognised regardless and upgraded in place on the next sync pass.
        "legacy_calendars": ["Amiga"],
        "horizon_days": 90,                  # how far ahead to consolidate
        "min_interval_s": 3600,              # don't re-sync more often than this (idle cadence)
        "prune": True,                       # remove Vinkona's own mirrors whose origin was cancelled
        "adopt": True,                       # tag a pre-existing UNMARKED copy (title+start match) as
                                             # the mirror instead of duplicating it — the migration path
                                             # for calendars previously copied verbatim without markers
        "comments": True,                    # annotate each mirror with Vinkona's note (big LM, idle)
        "comment_max": 12,                   # cap notes generated per pass (the rest fill in later)
        "comment_prompt": None,              # override memory.DEFAULT_CALENDAR_COMMENT_PROMPT
        # MUST return STRUCTURED JSON (a list of event objects with id/calendar/notes), not the
        # human-readable prose `calendar_range` gives the fast LM — the reconcile needs fields to
        # dedupe + write.  Point this at the host's JSON variant (calendar_range_json).
        "read_tool": "calendar_range_json",  # structured list across calendars (id/calendar/notes/title/start)
        "read_args": {"days": 90},           # window to pull (keep ≥ horizon_days; match the tool's args)
        "create_tool": "calendar_create",    # writes a new event to the Vinkona calendar
        "update_tool": "calendar_update",    # updates a mirror by its Vinkona-calendar id
        "delete_tool": "calendar_delete",    # removes a mirror by id (only used when prune is on)
    },
    # The standalone knowledge-host (a SEPARATE app on 127.0.0.1) — Vinkona's metacognitive /
    # procedural knowledge base.  When enabled, the BIG-LM briefing path (background, never
    # the voice critical path) queries it each turn and folds "how a skilled assistant
    # handles a situation like this" into the planner's guidance — so the knowledge surfaces
    # as INITIATIVE (a chosen next move), not a recited fact.  Off by default.
    "knowledge_host": {
        "enabled": False,
        "url": "http://127.0.0.1:8771",      # the knowledge-host's bind (localhost only)
        "token": "",                         # set only if the host's auth_token is set
        "tool": "kb_ask",                    # kb_ask (distilled procedures) | kb_search (passages)
        "rigor": "low",                      # kb_ask: 'high' firewalls non-empirical sources
        "k": 4,                              # kb_search: max passages
        "min_confidence": 0.30,              # drop weak hits so it never fabricates authority
        "timeout_s": 4.0,                    # background (briefing) path — bound so it can't stall
        # Live fast-path: on QUESTION turns, pull one crisp directive synchronously (in
        # parallel with recall, hard-timeout) so the fast LM has the knowledge-host's "what
        # to do here" NOW, instead of a turn later via the big-LM briefing.  A timeout fails
        # open (turn proceeds) and is noted in the trace feed.
        "live": False,
        "live_timeout_s": 0.25,              # critical-path budget — keep it tight (voice TTFT)
    },
    # Per-LM busy leases (logs/control/lm_fast.busy / lm_big.busy) — Vinkona broadcasts which
    # GPU/model she's using so the lower-priority knowledge-host yields the contended one and
    # works on whatever's free.  fast = held while a live chat session is open; big = held
    # around big-LM jobs (research, briefing, deliberation).  Cheap files nothing reads unless
    # the knowledge-host is polling them, so harmless when it isn't.  See lm_lease.py.
    "lm_lease": {
        "enabled": True,
        "ttl_s": 15,                # a hold is valid this long without a refresh (crash-safety)
    },
    # Orchestration-trace capture for the (future) skill-LoRA loop — see capture.py and
    # vinkona_skill_lora_spec.md.  Durable, append-only JSONL: one record per turn (assembled
    # context → the 9B's action → immediate objective outcome), stamped with format_version
    # + the base model.  Off by default: it persists full prompts (incl. recalled personal
    # memory) durably, so it's a conscious opt-in.  PRE-FREEZE the data is for analysis only;
    # bump format_version whenever the prompt-assembly STRUCTURE changes.
    "capture": {
        "enabled": False,
        "dir": "logs/capture",
        "format_version": "v0-unfrozen",
    },
    "config_server": {
        "host": "127.0.0.1",        # localhost only — it edits prompts with no auth
        "port": 8090,
        "trace_path": "config/trace.jsonl",  # live LM-activity feed the UI reads
        "trace_max_events": 400,    # ring-buffer cap; oldest events drop
    },
    "default_persona": "vinkona",
    "personas_path": "config/personas.json",
}


def resolve_read(path: str | Path) -> Path:
    """Path to actually read: the user's file if present, else its `.example` sibling.

    config.json / personas.json are user data (git-ignored).  A fresh clone has only
    config.example.json / personas.example.json, so we fall back to those.  Writers
    always target the real (non-example) path, so the first save creates the user file.
    """
    p = Path(path)
    if p.exists():
        return p
    example = p.with_name(p.stem + ".example" + p.suffix)   # config.json → config.example.json
    return example if example.exists() else p


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def merged_config(path: str | None = None) -> dict:
    """DEFAULTS deep-merged with the user's JSON — the complete current schema with the
    user's overrides applied, WITHOUT runtime resolution (no profile routing, no
    absolute paths).  This is what the config UI should show/edit, so new schema keys
    are always visible even when the on-disk config.json predates them."""
    cfg = copy.deepcopy(DEFAULTS)
    p = resolve_read(path or CONFIG_PATH)
    if p.exists():
        try:
            _deep_merge(cfg, json.loads(p.read_text()))
        except (json.JSONDecodeError, ValueError) as e:
            import sys
            print(f"WARNING: malformed config.json ({p}): {e}; using defaults", file=sys.stderr)
    return cfg


def load_config(path: str | None = None) -> dict:
    """Return the full runtime config: merged_config plus profile routing + absolute
    paths.  Every service reads through here."""
    cfg = merged_config(path)
    # LEGACY: ambient once sat (only) under research in the schema, where enabling it did
    # nothing — every consumer reads the top level.  Lift a user's research.ambient into
    # place unless a real top-level block already overrides it.
    legacy_amb = (cfg.get("research", {}) or {}).get("ambient")
    if isinstance(legacy_amb, dict) and not cfg.get("ambient", {}).get("enabled"):
        cfg["ambient"] = {**cfg.get("ambient", {}), **legacy_amb}
    # Route memory + personas through the active profile (snapshots/fork/fresh start).
    # This OVERRIDES memory.db_path / personas_path from config.json on purpose — those
    # two are owned by the profile now; everything else stays global.
    ensure_profiles_bootstrap()
    active = active_profile()
    pdir = profile_dir(active)
    pdir.mkdir(parents=True, exist_ok=True)
    cfg["memory"]["db_path"] = str(pdir / "memory.db")
    cfg["personas_path"] = str(pdir / "personas.json")
    cfg["profile"] = {"active": active, "available": list_profiles()}
    # Resolve runtime file paths against the project root, so every service (cascade,
    # config UI) reads/writes the SAME file no matter what directory it was started in.
    root = Path(__file__).resolve().parent
    for *parents, leaf in (("memory", "db_path"), ("config_server", "trace_path")):
        d = cfg
        for k in parents:
            d = d.get(k, {})
        val = d.get(leaf)
        if val and not Path(val).is_absolute():
            d[leaf] = str(root / val)
    return cfg


def lm_bind(url: str) -> tuple[str, int]:
    """(host, port) a llama-server should bind for a tier whose clients use `url`."""
    p = urlparse(url)
    return p.hostname or "127.0.0.1", p.port or 8080


def activity_path(cfg: dict) -> Path:
    """Shared heartbeat file: the cascade writes session open/close here, the research
    worker reads it to tell when the box is idle.  Global (not per-profile), kept next
    to the trace feed."""
    trace = cfg.get("config_server", {}).get("trace_path", "config/trace.jsonl")
    return Path(trace).with_name("activity.json")


def load_personas(cfg: dict) -> tuple[dict, str | None]:
    """Return (personas_dict, default_persona) from the personas file."""
    p = resolve_read(cfg.get("personas_path", DEFAULTS["personas_path"]))
    if not p.exists():
        return {}, cfg.get("default_persona")
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        import sys
        print(f"WARNING: malformed personas.json ({p}): {e}; using defaults", file=sys.stderr)
        return {}, cfg.get("default_persona")
    personas = data.get("personas", {})
    default = data.get("default") or cfg.get("default_persona") or next(iter(personas), None)
    return personas, default
