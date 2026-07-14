#!/usr/bin/env python
"""
Cascade voice server — Vinkona's realtime voice loop.

Flow:  mic PCM → RNNoise → VAD → faster-whisper ASR → user text
       → LLM bridge (fast LM) → response sentences
       → TTS HTTP service → 24 kHz PCM → client

WebSocket protocol (kept wire-compatible with the client's original server, so
the Flutter client works unchanged):
  server→client : 0x00 handshake | 0x02 JSON bubble {role,text} | 0x03 float32 PCM
  client→server : 0x03 float32 PCM (mic) | 0x04 JSON {text} (typed, text mode)

Config-driven: all settings come from config/config.json (see config.py).  Tunables
(VAD, audio, voice, LM/TTS URLs) and personas are re-read PER CONNECTION, so edits
in the config web UI take effect on the next call without a restart.  Structural
bits (ports, SSL, the ASR model) are read once at startup.

Standalone: reuses the project's own modules (rnnoise_frontend, asr, llm_bridge,
config); no torch in-process.  TTS runs as a separate HTTP service.

  python cascade_server.py --config config/config.json
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import ssl
import subprocess
import time
import types
import uuid
from collections import deque
from pathlib import Path

import aiohttp
from aiohttp import web
import numpy as np


def _load(modname: str):
    spec = importlib.util.spec_from_file_location(modname, Path(__file__).parent / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [cascade] {msg}", flush=True)


def _parse_iso(s):
    """Parse an ISO-8601 datetime (tolerating a trailing 'Z') → unix seconds, or None."""
    import datetime
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _humanize_when(start_ts, lead_min):
    """A short, speakable 'when' for a reminder, e.g. 'tomorrow at 15:00' / 'in 1 hour'."""
    import datetime
    dt = datetime.datetime.fromtimestamp(start_ts)
    today = datetime.date.today()
    day = ("today" if dt.date() == today
           else "tomorrow" if dt.date() == today + datetime.timedelta(days=1)
           else dt.strftime("%a %d %b"))
    return f"{day} at {dt:%H:%M}"


# Fixed framing (tied to the 24 kHz mimi-compatible pipeline; not user config).
SAMPLE_RATE = 24000
FRAME_SIZE = 1920                     # 80 ms @ 24 kHz
FRAME_DUR = FRAME_SIZE / SAMPLE_RATE

# Common ≥4-letter words to drop when widening the knowledge-host query nucleus with
# prior-turn vocab (the <4-letter ones fall out by the length filter).  Kept small on
# purpose — we want discriminators through, only the filler stripped.
_NUCLEUS_STOP = frozenset(
    "this that these those there their them they then than with from into have has had "
    "been were will would could should about here what when where which while whom whose "
    "your yours just like only also even still much more most some such very able want "
    "need know think really going okay yeah well done thing things stuff kind sort being "
    "does doing because would".split())


class TraceLog:
    """Append-only ring buffer of LM-activity events (JSONL), read by the config UI.

    Single writer (this process); the config web service only reads.  Capped at
    `max_events` lines — when it grows past the cap we rewrite, keeping the tail.
    """

    def __init__(self, path: str, max_events: int = 400):
        self.path = Path(path)
        self.max_events = max_events
        self._n = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")          # start each run with a clean feed
        except Exception as e:
            _log(f"trace: cannot open {path}: {e}")
            self.path = None

    def write(self, event: dict):
        if self.path is None:
            return
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(event) + "\n")
            self._n += 1
            if self._n > self.max_events * 2:     # amortised compaction
                lines = self.path.read_text().splitlines()[-self.max_events:]
                self.path.write_text("\n".join(lines) + ("\n" if lines else ""))
                self._n = len(lines)
        except Exception:
            pass


class CascadeServer:
    def __init__(self, config_path, denoiser, asr, soxr, memory=None, trace=None):
        self.config_path = config_path
        self.denoiser = denoiser
        self.asr = asr
        self.soxr = soxr
        self.memory = memory
        self.trace = trace
        self._cfgmod = _load("config")
        self._bridge_mod = _load("llm_bridge")
        self._tools_mod = _load("tools_client")
        self._asr_mod = _load("asr")            # for should_clarify (pure helper)
        self._ambient_mod = _load("ambient")    # ambient-snapshot formatters
        self._kh_mod = _load("knowledge_host")  # client for the standalone knowledge-host
        self._lease_mod = _load("lm_lease")     # per-LM busy leases (yield to the live path)
        self._capture_mod = _load("capture")    # durable orchestration-trace capture (skill-LoRA)
        self._cal_sync_mod = _load("calendar_sync")  # mirror-aware calendar folding (dedupe + notes)
        self._facade_mod = _load("tool_facade")   # simplified fast-LM tool surface (wrappers)
        self._wsauth = _load("wsauth")
        # Pre-shared WS access token (P1): the cascade listens on the network, so a client
        # must present this token (its first frame) before we'll talk.  Resolved/persisted
        # once here; None when auth is disabled.
        self.ws_token = None
        self.ws_handshake_timeout = 10
        try:
            acfg = self._cfgmod.load_config(config_path).get("server", {}).get("auth", {})
            self.ws_handshake_timeout = acfg.get("handshake_timeout_s", 10)
            if acfg.get("require_auth", True):
                self.ws_token = (acfg.get("token")
                                 or self._wsauth.load_or_create(
                                     acfg.get("token_file", "config/ws_token.txt")))
                _log("┌─ WS access token — enter this once in your client ─┐")
                _log(f"│   {self.ws_token}")
                _log("└────────────────────────────────────────────────────┘")
        except Exception as e:
            _log(f"WS auth setup failed: {e}")
        # Upcoming calendar events, cached by the scheduler scan, for the proactive feed.
        self._calendar_cache: dict = {"events": [], "at": 0.0}
        self.lock = asyncio.Lock()
        # Kind of the in-progress conversation ("audio"|"text") or None when idle.
        # One conversation at a time: the web text chat is only available while no
        # audio session is running, and vice versa.
        self.active_kind = None
        # Degraded-mode notices (e.g. TLS fell back to plain ws://) — shown in
        # the web UI via /api/status and the trace feed.
        self.startup_warnings: list = []
        # Heartbeat file the idle research worker reads to know when to stand down.
        try:
            self._activity_path = self._cfgmod.activity_path(
                self._cfgmod.load_config(config_path))
        except Exception:
            self._activity_path = None
        # Fast-LM busy lease: held while a live chat session is open so the knowledge-host
        # pauses its (fast-LM) distillation and never makes the voice path queue behind it.
        try:
            _lease_cfg = self._cfgmod.load_config(config_path).get("lm_lease", {})
        except Exception:
            _lease_cfg = {}
        self._lease_on = bool(_lease_cfg.get("enabled", True))
        self._lease_ttl = float(_lease_cfg.get("ttl_s", 15))
        # Ephemeral ambient cache: wipe stale rows from a previous run at startup unless
        # the user wants it persisted across restarts.
        try:
            acfg0 = self._cfgmod.load_config(config_path).get("ambient", {})
            if self.memory and acfg0.get("enabled") and not acfg0.get("persist", True):
                self.memory.ambient.clear()
        except Exception:
            pass

    def mark_activity(self, open: bool):
        """Record session open/close so the idle worker knows the box is in use.  The
        worker treats 'not open AND quiet for idle_after_s' as idle.  Best-effort.

        Also drives the fast-LM lease: while a session is open the knowledge-host yields
        the fast LM.  Repeated open=True calls (per turn) refresh it; a keepalive task
        keeps it warm through quiet stretches; close releases it."""
        if self._lease_on:
            if open:
                self._lease_mod.acquire(self._lease_mod.FAST, ttl=self._lease_ttl)
            else:
                self._lease_mod.release(self._lease_mod.FAST)
        if not self._activity_path:
            return
        try:
            self._activity_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: the idle worker treats an unreadable/torn file as "idle",
            # so a partial write could let big-LM work start mid-conversation.
            tmp = self._activity_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"open": bool(open), "ts": time.time()}))
            os.replace(tmp, self._activity_path)
        except Exception:
            pass

    def reload(self):
        """Fresh config + personas from disk (called per connection for live edits)."""
        cfg = self._cfgmod.load_config(self.config_path)
        personas, default = self._cfgmod.load_personas(cfg)
        return cfg, personas, default

    async def handle_personas(self, _req):
        _cfg, personas, default = self.reload()
        items = [{"name": n, "description": p.get("description", ""),
                  "default": n == default} for n, p in personas.items()]
        return web.json_response({"personas": items, "default": default})

    async def handle_status(self, _req):
        """Is a conversation in progress?  The text-chat page polls this to gate itself."""
        return web.json_response({"active": self.active_kind is not None,
                                  "kind": self.active_kind,
                                  "warnings": self.startup_warnings})

    async def handle_chat_page(self, _req):
        """Serve the web text-chat UI (intranet-reachable, same origin as the WS)."""
        return web.Response(text=(Path(__file__).parent / "chat_ui.html").read_text(),
                            content_type="text/html")

    # ── Notifications: the client polls this; the scheduler below fills the queue ──
    async def handle_notifications(self, request):
        """Return notifications whose time has come.  ?peek=1 leaves them unread (for a
        bell badge); a normal GET hands them over and marks them delivered."""
        if not self.memory:
            return web.json_response({"notifications": []})
        peek = request.query.get("peek") in ("1", "true", "yes")
        items = self.memory.due_notifications(peek=peek)
        return web.json_response({"notifications": items})

    async def notification_scheduler(self):
        """Background loop: scan the calendar (via the tool host) for upcoming events
        and queue reminders at the configured lead times.  Runs whether or not anyone's
        in a session — that's the point of a notification.  Best-effort; re-reads config
        each pass so it can be toggled live."""
        await asyncio.sleep(5)                       # let services settle first
        while True:
            poll = 60
            try:
                cfg = self._cfgmod.load_config(self.config_path)
                ncfg = cfg.get("notifications", {})
                pcfg = cfg.get("proactive", {})
                poll = ncfg.get("poll_interval_s", 60)
                # Scan if the bell OR the proactive feed is on (the scan both caches upcoming
                # events for the feed and queues lead-time reminders).
                if self.memory and (ncfg.get("enabled") or pcfg.get("enabled", True)):
                    await self._scan_calendar(cfg, ncfg, pcfg)
                if self.memory and cfg.get("ambient", {}).get("enabled"):
                    await self._refresh_ambient(cfg)
            except Exception as e:
                _log(f"notification scheduler: {e}")
            await asyncio.sleep(max(15, poll))

    async def _refresh_ambient(self, cfg):
        """Pull each ambient source whose snapshot is older than its TTL and store the
        mechanically-formatted result.  No LM.  Best-effort per source — one bad tool
        doesn't stall the rest."""
        acfg = cfg.get("ambient", {})
        tools = self._tools_mod.ToolHost(cfg["tools"])
        if not tools.active:
            return
        amb = self.memory.ambient
        default_max = acfg.get("max_items_per_source", 4)
        now = time.time()
        for src in acfg.get("sources", []):
            stype = (src.get("type") or src.get("tool") or "ambient")
            source = src.get("source") or stype
            ttl = float(src.get("ttl_s", 1800))
            if now - amb.last_fetch(source) < ttl:
                continue                                 # still fresh — skip
            tool = src.get("tool")
            if not tool:
                continue
            try:
                raw = await tools.call(tool, src.get("arguments", {}))
            except Exception as e:
                _log(f"ambient '{source}': tool call failed ({e})")
                continue
            if not raw or str(raw).startswith("("):       # tool error strings start with "("
                continue
            trust = src.get("trust") or self._ambient_mod.default_trust(stype)
            items = self._ambient_mod.format_source(stype, raw, src.get("max_items", default_max))
            amb.replace_source(source, items, ttl, trust=trust, priority=src.get("priority", 5))
            if self.trace:
                self.trace.write({"ts": now, "session": "scheduler", "kind": "ambient_refresh",
                                  "source": source, "items": len(items), "trust": trust})

    async def _scan_calendar(self, cfg, ncfg, pcfg):
        tools = self._tools_mod.ToolHost(cfg["tools"])
        if not tools.active:
            return
        args = ncfg.get("calendar_args") if ncfg.get("enabled") else None
        args = args or pcfg.get("calendar_args") or {"days": 2}
        raw = await tools.call(ncfg.get("calendar_tool", "calendar_range"), args)
        try:
            data = json.loads(raw)
            events = data.get("events", data) if isinstance(data, dict) else data
        except Exception:
            return                                   # tool didn't return JSON events
        if not isinstance(events, list):
            return
        # Collapse Vinkona's own calendar mirrors onto their originals (carrying her note) so a
        # consolidated calendar doesn't double-count — a no-op until calendar_sync is on.
        if self._cal_sync_mod:
            vinkona_cal = cfg.get("calendar_sync", {}).get("vinkona_calendar", "Vinkona")
            events = self._cal_sync_mod.fold_mirrors(events, vinkona_cal)
        # Cache a minimal, sorted view for the proactive situation feed.
        cache = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            start = _parse_iso(ev.get("start"))
            if start is None:
                continue
            title = (ev.get("title") or ev.get("summary") or "an event").strip()
            cache.append({"start": start, "title": title, "note": ev.get("note", ""),
                          "id": str(ev.get("id") or ev.get("uid") or f"{title}@{ev.get('start')}")})
        cache.sort(key=lambda e: e["start"])
        self._calendar_cache = {"events": cache, "at": time.time()}
        # Queue lead-time reminders only when the bell is enabled.
        if not ncfg.get("enabled"):
            return
        now = time.time()
        made = 0
        for ev in cache:
            for lead in ncfg.get("lead_times_min", [1440, 60]):
                deliver_at = ev["start"] - lead * 60
                if deliver_at < now - 60:            # the lead time already passed
                    continue
                when = _humanize_when(ev["start"], lead)
                if self.memory.add_notification(
                        f"{ev['title']} — {when}", deliver_at, kind="appointment",
                        source="calendar", dedup_key=f"cal:{ev['id']}:{lead}"):
                    made += 1
        if made and self.trace:
            self.trace.write({"ts": time.time(), "session": "scheduler",
                              "kind": "notify_scheduled", "count": made})

    async def _authenticate(self, ws) -> bool:
        """Gate a fresh connection on the pre-shared token: the client's FIRST frame must
        be 0x01 + the token (UTF-8).  Returns True if it checks out; otherwise tells the
        client and closes.  Runs before anything else, so an unauthenticated peer learns
        nothing (not even whether a session is busy)."""
        try:
            msg = await ws.receive(timeout=self.ws_handshake_timeout)
        except asyncio.TimeoutError:
            msg = None
        data = msg.data if (msg and msg.type == aiohttp.WSMsgType.BINARY) else None
        ok = (isinstance(data, (bytes, bytearray)) and len(data) >= 1 and data[0] == 0x01
              and self._wsauth.verify(bytes(data[1:]).decode("utf8", "ignore"), self.ws_token))
        if not ok:
            _log("refused connection — missing or invalid access token")
            try:
                await ws.send_bytes(b"\x02" + json.dumps(
                    {"role": "system",
                     "text": "Access denied: invalid or missing token."}).encode("utf8"))
                await ws.close(code=4401, message=b"unauthorized")
            except Exception:
                pass
        return ok

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.ws_token and not await self._authenticate(ws):
            return ws
        kind = "text" if request.query.get("mode") == "text" else "audio"
        # Mutual exclusion: one conversation at a time.  Refuse (rather than
        # silently queue) a second connection so the text-chat page can show busy.
        # The check-and-set below has no await between, so it's race-free in asyncio.
        if self.active_kind is not None:
            _log(f"refusing {kind} connection — {self.active_kind} session active")
            await ws.send_bytes(b"\x02" + json.dumps(
                {"role": "system",
                 "text": f"Busy: a {self.active_kind} session is in progress."}).encode("utf8"))
            await ws.close()
            return ws
        self.active_kind = kind
        _log(f"accepted {kind} connection")
        try:
            async with self.lock:
                cfg, personas, default = self.reload()
                if self.memory:
                    self.memory.reload(cfg)     # pick up UI/reflection edits + tuned knobs
                await _Session(self, ws, request, cfg, personas, default).run()
        finally:
            self.active_kind = None
        _log("done with connection")
        return ws


class _Session:
    """Per-connection state and tasks, configured from a freshly-loaded config."""

    def __init__(self, server, ws, request, cfg, personas, default_persona):
        self.s = server
        self.ws = ws
        self.request = request
        self.cfg = cfg
        self.personas = personas
        self.default_persona = default_persona

        self.out_q: asyncio.Queue = asyncio.Queue()          # float32 frames → client
        self.sentence_q: asyncio.Queue = asyncio.Queue()     # LLM sentences → speaker
        self.user_turn_queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        self.interrupt = False                                # barge-in: drop current speech
        self._just_clarified = False                          # last turn was a "say again?" re-ask
        # Shared with the LLM bridge: it sets .deliberating while the big LM is thinking
        # harder (the "let me think about that" pause) so the barge-in path holds off
        # cutting the reply until it starts answering.  See _recv_loop and _deliberate.
        self._shared = types.SimpleNamespace(user_turn_queue=self.user_turn_queue,
                                             deliberating=False)
        self.session_id = uuid.uuid4().hex
        self.active_tags: set = set()                         # rolling conversation tags
        self._rhythm = ""                                     # usage-rhythm line (set at session open)
        self._persona_name = default_persona

        v = cfg["vad"]
        self.onset_prob = v["onset_prob"]
        self.offset_prob = v["offset_prob"]
        self.onset_rms = v["onset_rms"]
        self.onset_frames = v["onset_frames"]
        self.barge_in_frames = max(v.get("barge_in_frames", v["onset_frames"]), v["onset_frames"])
        self.offset_frames = v["offset_frames"]
        self.asr_min_frames = v["asr_min_frames"]
        self.preroll_frames = v["asr_preroll_frames"]
        self.mic_gain = cfg["audio"]["mic_gain"]
        self.out_prime = max(1, cfg["audio"]["out_prime"])
        # Onset grace: at speech start the GPU is contended, so frames arrive slower than
        # realtime; wait up to this long for the buffer to reach out_prime before pacing
        # begins, so the first word doesn't underrun.  One-time per utterance.
        self.out_prime_fill_s = cfg["audio"].get("out_prime_fill_ms", 500) / 1000.0
        self.tts_url = cfg["tts"]["url"]
        self.voice = cfg["tts"]["default_voice"]
        self.speak = True                                    # voice mode vs text-only
        self._http: aiohttp.ClientSession | None = None

        t = cfg["tts"]
        self.trim_silence = t.get("trim_silence", True)
        self.silence_threshold = t.get("silence_threshold", 0.01)
        self.tail_keep_frames = max(0, round(t.get("tail_keep_ms", 160) / (FRAME_DUR * 1000)))
        self.gap_frames = max(0, round(t.get("sentence_gap_ms", 80) / (FRAME_DUR * 1000)))
        # Cap per-TTS-call text so one synthesis never overruns the engine's audio-token
        # budget and truncates mid-word; long sentences are split at clause boundaries.
        self.tts_max_chars = max(40, t.get("max_tts_chars", 240))

    async def run(self):
        pname = self.request.query.get("persona") or self.default_persona
        self._persona_name = pname
        persona = self.personas.get(pname) or {}
        if persona.get("voice"):
            self.voice = persona["voice"]
        # Output audio (TTS) is always on for voice mode; for text mode it can be turned
        # on with ?speak=1 — you type your turns and hear + read the reply.  The browser
        # (running on the server) plays the frames, so it comes out the box's own
        # headphones; no container audio device needed.
        self.speak = (self.request.query.get("mode") != "text"
                      or self.request.query.get("speak") in ("1", "true", "yes"))
        # Roleplay context surfaces Vinkona's embodiment (appearance/bio) in the identity
        # block; per-persona "roleplay", a ?roleplay=1 query flag, or the people default.
        people_cfg = self.cfg.get("people", {})
        self.roleplay = bool(persona.get("roleplay")) \
            or self.request.query.get("roleplay") in ("1", "true", "yes") \
            or bool(people_cfg.get("roleplay_default", False))
        _log(f"persona: {pname}  voice: {self.voice}  "
             f"input: {'mic' if self.request.query.get('mode') != 'text' else 'text'}  "
             f"speak: {self.speak}  roleplay: {self.roleplay}")

        # Privileged identity layer: seed Vinkona's character from her persona (first run
        # only — never clobbers a self-determined character) and make sure the user record
        # exists.  Both are then injected, always-on, into every turn.
        if self.s.memory and people_cfg.get("enabled", True):
            seed = persona.get("identity") or people_cfg.get("seed") or {}
            self.s.memory.people.seed_self(
                name=seed.get("name", pname.capitalize()),
                pronouns=seed.get("pronouns", "they/them"),
                summary=seed.get("summary", persona.get("description", "")),
                traits=seed.get("traits"), style=seed.get("style"))
            self.s.memory.people.ensure_user()

        await self.ws.send_bytes(b"\x00")                     # handshake

        # Tool hosts: the Mac host, plus the music host and the knowledge host when each is
        # enabled (their own ports).  MultiHost aggregates them so the fast LM can call
        # tools from any of them.
        _tm = self.s._tools_mod
        _hosts = [_tm.ToolHost(self.cfg["tools"])]
        music_cfg = self.cfg.get("music", {})
        if music_cfg.get("enabled") and music_cfg.get("tool_url"):
            _hosts.append(_tm.ToolHost({"enabled": True, "url": music_cfg["tool_url"],
                                        "timeout_s": music_cfg.get("timeout_s", 20),
                                        "auth_token": music_cfg.get("auth_token")}))
        knowledge_cfg = self.cfg.get("knowledge", {})
        if knowledge_cfg.get("enabled") and knowledge_cfg.get("tool_url"):
            _hosts.append(_tm.ToolHost({"enabled": True, "url": knowledge_cfg["tool_url"],
                                        "timeout_s": knowledge_cfg.get("timeout_s", 20),
                                        "auth_token": knowledge_cfg.get("auth_token")}))
        tool_host = _tm.MultiHost(_hosts) if len(_hosts) > 1 else _hosts[0]
        # Wrap the host in the simplified facade for the FAST LM only (the research worker
        # builds its own un-wrapped host, so it keeps the full granular set).  The facade
        # reshapes only what's catalogued; every primitive stays callable underneath.
        fac_cfg = self.cfg.get("tool_facade", {})
        if fac_cfg.get("enabled") and self.s._facade_mod:
            tool_host = self.s._facade_mod.FacadeHost(
                tool_host, hide=fac_cfg.get("hide"), passthrough=fac_cfg.get("passthrough"),
                trace=(self._trace if self.s.trace else None))

        big = self.cfg["big_lm"]
        cap_cfg = self.cfg.get("capture", {})
        capture = None
        if cap_cfg.get("enabled", False):
            cap_dir = Path(cap_cfg.get("dir", "logs/capture"))
            if not cap_dir.is_absolute():                  # resolve against the repo root
                cap_dir = Path(__file__).resolve().parent / cap_dir
            capture = self.s._capture_mod.TraceCapture(
                cap_dir, format_version=cap_cfg.get("format_version", "v0-unfrozen"),
                base_model=self.cfg["fast_lm"].get("model", ""), enabled=True)
        kc = self.cfg.get("knowledge_host", {})
        self._kh = (self.s._kh_mod.KnowledgeHost(
                        kc.get("url", ""), token=kc.get("token", ""),
                        timeout_s=kc.get("timeout_s", 4.0))
                    if kc.get("enabled") else None)
        bridge = self.s._bridge_mod.LLMBridge(
            server_state=self._shared,
            fast_lm_url=self.cfg["fast_lm"]["url"],
            big_lm_url=big.get("url"),
            fast_model=self.cfg["fast_lm"]["model"],
            big_model=big.get("model", "qwen2.5:32b"),
            lease_big=self.s._lease_on,
            lease_ttl=self.s._lease_ttl,
            live_guidance=bool(self._kh and kc.get("live", False)),
            live_guidance_timeout=float(kc.get("live_timeout_s", 0.25)),
            working_memory=bool(self.cfg.get("working_memory", {}).get("enabled", True)),
            working_memory_max=int(self.cfg.get("working_memory", {}).get("max_items", 12)),
            calculator=bool(self.cfg.get("tools", {}).get("calculator", True)),
            capture=capture,
            briefing_prompt=big.get("briefing_prompt"),
            lead=big.get("lead", 1),
            deliberate=big.get("deliberate"),
            speak_sink=self._on_reply_sentence,
            recall_hook=self._recall if self.s.memory else None,
            document_hook=(self._recall_document
                           if (self.s.memory and big.get("ground_in_docs", True)) else None),
            guidance_hook=(self._guidance if self._kh else None),
            self_hook=self._self_knowledge if self.s.memory else None,
            reminder_hook=(self._due_reminders
                           if (self.s.memory and self.cfg.get("notifications", {}).get("enabled"))
                           else None),
            user_questions_hook=(self._pending_user_questions
                                 if (self.s.memory
                                     and self.cfg.get("research", {}).get("plans", {}).get("enabled", True))
                                 else None),
            log_hook=self._log_turn if self.s.memory else None,
            trace_hook=self._trace if self.s.trace else None,
            inject_time=self.cfg["awareness"]["inject_time"],
            location=self.cfg["awareness"]["location"],
            time_meaning=self.cfg["awareness"].get("time_meaning", True),
            latitude=self.cfg["awareness"].get("latitude"),
            longitude=self.cfg["awareness"].get("longitude"),
            holidays_country=self.cfg["awareness"].get("holidays_country"),
            tools=tool_host,
            tool_max_rounds=self.cfg["tools"]["max_rounds"],
            tool_filler=self.cfg["tools"]["filler"],
            confirm_required=self.cfg["tools"].get("confirm_required", True),
            confirm_tools=self.cfg["tools"].get("confirm_tools"),
            announce_tools=self.cfg["tools"].get("announce_tools"),
            verify_writes=self.cfg["tools"].get("verify_writes", True),
            calendar_read_tool=self.cfg["tools"].get("calendar_read_tool", "calendar_range"),
            calendar_cfg=self.cfg["tools"].get("calendar"),
            mail_guidance=self.cfg["tools"].get("mail_guidance"),
            # When the local knowledge base is on, prefer its kb_search over the redundant
            # remote (Mac) Wikipedia tool — drop the latter from the fast LM's catalogue so
            # the local, faster, offline copy wins (only when kb_search is actually offered).
            prefer_tool=(knowledge_cfg.get("prefer_tool", "kb_search")
                         if knowledge_cfg.get("enabled") else None),
            supersede_tools=(knowledge_cfg.get("supersede_tools")
                             if knowledge_cfg.get("enabled") else None),
            research_enqueue=(self._queue_research
                              if (self.s.memory and self.cfg.get("research", {}).get("enabled"))
                              else None),
            news_search=(self._news_search
                         if (self.s.memory
                             and self.cfg.get("research", {}).get("rss", {}).get("enabled"))
                         else None),
            schedule_notification=(self._schedule_reminder
                                   if (self.s.memory and self.cfg.get("notifications", {}).get("enabled"))
                                   else None),
            identity_hook=(self._identity_block
                           if (self.s.memory and people_cfg.get("enabled", True)) else None),
            identity_detail_hook=(self._identity_detail
                                  if (self.s.memory and people_cfg.get("enabled", True)) else None),
            situation_hook=(self._situation
                            if self.cfg.get("proactive", {}).get("enabled", True) else None),
            ambient_hook=(self._ambient_block
                          if (self.s.memory and self.cfg.get("ambient", {}).get("enabled")) else None),
            rhythm_hook=(self._rhythm_block
                         if (self.s.memory and self.cfg.get("awareness", {}).get("time_meaning", True))
                         else None),
            affect_hook=(self._affect_block
                         if (self.s.memory and self.cfg.get("affect", {}).get("enabled", True)) else None),
            affect_update=(self._set_affect
                           if (self.s.memory and self.cfg.get("affect", {}).get("enabled", True)
                               and self.cfg.get("affect", {}).get("live", True)) else None),
            affect_objective=self.cfg.get("affect", {}).get("objective", ""),
            revise_self=(self._revise_self
                         if (self.s.memory and people_cfg.get("enabled", True)) else None),
            note_person=(self._note_person
                         if (self.s.memory and people_cfg.get("enabled", True)) else None),
            confirm_self_edits=people_cfg.get("confirm_self_edits", True),
            roleplay_default=self.roleplay,
            roleplay_adaptive=people_cfg.get("roleplay_adaptive", True),
        )
        bridge.apply_persona(system_prompt=persona.get("system_prompt"),
                             greeting=persona.get("greeting"),
                             voice_examples=(persona.get("identity") or {}).get("voice_examples"))

        self.s.mark_activity(open=True)                       # idle worker stands down
        if self.s.memory and self.cfg.get("awareness", {}).get("time_meaning", True):
            try:                                              # log this session + read the rhythm
                self.s.memory.usage.log("session")
                parts = [self.s.memory.usage.summary(),       # standing rhythm (Phase 2)
                         self.s.memory.rhythms.relevant()]    # what's due about now (Phase 3)
                self._rhythm = " ".join(p for p in parts if p)
            except Exception:
                self._rhythm = ""
        async with aiohttp.ClientSession() as http:
            self._http = http
            out_task = asyncio.create_task(self._output_loop())
            speaker_task = asyncio.create_task(self._speaker_loop())
            bridge_task = asyncio.create_task(bridge.run())
            lease_task = asyncio.create_task(self._fast_lease_keepalive())
            try:
                await self._recv_loop()
            finally:
                for t in (out_task, speaker_task, bridge_task, lease_task):
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        # A task that already died (e.g. the client reset mid-speech)
                        # re-raises here — swallow it or the loop aborts, the fast-LM
                        # lease keepalive leaks, and the session is never reflected.
                        _log(f"session task ended with error (cleanup continues): {e}")
                self.s.mark_activity(open=False)              # session closed; idle clock starts
                await self._reflect()

    # ── Introspection ─────────────────────────────────────────────────────────
    def _trace(self, event: dict):
        event["session"] = self.session_id[:8]
        event["persona"] = self._persona_name
        self.s.trace.write(event)

    # ── Memory hooks ─────────────────────────────────────────────────────────
    async def _recall(self, user_text: str) -> str:
        # Optionally enrich the QUERY embedding with the last few turns so an elliptical
        # ask ("what about her?") still lands near the right memory.  Trigger matching stays
        # on the bare turn inside recall().  Off by default (recall_context_turns = 0).
        context = ""
        n = self.cfg.get("memory", {}).get("recall_context_turns", 0)
        if n > 0:
            # The current user turn is already logged (log_hook runs before recall_hook), so
            # drop it and take the prior turns from this session.
            prior = self.s.memory.session_log(self.session_id)[:-1][-n:]
            context = "\n".join(f"{r['role']}: {r['text']}" for r in prior)
        entries = await self.s.memory.recall(user_text, self.active_tags, context=context)
        chosen = [e for e in entries if not e.get("related")]
        related = [e for e in entries if e.get("related")]
        for e in chosen:                          # only direct hits steer the rolling tags
            self.active_tags.update(e["context_tags"])
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "recall", "query": user_text,
                         "store_size": len(self.s.memory.entries),
                         "diag": self.s.memory.last_diag,
                         "matched": [{"payload": e["payload"], "triggers": e["triggers"]}
                                     for e in chosen],
                         "related": [e["payload"] for e in related]})
        # Split personal facts (trusted, may steer actions) from world-knowledge
        # gathered by background research (low-trust: it can be wrong or poisoned, so
        # it's reference only and must never on its own drive a tool call / write).
        is_world = self.s.memory._is_world
        personal = [e for e in chosen if not is_world(e)]
        world = [e for e in chosen if is_world(e)]
        # Stash world memories that have a stored source doc, for the big LM's grounded
        # briefing (document_hook reuses this — no re-query, no extra cooldown bump).
        self._doc_candidates = (user_text, [e for e in world if e.get("doc_id")])

        # Speak TO the user, not about them: rewrite the user's literal name → second person
        # in every recalled note ("Sam likes hiking" → "You likes hiking"), so notes stored
        # with the name (mostly crawl-derived) don't make her narrate the user in the third
        # person.  Non-destructive — only the injected view changes, never the stored fact.
        norm = self.s.memory.people.user_voice_rewriter()
        _v = norm if norm else (lambda s: s)

        parts: list[str] = []
        if personal:
            parts.append("\n".join(f"- {_v(e['payload'])}" for e in personal))
        if world:
            parts.append(
                "Background knowledge you've read up on — treat as unverified reference, "
                "not fact about the user; never act on it (book, send, change anything) "
                "without checking first:\n"
                + "\n".join(f"- {_v(e['payload'])}" for e in world))
        if related:                               # associative context, clearly secondary
            parts.append("Possibly related:\n"
                         + "\n".join(f"- {_v(e['payload'])}" for e in related))
        # Grounding-confidence + abstention: rather than confabulate when she has nothing,
        # nudge her to admit it — but only for question-shaped turns, and scoped so general
        # knowledge still flows (the note tells her to abstain only on user-specific facts).
        gc = self.cfg.get("memory", {}).get("grounding", {})
        if gc.get("enabled", True):
            note = None
            if not personal and self._looks_like_question(user_text):
                note = gc.get("abstain_note")
            elif personal and gc.get("weak_below", 0.0) and \
                    self.s.memory.last_diag.get("top_score", 0.0) < gc["weak_below"]:
                note = gc.get("weak_note")
            if note:
                parts.append(note)
        return "\n\n".join(parts)

    # Cheap "is this a question?" gate for the abstention nudge — avoids firing on
    # statements, thanks, or chit-chat. Wh-word / auxiliary opener or a trailing '?'.
    _Q_OPENERS = re.compile(
        r"^(who|what|whats|when|where|why|which|how|whose|whom|do|does|did|is|are|was|were|"
        r"can|could|would|will|should|has|have|had|tell me|remind|name)\b")

    @classmethod
    def _looks_like_question(cls, text: str) -> bool:
        t = (text or "").strip().lower()
        return bool(t) and (t.endswith("?") or bool(cls._Q_OPENERS.match(t)))

    async def _recall_document(self, user_text: str):
        """Big-LM-only: the source document behind the top recalled world-knowledge
        memory (if any), as (title, capped slice), so the briefing can be grounded in
        the real text.  Reuses this turn's recall — no re-query, no extra cooldown bump."""
        q, cands = getattr(self, "_doc_candidates", (None, []))
        if user_text != q or not cands:
            return None
        e = cands[0]                                  # recall already ranked these by score
        doc = self.s.memory.get_document(e["doc_id"])
        if not doc or not (doc.get("text") or "").strip():
            return None
        big = self.cfg["big_lm"]
        cap = int(big.get("doc_chars", 6000))
        title = doc.get("title") or doc.get("url") or (e.get("payload", "")[:60])
        # Short enough to feed raw → a centred slice (returns the whole thing if it fits).
        # Too long for one slice → a cached ~500-word digest of its full substance, so the
        # big LM speaks to what the document actually contains, not just a keyhole.
        if len(doc["text"]) > cap:
            digest = await self.s.memory.summarize_document(e["doc_id"], big.get("url"),
                                                            big.get("model"))
            if digest:
                return (f"{title} (digest)", digest)
        return (title, self._slice_document(doc["text"], e.get("payload", ""), cap))

    @staticmethod
    def _slice_document(text: str, around: str, cap: int) -> str:
        """Cap a document to `cap` chars, centred on where the memory's wording first
        appears (so we keep the relevant passage), else take the head."""
        text = (text or "").strip()
        if len(text) <= cap:
            return text
        start = 0
        for w in sorted((around or "").split(), key=len, reverse=True)[:3]:
            if len(w) > 4:
                i = text.lower().find(w.lower())
                if i >= 0:
                    start = max(0, i - cap // 3)
                    break
        snippet = text[start:start + cap].strip()
        return ("…" if start else "") + snippet + "…"

    @staticmethod
    def _last_user_line(conversation_text: str) -> str:
        """The most recent user utterance from the briefing transcript (lines are
        'USER:' / 'ASSISTANT:'), to use as the knowledge-host query nucleus."""
        for line in reversed((conversation_text or "").splitlines()):
            if line.startswith("USER:"):
                return line[len("USER:"):].strip()
        return (conversation_text or "").strip()

    @staticmethod
    def _query_nucleus(conversation_text: str, prior_turns: int = 2, max_extra: int = 8) -> str:
        """The knowledge-host query: the latest user utterance, WIDENED with salient content
        words from the previous user turn(s) — so a discriminator dropped a sentence earlier
        ('multipara', 'emergency') still reaches the host's keyword fit.  The host isn't
        AI-driven (retrieval + a small ranker), so extra discriminator vocab in the query is
        exactly what sharpens the match.  Deterministic, no LM, fits the <200ms budget."""
        lines = [l for l in (conversation_text or "").splitlines() if l.strip()]
        last, last_idx = "", -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("USER:"):
                last, last_idx = lines[i][len("USER:"):].strip(), i
                break
        if not last:
            return (conversation_text or "").strip()
        have = set(re.findall(r"[a-z0-9]+", last.lower()))
        extra, seen, used = [], set(), 0
        for line in reversed(lines[:last_idx]):
            if not line.startswith("USER:"):
                continue                            # situation is described by the user, not Vinkona
            used += 1
            for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", line[len("USER:"):]):
                lw = w.lower()
                if lw in _NUCLEUS_STOP or lw in have or lw in seen:
                    continue
                seen.add(lw); extra.append(w)
                if len(extra) >= max_extra:
                    break
            if len(extra) >= max_extra or used >= prior_turns:
                break
        return (last + " " + " ".join(extra)).strip() if extra else last

    def _trace_kb(self, tool: str, query: str, live: bool, res, block: str = "") -> None:
        """Declare a knowledge-host call in the Live feed — the question asked and what came
        back — for EVERY call (hit, abstain, low-confidence, miss), since the misses are the
        most informative.  `block` is the answer actually used (empty on a miss)."""
        if not self.s.trace:
            return
        r = res or {}
        if tool == "kb_search":
            outcome = ("no_response" if res is None else "low_confidence" if r.get("low_confidence")
                       else "empty" if not r.get("passages") else "ok")
            count = len(r.get("passages") or [])
        else:
            outcome = ("no_response" if res is None else "abstain" if r.get("abstain")
                       else "empty" if not r.get("items") else "ok")
            count = len(r.get("items") or [])
        ans = (block or "").strip()
        if not ans and r:                            # show the top item/passage even on a near-miss
            top = (r.get("items") or r.get("passages") or [{}])[0]
            ans = (top.get("text") or top.get("label") or "").strip()
        self._trace({"ts": time.time(), "kind": "kb_call", "tool": tool, "live": live,
                     "by": "auto", "query": (query or "")[:240], "outcome": outcome,
                     "confidence": r.get("confidence"), "count": count, "answer": ans[:500]})

    async def _guidance(self, conversation_text: str, live: bool = False):
        """Procedural/metacognitive guidance from the standalone knowledge-host for the
        current situation — 'how a skilled assistant handles this' — or None.  Fail-soft,
        confidence-gated.  The host does the retrieval + intent-conditioned rerank; we gate
        and format.

        live=False (default): the background briefing path — a few items as a block for the
          big-LM planner ('fold one into the next move').
        live=True: the synchronous fast-path — QUESTION turns only, ONE crisp directive for
          the fast LM to use right now (the bridge calls this in parallel with recall, under
          a hard timeout).  Keeping it to a single line is what makes it cheap to read and
          safe under the latency budget."""
        kh = getattr(self, "_kh", None)
        if not kh or not kh.enabled:
            return None
        last = self._last_user_line(conversation_text)
        if not last:
            return None
        if live and not self._looks_like_question(last):
            return None                         # gate on the raw utterance (the '?' survives)
        query = self._query_nucleus(conversation_text)   # widened with prior-turn discriminators
        kc = self.cfg.get("knowledge_host", {})
        min_conf = float(kc.get("min_confidence", 0.30))
        n = 1 if live else 3
        block = ""
        res = None
        if kc.get("tool", "kb_ask") == "kb_search":
            intent = ("Assisting this user — surface what to DO next and how (actionable "
                      "procedure), not glossary definitions.")
            res = await kh.search(query, intent=intent, k=int(kc.get("k", 4)), http=self._http)
            if not res or res.get("low_confidence") or float(res.get("confidence") or 0) < min_conf:
                self._trace_kb("kb_search", query, live, res)
                return None
            picks = [p for p in (res.get("passages") or []) if (p.get("text") or "").strip()][:n]
            if live:
                block = (picks[0].get("text") or "").strip() if picks else ""
            else:
                block = "\n".join(
                    f"- {p['text'].strip()}"
                    + (f"  [{(p.get('title') or p.get('source_type') or '').strip()}]"
                       if (p.get('title') or p.get('source_type')) else "")
                    for p in picks)
        else:
            res = await kh.ask(query, rigor=kc.get("rigor", "low"), http=self._http)
            if not res or res.get("abstain") or float(res.get("confidence") or 0) < min_conf:
                self._trace_kb("kb_ask", query, live, res)
                return None
            picks = [it for it in (res.get("items") or []) if (it.get("text") or "").strip()][:n]
            if live:
                it = picks[0] if picks else {}
                label, t = (it.get("label") or "").strip(), (it.get("text") or "").strip()
                block = f"{label}: {t}" if label else t
                steps = it.get("steps") or []
                if steps:                       # one step inline keeps it a single directive
                    s = (steps[0] if isinstance(steps[0], str) else str(steps[0])).strip()
                    if s:
                        block += f" → {s}"
            else:
                lines = []
                for it in picks:
                    label = (it.get("label") or "").strip()
                    lines.append(f"- {label}: {it['text'].strip()}" if label else f"- {it['text'].strip()}")
                    for step in (it.get("steps") or [])[:5]:
                        s = (step if isinstance(step, str) else str(step)).strip()
                        if s:
                            lines.append(f"    • {s}")
                block = "\n".join(lines)
        block = block.strip()
        self._trace_kb(kc.get("tool", "kb_ask"), query, live, res, block)
        if not block:
            return None
        return block[:240 if live else 1200]

    def _rhythm_block(self) -> str:
        """The learned usage-rhythm line for this session (computed once at session open),
        e.g. 'You tend to be around in the evenings — it's later than you usually talk to me.'"""
        return self._rhythm

    async def _fast_lease_keepalive(self):
        """While the audio session is open, refresh the fast-LM lease so the knowledge-host
        keeps yielding the fast LM through quiet stretches between turns (without this, a
        long silence would let the lease expire and the voice path could queue behind a
        distillation chunk)."""
        if not self.s._lease_on:
            return
        lease, ttl = self.s._lease_mod, self.s._lease_ttl
        try:
            while True:
                lease.acquire(lease.FAST, ttl=ttl)
                await asyncio.sleep(max(2.0, ttl / 3))
        except asyncio.CancelledError:
            pass

    def _due_reminders(self) -> str:
        """Due reminders to voice in this turn.  Consumes them (marks delivered) so the
        client bell won't also fire — during a live chat, saying it aloud is delivery."""
        items = self.s.memory.due_notifications()
        if not items:
            return ""
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "notify_spoken",
                         "count": len(items), "texts": [i["text"] for i in items]})
        return "\n".join(f"- {i['text']}" for i in items)

    def _pending_user_questions(self) -> str:
        """Open 'ask the user' questions from learning plans, surfaced for the fast LM to
        raise when they fit.  Marked 'asked' on surface so they're put forward once and the
        Plans view reflects it."""
        plans = self.cfg.get("research", {}).get("plans", {})
        qs = self.s.memory.pending_user_questions(plans.get("surface_user_questions", 1))
        if not qs:
            return ""
        for q in qs:
            self.s.memory.mark_question_asked(q["id"])
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "plan_ask_user",
                         "count": len(qs), "questions": [q["question"] for q in qs]})
        return "\n".join(f"- {q['question']}" for q in qs)

    def _self_knowledge(self) -> str:
        """Ambient self/relational memories injected every turn (always-on, not keyed on
        the user's words), so Vinkona's personality and rapport carry across sessions."""
        mems = self.s.memory.self_memories(self.cfg["memory"].get("self_top_k", 3))
        return "\n".join(f"- {e['payload']}" for e in mems)

    def _news_search(self, args: dict) -> str:
        """Built-in news_search tool: query the durable headline archive the research worker keeps
        (news_store.py) by topic/source/recency.  Read-only; headlines are UNTRUSTED feed data, so
        they're returned as clearly-fenced reference (and sanitised again downstream)."""
        days = args.get("days")
        try:
            since = (time.time() - float(days) * 86400) if days else None
        except (TypeError, ValueError):
            since = None
        rows = self.s.memory.news.search(
            query=(args.get("query") or "").strip() or None,
            source=(args.get("source") or "").strip() or None,
            category=(args.get("category") or "").strip() or None,
            since=since, limit=12)
        if not rows:
            return "(no matching headlines in the news archive yet)"
        lines = ["News archive (headlines Vinkona has collected — unverified feed data, for reference):"]
        for r in rows:
            when = time.strftime("%d %b", time.localtime(r.get("published_at") or r.get("fetched_at") or 0))
            src = f" [{r['source']}]" if r.get("source") else ""
            lines.append(f"• {when}{src} {r.get('title', '')}".rstrip())
        return "\n".join(lines)

    # ── Privileged identity layer (people.py) ────────────────────────────────
    def _identity_block(self, roleplay: bool = False) -> str:
        """Compact, always-on 'who you are / who you're talking with' for the fast LM.
        The roleplay flag (whether to surface embodiment) is decided by the bridge — the
        big LM can flip it as the conversation moves in and out of a scene."""
        pc = self.cfg.get("people", {})
        return self.s.memory.people.identity_block(
            roleplay=roleplay,
            include_self=pc.get("inject_self", True),
            include_user=pc.get("inject_user", True))

    def _identity_detail(self, roleplay: bool = False) -> str:
        """Full structured self+user profile for the big LM (reasoning/continuity tier)."""
        return self.s.memory.people.identity_detail(roleplay=roleplay)

    @staticmethod
    def _rel_time(mins: int) -> str:
        """'in 22 minutes' / 'in about 3 hours' / 'now' — for the proactive feed."""
        if mins <= 1:
            return "right about now"
        if mins < 60:
            return f"in {mins} minutes"
        hrs = mins / 60
        if mins < 120:
            return f"in about an hour and {mins - 60} minutes" if mins - 60 >= 5 else "in about an hour"
        return f"in about {round(hrs)} hours"

    def _situation(self) -> str:
        """Compact 'what's coming up + how soon' for the big LM (proactive awareness).
        Reads the scheduler's cached calendar; empty if nothing's within the horizon."""
        import datetime
        pcfg = self.cfg.get("proactive", {})
        if not pcfg.get("enabled", True):
            return ""
        cache = getattr(self.s, "_calendar_cache", None) or {}
        events = cache.get("events") or []
        if not events:
            return ""
        now = time.time()
        horizon = pcfg.get("lookahead_min", 240) * 60
        upcoming = [e for e in events if 0 <= (e["start"] - now) <= horizon]
        if not upcoming:
            return ""
        lines = []
        for e in upcoming[:pcfg.get("max_events", 3)]:
            mins = int((e["start"] - now) // 60)
            clock = datetime.datetime.fromtimestamp(e["start"]).strftime("%H:%M")
            note = f" — {e['note']}" if e.get("note") else ""
            lines.append(f"- {e['title']} at {clock} ({self._rel_time(mins)}){note}")
        nowstr = datetime.datetime.now().strftime("%A %H:%M")
        return f"It's {nowstr}.\n" + "\n".join(lines)

    def _ambient_block(self) -> str:
        """The disposable 'right now' snapshot (calendar/weather/news) for the fast prompt,
        served straight from the cache the scheduler keeps fresh — no LM, no tool call."""
        try:
            acfg = self.cfg.get("ambient", {})
            return self.s.memory.ambient.block(acfg.get("max_chars", 600),
                                               acfg.get("max_items_per_source", 4))
        except Exception:
            return ""

    def _affect_block(self) -> str:
        """Vinkona's current inner-state line (mood), for the fast prompt.  '' if disabled/empty."""
        try:
            acfg = self.cfg.get("affect", {})
            if not acfg.get("enabled", True):
                return ""
            return self.s.memory.people.self_state()[:acfg.get("max_chars", 280)]
        except Exception:
            return ""

    def _set_affect(self, text: str):
        """Persist a shifted inner state (from the big-LM director mid-conversation)."""
        try:
            if self.s.memory.people.set_self_state(text, source="conversation"):
                if self.s.trace:
                    self._trace({"ts": time.time(), "kind": "affect", "source": "conversation",
                                 "text": text})
        except Exception as e:
            _log(f"affect set failed: {e}")

    def _revise_self(self, args: dict):
        """Commit a change to Vinkona's own character (the revise_self tool)."""
        line = self.s.memory.people.revise_self(
            args.get("attribute", ""), args.get("value", ""),
            layer=args.get("layer", "core"))
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "identity_write", "target": "self",
                         "attribute": args.get("attribute"), "value": args.get("value"),
                         "layer": args.get("layer", "core")})
        return line

    def _note_person(self, args: dict):
        """Record a lasting fact about a person (the note_person tool)."""
        line = self.s.memory.people.note(
            args.get("person", "the user"), args.get("note", ""),
            args.get("facet", "social"))
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "identity_write", "target": "person",
                         "person": args.get("person"), "note": args.get("note")})
        return line

    def _schedule_reminder(self, text: str, when: str) -> str:
        """Built-in remind_me tool: queue a notification the client will surface at `when`
        (an ISO-8601 datetime the LM derived from the current time)."""
        ts = _parse_iso(when)
        if ts is None:
            return "(couldn't understand that time)"
        self.s.memory.add_notification(text, ts, kind="reminder", source="user")
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "notify_scheduled", "count": 1,
                         "text": text, "when": when})
        _log(f"reminder set: '{text}' at {when}")
        return f"Okay, I'll remind you: {text}, {_humanize_when(ts, 0)}."

    def _log_turn(self, role: str, text: str):
        self.s.memory.log_turn(self.session_id, role, text)

    def _queue_research(self, topic: str, query: str, reason: str) -> str:
        """Built-in queue_research tool: the assistant drops a topic into the Tier-3
        queue mid-conversation ("learn about growing potatoes"); the worker handles it
        offline.  Returns a short line the assistant can speak back."""
        n = self.s.memory.enqueue_research(
            self.session_id, [{"topic": topic, "query": query, "reason": reason}])
        if self.s.trace:
            self._trace({"ts": time.time(), "kind": "research_queued", "source": "user_request",
                         "topics": [{"topic": topic, "reason": reason}]})
        _log(f"research: user asked to learn about '{topic}'")
        if not n:
            return "(couldn't queue that)"
        return f"Okay — I'll read up on {topic} later and remember what I learn."

    async def _reflect(self):
        big = self.cfg["big_lm"]
        if not (self.s.memory and big.get("url")):
            return
        try:
            n = await self.s.memory.reflect(self.session_id, big["url"], big["model"],
                                            self.cfg["memory"].get("reflection_prompt"))
            if self.s.trace:
                self._trace({"ts": time.time(), "kind": "reflect", "applied": n,
                             "store_size": len(self.s.memory.entries)})
            if n:
                _log(f"reflection: applied {n} memory ops")
        except Exception as e:
            _log(f"reflection failed: {e}")

        # Tier-3: after basic memories are stashed, queue topics for the research
        # worker to deepen in the background (it runs as a separate process).
        rcfg = self.cfg.get("research", {})
        if not rcfg.get("enabled"):
            return
        try:
            topics = await self.s.memory.propose_research(
                self.session_id, big["url"], big["model"],
                rcfg.get("max_topics_per_session", 3), rcfg.get("research_prompt"))
            if self.s.trace:
                self._trace({"ts": time.time(), "kind": "research_queued",
                             "topics": [{"topic": t.get("topic"), "reason": t.get("reason", "")}
                                        for t in topics]})
            if topics:
                _log(f"research: queued {len(topics)} topic(s): "
                     + ", ".join(t.get('topic', '?') for t in topics))
        except Exception as e:
            _log(f"research proposal failed: {e}")

    # ── Output: paced PCM → client, with a small jitter buffer ────────────────
    async def _output_loop(self):
        prime = self.out_prime
        while True:
            first = await self.out_q.get()
            pending = [first]
            # Fill the jitter buffer to `prime` frames before pacing.  out_prime_fill_s is
            # an OPTIONAL extra onset grace (default 0): it buys smoothness on a
            # GPU-contended first word but adds latency at EVERY burst start (each sentence
            # boundary re-enters here), so keep it small or 0 and lean on out_prime instead.
            deadline = time.monotonic() + max(FRAME_DUR * prime, self.out_prime_fill_s)
            while len(pending) < prime:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.out_q.get(), rem))
                except asyncio.TimeoutError:
                    break
            next_t = time.monotonic()
            i = 0
            while True:
                if i < len(pending):
                    frame = pending[i]
                    i += 1
                else:
                    try:
                        frame = await asyncio.wait_for(self.out_q.get(), timeout=FRAME_DUR * prime)
                    except asyncio.TimeoutError:
                        break                       # genuine gap — end of burst
                    pending = [frame]
                    i = 1
                now = time.monotonic()
                delay = next_t - now
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -FRAME_DUR:
                    next_t = now              # fell >1 frame behind (slow patch) — resync
                                              # instead of burst-flooding the client (clips)
                next_t += FRAME_DUR
                await self.ws.send_bytes(b"\x03" + frame.astype(np.float32).tobytes())

    # ── Text bubbles (kind 0x02): {"role": "user"|"assistant", "text": ...} ──
    async def _send_bubble(self, role: str, text: str):
        await self.ws.send_bytes(b"\x02" + json.dumps({"role": role, "text": text}).encode("utf8"))

    @staticmethod
    def _tts_chunks(text: str, max_chars: int) -> list:
        """Split a long sentence into TTS-sized pieces so no single synthesis call
        overruns the engine's per-utterance token budget (which truncates mid-word).
        Splits at clause boundaries (, ; : — –) first, packs greedily up to max_chars,
        and word-splits anything still too long.  Short sentences pass through whole."""
        text = (text or "").strip()
        if len(text) <= max_chars:
            return [text] if text else []
        parts = re.split(r"(?<=[,;:—–])\s+", text)
        chunks, cur = [], ""
        for p in parts:
            if cur and len(cur) + 1 + len(p) <= max_chars:
                cur += " " + p
            else:
                if cur:
                    chunks.append(cur)
                cur = p
            while len(cur) > max_chars:                 # a single clause still too long
                cut = cur.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                chunks.append(cur[:cut].strip())
                cur = cur[cut:].strip()
        if cur:
            chunks.append(cur)
        return chunks

    async def _on_reply_sentence(self, sentence: str):
        if not sentence:
            return
        await self._send_bubble("assistant", sentence)         # the bubble shows the whole sentence
        if self.speak and not self.interrupt:
            for chunk in self._tts_chunks(sentence, self.tts_max_chars):
                await self.sentence_q.put(chunk)

    async def _speaker_loop(self):
        while True:
            sentence = await self.sentence_q.get()
            if self.interrupt:
                continue
            await self._speak(sentence)

    async def _push_frame(self, frame: np.ndarray, state: dict):
        """Enqueue one 80 ms frame, trimming the sentence's silent head/tail.

        Leading silence is dropped; interior silence (real pauses) is held and
        flushed when speech resumes; trailing silence is never flushed (dropped at
        end of sentence).  This collapses the dead air Orpheus bakes around each
        utterance, leaving a short uniform gap added by the caller instead.
        """
        if not self.trim_silence:
            await self.out_q.put(frame)
            return
        loud = float(np.sqrt(np.mean(frame ** 2))) >= self.silence_threshold
        if not state["started"]:
            if loud:
                state["started"] = True
                await self.out_q.put(frame)
        elif loud:
            for held in state["held"]:
                await self.out_q.put(held)
            state["held"].clear()
            await self.out_q.put(frame)
        else:
            state["held"].append(frame)

    async def _speak(self, sentence: str):
        """Stream a sentence from the TTS service, enqueuing frames as they arrive."""
        buf = b""
        state = {"started": False, "held": []}
        try:
            async with self._http.post(f"{self.tts_url}/synthesize_stream",
                                       json={"text": sentence, "voice": self.voice},
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    _log(f"TTS {resp.status}: {(await resp.text())[:160]}")
                    return
                async for data in resp.content.iter_chunked(FRAME_SIZE * 2):
                    if self.interrupt:
                        return
                    buf += data
                    while len(buf) >= FRAME_SIZE * 2:        # 16-bit → 2 bytes/sample
                        frame = np.frombuffer(buf[:FRAME_SIZE * 2], dtype=np.int16)
                        buf = buf[FRAME_SIZE * 2:]
                        await self._push_frame(frame.astype(np.float32) / 32768.0, state)
        except Exception as e:
            _log(f"TTS stream failed: {e}")
            return
        if buf and not self.interrupt:
            frame = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
            await self._push_frame(np.pad(frame, (0, FRAME_SIZE - len(frame))), state)
        # End of sentence.  With trimming on, first keep a short hangover of the trailing
        # audio so soft word-endings aren't clipped (held only fills when trimming).  Then
        # add the uniform inter-sentence gap REGARDLESS of trimming — previously it was
        # gated on state["started"], which is only set when trim_silence is on, so with
        # trimming off sentences ran together and the last word got run over.
        if not self.interrupt:
            if self.trim_silence and state["started"]:
                for held in state["held"][:self.tail_keep_frames]:
                    await self.out_q.put(held)
            for _ in range(self.gap_frames):
                await self.out_q.put(np.zeros(FRAME_SIZE, dtype=np.float32))

    def _drain(self, q: asyncio.Queue) -> int:
        n = 0
        while not q.empty():
            try:
                q.get_nowait()
                n += 1
            except asyncio.QueueEmpty:
                break
        return n

    # ── Input: mic PCM → denoise → VAD → ASR → user_turn_queue ───────────────
    async def _recv_loop(self):
        all_pcm = None
        vad_speaking = False
        silence_frames = 0
        voiced = 0                     # consecutive voice frames (resets on non-voice)
        user_buf: list = []
        preroll = deque(maxlen=self.preroll_frames)

        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break
                continue
            data = msg.data
            if not data:
                continue
            if data[0] == 4:                      # kind 0x04: typed text (text mode)
                try:
                    text = (json.loads(data[1:] or b"{}").get("text") or "").strip()
                except Exception:
                    continue
                if text:
                    await self._send_bubble("user", text)
                    self.s.mark_activity(open=True)        # keep the idle worker at bay
                    try:
                        self.user_turn_queue.put_nowait(text)
                    except asyncio.QueueFull:
                        _log("user_turn_queue full — dropping typed turn")
                continue
            if data[0] != 3:                      # otherwise only kind 0x03 raw PCM
                continue
            pcm = np.frombuffer(data[1:], dtype=np.float32)
            if self.mic_gain != 1.0:
                pcm = np.tanh(pcm * self.mic_gain)
            all_pcm = pcm if all_pcm is None else np.concatenate((all_pcm, pcm))

            while all_pcm.shape[-1] >= FRAME_SIZE:
                chunk = np.array(all_pcm[:FRAME_SIZE], dtype=np.float32)
                all_pcm = all_pcm[FRAME_SIZE:]

                speech_prob = None
                if self.s.denoiser is not None:
                    chunk, speech_prob = self.s.denoiser.process(chunk)

                if speech_prob is not None:
                    thr = self.offset_prob if vad_speaking else self.onset_prob
                    is_voice = speech_prob >= thr
                else:
                    is_voice = float(np.sqrt(np.mean(chunk ** 2))) >= self.onset_rms

                if is_voice:
                    voiced += 1
                    # Start capturing once we've seen a short debounced run of voice
                    # (grab preroll so the first word isn't clipped).
                    if not vad_speaking and voiced >= self.onset_frames:
                        user_buf = list(preroll)
                        vad_speaking = True
                        silence_frames = 0
                    if vad_speaking:
                        silence_frames = 0
                        # Only interrupt the AI once voice is *sustained* — brief
                        # noise blips start a (discardable) capture but don't cut it.
                        if (not self.interrupt and voiced >= self.barge_in_frames
                                and not getattr(self._shared, "deliberating", False)):
                            self.interrupt = True
                            dropped = self._drain(self.out_q) + self._drain(self.sentence_q)
                            if dropped:
                                _log(f"barge-in: dropped {dropped} queued items")
                else:
                    voiced = 0
                    if vad_speaking:
                        silence_frames += 1
                        if silence_frames >= self.offset_frames:
                            vad_speaking = False
                            silence_frames = 0
                            if self.s.asr is not None and len(user_buf) >= self.asr_min_frames:
                                asyncio.create_task(self._transcribe(np.concatenate(user_buf)))
                            user_buf = []

                if self.s.asr is not None and vad_speaking:
                    user_buf.append(chunk)
                preroll.append(chunk)

    def _asr_name_bias(self) -> str | None:
        """An initial_prompt for Whisper built from the names Vinkona knows (people store),
        so it spells proper nouns it would otherwise mangle.  None if disabled/empty."""
        a = self.cfg.get("asr", {})
        if not a.get("name_bias", True) or not self.s.memory:
            return None
        names = self.s.memory.people.vocabulary(a.get("name_bias_limit", 24))
        names = list(dict.fromkeys([*names, *a.get("name_bias_extra", [])]))
        return ("People in this conversation: " + ", ".join(names) + ".") if names else None

    async def _transcribe(self, clip: np.ndarray):
        loop = asyncio.get_running_loop()
        prompt = self._asr_name_bias()
        try:
            text, conf = await loop.run_in_executor(
                None, self.s.asr.transcribe, clip, self.cfg["asr"], prompt)
        except Exception as e:
            _log(f"ASR failed: {e}")
            return
        text = (text or "").strip()
        if not text:
            return
        # Confidence gate: if Whisper was genuinely unsure on a non-trivial turn, don't
        # feed the likely-garbled text to the LM (where it would also harden into memory) —
        # ask the user to repeat instead.  Guarded so a bad mic can't trap them in a loop.
        if self.s._asr_mod.should_clarify(text, conf, self.cfg["asr"], self._just_clarified):
            self._just_clarified = True
            line = (self.cfg["asr"].get("clarify_prompt")
                    or "Sorry, I didn't quite catch that — could you say it again?")
            _log(f"You(?): {text!r}  (low confidence {conf:.2f} → asking to repeat)")
            if self.s.trace:
                self._trace({"ts": time.time(), "kind": "asr_clarify",
                             "heard": text, "confidence": round(conf, 2)})
            await self._on_reply_sentence(line)   # bubble + speak; do NOT enqueue the turn
            return
        self._just_clarified = False
        _log(f"You: {text}")
        await self._send_bubble("user", text)
        self.interrupt = False               # user finished — let the reply play
        self.s.mark_activity(open=True)      # keep the idle worker at bay
        try:
            self.user_turn_queue.put_nowait(text)
        except asyncio.QueueFull:
            _log("user_turn_queue full — dropping turn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfgmod = _load("config")
    cfg = cfgmod.load_config(args.config)

    _log("loading RNNoise + ASR ...")
    denoiser = _load("rnnoise_frontend").RNNoiseFrontend(in_rate=SAMPLE_RATE)
    asr = _load("asr").WhisperASR(model=cfg["asr"]["model"], in_rate=SAMPLE_RATE)
    import soxr

    memory = None
    if cfg["memory"]["enabled"]:
        memory = _load("memory").MemoryStore(cfg)
        _log(f"memory: {len(memory.entries)} entries  (db {cfg['memory']['db_path']})")

    cs = cfg["config_server"]
    trace = TraceLog(cs.get("trace_path", "config/trace.jsonl"),
                     cs.get("trace_max_events", 400))
    _log(f"trace feed → {cs.get('trace_path', 'config/trace.jsonl')}")

    srv = CascadeServer(args.config, denoiser, asr, soxr, memory, trace)
    _, personas, default = srv.reload()
    _log(f"personas: {', '.join(personas) or '(none)'}  default={default}")

    app = web.Application()
    app.router.add_get("/api/chat", srv.handle_chat)
    app.router.add_get("/api/personas", srv.handle_personas)
    app.router.add_get("/api/status", srv.handle_status)
    app.router.add_get("/api/notifications", srv.handle_notifications)
    app.router.add_get("/chat", srv.handle_chat_page)

    async def _start_scheduler(app):
        app["notif_task"] = asyncio.create_task(srv.notification_scheduler())

    async def _stop_scheduler(app):
        t = app.get("notif_task")
        if t:
            t.cancel()
    app.on_startup.append(_start_scheduler)
    app.on_cleanup.append(_stop_scheduler)

    ssl_ctx = None
    ssl_dir = cfg["server"].get("ssl_dir")
    if ssl_dir:
        cert, key = Path(ssl_dir) / "cert.pem", Path(ssl_dir) / "key.pem"
        if not (cert.exists() and key.exists()):
            # Self-heal: generate the self-signed pair right here (same command
            # ./install.sh core runs). Only if that fails do we fall back to
            # plain ws:// — running beats refusing to start, but never silently:
            # it's logged AND pushed to the web UI's Live feed via the trace.
            _log(f"TLS is on (server.ssl_dir = {ssl_dir!r}) but {cert} is missing — generating a self-signed pair ...")
            try:
                Path(ssl_dir).mkdir(parents=True, exist_ok=True)
                subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:4096",
                                "-keyout", str(key), "-out", str(cert),
                                "-days", "3650", "-nodes", "-subj", "/CN=vinkona"],
                               check=True, capture_output=True, timeout=60)
                _log(f"created {cert} (self-signed, 10 years) — TLS is on")
            except Exception as e:
                warning = ("TLS certs are missing and could not be generated — running UNENCRYPTED ws:// "
                           "(the access token still gates connections, but audio is in the clear on your LAN). "
                           "Fix: re-run './install.sh core', or create certs/ per the README.")
                _log(f"cert generation failed ({e}).")
                _log(warning)
                trace.write({"ts": time.time(), "session": "startup",
                             "kind": "warning", "text": warning})
                srv.startup_warnings.append(warning)
        if cert.exists() and key.exists():
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    _log(f"cascade server on {'https' if ssl_ctx else 'http'}://{host}:{port}  tts={cfg['tts']['url']}")
    web.run_app(app, host=host, port=port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    main()
