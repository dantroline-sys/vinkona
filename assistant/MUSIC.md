# Vinkona Music — feature scope & integration contract

A scoping spec for a **music playback** feature, designed to be built largely in its own
repo/chat and bolted onto Vinkona through small, well-defined seams. Hand this to the music
build; it defines the contract on both sides.

## The core idea: music is a *mode*, not a mixed stream

Vinkona's audio pipe carries 24 kHz mono TTS frames, tuned for speech. Rather than mix
44.1 kHz stereo music into that (real-time resampling, ducking, format juggling — a
nightmare), **the voice session hands off to a player and suspends.** The pipe does one
thing at a time: *voice* or *music*. When the track ends (or you press stop) the voice
session **resumes with its context intact** — same conversation `history`, same memory,
just an I/O-mode round trip.

Two consequences fall straight out, both desirable:
- **No mixing.** Hi-fi stereo is the player's job, not the cascade's.
- **Music time = idle time.** While you listen you aren't talking, so the cascade marks
  the session idle and the Tier-3 worker is free to research / consolidate. Vinkona can
  even open with "while that was playing I read up on X" when voice resumes.

## Two surfaces (build them independently)

```
                      tool calls (GET /tools, POST /call)
   ┌──────────────┐  ───────────────────────────────►  ┌────────────────────────┐
   │  Vinkona core  │   Surface 1: Music tool host        │  Music tool host       │
   │  (fast LM,   │  ◄───────────────────────────────   │  (SEPARATE REPO)       │
   │   bridge)    │   {track_id, title, …, url}         │  indexes tidaler lib   │
   └──────┬───────┘                                     └────────────────────────┘
          │ player-mode protocol (WS control frames)
          ▼            Surface 2: small addition here + in the Flutter client
   ┌──────────────┐   enter_music / player_cmd / state   ┌────────────────────────┐
   │  cascade WS  │  ◄──────────────────────────────►    │  Flutter player        │
   │  _Session    │   audio: client pulls the URL        │  (client repo)         │
   └──────────────┘                                      └────────────────────────┘
```

- **Surface 1 — the music brain.** A standalone **tool host** speaking the exact same
  contract as the Mac tools ([MAC_TOOLS.md](MAC_TOOLS.md)): `GET /tools` + `POST /call`.
  It indexes the [tidaler](https://pypi.org/project/tidaler/) download tree, searches it,
  and starts/controls playback. Vinkona already knows how to call tool hosts, so this needs
  **zero changes to Vinkona core** — it's just another host in `tools.url` (or a second one).
  This is where the featureset lives and grows; build it first, test it with `curl`.
- **Surface 2 — the player handoff.** A small, documented addition to the cascade WS and
  the Flutter client: the mode switch (suspend/resume voice) and transport control. This
  is the only part that touches this repo + the client repo, and it's deliberately thin.

## Audio transport: send the FLAC file through the pipe (DECIDED)

Tidal downloads are **all FLAC**, so there's no codec matrix to handle and no need to
decode to PCM. The chosen transport is the simplest one: **transfer the FLAC file over the
existing WS connection** and let the phone decode it natively.

- No separate HTTP server, no second auth story — it reuses the WS that's already
  authenticated through your tunnel.
- It's a **file transfer, not a real-time stream**, so there's no jitter/latency pressure:
  send it **chunked** (binary frames) so the player can start once it has enough buffered,
  and seeking is local on the phone once the bytes are in hand.
- The phone needs only a **FLAC decoder** + a buffer; full 44.1 kHz stereo, no DSP on
  Vinkona's side.

Flow: `play_music` → the music host returns the resolved track (metadata + a FLAC path or
the bytes) → the cascade sends `enter_music` (metadata + total size), then streams the FLAC
as `music_data` chunks until complete; the phone assembles and plays.

(A URL-pull / HTTP-range alternative exists and is lighter on the WS if you ever serve very
large files or want native seek-before-download, but it costs you a second authenticated
endpoint. Not needed for all-FLAC, file-sized tracks.)

## Surface 1 — Music tool host (standard `/tools` + `/call`)

Same envelope as MAC_TOOLS.md: every `result` is a string; crawl-style listers return a
**JSON array string** of objects (see that doc's "Crawl list tools" — same rule applies).

```jsonc
// Find tracks/albums/playlists in the local library.  result = JSON array string.
{ "name": "music_search",
  "description": "Search the user's local music library by artist, album, track, or mood.",
  "parameters": { "type":"object", "properties": {
    "query": {"type":"string"},
    "kind":  {"type":"string", "enum":["track","album","artist","playlist"], "default":"track"},
    "limit": {"type":"integer", "default":8} }, "required":["query"] } }
// result: [{"track_id":"…","title":"…","artist":"…","album":"…","duration":312,
//           "url":"http://<host>/stream/<id>","art":"http://…"}, …]

// Start playback (say-back tool — speaks an ack, kicks off the side-effect, returns nothing
// to the LM).  Accepts a resolved id OR a free query it resolves itself.
{ "name": "play_music",
  "description": "Play music from the user's library — a track, album, artist, or playlist.",
  "parameters": { "type":"object", "properties": {
    "track_id": {"type":"string"},
    "query":    {"type":"string", "description":"used if no track_id"},
    "mode":     {"type":"string", "enum":["replace","queue","shuffle"], "default":"replace"} } } }
// result: {"now_playing":{…}, "queued":N}  (the cascade turns this into the enter_music handoff)

{ "name": "music_control",
  "description": "Control playback: pause, resume, stop, next, previous, seek.",
  "parameters": { "type":"object", "properties": {
    "action": {"type":"string", "enum":["pause","resume","stop","next","previous","seek"]},
    "value":  {"type":"number", "description":"seconds, for seek"} }, "required":["action"] } }

{ "name": "now_playing",
  "description": "What's playing now, position, and the upcoming queue.",
  "parameters": { "type":"object", "properties": {} } }
// result: {"playing":true,"track":{…},"position":47,"queue":[…]}
```

Library / tidaler notes:
- Index the tidaler download tree by reading on-disk tags (`mutagen` for FLAC/AAC) — don't
  depend on tidaler's internal API, just its output folder. Typical layout is
  `Artist/Album/NN - Title.flac`; make the root and the layout config, not assumptions.
- Cache the index (sqlite or a json) and refresh on a watch or a periodic rescan.
- **Trust:** the library is the user's own, but tag text is still free-form data — sanitize
  it (strip control/turn tokens) before any of it reaches the LM, consistent with the
  project's untrusted-data posture. The stream HTTP endpoint must be access-controlled.
- **v2:** if a requested track isn't local, kick off a tidaler download (needs network — so
  not for radio-silent areas, hence later).

## Surface 2 — Player-mode protocol (cascade WS + Flutter client)

New WS message kinds, JSON payload after a one-byte kind (mirrors the existing `0x02`
bubble / `0x04` typed-text). Existing kinds in use: `0x00` handshake, `0x02` bubble,
`0x03` audio frame, `0x04` typed text — so music control starts at `0x05`.

| kind | dir | name | payload |
|------|-----|------|---------|
| `0x05` | server→client | `enter_music` | `{track:{title,artist,album,duration,art}, size_bytes}` |
| `0x0A` | server→client | `music_data` | raw FLAC chunk (kind byte + bytes); repeated until `size_bytes` sent |
| `0x06` | server→client | `player_state` | `{playing, position, duration, track}` (server push) |
| `0x07` | client→server | `player_cmd` | `{action: stop}` (v1 is stop-only; pause/seek are client-local) |
| `0x08` | client→server | `player_progress` | `{position, state: playing\|ended\|error}` |
| `0x09` | server→client | `exit_music` | `{reason: ended\|stopped\|error}` |

Flutter client gains a **player component**: on `enter_music` it shows transport controls +
art, buffers the `music_data` chunks, decodes/plays the FLAC, emits `player_progress`, and
its **stop button** sends `player_cmd:{stop}`. (This is the part you'll add in the client
repo.) Pause/seek/volume are the player's own local controls — they don't need to involve
the server in v1.

## Suspend / resume — the "don't blank the context" requirement

In the cascade `_Session`:
- Add a `mode` flag (`voice` | `music`). On `play_music` → `enter_music`: set `mode=music`,
  **keep the bridge and its `history` alive** (do NOT tear it down), and have the recv loop
  ignore mic/VAD-driven turns while in music mode. `mark_activity(open=False)` so the idle
  worker may run.
- On `exit_music` (track ended, stop pressed, or error): set `mode=voice`,
  `mark_activity(open=True)`, and resume — the next user turn continues the *same*
  conversation. Optionally Vinkona speaks a one-liner on resume (config: silent | "hope you
  enjoyed that" | "while that played I looked into …").

### Mic and stop (DECIDED)

- **Mic is fully OFF while music plays.** Don't keep VAD/ASR running — the music itself
  would constantly false-trigger it. The cascade stops reading mic input the moment it
  enters music mode and only re-enables it on `exit_music`.
- **No stop-word / no hands-free voice control during music** (same reason — the audio
  would trip it).
- **The track plays to the end unless the user presses the stop button.** Stop is the
  *only* mid-track exit: button → `player_cmd:{stop}` → the Flutter app tears down its
  player AND the cascade sends `exit_music`, and **both resume the chat** (voice back on,
  conversation context intact). A natural track-end does the same via `player_progress:
  {ended}`.

## State machine

```
  voice ──play_music──► music ──(ended│stop│error)──► voice
            ▲                │
            └──play_music────┘   (new request while playing → swap/queue track, stay in music)
  disconnect-during-music → on reconnect, client resumes the URL or returns to voice (config).
```

## Config (a `music` block in config.py)

```jsonc
"music": {
  "enabled": false,
  "tool_url": "http://127.0.0.1:8770",   // the music tool host (its own process/tunnel)
  "transport": "flac_file",              // DECIDED: chunked FLAC over the WS (url = future alt)
  "mic_during_music": false,             // DECIDED: mic off — music would false-trigger VAD
  "stop": "button",                      // DECIDED: button-only stop; no wake-word
  "resume_announce": "learned",          // silent | greeting | learned
  "research_while_playing": true,        // let the Tier-3 worker use the idle window
  "library_root": "~/Music/tidaler"      // informational; the host owns the real path
}
```

## Repo boundaries / what gets built where

- **Separate repo (the new chat):** the **music tool host** — library indexer (tidaler
  tree + mutagen), search, the HTTP stream endpoint, and the four tools above. Fully
  testable with `curl` against the contract, no Vinkona changes. You can even prove the
  library/search end-to-end with **server-local `mpv` playback** before the phone player
  exists.
- **This repo (small, later):** the player-mode protocol in `cascade_server.py`
  (the `mode` flag, the `0x05`–`0x09` frames, suspend/resume, idle handoff) and the `music`
  config block. Vinkona core/bridge is untouched — `play_music` is just a say-back tool call
  routed to the music host.
- **Client repo:** the Flutter player component + transport UI.

## Build phases

1. **Music host, standalone.** Index + search + `mpv`/local playback + the tool contract.
   Curl-testable. (No Vinkona, no phone.)
2. **Handoff.** Player-mode protocol in cascade + Flutter player; URL transport; suspend/
   resume with context; stop button.
3. **Polish.** Idle-research-during-playback surfaced on resume, `now_playing` to Vinkona,
   spoken stop, queues/playlists, on-demand tidaler download.

## Decisions (settled)

1. **Transport:** ✅ chunked **FLAC file over the WS** (all Tidal downloads are FLAC; no
   transcode, no second HTTP server). URL-pull kept only as a future alternative.
2. **Mic during music:** ✅ **fully off**; **no stop-word** (the music would false-trigger it).
3. **Stop:** ✅ **button only** → both the Flutter app and Vinkona resume the chat. Tracks
   otherwise play to the end.
4. **Context:** ✅ **suspend, don't tear down** — voice resumes the same conversation.

## Still open

1. **One tool host or two:** fold music tools into the existing Mac host, or run a second
   host on its own port/tunnel (cleaner separation — recommended).
2. **Where the music host runs** (and where the FLAC library lives): same Mac as the other
   tools, or the Linux box. Wherever it is, the cascade reads the FLAC and chunks it to the
   phone — so the host just needs to return the file (path or bytes).
3. **Progressive vs buffered play:** start playback after N seconds buffered, or wait for
   the full file. (Progressive is nicer; either is fine for v1.)
