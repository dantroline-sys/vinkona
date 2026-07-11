# Cascade WebSocket access token — client integration

The cascade voice/text server listens on the network so the phone can reach it. It used
to accept **any** connection, which meant anyone who could reach the port could talk to
the assistant and pull the user's private memories. It now requires a **pre-shared
token** that the client presents as its first WebSocket frame. This document is the
contract the Flutter client must meet.

## How the token is created and shared

- The **server generates** the token on first run and writes it to
  `config/ws_token.txt` (mode 600). It persists across restarts, so you set it once.
- It's also printed to the server log on every start:

  ```
  ┌─ WS access token — enter this once in your client ─┐
  │   K7M2-9PQR-4XYZ-AB3C
  └────────────────────────────────────────────────────┘
  ```

- Format: **Crockford-style base32, 4 groups of 4**, e.g. `K7M2-9PQR-4XYZ-AB3C`
  (80 bits of entropy). The alphabet deliberately excludes `0 1 I L O U`, so there are no
  ambiguous characters to read or type.
- The user copies it into the Flutter app **once** (a settings field). Store it on-device
  (e.g. secure storage / shared prefs) and reuse it for every connection.

## What the client must do

Immediately on WebSocket **open**, before sending anything else, send **one binary
frame** carrying the token:

```
byte 0      : 0x01            // frame kind = AUTH
bytes 1..n  : token, UTF-8    // the string the user entered
```

- It **must be a BINARY frame** (not text).
- The token may be sent exactly as displayed (with hyphens) or normalized — the server
  compares **case-insensitively and ignores hyphens/spaces**. So `k7m2 9pqr 4xyz ab3c`
  and `K7M29PQR4XYZAB3C` both match `K7M2-9PQR-4XYZ-AB3C`.
- Send it within **`handshake_timeout_s`** (default 10s) of connecting, or you're closed.

Dart sketch:

```dart
final token = await loadStoredToken();              // user-entered, persisted
channel.sink.add(Uint8List.fromList([0x01, ...utf8.encode(token)]));
// ...then proceed exactly as before (mic frames 0x03, typed text 0x04, etc.)
```

## Server responses

- **Accepted** → the server continues the normal protocol: it sends the `0x00` handshake
  byte and the session begins as it does today. **No protocol change after auth.**
- **Rejected** (missing/invalid token, wrong frame type, or timeout) → the server sends a
  `0x02` system bubble `{"role":"system","text":"Access denied: invalid or missing token."}`
  and closes with WebSocket close code **`4401`**. On `4401`, the client should discard
  the stored token and re-prompt the user for a new one.

## Notes & ordering

- The auth frame is the **first** thing the client sends, before mic audio or typed text.
  Existing frame kinds are unchanged: `0x03` = mic PCM, `0x04` = typed text, music
  `0x05–0x0A`. `0x01` (client→server auth) and `0x02` (server→client bubble) /
  `0x00` (server→client handshake) round out the set.
- The query string (`?mode=…&persona=…&speak=…`) is unchanged; the token does **not** go
  in the URL (keeps it out of logs/proxies).
- To disable auth on a trusted, isolated host, set `server.auth.require_auth: false` in
  config — the server then ignores any leading `0x01` frame, so a client that always
  sends one still works.
- The bundled browser text-chat page (`/chat`) already implements this (prompts for the
  token, stores it in `localStorage`, sends the `0x01` frame, clears it on `4401`); use it
  as a reference implementation.

## Server-side config (`server.auth`)

```json
"auth": {
  "require_auth": true,
  "token": null,                      // null → generate + persist to token_file
  "token_file": "config/ws_token.txt",
  "handshake_timeout_s": 10
}
```

Set `token` explicitly if you'd rather pin a known value instead of the generated one.
