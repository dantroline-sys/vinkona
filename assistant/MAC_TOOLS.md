# Vinkona Tool Host — contract for the Mac side

Vinkona (the voice assistant, on the Linux GPU box) can call **tools** mid-conversation
to fetch live info — calendar, files, mail — that lives on the **Mac Mini**. The Mac
runs a small **tool host**: one HTTP service that exposes those tools. Vinkona is the
client; it fetches the tool catalogue, hands it to the language model, and forwards
any tool call the model decides to make.

You can implement the host however you like. **Recommended:** back it with existing
**MCP servers** (filesystem, Apple Calendar/Reminders/Notes, mail) and let the host
translate — so you don't write tool logic, you just wrap servers. A thin
FastAPI/Express app works equally well if you'd rather call EventKit/AppleScript
directly. Either way, you only have to satisfy the contract below.

## Network — bind localhost, reach over SSH (do this)
- **Bind the tool host to `127.0.0.1:8765` on the Mac** — do NOT expose it on the LAN.
  Vinkona reaches it through an SSH tunnel, so the only access path is an authenticated
  SSH connection.
- Vinkona runs `./serve_tunnel.sh`, which forwards its `localhost:8765` → the Mac's
  `127.0.0.1:8765`. Vinkona's `tools.url` stays `http://127.0.0.1:8765` (the tunnel's
  local end). Nothing else changes on Vinkona's side.
- Keep responses fast (Vinkona waits on them in-conversation); target < a few seconds.

### What you (Mac) need to set up
1. **Enable Remote Login** (System Settings → General → Sharing → Remote Login), i.e. sshd.
2. **Install Vinkona's tunnel key.** Vinkona generates an `ed25519` keypair
   (`~/.ssh/vinkona_tunnel` + `.pub`) and sends you the **`.pub`**. Append it to the
   Mac login user's `~/.ssh/authorized_keys`.
3. **Harden it to forward-only** (recommended) — prefix that line in `authorized_keys`
   so the key can ONLY tunnel to the tool host, nothing else:
   ```
   no-pty,no-agent-forwarding,no-x11-forwarding,permitopen="127.0.0.1:8765" ssh-ed25519 AAAA…vinkona_tunnel
   ```
   With that, the key can't open a shell or forward anywhere except the tool host port.

   **Allowlist every port you forward.** `permitopen` is an allowlist — if you add more
   forwards on Vinkona's side (`tools.tunnel.extra_forwards`, e.g. a SearXNG at
   `127.0.0.1:8888` for research), add a matching `permitopen` for each, comma-separated
   on the same line:
   ```
   no-pty,no-agent-forwarding,no-x11-forwarding,permitopen="127.0.0.1:8765",permitopen="127.0.0.1:8888" ssh-ed25519 AAAA…vinkona_tunnel
   ```
   A port that's forwarded but not allowlisted fails with `channel N: open failed:
   administratively prohibited` on the tunnel. `authorized_keys` is re-read per
   connection, so just reconnect the tunnel after editing — no sshd restart.
4. Tell Vinkona the Mac's **IP/hostname** and **ssh user** (they go in `tools.tunnel`).

## Endpoints

### `GET /tools` — the catalogue
Return the tools you offer, as OpenAI-style function specs:

```json
{
  "tools": [
    {
      "name": "calendar_today",
      "description": "List the user's calendar events for today.",
      "parameters": { "type": "object", "properties": {} }
    },
    {
      "name": "file_search",
      "description": "Search the user's files by keyword and return matches.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "Search text" },
          "limit": { "type": "integer", "description": "Max results", "default": 5 }
        },
        "required": ["query"]
      }
    }
  ]
}
```

- `name`: snake_case, stable.
- `description`: write it for the model — say exactly when to use it. This is what
  makes the model call the right tool at the right time, so be precise.
- `parameters`: JSON Schema (object). Empty `properties` if it takes no args.

### `POST /call` — run a tool
Request:
```json
{ "name": "file_search", "arguments": { "query": "tax return 2024", "limit": 5 } }
```
Response (success):
```json
{ "ok": true, "result": "Found 2 files:\n- ~/Documents/tax-2024.pdf\n- ~/Downloads/hmrc.pdf" }
```
Response (failure):
```json
{ "ok": false, "error": "calendar access not granted" }
```

- **`result` is a STRING.** For **interactive tools** the model reads it and speaks an
  answer, so render it nicely — a short list, a sentence — and keep it concise.
  **Exception:** the **crawl list tools** (`mail_list`, `file_index`, …) are read by a
  program, not the model, and must put **machine-readable JSON _inside_ that string** —
  see [Crawl list tools](#crawl-list-tools--machine-readable-not-prose). Don't conflate
  the two: a browsing tool returns prose; a crawl lister returns JSON-as-a-string.
- Always return `ok: true/false`; on error put a short reason in `error`. Never hang —
  honour a timeout and return an error string instead.

### `GET /health` (optional)
Return `200` so we can check the host is up.

## Scope for v1 (please)
- **Read-only tools first**: `calendar_today`, `calendar_range`, `file_search`,
  `file_read`, `mail_search`, `mail_read`. Get the round-trip solid before any writes.
- **Writes** (create event, send mail, modify files) come later and will go behind a
  spoken confirmation on Vinkona's side — flag any write tool clearly in its description
  (e.g. "Creates a calendar event — destructive") so we can gate it.
- Permissions: the Mac will prompt for Calendar/Contacts/Full-Disk access the first
  time — make sure the host process has them granted.

## Paginated read tools — for the idle mail/file crawl

Vinkona can slowly learn about the user from their mail and files during idle time
(`research.ingest.crawls`). It walks the corpus a small batch at a time, keeping a
**cursor** (an offset) per source and advancing it each cycle, so it covers everything
over days rather than re-reading the same items. Each source is a **list tool** (gives a
page of items) plus a **read tool** (gives one item's full text).

> **This is a different job from interactive browsing.** Your nice "start here, drill in,
> here's a tidy list" tools are for the *live model*. The crawl is a *program* walking
> everything by offset — it needs structured, stable, paginated JSON, not prose. If you
> already have a browsing `file_list`, **leave it** and add a separate crawl lister
> (`file_index`) rather than trying to make one tool do both.

### Crawl list tools — machine-readable, not prose

A crawl list tool's `result` string must contain **a JSON array of objects** (it can be a
bare array, or wrapped under `items`/`mails`/`emails`/`files`/`results`). Vinkona
`json.loads()` the `result` and keeps only the objects; each object **must carry the field
named by the job's `id_field`** (`id` for mail, `path` for files) — that's the handle the
read tool and the de-dup registry use.

**What Vinkona sends** (it injects `offset`/`limit` itself — you only declare them):
```json
{ "name": "file_index", "arguments": { "path": "~/Documents", "offset": 0, "limit": 8 } }
```
**What you must return** — JSON **encoded as a string** in `result`:
```json
{ "ok": true, "result": "[{\"path\": \"/Users/you/Documents/notes.md\", \"size\": 1820}, {\"path\": \"/Users/you/Documents/cv.pdf\", \"size\": 40211}]" }
```

#### ✓ Right vs ✗ wrong (the exact trap to avoid)

```jsonc
// ✗ WRONG — prose. Vinkona can't parse it → learns nothing, logs `learned:0, wrapped:true`.
{ "ok": true, "result": "Browsable directories:\n- ~/Documents\n- ~/Downloads" }

// ✗ WRONG — a real JSON array, not a string. Vinkona coerces the value with str(), which
//   yields Python single-quotes ("[{'path': …}]") and then fails to parse. Must be a STRING.
{ "ok": true, "result": [ { "path": "…" } ] }

// ✗ WRONG — array of bare filenames (strings, not objects). Non-objects are dropped → empty.
{ "ok": true, "result": "[\"notes.md\", \"cv.pdf\"]" }

// ✓ RIGHT — JSON array of objects, json.dumps()'d into the result string, each with `path`.
{ "ok": true, "result": "[{\"path\": \"/Users/you/Documents/notes.md\", \"size\": 1820}]" }
```

### The tools

```jsonc
{ "name": "mail_list",
  "description": "CRAWL lister: a page of emails in a folder as JSON, oldest first, for offset paging.",
  "parameters": { "type": "object", "properties": {
    "folder": {"type":"string", "description":"e.g. inbox, sent"},
    "offset": {"type":"integer", "default":0},
    "limit":  {"type":"integer", "default":8} }, "required": ["folder"] } }
// result (JSON string): [{"id":"…","from":"…","subject":"…","date":"…","snippet":"…"}, …]
//                       (or {"items":[ … ]} / {"mails":[ … ]} — same thing)

{ "name": "mail_read",
  "description": "Read one email's full text by id.",
  "parameters": { "type":"object", "properties": {"id":{"type":"string"}}, "required":["id"] } }
// result (string): the email body text — prose is fine here (one item, read by the crawler then distilled)

{ "name": "file_index",
  "description": "CRAWL lister: files under a path as JSON, stable order, for offset paging. Separate from the interactive file_list.",
  "parameters": { "type":"object", "properties": {
    "path": {"type":"string"}, "offset":{"type":"integer","default":0},
    "limit":{"type":"integer","default":8} }, "required":["path"] } }
// result (JSON string): [{"path":"/abs/path/file.pdf","name":"file.pdf","size":12345}, …]
//                       each object MUST include "path" (the job's id_field)

{ "name": "file_read",
  "description": "Return the text of a readable file (txt/md/pdf/docx…); empty/error for binaries.",
  "parameters": { "type":"object", "properties": {"path":{"type":"string"}}, "required":["path"] } }
// result (string): the file's extracted text (keep it bounded; Vinkona also caps it)
```

The matching Vinkona-side config (note the arg name must match what you declare —
`path`/`folder`, not `root`; Vinkona adds `offset`/`limit`):
```json
{ "source": "files-documents", "list_tool": "file_index", "list_args": {"path": "~/Documents"},
  "read_tool": "file_read", "id_field": "path", "category": "profile",
  "batch": 8, "read_chars": 2000, "fingerprint_fields": ["size"], "recrawl_after_days": 30 }
```

### Hard requirements (all four, or the crawl learns nothing)
1. **`result` is a JSON-encoded *string*** — `json.dumps(array)`. A raw array breaks (see above).
2. **Items are objects**, each with the `id_field` the job names (`id` / `path`).
3. **Stable ordering** across calls (oldest-first by date, or sorted by path) — offset paging
   shuffles and double-reads/skips otherwise. New items should land at the **end**, so they're
   caught when the cursor wraps back to 0.
4. **Honour `offset`/`limit`** — return *that slice*, not the whole corpus and not a fixed
   first page. Return an **empty array** (`"[]"`) once `offset` runs past the end; Vinkona
   reads that as "done", resets the cursor to 0, and starts the next sweep.

### Troubleshooting — `crawl … learned:0, wrapped:true`
That trace line means *zero parseable items came back*. Walk down:
- `result` is prose (a sentence / bullet list) → make it a JSON array string. **Most common.**
- `result` is a real array, not a string → `json.dumps()` it.
- items are filename/id strings, not objects → wrap each as `{"path": …}` / `{"id": …}`.
- the arg name you declared ≠ what the job sends (`path` vs `root`) → the host ignored it and
  listed its default/empty location → align the names.
- `offset` ignored and you always return page 1 → the cursor never advances; it re-reads and,
  past your first page, looks "empty".

Test one in isolation, exactly as the crawler calls it:
```bash
curl -s http://localhost:8765/call -X POST -H 'content-type: application/json' \
  -d '{"name":"file_index","arguments":{"path":"~/Documents","offset":0,"limit":8}}'
# want: {"ok":true,"result":"[{\"path\":\"…\",\"size\":…}, …]"}  ← a JSON array INSIDE the string
```

### General notes
- `read_chars` is capped Vinkona-side per item; you can also cap on your end.
- Everything Vinkona ingests from these is treated as **untrusted, hostile-by-default** data
  (mail/files are a prime prompt-injection vector) and distilled into low-priority facts
  about the user only — but still scope crawl roots sensibly and never return secrets you
  wouldn't want summarised into a local memory store.

## Calendar write — a dedicated "Vinkona" calendar

Vinkona can record appointments, but to keep your real calendars safe it writes to **its
own calendar inside your existing suite** (iCloud/Google), not your personal ones.

1. **Create one calendar named `Vinkona`** in the account you want (it then syncs to all
   your devices and shows alongside your other calendars).
2. **Reads span all calendars** (so conflicts are caught); **writes go ONLY to the
   `Vinkona` calendar** — enforce this in the host: refuse to create/update/delete events
   on any other calendar, even if asked. This is the whole safety story; please don't
   make it configurable away.

### Write tools to expose
```jsonc
{ "name": "calendar_create",
  "description": "Add an event to the user's 'Vinkona' calendar (the assistant's own calendar). Writes ONLY to that calendar. Checks for conflicts across all calendars first. Times are ISO-8601 in the user's local timezone.",
  "parameters": { "type": "object",
    "properties": {
      "title":  {"type":"string"},
      "start":  {"type":"string", "description":"ISO-8601, e.g. 2026-06-23T15:00"},
      "end":    {"type":"string", "description":"ISO-8601; default +1h if omitted"},
      "notes":  {"type":"string"},
      "force":  {"type":"boolean", "description":"create even if it clashes (default false)"}
    }, "required": ["title","start"] } }
```
Also useful: `calendar_update` (by event id, Vinkona calendar only) and
`calendar_delete` (by event id, Vinkona calendar only).

### Conflict check is yours to enforce (don't trust the LM on times)
On `calendar_create`, **before writing**, scan all calendars for events overlapping
`[start, end)`. If any and `force` is not true, **do not create** — return the clash so
Vinkona can tell the user:
```json
{ "ok": true, "result": "{\"created\": false, \"conflicts\": [\"Dentist 15:00–16:00\"]}" }
```
On success:
```json
{ "ok": true, "result": "{\"created\": true, \"id\": \"ABC123\", \"when\": \"Tue 23 Jun 15:00–16:00\"}" }
```
(`result` is a string, as for every tool — JSON inside a string is fine; Vinkona reads it
back to the user.)

### Vinkona verifies the write — make these contracts hold
Vinkona does **not** trust the model to announce the outcome. After a confirmed write it:
1. Honours the envelope `ok` — a non-200, a timeout, or `ok:false` is reported to the
   user as a failure, never as success. Put a short reason in `error`.
2. Parses the `result` JSON. `{"created": true, "id", "when"}` → booked;
   `{"created": false, "conflicts": [...]}` → it tells the user the clash and does **not**
   claim success; anything else `false` (or an `"error"` field inside `result`) → failure.
   So `ok:true` alone is **not** "created" — set `created` correctly.
3. For `calendar_create`/`calendar_update`, **if your result includes `"verified": true`**
   (you re-read the event server-side after writing), Vinkona trusts it and confirms the
   booking — this is the reliable path, since the host has no sync lag against itself.
   **Preferred:** create the event, read it straight back on the Mac, and return
   `{"created": true, "verified": true, "id": …, "when": …}`.
4. Only if you DON'T return `verified` does Vinkona do its own read-back
   (`calendar_read_tool`, default `calendar_range`, day window covering the event),
   matching **by `id`** (else exact title). That can race calendar sync and produce a
   false "couldn't find it", so self-verifying (step 3) is strongly preferred. Either way,
   return a stable `id` on create.

### Naming matters — Vinkona gates writes by tool name
Vinkona asks the user for spoken confirmation **before** running any tool whose name
contains `create`/`update`/`delete`/`book`/`schedule`/`send`/etc. Keep those verbs in
the names (`calendar_create`, not `calendar_event`) so the guard triggers. Read tools
(`calendar_range`, `calendar_today`) must NOT contain those verbs.

## How to test without Vinkona
```bash
curl http://localhost:8765/tools
curl -X POST http://localhost:8765/call -H 'Content-Type: application/json' \
     -d '{"name":"calendar_today","arguments":{}}'
```
If those two return the shapes above, Vinkona will work. On our side we flip
`tools.enabled` to true and point `tools.url` at your host.
