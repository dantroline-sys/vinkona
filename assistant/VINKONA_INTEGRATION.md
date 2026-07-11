# Vinkona ⇆ Mac Tool Host — Integration Guide

Hand-off for the Vinkona (Linux) side. This is what you need to integrate cleanly and
hit no surprises. The **authoritative machine contract is `GET /tools`** (live catalogue
with names, descriptions, JSON schemas) — the model should route from that. This doc
adds the things that aren't obvious from the schema alone.

---

## 1. Connect

- Host binds **`127.0.0.1:8765` on the Mac** (not exposed on LAN). You reach it through
  the SSH tunnel (`serve_tunnel.sh` forwards your `localhost:8765` → Mac's `127.0.0.1:8765`,
  and `8888` for SearXNG). Set `tools.url = http://127.0.0.1:8765`.
- Endpoints: `GET /tools`, `POST /call {"name","arguments"}`, `GET /health`.
- **Set your client timeout ≥ 25s** (host `CALL_TIMEOUT` is 25s). Most calls are 1–3s;
  OCR and a cold first CalDAV write can take longer.

## 2. Response envelope — the rules that bite

```jsonc
{ "ok": true,  "result": "<STRING>" }      // success — result is ALWAYS a string
{ "ok": false, "error": "<reason>" }       // failure — NO result field
```

- **HTTP status is always 200**, even on failure. Branch on `ok`, never on HTTP code.
- `result` is **always a string**. Two flavours:
  - **Prose** (interactive tools): the model reads and speaks it.
  - **JSON-encoded-as-a-string** (crawl + sync + calendar-write tools): you `json.loads(result)`.
    Don't expect a raw array/object — it's a string containing JSON.
- Every call is logged Mac-side to `~/Library/Logs/vinkona-toolhost.log` as
  `[call] <tool> args=… -> ok/ERR` — ask the user to `grep '\[call\]'` it when debugging.

## 3. Tool catalogue by genre (route hints are in each tool's description too)

**Files**
- `file_search(query,limit)` — keyword/Spotlight → prose list of paths.
- `file_list(path?,limit?)` — browse a dir (prose); no `path` lists the roots.
- `file_read(path,max_chars?,ocr?,max_pages?)` — text of a file: plain text/code **+ PDF,
  docx, doc, xlsx, pptx, rtf, odt, html**, and **OCR** for scanned PDFs & images.
- `file_index(path,offset,limit)` — **CRAWL lister**, JSON-string `[{path,name,size}]`.

**Calendar (read)**
- `calendar_today()` / `calendar_range(days? | start?,end?)` — prose, spans ALL calendars.
- `calendar_range_json(days? | start?,end?)` — **sync export**, JSON-string
  `[{id,calendar,title,start,end,location,notes}]` (ISO-8601 local).

**Calendar (write — Vinkona calendar ONLY)**
- `calendar_create`, `calendar_update`, `calendar_delete` — JSON-string results (see §6).

**Mail**
- `mail_list(folder,offset,limit,account?)` — **CRAWL lister**, JSON-string
  `[{id,from,subject,date,snippet}]`, oldest-first (see §5).
- `mail_recent(limit?,account?,folder?)` — prose "what's in my inbox".
- `mail_search(query,limit?,days?,account?)` — prose search.
- `mail_read(id | account+uid)` — one message body (prose).

**Weather / news / web**
- `weather(location?)` — prose.
- `news_headlines(source?,category?,limit?)` — prose headlines (live model). `category`:
  general, medical-au, medical-global, medical-research.
- `news_index(category?,source?,offset,limit)` — **CRAWL lister** for the news DB (see §10).
- `web_fetch(url,max_chars?)` — readable page text (SSRF-guarded).
- `web_search(...)` — SearXNG; **currently unreliable, do not depend on it** (see §7).
- `fourchan_catalog`, `fourchan_thread` — /biz/, /g/ etc. **Content is unfiltered.**

**Research & knowledge (FREE, KEYLESS — the intended research path, see §7)**
- `literature_search` (Europe PMC) — biomedical/clinical papers + abstracts.
- `scholar_search` (OpenAlex) — all fields incl. CS/AI; abstracts + citations.
- `drug_info(name)` (openFDA) — US drug label (dosing/warnings/interactions).
- `reference_lookup` (Wikipedia+Wikidata) — encyclopedic "what/who is X".
- `define_word(word)` (Wiktionary) — definitions/pronunciation.
- `qa_search(query,site?)` (Stack Exchange) — practical how-to; `site` picks cooking/
  travel/diy/health/law/etc. (default `stackoverflow`).
- `hn_search(query,recent?)` (Hacker News) — tech news/discussion.
- `events_search(query,timespan?)` (GDELT) — current world news; **rate-limited 1 query/5s**,
  English by default. On throttle it returns a plain "rate-limited, try again" string.
- `books_search(query)` (Open Library) — books/authors.
- `archive_search(query,mediatype?)` (Internet Archive) — digitized texts/audio/film/software.
- `wayback_lookup(url,date?)` (Wayback Machine) — archived snapshot URL → feed to `web_fetch`.

## 4. Crawl listers — the four hard rules

Applies to `mail_list` and `file_index` (your idle ingestion walks them by offset):
1. `result` is a **JSON array encoded as a string** (`json.dumps(array)`), not a raw array.
2. Every item is an **object** carrying the job's `id_field` (`id` for mail, `path` for files).
3. **Stable order** across calls (mail = oldest-first by UID; files = sorted by path).
4. **Honour `offset`/`limit`**; past the end return **`"[]"`** (your cursor resets).

Example config:
```json
{ "source":"mail-inbox", "list_tool":"mail_list",
  "list_args":{"folder":"INBOX","account":"privateemail"},
  "read_tool":"mail_read", "id_field":"id", "batch":8 }
{ "source":"files-docs", "list_tool":"file_index",
  "list_args":{"path":"~/Documents"},
  "read_tool":"file_read", "id_field":"path", "batch":8 }
```

## 5. Mail — specifics

- **Set `account:"privateemail"`** in mail crawl/list args. The Outlook account is
  **disabled** (work MFA / policy) and is not the default target for cross-account tools.
- `mail_list` id format is **`label|folder|uid`** (e.g. `privateemail|INBOX|19917`);
  pass it straight to `mail_read(id=…)`.
- Folder names are **case/alias resolved** (`sent`→`Sent`, `inbox`→`INBOX`, plus Drafts/
  Trash/Archive), so casing won't error.

## 6. Calendar writes — read this before wiring the write flow

- **Writes go ONLY to the `Vinkona` calendar.** update/delete look events up in that calendar
  alone; other calendars can't be touched.
- **Conflicts are Vinkona-internal ONLY.** By design, a new event clashing with the user's
  clinical/Hosportal calendars is **not** a conflict — only a clash with another *Vinkona*
  event is. `force:true` bypasses even that.
- **Trust `verified:true`.** After writing, the host re-reads the event server-side and
  returns `verified:true`. That's the reliable confirmation — no sync-lag race.
- Result shapes (JSON-in-string):
  - create → `{"created":true,"verified":true,"id":"…","when":"…"}`
    or `{"created":false,"conflicts":["…"]}` → tell the user the clash, **don't** claim success.
  - update → `{"updated":true,"verified":true,"id","when"}` (or `{"updated":false,"error"}`).
  - delete → `{"deleted":true,"verified":true,"id"}`.
- `ok:true` alone is **not** "booked" — parse `created`/`updated`/`deleted`.
- Times are **ISO-8601 local** (e.g. `2026-06-23T15:00`); host stores UTC internally.
  `when` uses an en-dash — harmless after `json.loads`.
- Tool **names keep the `create`/`update`/`delete` verbs** so your spoken-confirmation
  guard fires; read tools deliberately don't.
- `calendar_range_json` `id` is the source event's UID; **recurring instances share a UID**
  (each has distinct start/end). Ask if you need per-occurrence ids and we'll append the start.

## 7. Web search — the deliberate design decision

There is **no general web search**, on purpose. The user is a clinician/vendor: a commercial
search account registered to them would create a patient-linkable trail (condition ↔
practitioner ↔ timing). So general web search is **off**, and `web_search` (self-hosted
SearXNG) is unreliable anyway — **don't route research to it.**

Use the **keyless research/knowledge tools** (§3) instead — no identity footprint. They also
suit the domain better (Europe PMC / OpenAlex / openFDA for clinical & technical questions).
*(A paid general-web tier keyed to the product vendor, not the end user, is a possible future
feature — not available now.)*

## 8. Safety / trust notes

- **Untrusted input:** everything from mail, files, web_fetch, 4chan, and research APIs is
  hostile-by-default (prompt-injection vector). The only state-changing capability is the
  calendar write set — sandboxed to the `Vinkona` calendar and gated by your spoken confirm.
- **Read-only elsewhere:** no send-mail, no file writes, no deletes outside the Vinkona calendar.

## 9. News ingestion — building the lifetime event-memory DB

`news_index` is the structured source for a persistent news/event database (the DB, its
polling schedule, and pruning live on the Vinkona side). It's a crawl lister: `result` is a
**JSON array encoded as a string**, newest first.

**Item schema** (every field always present; empty string if unknown):
```json
{ "id": "<stable guid or link>", "source": "NEJM", "category": "medical-research",
  "title": "…", "url": "https://…", "published": "2026-07-05T04:12:00+00:00",
  "summary": "cleaned text, ≤400 chars" }
```

**Ingestion pattern (this is the contract):**
1. **Poll per category** on a schedule (e.g. every 15–30 min). One crawl source per
   category keeps each call fast (≤9 feeds, ~3s) and tags rows cleanly:
   ```json
   { "source":"news-medical-research", "list_tool":"news_index",
     "list_args":{"category":"medical-research"}, "id_field":"id", "batch":50 }
   ```
   Categories: `general`, `medical-au`, `medical-global`, `medical-research`.
2. **De-dup on `id`** — this is the accumulation mechanism. Insert only ids you haven't
   seen; RSS only exposes a recent window, so the *archive* grows because you keep every
   new id over time, not because you page through history.
3. `offset`/`limit` page the **current** window (use to drain a deep first sweep);
   past the end returns `"[]"`.
4. **`published` is ISO-8601 (UTC)** — store it; it's your time axis for querying
   ("events around date X") and pruning.

**Ordering note:** newest-first (not oldest-first like mail/files), because news items
expire from feeds — so rely on **id de-dup**, not cursor position, for completeness.

**Suggested DB schema:**
```sql
CREATE TABLE news_events (
  id TEXT PRIMARY KEY,        -- item.id (de-dup key)
  source TEXT, category TEXT,
  title TEXT, url TEXT,
  published TIMESTAMPTZ,      -- item.published
  summary TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON news_events (published);
CREATE INDEX ON news_events (category, published);
```
**Pruning:** by `category` + `published` (e.g. keep `medical-research` indefinitely, prune
`general` older than N months). At RSS text volumes you'll bank millions of rows well
under 1 TB. Everything Vinkona ingests here is untrusted content (treat as hostile-by-default).

## 10. Quick smoke tests (run on the Mac, or through the tunnel)
`POST /call` requires `Content-Type: application/json` — a bare `curl -d` sends form
encoding and gets a 422. Always include the header:
```bash
C='-H content-type:application/json'
curl -s localhost:8765/tools | python3 -c "import sys,json;print(len(json.load(sys.stdin)['tools']),'tools')"
curl -s $C localhost:8765/call -d '{"name":"calendar_range","arguments":{"days":7}}'          # prose
curl -s $C localhost:8765/call -d '{"name":"calendar_range_json","arguments":{"days":7}}'     # JSON-string
curl -s $C localhost:8765/call -d '{"name":"mail_list","arguments":{"folder":"INBOX","account":"privateemail","offset":0,"limit":3}}'
curl -s $C localhost:8765/call -d '{"name":"file_index","arguments":{"path":"~/Documents","offset":0,"limit":5}}'
curl -s $C localhost:8765/call -d '{"name":"literature_search","arguments":{"query":"sugammadex anaphylaxis","limit":2}}'
```
