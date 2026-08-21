# Local tools — the bundled toolset (VIN-LOCAL-01)

The Mac tool host's genres, served from the machine Vinkona runs on — for a
setup with **no Mac** (or no wish to keep one awake).  Same tool names, same
result shapes, same client contract (MAC_TOOLS.md); the implementation lives
in `local_tools/` and joins the cascade + research worker as one more host in
`tools_client.MultiHost`.  A real Mac host, when also connected, **wins any
name clash** — it keeps Spotlight search and OCR; the local genres cover
everything else.

## The genres (each its own opt-in, all off by default)

| Genre | Tools | Backed by |
|---|---|---|
| files | file_search / file_list / file_read / file_index | the folders you list — and ONLY those; text/code, docx, odt, xlsx, pptx, html; PDF with optional `pypdf` |
| news | news_headlines / news_index (+ background feed poller) | your RSS/Atom feeds → the durable NewsStore archive (guid-deduped) |
| weather | weather | Open-Meteo (keyless) |
| research | literature_search, scholar_search, drug_info, reference_lookup, define_word, qa_search, hn_search, events_search, books_search, archive_search, wayback_lookup | the same keyless APIs the Mac host uses |
| mail | mail_list / mail_recent / mail_search / mail_read | IMAP, **strictly read-only** (mailboxes open EXAMINE — even a read can't mark a message seen); app passwords recommended; **no mail_send exists** |
| calendar | calendar_today / calendar_range / calendar_range_json / calendar_create / calendar_update / calendar_delete | CalDAV (iCloud / Google / Nextcloud / Radicale); dateutil expands RRULEs |

Not bundled, deliberately: `web_search`/`web_fetch` (general web stays off by
design — §7 of the integration guide), 4chan, Spotlight-grade file search, OCR.

## Write containment (the part that is not configurable away)

Reads span everything a genre is given; **writes exist only in the calendar
genre**, and only onto the calendar named in
`tools.local.calendar.vinkona_calendar` (default `Vinkona`) — create that
calendar in the account first.  `calendar_create` conflict-checks across ALL
calendars and returns `{"created": false, "conflicts": [...]}` rather than
double-booking (`force` overrides); a successful write is **read back from the
server** and returns `verified: true`, so the spoken confirmation loop never
trusts the model's word for it.  `calendar_update`/`calendar_delete` refuse
any event id outside the Vinkona collection.  The write-verb names keep the
cascade's spoken-confirmation gate triggering, exactly as with a Mac host.

## Network posture

All HTTP genres go through the **amiga_net broker**: `egress_sync.py`
maintains a clearly marked managed block in `egress.toml`, derived from what
is actually configured — the research/weather API hosts when those genres are
on, the hosts of the feed URLs you listed, the host of your CalDAV address.
Disable a genre and its rules are withdrawn on the next save.  Deny-by-default
stays true; every call lands in the egress audit log (Network tab).  **IMAP is
the one non-HTTP lane** — a direct TLS connection to the server you configured
— and the managed block documents it in a comment so reading `egress.toml`
still tells the whole outbound story.

Enabling the research genre is a posture choice: with a Mac host those
lookups leave the Mac; with this genre they leave THIS machine.

## Configuring

- **Web panel** → Tools tab → *Local tools*: per-genre cards, Save, then a
  per-genre **Test** button (signs in / lists calendars / fetches a feed) with
  a person-readable verdict (`POST /api/local_tools/test`).
- **Desk app** → Settings → *Local tools*: the same surface.
- Raw config: the `tools.local` block in `config.json` (all keys documented in
  config.py).  A new chat picks changes up immediately; the research worker
  (feed poller, crawls) after a restart.

## What lights up downstream

Because every consumer already speaks `call_raw`, enabling local genres feeds
the existing machinery unchanged: ambient orientation (weather/calendar/news
in her morning pull), calendar sync + notifications, the mail/file idle crawls
(`file_index`/`mail_list` follow the §4 crawl-lister rules: JSON-in-a-string,
stable oldest-first order, honest offset/limit, `"[]"` past the end), research
source routing, and the news event-memory DB (the poller fills NewsStore;
`news_index` serves from it).

Tests: `test_local_tools.py` — offline on canned transports, including a full
fake-CalDAV conversation and a fake IMAP server.
