#!/usr/bin/env python
"""VIN-LOCAL-01 — the bundled local toolset (local_tools/).

Everything runs offline on canned transports: the catalogue and dispatch, the
files genre's root containment and §4-compliant crawl lister, feed parsing +
archive round trip, weather/research parsers (including Stack Exchange's
unconditional gzip and GDELT's self rate-limit), read-only IMAP against a fake
server, a full fake-CalDAV conversation (discovery, RRULE expansion, conflict
refusal, verified create, and the write-containment gate), and egress.toml's
managed block.

    vinkona_env/bin/python test_local_tools.py
"""
import asyncio
import datetime
import gzip
import json
import os
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import local_tools
from local_tools import caldav, egress_sync, extract, files, mail, news, research, weather

# An offline test must never touch the REAL egress.toml: LocalHost defensively
# syncs the managed block on build, so default-path ensure() becomes a no-op
# here; test_egress_sync exercises the real writer against explicit tmp paths.
_real_ensure = egress_sync.ensure
egress_sync.ensure = lambda cfg, path=None: (_real_ensure(cfg, path) if path else False)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


def _no_net(*a, **k):
    raise AssertionError("network was touched in an offline test")


# ── catalogue & dispatch ─────────────────────────────────────────────────────

def _cfg(**genres):
    local = {"enabled": True}
    for g, v in genres.items():
        local[g] = v
    return {"tools": {"local": local, "calendar": {"timezone": ""}}}


def test_host_shape():
    h = local_tools.LocalHost({"tools": {"local": {"enabled": False}}})
    check("master switch off → host inactive", not h.active)
    h = local_tools.LocalHost({"tools": {"local": {"enabled": True}}})
    check("no genre enabled → still inactive", not h.active)

    h = local_tools.LocalHost(_cfg(weather={"enabled": True, "location": "X"}),
                              env={"fetch": _no_net})
    cat = asyncio.run(h.catalogue())
    names = [t["function"]["name"] for t in cat]
    check("one genre on → exactly its tools catalogued", names == ["weather"])
    got = asyncio.run(h.call_raw("no_such_tool", {}))
    check("an unknown tool is a clean refusal",
          not got["ok"] and "no local tool" in got["error"])

    import tools_client
    class FakeMac:
        active = True
        async def catalogue(self):
            return [{"type": "function", "function": {"name": "weather",
                                                      "description": "mac wins"}}]
        async def call_raw(self, name, args):
            return {"ok": True, "result": "from the mac", "error": ""}
    multi = tools_client.MultiHost([FakeMac(), h])
    cat = asyncio.run(multi.catalogue())
    descs = [t["function"]["description"] for t in cat
             if t["function"]["name"] == "weather"]
    check("a real Mac host wins the name clash (listed first)",
          descs == ["mac wins"])


# ── files ────────────────────────────────────────────────────────────────────

def _docx(path):
    doc = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           "<w:body><w:p><w:r><w:t>Hello from the docx.</w:t></w:r></w:p>"
           "<w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p></w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", doc)


def test_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "docs"
        (root / "sub").mkdir(parents=True)
        (root / "notes.txt").write_text("tax return 2024 notes")
        (root / "sub" / "recipe.md").write_text("mushroom soup recipe")
        (root / "binary.bin").write_bytes(b"\x00\x01\x02" * 100)
        _docx(root / "letter.docx")
        outside = Path(td) / "secret.txt"
        outside.write_text("not shared")

        gcfg = {"enabled": True, "roots": [str(root)]}
        t = {s["name"]: fn for s, fn in files.tools(gcfg, {})}

        got = t["file_search"]({"query": "recipe"})
        check("file_search finds by name inside the roots", "recipe.md" in got)
        got = t["file_list"]({})
        check("file_list with no path lists the shared roots", str(root) in got)
        bad = t["file_list"]({"path": td})
        check("file_list refuses a path outside the roots",
              isinstance(bad, dict) and not bad["ok"])

        got = t["file_read"]({"path": str(root / "notes.txt")})
        check("file_read reads plain text", "tax return" in got)
        got = t["file_read"]({"path": str(root / "letter.docx")})
        check("file_read extracts docx text (stdlib zip+xml)",
              "Hello from the docx." in got and "Second paragraph." in got)
        bad = t["file_read"]({"path": str(outside)})
        check("file_read refuses a file outside the roots — containment",
              isinstance(bad, dict) and not bad["ok"] and "not inside" in bad["error"])
        bad = t["file_read"]({"path": str(root / "binary.bin")})
        check("a binary file reads as a clear refusal, not garbage",
              isinstance(bad, dict) and not bad["ok"])

        # §4: JSON-array-as-a-string, objects with `path`, stable order,
        # honest offset/limit, [] past the end.
        page1 = json.loads(t["file_index"]({"path": str(root), "offset": 0, "limit": 2}))
        page2 = json.loads(t["file_index"]({"path": str(root), "offset": 2, "limit": 2}))
        allp = [i["path"] for i in page1 + page2]
        check("file_index pages in stable path order, objects carry `path`",
              len(page1) == 2 and allp == sorted(allp) and
              all("path" in i and "size" in i for i in page1))
        again = json.loads(t["file_index"]({"path": str(root), "offset": 0, "limit": 2}))
        check("the same offset returns the same page (stable across calls)",
              again == page1)
        past = t["file_index"]({"path": str(root), "offset": 99, "limit": 8})
        check('past the end returns "[]" — the cursor-reset signal', past == "[]")

        got = files.probe(gcfg, {})
        check("files probe counts the readable tree", got["ok"] and "4 files" in got["detail"])
        got = files.probe({"enabled": True, "roots": []}, {})
        check("files probe says plainly when nothing is shared",
              not got["ok"] and "no folders" in got["detail"])


# ── news ─────────────────────────────────────────────────────────────────────

_RSS = ("<rss><channel><item><title>Rain over Alpha</title>"
        "<link>http://n/1</link><guid>g-1</guid>"
        "<description>&lt;p&gt;Heavy rain tonight.&lt;/p&gt;</description>"
        "<pubDate>Mon, 01 Jun 2026 10:00:00 +0000</pubDate></item>"
        "<item><title>Beta wins cup</title><link>http://n/2</link><guid>g-2</guid>"
        "<pubDate>Tue, 02 Jun 2026 10:00:00 +0000</pubDate></item>"
        "</channel></rss>")
_ATOM = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
         "<title>Gamma launch</title><id>g-3</id>"
         '<link href="http://n/3"/><summary>Gamma lifts off.</summary>'
         "<updated>2026-06-03T10:00:00Z</updated></entry></feed>")


def test_news():
    got = news.parse_feed(_RSS, source="AlphaNews", category="general")
    check("RSS parses: title/link/guid/summary/date", len(got) == 2
          and got[0]["guid"] == "g-1" and got[0]["summary"] == "Heavy rain tonight."
          and got[0]["source"] == "AlphaNews")
    got = news.parse_feed(_ATOM, source="SpaceWire")
    check("Atom parses too (href links, id as guid)",
          len(got) == 1 and got[0]["link"] == "http://n/3" and got[0]["guid"] == "g-3")

    def fetch(url, **k):
        if url == "http://a/feed":
            return 200, _RSS.encode()
        if url == "http://b/feed":
            return 200, _ATOM.encode()
        return 404, b""
    feeds = [{"url": "http://a/feed", "source": "AlphaNews", "category": "general"},
             {"url": "http://b/feed", "source": "SpaceWire", "category": "space"},
             {"url": "http://dead/feed", "source": "Dead"}]
    items, errors = news.poll_feeds(fetch, feeds)
    check("the poller gathers every live feed and reports the dead one",
          len(items) == 3 and len(errors) == 1 and errors[0][0] == "http://dead/feed")

    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "memory.db")
        import news_store
        db = sqlite3.connect(db_path)
        store = news_store.NewsStore(db)
        n1 = store.ingest(items, now=1780500000.0)
        n2 = store.ingest(items, now=1780500600.0)
        check("archive ingest dedupes by guid (re-poll is a no-op)",
              n1 == 3 and n2 == 0)
        db.close()

        rows = news_store.index_readonly(db_path, offset=0, limit=2)
        rows2 = news_store.index_readonly(db_path, offset=2, limit=2)
        check("index_readonly pages oldest-first with a stable cursor",
              len(rows) == 2 and rows[0]["guid"] == "g-1" and len(rows2) == 1
              and news_store.index_readonly(db_path, offset=9, limit=2) == [])
        rows = news_store.index_readonly(db_path, category="space", offset=0, limit=8)
        check("category filters the crawl page", [r["guid"] for r in rows] == ["g-3"])

        gcfg = {"enabled": True, "feeds": feeds}
        t = {s["name"]: fn for s, fn in news.tools(gcfg, {"news_db_path": db_path})}
        got = t["news_headlines"]({"limit": 5})
        check("news_headlines renders prose, newest first",
              got.startswith("Recent headlines:") and "Gamma launch" in got.split("\n")[1])
        page = json.loads(t["news_index"]({"offset": 0, "limit": 8}))
        check("news_index is a §4 lister (id=guid, oldest first)",
              [i["id"] for i in page] == ["g-1", "g-2", "g-3"]
              and all("title" in i and "published_at" in i for i in page))


# ── weather ──────────────────────────────────────────────────────────────────

def test_weather():
    geocode = {"results": [{"latitude": -42.9, "longitude": 147.3,
                            "name": "Hobart", "country": "Australia"}]}
    forecast = {"current": {"temperature_2m": 11.2, "apparent_temperature": 8.9,
                            "weather_code": 61, "wind_speed_10m": 22.0,
                            "relative_humidity_2m": 78},
                "daily": {"temperature_2m_max": [12.0, 14.5],
                          "temperature_2m_min": [6.0, 7.5],
                          "precipitation_probability_max": [80, 20],
                          "weather_code": [61, 2]}}

    def fetch(url, **k):
        return 200, json.dumps(geocode if "geocoding" in url else forecast).encode()
    weather._geocache.clear()
    got = weather.forecast_prose(fetch, "Hobart")
    check("weather renders now + today + tomorrow",
          "Hobart, Australia" in got and "light rain" in got
          and "feels like 9°" in got and "Tomorrow: partly cloudy" in got)

    def fetch_none(url, **k):
        return 200, json.dumps({"results": []} if "geocoding" in url else forecast).encode()
    weather._geocache.clear()
    t = {s["name"]: fn for s, fn in weather.tools({"enabled": True, "location": ""},
                                                  {"fetch": fetch_none})}
    bad = t["weather"]({})
    check("no default location → a config pointer, not a guess",
          isinstance(bad, dict) and not bad["ok"] and "no location" in bad["error"])
    try:
        weather.forecast_prose(fetch_none, "Atlantis")
        check("an unknown place raises", False)
    except RuntimeError as e:
        check("an unknown place raises with the place named", "Atlantis" in str(e))
    weather._geocache.clear()


# ── research ─────────────────────────────────────────────────────────────────

def test_research():
    canned = {
        "europepmc": {"resultList": {"result": [
            {"title": "Sleep and memory", "authorString": "Smith J.",
             "journalTitle": "Sleep", "pubYear": "2024",
             "abstractText": "We show consolidation."}]}},
        "openalex": {"results": [
            {"display_name": "Attention is all you need",
             "publication_year": 2017, "cited_by_count": 100000,
             "authorships": [{"author": {"display_name": "Vaswani"}}],
             "abstract_inverted_index": {"Attention": [0], "wins": [1]}}]},
        "wiki_summary": {"title": "Earth", "description": "third planet",
                         "extract": "Earth is the third planet from the Sun."},
        "stackex": {"items": [{"title": "How to peel garlic fast?", "score": 12,
                               "is_answered": True, "link": "https://c.se/q/1"}]},
        "hn": {"hits": [{"title": "Show HN: tiny CalDAV", "points": 321,
                         "num_comments": 88, "url": "http://x"}]},
        "gdelt": {"articles": [{"title": "Flooding in the valley",
                                "domain": "example.org", "seendate": "20260820T101500Z"}]},
        "openlibrary": {"docs": [{"title": "The Left Hand of Darkness",
                                  "author_name": ["Ursula K. Le Guin"],
                                  "first_publish_year": 1969}]},
        "archive": {"response": {"docs": [{"identifier": "nightfilm00", "title":
                                           "Night Film", "mediatype": "texts",
                                           "year": "2013"}]}},
        "wayback": {"archived_snapshots": {"closest": {
            "url": "http://web.archive.org/web/2019/http://x", "timestamp": "20190101000000"}}},
    }

    def fetch(url, **k):
        if "ebi.ac.uk" in url:
            return 200, json.dumps(canned["europepmc"]).encode()
        if "openalex" in url:
            return 200, json.dumps(canned["openalex"]).encode()
        if "wikipedia.org/api/rest_v1/page/summary" in url:
            return 200, json.dumps(canned["wiki_summary"]).encode()
        if "stackexchange" in url:                      # SE gzips unconditionally
            return 200, gzip.compress(json.dumps(canned["stackex"]).encode())
        if "hn.algolia" in url:
            return 200, json.dumps(canned["hn"]).encode()
        if "gdeltproject" in url:
            return 200, json.dumps(canned["gdelt"]).encode()
        if "openlibrary" in url:
            return 200, json.dumps(canned["openlibrary"]).encode()
        if "advancedsearch" in url:
            return 200, json.dumps(canned["archive"]).encode()
        if "wayback/available" in url:
            return 200, json.dumps(canned["wayback"]).encode()
        return 404, b""

    got = research.literature_search(fetch, {"query": "sleep"})
    check("Europe PMC renders title + cite + abstract",
          "Sleep and memory" in got and "We show consolidation." in got)
    got = research.scholar_search(fetch, {"query": "attention"})
    check("OpenAlex re-inverts the abstract index",
          "Attention wins" in got and "cited 100000×" in got)
    got = research.reference_lookup(fetch, {"query": "Earth"})
    check("reference_lookup gives the sourced summary",
          got.startswith("Earth (third planet)") and "third planet from the Sun" in got)
    got = research.qa_search(fetch, {"query": "garlic", "site": "cooking"})
    check("Stack Exchange survives its unconditional gzip",
          "peel garlic" in got and "answered" in got)
    got = research.hn_search(fetch, {"query": "caldav"})
    check("Hacker News lists points + comments", "321 points" in got)
    got = research.books_search(fetch, {"query": "left hand"})
    check("Open Library names author and year", "Le Guin" in got and "1969" in got)
    got = research.archive_search(fetch, {"query": "night film", "mediatype": "texts"})
    check("Internet Archive links the item", "archive.org/details/nightfilm00" in got)
    got = research.wayback_lookup(fetch, {"url": "http://x", "date": "2019-01-01"})
    check("wayback returns the snapshot URL", "web.archive.org" in got)

    research._gdelt_last[0] = 0.0
    got = research.events_search(fetch, {"query": "flood"}, now=lambda: 1000.0)
    got2 = research.events_search(fetch, {"query": "flood"}, now=lambda: 1001.0)
    check("GDELT: first call answers, an immediate second is rate-limited",
          "Flooding in the valley" in got and got2 == "rate-limited, try again in a few seconds")
    research._gdelt_last[0] = 0.0


# ── mail (fake IMAP) ─────────────────────────────────────────────────────────

_MAILS = {
    1: b"From: anna@example.com\r\nSubject: Lunch?\r\nDate: Mon, 01 Jun 2026 10:00:00 +0000\r\n\r\nSoup at noon?\r\n",
    2: (b"From: bank@example.com\r\nSubject: =?utf-8?q?Statement_f=C3=BCr_June?=\r\n"
        b"Date: Tue, 02 Jun 2026 10:00:00 +0000\r\nContent-Type: text/html\r\n\r\n"
        b"<p>Your <b>statement</b> is ready.</p>\r\n"),
    3: b"From: club@example.com\r\nSubject: Meeting\r\nDate: Wed, 03 Jun 2026 10:00:00 +0000\r\n\r\nThursday 7pm.\r\n",
}


class FakeIMAP:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.selected = None
        self.readonly_flags = []

    def login(self, user, pw):
        if pw != "app-pass":
            raise RuntimeError("AUTHENTICATIONFAILED")
        return "OK", [b"ok"]

    def list(self):
        return "OK", [b'(\\HasNoChildren) "." "INBOX"',
                      b'(\\HasNoChildren) "." "INBOX.Sent"']

    def select(self, mailbox, readonly=False):
        self.readonly_flags.append(readonly)
        name = mailbox.strip('"')
        if name not in ("INBOX", "INBOX.Sent"):
            return "NO", [None]
        self.selected = name
        return "OK", [b"3" if name == "INBOX" else b"0"]

    def uid(self, cmd, first, *criteria):
        # imaplib's uid(command, *args): search gets (charset, *criteria),
        # fetch gets (uid, message_parts).
        if cmd == "search":
            if self.selected != "INBOX":
                return "OK", [b""]
            if "TEXT" in criteria:
                q = criteria[-1].strip('"').lower()
                hits = [str(u) for u, raw in _MAILS.items() if q in raw.decode().lower()]
                return "OK", [" ".join(hits).encode()]
            return "OK", [b"3 1 2"]              # deliberately unsorted
        if cmd == "fetch":
            uid, spec = int(first), criteria[0]
            raw = _MAILS.get(uid)
            if raw is None:
                return "OK", [None]
            if "HEADER.FIELDS" in spec:
                head = raw.split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
                return "OK", [(b"x", head), b")"]
            return "OK", [(b"x", raw), b")"]
        return "NO", [None]

    def logout(self):
        return "BYE", [b""]


def test_mail():
    gcfg = {"enabled": True, "accounts": [{
        "label": "personal", "host": "imap.example.com", "port": 993,
        "user": "me@example.com", "password": "app-pass"}]}
    env = {"imap_factory": FakeIMAP}
    t = {s["name"]: fn for s, fn in mail.tools(gcfg, env)}

    page = json.loads(t["mail_list"]({"folder": "inbox", "offset": 0, "limit": 2}))
    check("mail_list pages oldest-first with label|folder|uid ids",
          [i["id"] for i in page] == ["personal|INBOX|1", "personal|INBOX|2"]
          and page[0]["subject"] == "Lunch?")
    check("MIME-encoded headers decode", "für June" in page[1]["subject"])
    page2 = json.loads(t["mail_list"]({"folder": "inbox", "offset": 2, "limit": 2}))
    check('mail_list honours offset and returns "[]" past the end',
          len(page2) == 1 and
          t["mail_list"]({"folder": "inbox", "offset": 9, "limit": 2}) == "[]")

    got = t["mail_read"]({"id": "personal|INBOX|1"})
    check("mail_read returns headers + body", "Soup at noon?" in got
          and got.startswith("From: anna@example.com"))
    got = t["mail_read"]({"id": "personal|INBOX|2"})
    check("an html-only mail reads as stripped text",
          "Your statement is ready." in got and "<p>" not in got)
    bad = t["mail_read"]({"id": "nonsense"})
    check("a malformed id is refused with the format named",
          isinstance(bad, dict) and "label|folder|uid" in bad["error"])

    got = t["mail_recent"]({"limit": 2})
    check("mail_recent shows the newest first", got.index("Meeting") <
          got.index("Statement"))
    got = t["mail_search"]({"query": "soup"})
    check("mail_search finds by body text", "Lunch?" in got)

    conn_probe = []
    class SpyIMAP(FakeIMAP):
        def __init__(self, host, port):
            super().__init__(host, port)
            conn_probe.append(self)
    t2 = {s["name"]: fn for s, fn in mail.tools(gcfg, {"imap_factory": SpyIMAP})}
    t2["mail_list"]({"folder": "inbox"})
    check("every mailbox opens READ-ONLY (EXAMINE — reads can't mark seen)",
          conn_probe and all(all(f is True for f in c.readonly_flags)
                             for c in conn_probe))

    got = mail.probe(gcfg, env)
    check("mail probe signs in and counts the inbox",
          got["ok"] and "INBOX holds 3" in got["detail"])
    bad_cfg = {"enabled": True, "accounts": [{**gcfg["accounts"][0], "password": "wrong"}]}
    got = mail.probe(bad_cfg, env)
    check("a wrong password fails the probe loudly",
          not got["ok"] and "FAILED" in got["detail"])


# ── calendar (fake CalDAV) ───────────────────────────────────────────────────

def _ics(uid, title, start, end, cal_extra=""):
    return ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n"
            f"UID:{uid}\r\nDTSTART:{start}\r\nDTEND:{end}\r\n"
            f"SUMMARY:{title}\r\n{cal_extra}END:VEVENT\r\nEND:VCALENDAR\r\n")


class FakeCalDav:
    """Just enough server: discovery PROPFINDs, per-collection REPORT,
    GET/PUT/DELETE on event objects."""

    def __init__(self):
        self.store = {
            "/cal/work/dentist.ics": _ics(
                "w1", "Dentist", "20260825T050000Z", "20260825T060000Z"),
            "/cal/work/standup.ics": _ics(
                "w2", "Standup", "20260824T000000Z", "20260824T003000Z",
                "RRULE:FREQ=DAILY;COUNT=5\r\nEXDATE:20260826T000000Z\r\n"),
            "/cal/vinkona/haircut.ics": _ics(
                "v1", "Haircut", "20260827T040000Z", "20260827T050000Z"),
        }
        self.deleted = []

    def __call__(self, url, *, method="GET", data=None, headers=None, **k):
        path = url.split("://", 1)[-1].split("/", 1)
        path = "/" + (path[1] if len(path) > 1 else "")
        body = (data or b"").decode() if isinstance(data, bytes) else (data or "")
        D = 'xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav"'
        if method == "PROPFIND":
            if "current-user-principal" in body:
                return 207, (f'<D:multistatus {D}><D:response><D:href>{path}</D:href>'
                             "<D:propstat><D:prop><D:current-user-principal>"
                             "<D:href>/principals/me/</D:href>"
                             "</D:current-user-principal></D:prop></D:propstat>"
                             "</D:response></D:multistatus>").encode()
            if "calendar-home-set" in body:
                return 207, (f'<D:multistatus {D}><D:response><D:href>{path}</D:href>'
                             "<D:propstat><D:prop><C:calendar-home-set>"
                             "<D:href>/cal/</D:href></C:calendar-home-set>"
                             "</D:prop></D:propstat></D:response></D:multistatus>").encode()
            resp = []
            for href, name in (("/cal/", None), ("/cal/work/", "Work"),
                               ("/cal/vinkona/", "Vinkona")):
                rt = "<D:collection/>" + ("<C:calendar/>" if name else "")
                dn = f"<D:displayname>{name}</D:displayname>" if name else ""
                resp.append(f"<D:response><D:href>{href}</D:href><D:propstat><D:prop>"
                            f"<D:resourcetype>{rt}</D:resourcetype>{dn}"
                            "</D:prop></D:propstat></D:response>")
            return 207, (f'<D:multistatus {D}>' + "".join(resp) + "</D:multistatus>").encode()
        if method == "REPORT":
            resp = []
            for href, ics in self.store.items():
                if href.startswith(path):
                    esc = ics.replace("&", "&amp;").replace("<", "&lt;")
                    resp.append(f"<D:response><D:href>{href}</D:href><D:propstat><D:prop>"
                                f"<C:calendar-data>{esc}</C:calendar-data>"
                                "</D:prop></D:propstat></D:response>")
            return 207, (f'<D:multistatus {D}>' + "".join(resp) + "</D:multistatus>").encode()
        if method == "GET":
            ics = self.store.get(path)
            return (200, ics.encode()) if ics else (404, b"")
        if method == "PUT":
            self.store[path] = body
            return 201, b""
        if method == "DELETE":
            if path in self.store:
                del self.store[path]
                self.deleted.append(path)
                return 204, b""
            return 404, b""
        return 405, b""


def test_caldav():
    server = FakeCalDav()
    gcfg = {"enabled": True, "caldav_url": "https://cal.example.com/",
            "user": "me", "password": "app-pass", "vinkona_calendar": "Vinkona"}
    env = {"fetch": server, "now": lambda: 1787000000.0, "user_tz": "UTC",
           "uuid": lambda: "fixed123"}
    t = {s["name"]: fn for s, fn in caldav.tools(gcfg, env)}
    win = {"start": "2026-08-24", "end": "2026-08-31"}

    evs = json.loads(t["calendar_range_json"](win))
    titles = [e["title"] for e in evs]
    check("range_json spans ALL calendars with the sync fields",
          "Dentist" in titles and "Haircut" in titles
          and all(k in evs[0] for k in ("id", "calendar", "title", "start", "end")))
    standups = [e for e in evs if e["title"] == "Standup"]
    check("an RRULE expands to occurrences minus EXDATE (5 - 1 = 4 in window)",
          len(standups) == 4)
    prose = t["calendar_range"](win)
    check("the prose view names calendar and time",
          "[Work] Dentist" in prose and "[Vinkona] Haircut" in prose)

    got = json.loads(t["calendar_create"]({
        "title": "Massage", "start": "2026-08-25T05:30", "end": "2026-08-25T06:30"}))
    check("a clashing create is refused with the clash named",
          got["created"] is False and any("Dentist" in c for c in got["conflicts"]))
    got = json.loads(t["calendar_create"]({
        "title": "Massage", "start": "2026-08-25T05:30", "end": "2026-08-25T06:30",
        "force": True}))
    check("force creates despite the clash, verified by server read-back",
          got["created"] is True and got["verified"] is True
          and got["id"].startswith("/cal/vinkona/"))
    check("the new event went to the Vinkona collection ONLY",
          "/cal/vinkona/vinkona-fixed123.ics" in server.store)

    free = json.loads(t["calendar_create"]({
        "title": "Quiet hour", "start": "2026-08-28T09:00"}))
    check("a clash-free create books with the default hour",
          free["created"] is True and "09:00" in free["when"])

    bad = t["calendar_update"]({"id": "/cal/work/dentist.ics", "title": "HAHA"})
    check("updating another calendar's event is REFUSED (write containment)",
          isinstance(bad, dict) and not bad["ok"] and "only ever changes" in bad["error"])
    bad = t["calendar_delete"]({"id": "/cal/work/dentist.ics"})
    check("deleting another calendar's event is REFUSED",
          isinstance(bad, dict) and not bad["ok"] and server.deleted == [])

    got = json.loads(t["calendar_update"]({"id": "/cal/vinkona/haircut.ics",
                                           "title": "Haircut (moved)",
                                           "start": "2026-08-27T06:00"}))
    back = caldav.parse_vevents(server.store["/cal/vinkona/haircut.ics"],
                                datetime.timezone.utc)[0]
    check("updating her own event rewrites it in place, uid kept",
          got["updated"] and back["title"] == "Haircut (moved)" and back["uid"] == "v1"
          and back["start"].hour == 6 and back["end"].hour == 7)
    got = json.loads(t["calendar_delete"]({"id": "/cal/vinkona/haircut.ics"}))
    check("deleting her own event works", got["deleted"]
          and server.deleted == ["/cal/vinkona/haircut.ics"])

    gone = dict(gcfg, vinkona_calendar="Nonexistent")
    t2 = {s["name"]: fn for s, fn in caldav.tools(gone, env)}
    bad = t2["calendar_create"]({"title": "X", "start": "2026-08-28T09:00"})
    check("no Vinkona calendar in the account → create refuses, tells the user",
          isinstance(bad, dict) and not bad["ok"] and "create one there first" in bad["error"])

    got = caldav.probe(gcfg, env)
    check("calendar probe lists calendars and confirms the writable one",
          got["ok"] and "Work" in got["detail"] and "can be written" in got["detail"])
    got = caldav.probe(gone, env)
    check("…and warns when the writable calendar is missing",
          got["ok"] and "create one in the" in got["detail"])


# ── egress managed block ─────────────────────────────────────────────────────

def test_egress_sync():
    cfg = {"tools": {"local": {
        "enabled": True,
        "research": {"enabled": True},
        "weather": {"enabled": True},
        "news": {"enabled": True, "feeds": [
            {"url": "https://feeds.bbci.co.uk/news/rss.xml"},
            {"url": "http://example.org:8080/feed"}]},
        "calendar": {"enabled": True, "caldav_url": "https://caldav.icloud.com/"},
        "mail": {"enabled": True, "accounts": [
            {"host": "imap.example.com", "port": 993, "user": "x"}]},
    }}}
    block = egress_sync.render(cfg)
    check("research rule carries the keyless API hosts",
          '"api.openalex.org"' in block and '"www.ebi.ac.uk"' in block)
    check("feed hosts are derived from the configured urls, split by port",
          '"feeds.bbci.co.uk"' in block and "local-news-8080" in block)
    check("caldav rule allows the DAV verbs on the configured host only",
          '"caldav.icloud.com"' in block and '"PROPFIND"' in block)
    check("mail is documented as the one un-brokered lane",
          "imap.example.com:993" in block and "not HTTP" in block)

    off = egress_sync.render({"tools": {"local": {"enabled": False,
                                                  "research": {"enabled": True}}}})
    check("master off → nothing granted", "local-research" not in off
          and "nothing granted" in off)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "egress.toml"
        p.write_text('[[rule]]\nname = "hand-made"\nhosts = ["x.example"]\nport = 443\n')
        changed = egress_sync.ensure(cfg, p)
        text = p.read_text()
        check("ensure appends the managed block, hand rules intact",
              changed and "hand-made" in text and egress_sync.BEGIN in text)
        check("ensure is idempotent", egress_sync.ensure(cfg, p) is False)
        cfg2 = json.loads(json.dumps(cfg))
        cfg2["tools"]["local"]["research"]["enabled"] = False
        egress_sync.ensure(cfg2, p)
        text2 = p.read_text()
        check("disabling a genre withdraws its rule but keeps hand rules",
              "local-research" not in text2 and "hand-made" in text2)
        import tomllib
        tomllib.loads(text2.replace(egress_sync.BEGIN, "").replace(egress_sync.END, ""))
        check("the result stays valid TOML", True)


# ── the assembled host, end to end ───────────────────────────────────────────

def test_assembled_host():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "docs"
        root.mkdir()
        (root / "a.txt").write_text("alpha")
        cfg = _cfg(files={"enabled": True, "roots": [str(root)]},
                   mail={"enabled": True, "accounts": [{
                       "label": "p", "host": "h", "port": 993,
                       "user": "u", "password": "app-pass"}]})
        h = local_tools.LocalHost(cfg, env={"fetch": _no_net,
                                            "imap_factory": FakeIMAP})
        cat = asyncio.run(h.catalogue())
        names = sorted(t["function"]["name"] for t in cat)
        check("the assembled catalogue is the union of enabled genres",
              names == ["file_index", "file_list", "file_read", "file_search",
                        "mail_list", "mail_read", "mail_recent", "mail_search"])
        got = asyncio.run(h.call_raw("file_read", {"path": str(root / "a.txt")}))
        check("call_raw returns the ToolHost envelope on success",
              got == {"ok": True, "result": "alpha", "error": ""})
        got = asyncio.run(h.call_raw("file_read", {"path": "/etc/passwd"}))
        check("…and a structured refusal on containment",
              not got["ok"] and "not inside" in got["error"])
        got = asyncio.run(h.call("mail_recent", {}))
        check("call() flattens errors into model-readable text",
              isinstance(got, str))


def main():
    test_host_shape()
    test_files()
    test_news()
    test_weather()
    test_research()
    test_mail()
    test_caldav()
    test_egress_sync()
    test_assembled_host()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
