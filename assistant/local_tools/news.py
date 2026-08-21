"""Local news genre — her own RSS poller + news_headlines / news_index.

The Mac host's news pipeline, relocated: a poller fetches the feeds the USER
listed (tools.local.news.feeds — url / source / category), parses them with
the same stdlib RSS+Atom approach as palette.rss_fetch, and appends into the
durable NewsStore archive (guid-deduped, so re-polls and even a Mac host
feeding the same store are loop-safe).  The two tools then serve straight
from that archive on their own read-only connections (thread-safe from any
loop): news_headlines is the prose "what's in the news" for the live model,
news_index the §4 CRAWL lister the event-memory ingestion walks by offset.
"""
import datetime
import json
import re
import xml.etree.ElementTree as ET

_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    import html
    return html.unescape(_TAG.sub(" ", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>",
                                              " ", s or ""))).strip()


def parse_feed(raw, *, source: str = "", category: str = "") -> list:
    """RSS 2.0 / Atom bytes-or-text → NewsStore-shaped items (normalize_item
    finishes the job on ingest).  A feed that does not parse raises — the
    poller records it against that feed and moves on."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    root = ET.fromstring(raw)
    items = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        d = {"title": "", "link": "", "summary": "", "published": "",
             "source": source, "category": category, "guid": ""}
        for c in node:
            tag = c.tag.rsplit("}", 1)[-1]
            if tag == "title":
                d["title"] = (c.text or "").strip()
            elif tag == "link":
                d["link"] = (c.text or c.get("href") or "").strip()
            elif tag in ("description", "summary", "content"):
                d["summary"] = d["summary"] or _strip_html(c.text or "")[:1000]
            elif tag in ("pubDate", "published", "updated"):
                d["published"] = d["published"] or (c.text or "").strip()
            elif tag in ("guid", "id"):
                d["guid"] = (c.text or "").strip()
        if d["title"]:
            items.append(d)
    return items


def poll_feeds(fetch, feeds: list, *, per_feed_limit: int = 30) -> tuple:
    """Fetch every configured feed through the broker; returns (items, errors)
    where errors is [(url, reason)] — one dead feed never starves the rest."""
    items, errors = [], []
    for feed in feeds or []:
        url = str(feed.get("url") or "").strip()
        if not url:
            continue
        try:
            status, body = fetch(url, timeout=20.0, purpose="local_news")
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            got = parse_feed(body, source=str(feed.get("source") or ""),
                             category=str(feed.get("category") or ""))
            items.extend(got[: max(1, int(per_feed_limit))])
        except Exception as e:
            errors.append((url, str(e)))
    return items, errors


def _iso(ts) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def tools(gcfg: dict, env: dict) -> list:
    db_path = env.get("news_db_path") or ""

    def _unconfigured():
        if not (gcfg.get("feeds") or []):
            return {"ok": False, "error": "no news feeds are configured yet — add "
                    "RSS/Atom feed addresses under Settings → Local tools → News"}
        return None

    def news_headlines(args):
        import news_store
        bad = _unconfigured()
        if bad and not db_path:
            return bad
        limit = max(1, min(int(args.get("limit") or 8), 30))
        rows = news_store.index_readonly(
            db_path, category=str(args.get("category") or "") or None,
            source=str(args.get("source") or "") or None,
            offset=0, limit=limit, newest_first=True)
        if not rows:
            return bad or "No stored headlines yet — the next feed poll fills the archive."
        lines = ["Recent headlines:"]
        for r in rows:
            src = f" — {r['source']}" if r.get("source") else ""
            when = _iso(r.get("ts"))
            summary = (r.get("summary") or "").strip()
            summary = f" {summary[:180]}" if summary else ""
            lines.append(f"- {r['title']}{src} ({when}){summary}")
        return "\n".join(lines)

    def news_index(args):
        import news_store
        offset = max(0, int(args.get("offset") or 0))
        limit = max(1, min(int(args.get("limit") or 8), 100))
        rows = news_store.index_readonly(
            db_path, category=str(args.get("category") or "") or None,
            source=str(args.get("source") or "") or None,
            offset=offset, limit=limit, newest_first=False)
        return json.dumps([{
            "id": r["guid"], "title": r["title"], "summary": r.get("summary", ""),
            "source": r.get("source", ""), "category": r.get("category", ""),
            "link": r.get("link", ""), "published_at": _iso(r.get("ts")),
        } for r in rows])

    return [
        ({"name": "news_headlines",
          "description": "Recent headlines from the news feeds the user follows "
                         "(the local archive). Optionally filter by source or category.",
          "parameters": {"type": "object", "properties": {
              "source": {"type": "string"}, "category": {"type": "string"},
              "limit": {"type": "integer", "default": 8}}}},
         news_headlines),
        ({"name": "news_index",
          "description": "CRAWL lister for the news archive: a page of stored "
                         "headlines as JSON, oldest first, for offset paging.",
          "parameters": {"type": "object", "properties": {
              "category": {"type": "string"}, "source": {"type": "string"},
              "offset": {"type": "integer", "default": 0},
              "limit": {"type": "integer", "default": 8}}}},
         news_index),
    ]


def probe(gcfg: dict, env: dict) -> dict:
    feeds = [f for f in (gcfg.get("feeds") or []) if str(f.get("url") or "").strip()]
    if not feeds:
        return {"ok": False, "detail": "no feeds configured yet — add at least one "
                "RSS/Atom address (url, plus an optional source name and category)"}
    items, errors = poll_feeds(env["fetch"], feeds[:3],
                               per_feed_limit=int(gcfg.get("per_feed_limit", 30)))
    parts = [f"{len(items)} headline(s) from {min(len(feeds), 3)} feed(s)"]
    if len(feeds) > 3:
        parts.append(f"(first 3 of {len(feeds)} tested)")
    for url, why in errors:
        parts.append(f"FAILED {url}: {why}")
    return {"ok": bool(items) or not errors, "detail": " ".join(parts)}
