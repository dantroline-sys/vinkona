"""
News/RSS headline store — a durable, queryable archive of the headlines the research worker crawls.

Distinct from `ambient` (a transient replace-on-refresh snapshot that keeps the *current* top
headlines in the prompt) and from `memories` (LM-curated durable learnings): this is a plain
append-only log of headlines, deduped by guid/link and timestamped, that Vinkona can QUERY as a tool
— by keyword/topic, source, and date — to answer "what's the latest on X", "any news from the BBC
yesterday", and to build a historical narrative of headlines over time.

Trust: headlines are attacker-controlled (hostile-by-default, see safety/§8) — titles and summaries
are sanitised on the way IN, and the tool that surfaces them fences them as untrusted data, never
instructions.
"""

import hashlib
import time
import typing as tp
from datetime import datetime

try:                                    # untrusted-content defense (prompt injection)
    from safety import sanitize_external
except Exception:                       # importlib-loaded context without cwd on sys.path
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _spec = _ilu.spec_from_file_location("safety", _Path(__file__).resolve().parent / "safety.py")
    _safety = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_safety)
    sanitize_external = _safety.sanitize_external


def to_epoch(s: tp.Any) -> tp.Optional[float]:
    """Parse an ISO-8601 / RFC-ish timestamp into a unix time; None if unparseable."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _field(d: dict, *keys: str) -> str:
    """First present, non-empty string among the given synonym keys."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "") and not isinstance(v, (dict, list)):
            return str(v).strip()
    return ""


def normalize_item(it: dict) -> tp.Optional[dict]:
    """Coerce one raw feed item (news_index / rss-style JSON) into our columns, tolerating the
    synonym keys different feeds emit.  Returns None if there's no title to key on."""
    if not isinstance(it, dict):
        return None
    title = _field(it, "title", "headline", "name")
    if not title:
        return None
    link = _field(it, "url", "link", "href")
    source = _field(it, "source", "feed", "outlet", "publisher", "site")
    category = _field(it, "category", "cat", "topic")
    summary = _field(it, "summary", "description", "content", "snippet", "text")
    published = to_epoch(it.get("published") or it.get("pubDate") or it.get("date")
                         or it.get("published_at") or it.get("updated"))
    guid = _field(it, "id", "guid") or link or ("h:" + hashlib.sha1(
        f"{source}|{title}".encode("utf-8")).hexdigest())
    return {"guid": guid, "title": title, "summary": summary, "source": source,
            "category": category, "link": link, "published_at": published}


class NewsStore:
    def __init__(self, db):
        self.db = db
        self._migrate()

    def _migrate(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS headlines (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              guid TEXT UNIQUE,
              title TEXT NOT NULL,
              summary TEXT DEFAULT '',
              source TEXT DEFAULT '',
              category TEXT DEFAULT '',
              link TEXT DEFAULT '',
              published_at REAL,
              fetched_at REAL NOT NULL)""")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(headlines)")}
        if "category" not in cols:                       # migrate an older archive in place
            self.db.execute("ALTER TABLE headlines ADD COLUMN category TEXT DEFAULT ''")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_headlines_fetched ON headlines(fetched_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_headlines_source ON headlines(source)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_headlines_published ON headlines(published_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_headlines_cat ON headlines(category, published_at)")
        self.db.commit()

    def ingest(self, items: tp.Iterable[dict], now: float | None = None) -> int:
        """Append NEW headlines (dedup by guid — INSERT OR IGNORE), sanitising untrusted text on
        the way in.  Returns how many were newly stored (repeats across polls are ignored)."""
        now = now or time.time()
        new = 0
        for raw in (items or []):
            it = normalize_item(raw)
            if not it:
                continue
            title = sanitize_external(it["title"], 300)
            if not title:
                continue
            summary = sanitize_external(it["summary"], 1000)
            source = sanitize_external(it["source"], 120)
            category = sanitize_external(it.get("category", ""), 60)
            cur = self.db.execute(
                "INSERT OR IGNORE INTO headlines"
                "(guid,title,summary,source,category,link,published_at,fetched_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (it["guid"], title, summary, source, category, it["link"], it["published_at"], now))
            new += cur.rowcount or 0
        self.db.commit()
        return new

    def search(self, query: str | None = None, source: str | None = None,
               category: str | None = None, since: float | None = None,
               until: float | None = None, limit: int = 20) -> list[dict]:
        """Query the archive by keyword/topic (AND over title+summary), source, category, and a
        date window (on the article's own time when known, else capture time).  Newest first."""
        where, params = [], []
        for term in (query or "").split():
            like = f"%{term}%"
            where.append("(title LIKE ? OR summary LIKE ?)")
            params += [like, like]
        if source:
            where.append("source LIKE ?"); params.append(f"%{source}%")
        if category:
            where.append("category = ?"); params.append(category)
        # a single "when" expression: prefer the article's published time, fall back to capture time
        when = "COALESCE(published_at, fetched_at)"
        if since is not None:
            where.append(f"{when} >= ?"); params.append(float(since))
        if until is not None:
            where.append(f"{when} <= ?"); params.append(float(until))
        sql = "SELECT * FROM headlines"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {when} DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    def between(self, since: float, until: float, limit: int = 500) -> list[dict]:
        """All headlines whose time falls in [since, until) — for the daily digest."""
        return self.search(since=since, until=until, limit=limit)

    def sources(self) -> list[dict]:
        """Distinct feeds with a count each (so 'query by news source' can enumerate them)."""
        rows = self.db.execute(
            "SELECT source, COUNT(*) n FROM headlines WHERE source<>'' "
            "GROUP BY source ORDER BY n DESC").fetchall()
        return [{"source": r["source"], "count": r["n"]} for r in rows]

    def categories(self) -> list[dict]:
        """Distinct categories with a count each."""
        rows = self.db.execute(
            "SELECT category, COUNT(*) n FROM headlines WHERE category<>'' "
            "GROUP BY category ORDER BY n DESC").fetchall()
        return [{"category": r["category"], "count": r["n"]} for r in rows]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM headlines").fetchone()["c"]

    def prune(self, max_age_days: float, category: str | None = None) -> int:
        """Bound the archive by age (0/None = keep the full history).  Optionally scope to one
        category, so different categories can have different retention (§9: keep medical-research
        indefinitely, prune general after N months)."""
        if not max_age_days or max_age_days <= 0:
            return 0
        cutoff = time.time() - float(max_age_days) * 86400
        sql = "DELETE FROM headlines WHERE COALESCE(published_at, fetched_at) < ?"
        params = [cutoff]
        if category:
            sql += " AND category = ?"; params.append(category)
        cur = self.db.execute(sql, params)
        self.db.commit()
        return cur.rowcount or 0


def _when_label(row: dict) -> str:
    ts = row.get("published_at") or row.get("fetched_at")
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def render(rows: list[dict], max_chars: int = 2000) -> str:
    """Format matched headlines into a compact, fenced-friendly block for a tool result."""
    lines = []
    for r in rows:
        when = _when_label(r)
        tag = " · ".join(x for x in (r.get("source"), r.get("category")) if x)
        src = f" [{tag}]" if tag else ""
        head = f"• {when}{src} {r.get('title', '')}".strip()
        summ = (r.get("summary") or "").strip()
        lines.append(head + (f"\n    {summ[:200]}" if summ else ""))
    return "\n".join(lines)[:max_chars].strip()
