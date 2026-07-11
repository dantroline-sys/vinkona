"""Tests for news_store.NewsStore — the durable, queryable RSS/headline archive (dedup, keyword/
source/date search, retention) and normalize_item's synonym tolerance."""
import sqlite3
import time

from news_store import NewsStore, normalize_item, render, to_epoch


def _db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return db


# ── ingest / dedup ────────────────────────────────────────────────────────────────

def test_ingest_dedups_by_link():
    ns = NewsStore(_db())
    a = ns.ingest([{"title": "Quake hits coast", "link": "http://x/1", "source": "BBC"}])
    b = ns.ingest([{"title": "Quake hits coast", "link": "http://x/1", "source": "BBC"}])
    assert a == 1 and b == 0 and ns.count() == 1


def test_ingest_hashes_when_no_guid_or_link():
    ns = NewsStore(_db())
    n1 = ns.ingest([{"title": "Budget passes", "source": "ABC"}])
    n2 = ns.ingest([{"title": "Budget passes", "source": "ABC"}])   # same source+title → same hash
    n3 = ns.ingest([{"title": "Budget passes", "source": "SBS"}])   # different source → new
    assert (n1, n2, n3) == (1, 0, 1)


def test_ingest_skips_titleless_items():
    ns = NewsStore(_db())
    assert ns.ingest([{"link": "http://z"}, {"title": "", "link": "q"}, "junk"]) == 0


def test_ingest_sanitizes_and_caps_long_title():
    ns = NewsStore(_db())
    ns.ingest([{"title": "Breaking " * 60, "link": "h"}])           # > 300 chars
    assert ns.search()[0]["title"].endswith("…(truncated)")         # the cap fired


# ── normalize_item ────────────────────────────────────────────────────────────────

def test_normalize_item_synonyms_and_published():
    it = normalize_item({"headline": "Rate cut", "url": "http://y", "publisher": "AFR",
                         "description": "RBA moves", "pubDate": "2026-07-05T09:00:00"})
    assert it["title"] == "Rate cut" and it["link"] == "http://y" and it["source"] == "AFR"
    assert it["summary"] == "RBA moves" and it["published_at"] == to_epoch("2026-07-05T09:00:00")


def test_normalize_item_needs_title():
    assert normalize_item({"link": "http://z"}) is None
    assert normalize_item("not a dict") is None


def test_normalize_item_news_index_schema():
    # The §9 news_index item shape: id (dedup key), source, category, title, url, published, summary.
    it = normalize_item({"id": "guid-1", "source": "NEJM", "category": "medical-research",
                         "title": "New trial", "url": "https://x", "summary": "abstract",
                         "published": "2026-07-05T04:12:00+00:00"})
    assert it["guid"] == "guid-1" and it["category"] == "medical-research"
    assert it["source"] == "NEJM" and it["link"] == "https://x"
    assert it["published_at"] == to_epoch("2026-07-05T04:12:00+00:00")


# ── category axis ─────────────────────────────────────────────────────────────────

def test_search_and_prune_by_category():
    ns = NewsStore(_db())
    old = time.time() - 200 * 86400
    ns.ingest([{"id": "g1", "title": "general politics", "category": "general"}], now=old)
    ns.ingest([{"id": "m1", "title": "clinical trial result", "category": "medical-research"}], now=old)
    ns.ingest([{"id": "g2", "title": "general sport tonight", "category": "general"}])
    assert {h["title"] for h in ns.search(category="general")} == {
        "general politics", "general sport tonight"}
    assert {c["category"]: c["count"] for c in ns.categories()} == {"general": 2, "medical-research": 1}
    # prune only the general category older than 180d — keep clinical indefinitely
    assert ns.prune(180, category="general") == 1
    assert ns.count() == 2 and {h["title"] for h in ns.search(category="general")} == {
        "general sport tonight"}
    assert ns.search(category="medical-research")[0]["title"] == "clinical trial result"


def test_migrates_pre_category_archive_in_place():
    db = _db()
    # simulate an older archive created before the category column existed
    db.execute("CREATE TABLE headlines (id INTEGER PRIMARY KEY AUTOINCREMENT, guid TEXT UNIQUE, "
               "title TEXT NOT NULL, summary TEXT DEFAULT '', source TEXT DEFAULT '', "
               "link TEXT DEFAULT '', published_at REAL, fetched_at REAL NOT NULL)")
    db.execute("INSERT INTO headlines(guid,title,fetched_at) VALUES ('old','legacy row',1.0)")
    db.commit()
    ns = NewsStore(db)                                   # _migrate adds the category column
    assert ns.count() == 1                               # existing row preserved
    assert ns.ingest([{"id": "n", "title": "new one", "category": "general"}]) == 1


# ── search: keyword / source / date ───────────────────────────────────────────────

def test_search_keyword_source_and_date():
    ns = NewsStore(_db())
    t0 = 1_700_000_000.0
    ns.ingest([{"title": "Election results in", "source": "BBC", "link": "l1"}], now=t0)
    ns.ingest([{"title": "Sports update tonight", "source": "ESPN", "link": "l2"}], now=t0 + 100)
    ns.ingest([{"title": "Election recount ordered", "source": "CNN", "link": "l3"}], now=t0 + 200)
    assert {h["title"] for h in ns.search(query="election")} == {
        "Election results in", "Election recount ordered"}
    assert [h["source"] for h in ns.search(source="ESPN")] == ["ESPN"]
    assert ns.search()[0]["title"] == "Election recount ordered"     # newest first
    assert len(ns.search(since=t0 + 50)) == 2                        # date window


def test_search_multi_term_is_and():
    ns = NewsStore(_db())
    ns.ingest([{"title": "Reserve Bank cuts rates", "link": "a"},
               {"title": "Reserve players named", "link": "b"}])
    assert [h["title"] for h in ns.search(query="reserve rates")] == ["Reserve Bank cuts rates"]


def test_search_orders_by_published_over_fetched():
    ns = NewsStore(_db())
    cap = 2_000_000_000.0
    ns.ingest([{"title": "Old news late capture", "link": "a", "published": "2020-01-01T00:00:00"}], now=cap)
    ns.ingest([{"title": "Fresh news", "link": "b", "published": "2026-07-05T00:00:00"}], now=cap)
    assert ns.search()[0]["title"] == "Fresh news"


# ── sources / retention / render ──────────────────────────────────────────────────

def test_sources_counts():
    ns = NewsStore(_db())
    ns.ingest([{"title": "a", "link": "1", "source": "BBC"},
               {"title": "b", "link": "2", "source": "BBC"},
               {"title": "c", "link": "3", "source": "CNN"}])
    assert {s["source"]: s["count"] for s in ns.sources()} == {"BBC": 2, "CNN": 1}


def test_prune_by_age_and_zero_keeps_all():
    ns = NewsStore(_db())
    ns.ingest([{"title": "ancient", "link": "o"}], now=time.time() - 40 * 86400)
    ns.ingest([{"title": "recent", "link": "r"}])
    assert ns.prune(30) == 1 and ns.count() == 1 and ns.search()[0]["title"] == "recent"
    assert ns.prune(0) == 0 and ns.count() == 1                      # 0 = keep the full history


def test_between_window():
    ns = NewsStore(_db())
    t0 = 1_700_000_000.0
    for i in range(5):
        ns.ingest([{"title": f"h{i}", "link": f"l{i}"}], now=t0 + i * 3600)
    assert len(ns.between(t0 + 3600, t0 + 3 * 3600)) == 3            # inclusive window


def test_render_block():
    rows = [{"title": "Quake", "source": "BBC", "summary": "big one", "published_at": 1_700_000_000.0}]
    out = render(rows)
    assert "Quake" in out and "[BBC]" in out and "big one" in out


def main():
    passed = failed = 0
    import types
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn(); passed += 1; print(f"  ok  {name}")
            except Exception as e:
                failed += 1; print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
