"""Wikipedia via a **Kiwix ZIM** (`enwikipedia_…zim`) — pre-rendered, cleaned,
sectioned HTML, read with **libzim**.  This sidesteps wikitext entirely; each
article is iterated and split on its ``<h2>/<h3>`` headings by the shared HTML
sectioner.  Monthly refresh = drop in a new ZIM, re-version, swap.

This is an article *iterator* (not the path-crawl extractor), driven directly by
``ingest.py`` because a ZIM is one file holding millions of articles.  The raw
XML-dump route (mwxml + mwparserfromhell) is intentionally not implemented — the
ZIM render is the recommended path; add the dump route only if you need
namespaces/templates the render drops.
"""
from __future__ import annotations

import logging
import threading

from . import MissingDependency
from . import html as html_src

log = logging.getLogger("knowledgehost.sources.wikipedia")

# Cached open Archives (libzim Archive is a read-only mmap — cheap to keep open and shared
# across the threaded server's requests).  One lock guards cache population.
_ARCHIVES: dict = {}
_ARCH_LOCK = threading.Lock()


def _archive(zim_path: str):
    a = _ARCHIVES.get(zim_path)
    if a is None:
        with _ARCH_LOCK:
            a = _ARCHIVES.get(zim_path)
            if a is None:
                from libzim.reader import Archive
                a = Archive(zim_path)
                _ARCHIVES[zim_path] = a
    return a


def has_fulltext(zim_path: str) -> bool:
    """True if this ZIM ships a built-in Xapian full-text index (Kiwix Wikipedia ZIMs do)."""
    try:
        return bool(getattr(_archive(zim_path), "has_fulltext_index", False))
    except Exception as e:
        log.debug("ZIM fulltext check failed (%s)", e)
        return False


def search(zim_path: str, query: str, *, articles: int = 8, chunk_chars: int = 1000,
           max_chunks: int = 120) -> list:
    """Tier-0 + slicing: query the ZIM's OWN Xapian full-text index for the top `articles`
    (broad, free, no ingestion), then SLICE each into section/length chunks and return them
    as passage dicts (id/text/title/section/path_or_url/source_type/score).  The caller
    semantically re-ranks the slices, so the relevant SECTION surfaces rather than just the
    article lead.  Returns [] if libzim is missing, the ZIM has no full-text index, or
    anything goes wrong (the chunk-store arms still answer)."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        from libzim.search import Query, Searcher
        archive = _archive(zim_path)
        if not getattr(archive, "has_fulltext_index", False):
            log.debug("ZIM has no full-text index — skipping the wikipedia arm")
            return []
        srch = Searcher(archive).search(Query().set_query(query))
        if not srch.getEstimatedMatches():
            return []
        paths = list(srch.getResults(0, int(articles)))
    except Exception as e:
        log.warning("ZIM search failed (%s) — skipping the wikipedia arm", e)
        return []

    out = []
    for rank, path in enumerate(paths):
        try:
            entry = archive.get_entry_by_path(path)
            if entry.is_redirect:
                entry = entry.get_redirect_entry()
            raw = bytes(entry.get_item().content).decode("utf-8", "replace")
            title, blocks = html_src.extract_from_string(raw, entry.title)
        except Exception:
            continue
        title = title or getattr(entry, "title", "")
        for si, (section, text) in enumerate(blocks):
            text = (text or "").strip()
            if not text:
                continue
            for off in range(0, len(text), chunk_chars):     # slice long sections
                seg = text[off:off + chunk_chars].strip()
                if len(seg) < 40:
                    continue
                out.append({
                    "id": f"zim://{path}#{si}.{off}",
                    "text": seg,
                    "title": title,
                    "section": section or "",
                    "path_or_url": f"zim://{path}",
                    "source_type": "wikipedia",
                    "score": 1.0 / (rank + 1),     # Xapian article rank; refined by rerank
                })
                if len(out) >= max_chunks:
                    return out
    return out


def iter_articles(zim_path: str):
    """Yield ``(article_url, title, blocks)`` for every content article in a ZIM."""
    try:
        from libzim.reader import Archive
    except Exception as e:
        raise MissingDependency(f"libzim required for Wikipedia ZIM: {e}")

    archive = Archive(zim_path)
    count = archive.all_entry_count
    log.info("ZIM opened: %s (%d entries)", zim_path, count)
    for i in range(count):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            mime = item.mimetype or ""
            if "html" not in mime:
                continue
            raw = bytes(item.content).decode("utf-8", "replace")
        except Exception:
            continue
        title, blocks = html_src.extract_from_string(raw, entry.title)
        if blocks:
            yield entry.path, (title or entry.title), blocks
