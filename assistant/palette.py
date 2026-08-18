"""
VIN-TOOL-01 palette v0 (§0.8): the news/document core — the use-case that
actually failed under free-form codegen, now as pre-verified blocks.

    rss_fetch → filter_predicate → dedupe → sort_rank → ranked_head
              → summarise (LM) | digest_render (mechanical)

Every block: pure stdlib, side effects only via the injected Ctx, ≥2 fixtures
(one an edge case).  Importing this module populates the registry.
"""
from __future__ import annotations

import email.utils
import html.parser
import xml.etree.ElementTree as ET
from datetime import datetime

from blocks import BlockError, block

# ── helpers (not blocks) ──────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _ts(raw: str) -> float:
    """RFC-822 (RSS) or ISO-8601 (Atom) → epoch seconds; unparseable → 0.0
    (degrade honestly, never invent a date)."""
    raw = (raw or "").strip()
    if not raw:
        return 0.0
    try:
        return email.utils.parsedate_to_datetime(raw).timestamp()
    except Exception:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


class _TextExtractor(html.parser.HTMLParser):
    _SKIP = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._out, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._out.append(data.strip())

    def text(self) -> str:
        return " ".join(self._out)


def _strip_html(s: str) -> str:
    p = _TextExtractor()
    p.feed(s or "")
    return p.text()


_RSS_XML = ("<rss><channel>"
            "<item><title>Alpha rains</title><link>http://a/1</link>"
            "<description>Rain in &amp; around Alpha.</description>"
            "<pubDate>Mon, 01 Jun 2026 10:00:00 GMT</pubDate></item>"
            "<item><title>Beta wins</title><link>http://a/2</link>"
            "<description>Beta took the cup.</description></item>"
            "</channel></rss>")

_ATOM_XML = ('<feed xmlns="http://www.w3.org/2005/Atom">'
             "<entry><title>Gamma launch</title>"
             '<link href="http://b/1"/>'
             "<summary>Gamma lifts off.</summary>"
             "<updated>2026-06-01T10:00:00Z</updated></entry></feed>")


@block(
    name="rss_fetch", version="1.0.0",
    summary="Fetch an RSS or Atom feed and return its entries as documents.",
    ports_in={"feed": "FeedRef"}, ports_out={"docs": "List[Document]"},
    params={"max_items": {"type": "integer", "default": 30,
                          "description": "Keep at most this many newest entries."}},
    capabilities=("net",),
    failure_modes=("network error → fail", "unparseable feed → fail",
                   "entry without a date → ts 0.0 (degrade, never invent)"),
    fixtures=(
        {"name": "rss2", "inputs": {"feed": {"url": "http://a/feed"}},
         "ctx": {"net": {"http://a/feed": _RSS_XML}},
         "expect": {"docs": [
             {"title": "Alpha rains", "url": "http://a/1",
              "text": "Rain in & around Alpha.", "ts": 1780308000.0},
             {"title": "Beta wins", "url": "http://a/2",
              "text": "Beta took the cup.", "ts": 0.0}]}},
        {"name": "atom", "inputs": {"feed": {"url": "http://b/feed"}},
         "ctx": {"net": {"http://b/feed": _ATOM_XML}},
         "expect": {"docs": [
             {"title": "Gamma launch", "url": "http://b/1",
              "text": "Gamma lifts off.", "ts": 1780308000.0}]}},
        {"name": "edge-malformed", "inputs": {"feed": {"url": "http://c/feed"}},
         "ctx": {"net": {"http://c/feed": "<rss><chan"}},
         "expect_error": "did not parse"},
    ))
def rss_fetch(inputs, params, ctx):
    raw = ctx.net(inputs["feed"]["url"])
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise BlockError(f"feed did not parse: {e}") from None
    docs = []
    for item in root.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        d = {"title": "", "url": "", "text": "", "ts": 0.0}
        for c in item:
            ctag = c.tag.rsplit("}", 1)[-1]
            if ctag == "title":
                d["title"] = (c.text or "").strip()
            elif ctag == "link":
                d["url"] = (c.text or c.get("href") or "").strip()
            elif ctag in ("description", "summary", "content"):
                d["text"] = _strip_html(c.text or "")
            elif ctag in ("pubDate", "updated", "published"):
                d["ts"] = d["ts"] or _ts(c.text or "")
        docs.append(d)
    docs.sort(key=lambda d: -d["ts"])
    return {"docs": docs[: params["max_items"]]}


@block(
    name="parse_html", version="1.0.0",
    summary="Strip HTML markup from a document's text, keeping readable prose.",
    ports_in={"doc": "Document"}, ports_out={"doc": "Document"},
    params={},
    capabilities=(),
    failure_modes=("script/style content is dropped, never surfaced",),
    fixtures=(
        {"name": "tags", "inputs": {"doc": {"title": "t", "url": "",
                                            "text": "<p>Hello <b>world</b></p>"}},
         "expect": {"doc": {"title": "t", "url": "", "text": "Hello world"}}},
        {"name": "edge-script-and-entities",
         "inputs": {"doc": {"text": "<script>evil()</script>Tom &amp; Jerry"}},
         "expect": {"doc": {"text": "Tom & Jerry"}}},
    ))
def parse_html(inputs, params, ctx):
    doc = dict(inputs["doc"])
    doc["text"] = _strip_html(doc.get("text", ""))
    return {"doc": doc}


@block(
    name="docs_to_passages", version="1.0.0",
    summary="Adapter: split documents into short passages for finer-grained work.",
    ports_in={"docs": "List[Document]"}, ports_out={"passages": "List[Passage]"},
    params={"max_chars": {"type": "integer", "default": 800,
                          "description": "Longest passage, in characters."}},
    capabilities=(),
    failure_modes=("a document with no text yields no passages (degrade)",),
    fixtures=(
        {"name": "split", "inputs": {"docs": [
            {"title": "t", "url": "u", "text": "One.\n\nTwo."}]},
         "expect": {"passages": [{"text": "One.", "title": "t", "url": "u"},
                                 {"text": "Two.", "title": "t", "url": "u"}]}},
        {"name": "edge-empty-doc", "inputs": {"docs": [{"text": "   "}]},
         "expect": {"passages": []}},
    ))
def docs_to_passages(inputs, params, ctx):
    out = []
    for d in inputs["docs"]:
        for para in (d.get("text") or "").split("\n\n"):
            para = para.strip()
            while para:
                chunk, para = para[: params["max_chars"]], para[params["max_chars"]:]
                out.append({"text": chunk,
                            "title": d.get("title", ""), "url": d.get("url", "")})
    return {"passages": out}


@block(
    name="filter_predicate", version="1.0.0",
    summary="Keep documents whose chosen field matches declared terms "
            "(case-insensitive substrings; any/all/none).",
    ports_in={"docs": "List[Document]"}, ports_out={"docs": "List[Document]"},
    params={"terms": {"type": "array", "items": {"type": "string"},
                      "description": "Substrings to look for."},
            "field": {"type": "string", "enum": ["title", "text", "url", "any"],
                      "default": "any", "description": "Which field to match."},
            "mode": {"type": "string", "enum": ["any", "all", "none"],
                     "default": "any",
                     "description": "Keep on any term, all terms, or no term."}},
    capabilities=(),
    failure_modes=("empty terms list → fail (a filter matching everything or "
                   "nothing by accident is the invisible failure §6 exists for)",),
    fixtures=(
        {"name": "any", "params": {"terms": ["rain"]},
         "inputs": {"docs": [{"title": "Alpha rains", "text": "wet"},
                             {"title": "Beta wins", "text": "cup"}]},
         "expect": {"docs": [{"title": "Alpha rains", "text": "wet"}]}},
        {"name": "none-mode", "params": {"terms": ["cup"], "mode": "none"},
         "inputs": {"docs": [{"title": "Alpha", "text": "wet"},
                             {"title": "Beta", "text": "cup"}]},
         "expect": {"docs": [{"title": "Alpha", "text": "wet"}]}},
        {"name": "edge-no-terms", "params": {"terms": []},
         "inputs": {"docs": []}, "expect_error": "no terms"},
    ))
def filter_predicate(inputs, params, ctx):
    terms = [_norm(t) for t in params["terms"] if _norm(t)]
    if not terms:
        raise BlockError("no terms declared for the filter")
    fields = ["title", "text", "url"] if params["field"] == "any" else [params["field"]]

    def hits(d):
        hay = " ".join(_norm(str(d.get(f, ""))) for f in fields)
        return [t in hay for t in terms]

    mode = params["mode"]
    keep = {"any": any, "all": all, "none": lambda h: not any(h)}[mode]
    return {"docs": [d for d in inputs["docs"] if keep(hits(d))]}


@block(
    name="dedupe", version="1.0.0",
    summary="Drop duplicate documents by url, title, or text (first one wins).",
    ports_in={"docs": "List[Document]"}, ports_out={"docs": "List[Document]"},
    params={"key": {"type": "string", "enum": ["url", "title", "text"],
                    "default": "url", "description": "Field that defines sameness."}},
    capabilities=(),
    failure_modes=("a document missing the key field is kept (cannot judge)",),
    fixtures=(
        {"name": "url-dupes", "inputs": {"docs": [
            {"title": "A", "url": "http://x", "text": "1"},
            {"title": "B", "url": "http://x", "text": "2"},
            {"title": "C", "url": "http://y", "text": "3"}]},
         "expect": {"docs": [{"title": "A", "url": "http://x", "text": "1"},
                             {"title": "C", "url": "http://y", "text": "3"}]}},
        {"name": "edge-case-and-spacing", "params": {"key": "title"},
         "inputs": {"docs": [{"title": "Big  News", "text": "1"},
                             {"title": "big news", "text": "2"}]},
         "expect": {"docs": [{"title": "Big  News", "text": "1"}]}},
    ))
def dedupe(inputs, params, ctx):
    seen, out = set(), []
    for d in inputs["docs"]:
        k = _norm(str(d.get(params["key"], "")))
        if not k:
            out.append(d)
            continue
        if k not in seen:
            seen.add(k)
            out.append(d)
    return {"docs": out}


@block(
    name="sort_rank", version="1.0.0",
    summary="Rank documents against a query by term overlap (title counts "
            "double), optionally weighting recency.",
    ports_in={"docs": "List[Document]", "query": "Query"},
    ports_out={"ranked": "Ranked[Document]"},
    params={"recency_weight": {"type": "number", "default": 0.0,
                               "description": "0 = relevance only; higher favours newer items."}},
    capabilities=(),
    failure_modes=("empty query → fail", "ties keep their input order (stable)"),
    fixtures=(
        {"name": "relevance", "inputs": {
            "query": {"text": "rain alpha"},
            "docs": [{"title": "Beta wins", "text": "cup", "ts": 0.0},
                     {"title": "Alpha rains", "text": "rain in alpha", "ts": 0.0}]},
         "expect": {"ranked": [
             {"title": "Alpha rains", "text": "rain in alpha", "ts": 0.0},
             {"title": "Beta wins", "text": "cup", "ts": 0.0}]}},
        {"name": "edge-empty-query", "inputs": {"query": {"text": "  "}, "docs": []},
         "expect_error": "empty query"},
    ))
def sort_rank(inputs, params, ctx):
    terms = _norm(inputs["query"]["text"]).split()
    if not terms:
        raise BlockError("empty query")
    docs = list(inputs["docs"])
    max_ts = max((d.get("ts") or 0.0 for d in docs), default=0.0) or 1.0

    def score(d):
        title, text = _norm(d.get("title", "")), _norm(d.get("text", ""))
        s = sum(2 * title.count(t) + text.count(t) for t in terms)
        return s + params["recency_weight"] * ((d.get("ts") or 0.0) / max_ts)

    return {"ranked": sorted(docs, key=score, reverse=True)}


@block(
    name="ranked_head", version="1.0.0",
    summary="Adapter: keep only the top N of a ranked list.",
    ports_in={"ranked": "Ranked[Document]"}, ports_out={"docs": "List[Document]"},
    params={"n": {"type": "integer", "default": 5,
                  "description": "How many of the best to keep."}},
    capabilities=(),
    failure_modes=("n < 1 → fail",),
    fixtures=(
        {"name": "head", "params": {"n": 1},
         "inputs": {"ranked": [{"text": "a"}, {"text": "b"}]},
         "expect": {"docs": [{"text": "a"}]}},
        {"name": "edge-n-past-end", "params": {"n": 9},
         "inputs": {"ranked": [{"text": "a"}]},
         "expect": {"docs": [{"text": "a"}]}},
        {"name": "edge-bad-n", "params": {"n": 0}, "inputs": {"ranked": []},
         "expect_error": "n must be"},
    ))
def ranked_head(inputs, params, ctx):
    if params["n"] < 1:
        raise BlockError("n must be at least 1")
    return {"docs": inputs["ranked"][: params["n"]]}


@block(
    name="summarise", version="1.0.0",
    summary="Summarise documents into one short digest using the language model.",
    ports_in={"docs": "List[Document]"}, ports_out={"digest": "Digest"},
    params={"style": {"type": "string", "default": "brief",
                      "description": "Tone of the summary (e.g. brief, chatty)."},
            "title": {"type": "string", "default": "Digest",
                      "description": "Heading for the digest."}},
    capabilities=(),          # LM access is capability-free by contract (§9)
    failure_modes=("no documents → fail (never an invented digest)",),
    fixtures=(
        {"name": "canned", "inputs": {"docs": [
            {"title": "Alpha rains", "text": "Rain in Alpha."}]},
         "ctx": {"llm": "It rained in Alpha."},
         "expect": {"digest": {"title": "Digest", "text": "It rained in Alpha.",
                               "count": 1}}},
        {"name": "edge-nothing", "inputs": {"docs": []},
         "expect_error": "nothing to summarise"},
    ))
def summarise(inputs, params, ctx):
    docs = inputs["docs"]
    if not docs:
        raise BlockError("nothing to summarise")
    lines = [f"- {d.get('title', '(untitled)')}: {(d.get('text') or '')[:300]}"
             for d in docs]
    reply = ctx.llm(f"Summarise these items in a {params['style']} digest:\n"
                    + "\n".join(lines))
    return {"digest": {"title": params["title"], "text": reply.strip(),
                       "count": len(docs)}}


@block(
    name="digest_render", version="1.0.0",
    summary="Render a ranked list into a plain-text digest, no language model "
            "involved (titles, first lines, links).",
    ports_in={"ranked": "Ranked[Document]"}, ports_out={"digest": "Digest"},
    params={"max_items": {"type": "integer", "default": 10,
                          "description": "Most items the digest lists."},
            "title": {"type": "string", "default": "Digest",
                      "description": "Heading for the digest."}},
    capabilities=(),
    failure_modes=("empty input renders count 0, distinct from 'no new data' (§6)",),
    fixtures=(
        {"name": "render", "inputs": {"ranked": [
            {"title": "Alpha rains", "url": "http://a/1", "text": "Rain came."}]},
         "expect": {"digest": {
             "title": "Digest", "count": 1,
             "text": "• Alpha rains — Rain came. (http://a/1)"}}},
        {"name": "edge-empty", "inputs": {"ranked": []},
         "expect": {"digest": {"title": "Digest", "count": 0, "text": ""}}},
    ))
def digest_render(inputs, params, ctx):
    items = inputs["ranked"][: params["max_items"]]
    lines = []
    for d in items:
        first = (d.get("text") or "").split(". ")[0].strip()
        if first and not first.endswith("."):
            first += "."
        bits = [d.get("title") or "(untitled)"]
        if first:
            bits.append(f"— {first}")
        if d.get("url"):
            bits.append(f"({d['url']})")
        lines.append("• " + " ".join(bits))
    return {"digest": {"title": params["title"], "text": "\n".join(lines),
                       "count": len(items)}}


@block(
    name="store_write", version="1.0.0",
    summary="Save a digest as a text file inside her tool store, so it can be "
            "read back later.",
    ports_in={"digest": "Digest"}, ports_out={"file": "FileRef"},
    params={"path": {"type": "string", "default": "digests/latest.txt",
                     "description": "Store-relative file path to write."}},
    capabilities=("fs-write",),
    rollback="the one store-relative file it writes (engine snapshots before "
             "overwrite; see graphrun.LiveCtx)",
    failure_modes=("a path escaping the store → fail (containment is the "
                   "engine's, not this block's)",),
    fixtures=(
        {"name": "write", "inputs": {"digest": {"title": "D", "text": "hello",
                                                "count": 1}},
         "expect": {"file": {"path": "(shadow)/digests/latest.txt"}}},
        {"name": "edge-own-path", "params": {"path": "digests/rain.txt"},
         "inputs": {"digest": {"title": "D", "text": "", "count": 0}},
         "expect": {"file": {"path": "(shadow)/digests/rain.txt"}}},
    ))
def store_write(inputs, params, ctx):
    d = inputs["digest"]
    body = (d.get("title") or "Digest") + "\n\n" + (d.get("text") or "")
    return {"file": {"path": ctx.write(params["path"], body)}}
