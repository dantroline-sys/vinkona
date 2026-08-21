"""Local research genre — the Mac host's keyless research tools, served here.

The same names the research router already selects between (see
research_worker.RESEARCH_SOURCES): literature_search (Europe PMC),
scholar_search (OpenAlex), drug_info (openFDA), reference_lookup
(Wikipedia/Wikidata), define_word (Wiktionary), qa_search (Stack Exchange),
hn_search (Hacker News/Algolia), events_search (GDELT, self-rate-limited),
books_search (Open Library), archive_search + wayback_lookup (Internet
Archive).  All keyless, all GETs through the broker.  NOTE the posture
change this genre opts into: with a Mac host these lookups leave the MAC;
enabling this genre makes them leave THIS box (still policy-checked and
audited in egress.jsonl, still no general web search — that stays off by
design).
"""
import gzip
import json
import re
import threading
import time
import urllib.parse

_TAG = re.compile(r"<[^>]+>")


def _q(s) -> str:
    return urllib.parse.quote(str(s or "").strip())


def _json_get(fetch, url, *, timeout=15.0, headers=None):
    status, body = fetch(url, timeout=timeout, purpose="local_research",
                         headers=headers)
    if body[:2] == b"\x1f\x8b":            # Stack Exchange gzips unconditionally
        body = gzip.decompress(body)
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    return json.loads(body.decode("utf-8", errors="replace"))


def _clean(s, cap=300) -> str:
    return _TAG.sub(" ", str(s or "")).replace("\n", " ").strip()[:cap]


def _delist(v) -> str:
    return ", ".join(str(x) for x in v[:4]) if isinstance(v, list) else str(v or "")


# ── the tools ────────────────────────────────────────────────────────────────

def literature_search(fetch, args):
    n = max(1, min(int(args.get("limit") or 5), 10))
    data = _json_get(fetch, "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                            f"?query={_q(args.get('query'))}&format=json"
                            f"&resultType=core&pageSize={n}")
    hits = (data.get("resultList") or {}).get("result") or []
    if not hits:
        return "No papers found."
    out = [f"{len(hits)} paper(s) from Europe PMC:"]
    for h in hits:
        cite = " · ".join(x for x in (
            _clean(h.get("authorString"), 120), h.get("journalTitle", ""),
            str(h.get("pubYear", ""))) if x)
        line = f"- {_clean(h.get('title'), 200)} ({cite})"
        abstract = _clean(h.get("abstractText"), 400)
        out.append(line + (f"\n  {abstract}" if abstract else ""))
    return "\n".join(out)


def _openalex_abstract(inv) -> str:
    """OpenAlex ships abstracts as {word: [positions]} — invert it back."""
    if not isinstance(inv, dict):
        return ""
    slots = {}
    for word, positions in inv.items():
        for p in positions or []:
            slots[int(p)] = word
    return " ".join(slots[i] for i in sorted(slots))


def scholar_search(fetch, args):
    n = max(1, min(int(args.get("limit") or 5), 10))
    data = _json_get(fetch, "https://api.openalex.org/works"
                            f"?search={_q(args.get('query'))}&per-page={n}")
    hits = data.get("results") or []
    if not hits:
        return "No works found."
    out = [f"{len(hits)} work(s) from OpenAlex:"]
    for h in hits:
        authors = _delist([(a.get("author") or {}).get("display_name", "")
                           for a in h.get("authorships") or []])
        line = (f"- {_clean(h.get('display_name'), 200)} "
                f"({authors} · {h.get('publication_year', '?')} · "
                f"cited {h.get('cited_by_count', 0)}×)")
        abstract = _clean(_openalex_abstract(h.get("abstract_inverted_index")), 400)
        out.append(line + (f"\n  {abstract}" if abstract else ""))
    return "\n".join(out)


def drug_info(fetch, args):
    name = str(args.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "give a drug name"}
    q = _q(f'openfda.brand_name:"{name}" openfda.generic_name:"{name}"')
    data = _json_get(fetch, f"https://api.fda.gov/drug/label.json?search={q}&limit=1")
    hits = data.get("results") or []
    if not hits:
        return f"No US drug label found for {name}."
    label = hits[0]
    fda = label.get("openfda") or {}
    out = [f"US drug label for {_delist(fda.get('brand_name')) or name}"
           f" ({_delist(fda.get('generic_name'))}):"]
    for key, title in (("indications_and_usage", "Uses"),
                       ("dosage_and_administration", "Dosing"),
                       ("warnings", "Warnings"),
                       ("drug_interactions", "Interactions")):
        v = label.get(key)
        text = _clean(v[0] if isinstance(v, list) and v else v, 600)
        if text:
            out.append(f"- {title}: {text}")
    return "\n".join(out)


def reference_lookup(fetch, args):
    query = str(args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "give something to look up"}
    try:
        data = _json_get(fetch, "https://en.wikipedia.org/api/rest_v1/page/summary/"
                                + _q(query.replace(" ", "_")))
    except Exception:
        data = None
    if not data or data.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
        found = _json_get(fetch, "https://en.wikipedia.org/w/rest.php/v1/search/page"
                                 f"?q={_q(query)}&limit=1")
        pages = found.get("pages") or []
        if not pages:
            return f"Nothing encyclopedic found for '{query}'."
        data = _json_get(fetch, "https://en.wikipedia.org/api/rest_v1/page/summary/"
                                + _q(pages[0].get("key", "")))
    title = data.get("title", query)
    extract = _clean(data.get("extract"), 1200)
    desc = _clean(data.get("description"), 120)
    head = f"{title} ({desc})" if desc else title
    return f"{head}\n{extract}" if extract else f"{head} — no summary available."


def define_word(fetch, args):
    word = str(args.get("word") or "").strip()
    if not word:
        return {"ok": False, "error": "give a word"}
    data = _json_get(fetch, "https://en.wiktionary.org/api/rest_v1/page/definition/"
                            + _q(word.lower()))
    senses = data.get("en") or next(iter(data.values()), []) if isinstance(data, dict) else []
    if not senses:
        return f"No definition found for '{word}'."
    out = [f"{word}:"]
    for block in senses[:3]:
        pos = block.get("partOfSpeech", "")
        for i, d in enumerate(block.get("definitions") or []):
            text = _clean(d.get("definition"), 220)
            if text:
                out.append(f"- ({pos}) {text}")
            if i >= 2:
                break
    return "\n".join(out)


def qa_search(fetch, args):
    site = str(args.get("site") or "stackoverflow").strip()
    n = max(1, min(int(args.get("limit") or 5), 10))
    data = _json_get(fetch, "https://api.stackexchange.com/2.3/search/advanced"
                            f"?order=desc&sort=relevance&q={_q(args.get('query'))}"
                            f"&site={_q(site)}&pagesize={n}")
    hits = data.get("items") or []
    if not hits:
        return f"No {site} questions found."
    out = [f"{len(hits)} question(s) on {site}:"]
    for h in hits:
        answered = "answered" if h.get("is_answered") else "unanswered"
        out.append(f"- {_clean(h.get('title'), 180)} "
                   f"(score {h.get('score', 0)}, {answered})\n  {h.get('link', '')}")
    return "\n".join(out)


def hn_search(fetch, args):
    n = max(1, min(int(args.get("limit") or 5), 10))
    endpoint = "search_by_date" if args.get("recent") else "search"
    data = _json_get(fetch, f"https://hn.algolia.com/api/v1/{endpoint}"
                            f"?query={_q(args.get('query'))}&hitsPerPage={n}")
    hits = data.get("hits") or []
    if not hits:
        return "Nothing on Hacker News for that."
    out = [f"{len(hits)} Hacker News item(s):"]
    for h in hits:
        title = _clean(h.get("title") or h.get("story_title"), 180)
        out.append(f"- {title} ({h.get('points', 0)} points, "
                   f"{h.get('num_comments', 0)} comments) {h.get('url') or ''}".rstrip())
    return "\n".join(out)


_gdelt_lock = threading.Lock()
_gdelt_last = [0.0]
GDELT_MIN_GAP_S = 5.0


def events_search(fetch, args, now=time.time):
    with _gdelt_lock:
        wait = GDELT_MIN_GAP_S - (now() - _gdelt_last[0])
        if wait > 0:
            return "rate-limited, try again in a few seconds"
        _gdelt_last[0] = now()
    n = max(1, min(int(args.get("limit") or 6), 15))
    span = str(args.get("timespan") or "3d").strip()
    data = _json_get(fetch, "https://api.gdeltproject.org/api/v2/doc/doc"
                            f"?query={_q(args.get('query'))}%20sourcelang:english"
                            f"&mode=artlist&maxrecords={n}&timespan={_q(span)}"
                            "&format=json")
    hits = data.get("articles") or []
    if not hits:
        return "No recent world-news coverage found."
    out = [f"{len(hits)} recent article(s):"]
    for h in hits:
        out.append(f"- {_clean(h.get('title'), 180)} "
                   f"({h.get('domain', '')}, {str(h.get('seendate', ''))[:8]})")
    return "\n".join(out)


def books_search(fetch, args):
    n = max(1, min(int(args.get("limit") or 5), 10))
    data = _json_get(fetch, "https://openlibrary.org/search.json"
                            f"?q={_q(args.get('query'))}&limit={n}"
                            "&fields=title,author_name,first_publish_year")
    hits = data.get("docs") or []
    if not hits:
        return "No books found."
    out = [f"{len(hits)} book(s) from Open Library:"]
    for h in hits:
        out.append(f"- {_clean(h.get('title'), 160)} — "
                   f"{_delist(h.get('author_name'))} ({h.get('first_publish_year', '?')})")
    return "\n".join(out)


def archive_search(fetch, args):
    n = max(1, min(int(args.get("limit") or 6), 15))
    q = str(args.get("query") or "").strip()
    mt = str(args.get("mediatype") or "").strip()
    if mt:
        q = f"{q} AND mediatype:{mt}"
    data = _json_get(fetch, "https://archive.org/advancedsearch.php"
                            f"?q={_q(q)}&rows={n}&output=json"
                            "&fl%5B%5D=identifier&fl%5B%5D=title"
                            "&fl%5B%5D=mediatype&fl%5B%5D=year")
    hits = (data.get("response") or {}).get("docs") or []
    if not hits:
        return "Nothing in the Internet Archive for that."
    out = [f"{len(hits)} Internet Archive item(s):"]
    for h in hits:
        out.append(f"- {_clean(h.get('title'), 160)} "
                   f"({h.get('mediatype', '?')}, {h.get('year', '?')}) "
                   f"https://archive.org/details/{h.get('identifier', '')}")
    return "\n".join(out)


def wayback_lookup(fetch, args):
    url = str(args.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "give a URL"}
    date = re.sub(r"\D", "", str(args.get("date") or ""))
    q = f"https://archive.org/wayback/available?url={_q(url)}"
    if date:
        q += f"&timestamp={date}"
    data = _json_get(fetch, q)
    snap = ((data.get("archived_snapshots") or {}).get("closest")) or {}
    if not snap.get("url"):
        return f"No archived snapshot of {url}."
    return (f"Closest snapshot ({str(snap.get('timestamp', ''))[:8]}): "
            f"{snap['url']}")


# ── catalogue ────────────────────────────────────────────────────────────────

_SPECS = [
    ("literature_search", literature_search,
     "Search biomedical/clinical papers (Europe PMC): abstracts + citations.",
     {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
     ["query"]),
    ("scholar_search", scholar_search,
     "Search scholarly works in ALL fields incl. CS/AI (OpenAlex): abstracts + citation counts.",
     {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
     ["query"]),
    ("drug_info", drug_info,
     "US drug label (openFDA): uses, dosing, warnings, interactions.",
     {"name": {"type": "string", "description": "brand or generic drug name"}},
     ["name"]),
    ("reference_lookup", reference_lookup,
     "Encyclopedic 'what/who is X' (Wikipedia): a sourced summary.",
     {"query": {"type": "string"}}, ["query"]),
    ("define_word", define_word,
     "Dictionary definition of a word (Wiktionary).",
     {"word": {"type": "string"}}, ["word"]),
    ("qa_search", qa_search,
     "Practical how-to Q&A (Stack Exchange); site picks the community, e.g. "
     "stackoverflow, cooking, diy, travel, health, law.",
     {"query": {"type": "string"}, "site": {"type": "string", "default": "stackoverflow"},
      "limit": {"type": "integer", "default": 5}}, ["query"]),
    ("hn_search", hn_search,
     "Tech news and discussion (Hacker News); recent=true sorts newest first.",
     {"query": {"type": "string"}, "recent": {"type": "boolean", "default": False},
      "limit": {"type": "integer", "default": 5}}, ["query"]),
    ("events_search", events_search,
     "Current world news coverage (GDELT). Rate-limited to one query every 5s.",
     {"query": {"type": "string"}, "timespan": {"type": "string", "default": "3d"},
      "limit": {"type": "integer", "default": 6}}, ["query"]),
    ("books_search", books_search,
     "Books and authors (Open Library).",
     {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
     ["query"]),
    ("archive_search", archive_search,
     "Digitised texts/audio/film/software (Internet Archive); mediatype narrows "
     "(texts, audio, movies, software).",
     {"query": {"type": "string"}, "mediatype": {"type": "string"},
      "limit": {"type": "integer", "default": 6}}, ["query"]),
    ("wayback_lookup", wayback_lookup,
     "Archived snapshot URL of a web page (Wayback Machine) — a lookup, not a search.",
     {"url": {"type": "string"}, "date": {"type": "string", "description": "YYYYMMDD"}},
     ["url"]),
]


def tools(gcfg: dict, env: dict) -> list:
    fetch = env["fetch"]
    out = []
    for name, fn, desc, props, required in _SPECS:
        spec = {"name": name, "description": desc,
                "parameters": {"type": "object", "properties": props}}
        if required:
            spec["parameters"]["required"] = required
        out.append((spec, (lambda args, _fn=fn: _fn(fetch, args))))
    return out


def probe(gcfg: dict, env: dict) -> dict:
    got = reference_lookup(env["fetch"], {"query": "Earth"})
    if isinstance(got, dict):
        return {"ok": False, "detail": got.get("error", "lookup failed")}
    return {"ok": True, "detail": "reference_lookup answered: " + got.split("\n")[0][:160]}
