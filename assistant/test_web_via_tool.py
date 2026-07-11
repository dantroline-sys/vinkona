"""Tests for research_worker.web_via_tool — web search through the Mac tool host, shape-tolerant."""
import asyncio
import json
import sys
import types

try:                                    # web_via_tool doesn't use aiohttp; stub it if absent so
    import aiohttp                      # this test runs where the full env isn't installed.
except Exception:
    _stub = types.ModuleType("aiohttp")
    _stub.ClientSession = object
    _stub.ClientTimeout = lambda **k: None
    sys.modules["aiohttp"] = _stub

import research_worker as rw


class FakeTools:
    """Minimal stand-in for the tool host: canned per-tool responses."""
    def __init__(self, responses, active=True):
        self.responses = responses
        self.active = active
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        r = self.responses.get(name)
        return r(args) if callable(r) else r


def _run(coro):
    return asyncio.run(coro)


def test_json_results_shape_builds_snippets_and_fetches_top():
    tools = FakeTools({
        "web_search": json.dumps({"results": [
            {"title": "KPI guidance", "content": "KPIs should be SMART.", "url": "https://x/kpi"},
            {"title": "More", "content": "second snippet", "url": "https://x/2"}]}),
        "web_fetch": "FULL PAGE BODY about KPI regulations, much longer than the snippet text here.",
    })
    out = _run(rw.web_via_tool(tools, "KPI regulations"))
    assert out is not None
    text, url = out
    assert url == "https://x/kpi"
    assert "FULL PAGE BODY" in text                      # web_fetch of the top hit deepened it
    assert ("web_search", {"query": "KPI regulations"}) in tools.calls
    assert any(n == "web_fetch" for n, _ in tools.calls)


def test_snippets_used_when_fetch_shorter():
    tools = FakeTools({
        "web_search": json.dumps({"results": [
            {"title": "T", "content": "a reasonably long snippet of content here", "url": "u"}]}),
        "web_fetch": "tiny",                             # shorter than the snippet → keep snippets
    })
    text, url = _run(rw.web_via_tool(tools, "q"))
    assert "reasonably long snippet" in text and url == "u"


def test_prose_result_used_as_is():
    tools = FakeTools({"web_search": "just prose, not JSON at all", "web_fetch": ""})
    text, url = _run(rw.web_via_tool(tools, "q", fetch_top=False))
    assert text == "just prose, not JSON at all" and url == "web"


def test_empty_and_inactive_return_none():
    assert _run(rw.web_via_tool(FakeTools({"web_search": ""}), "q")) is None
    assert _run(rw.web_via_tool(FakeTools({}, active=False), "q")) is None
    assert _run(rw.web_via_tool(None, "q")) is None


def test_list_shape_tolerated():
    tools = FakeTools({"web_search": json.dumps(
        [{"title": "A", "snippet": "alt snippet key", "link": "L"}]), "web_fetch": ""})
    text, url = _run(rw.web_via_tool(tools, "q"))
    assert "alt snippet key" in text and url == "L"


# ── kb / library sources (knowledge-host tools via call_raw) ──────────────────────

class RawTools:
    """Tool host whose call_raw returns {ok, result} per tool (None result = unknown tool)."""
    def __init__(self, results, active=True):
        self.results = results
        self.active = active

    async def call_raw(self, name, args):
        if name not in self.results:
            return {"ok": False, "result": "", "error": f"no tool named {name}"}
        return {"ok": True, "result": self.results[name]}


def test_kb_text_extracts_items_and_steps():
    raw = json.dumps({"items": [{"label": "Sepsis", "text": "give fluids",
                                 "steps": ["cultures", "antibiotics within 1h"]}], "confidence": 0.8})
    t = rw._kb_text(raw)
    assert "Sepsis: give fluids" in t and "antibiotics within 1h" in t


def test_kb_text_abstain_is_empty():
    assert rw._kb_text(json.dumps({"abstain": True, "items": [{"text": "x"}]})) == ""
    assert rw._kb_text(json.dumps({"low_confidence": True, "passages": [{"text": "x"}]})) == ""
    assert rw._kb_text("not json") == ""


def test_kb_source_combines_ask_and_search():
    tools = RawTools({
        "kb_ask": json.dumps({"items": [{"label": "P", "text": "procedure text"}]}),
        "kb_search": json.dumps({"passages": [{"title": "T", "text": "passage text"}]}),
    })
    out = _run(rw.kb_source(tools, "q"))
    assert out is not None and "procedure text" in out[0] and "passage text" in out[0]


def test_kb_source_none_when_host_absent():
    # Mac-only host: kb tools are unknown → call_raw ok=False → skipped → None (not an error blob).
    assert _run(rw.kb_source(RawTools({"web_search": "x"}), "q")) is None
    assert _run(rw.library_source(RawTools({"web_search": "x"}), "q")) is None


def test_library_source_reads_results():
    tools = RawTools({"library_search": json.dumps(
        {"results": [{"title": "Doc", "text": "reg clause 4.2 says..."}]})})
    out = _run(rw.library_source(tools, "q"))
    assert out is not None and "reg clause 4.2" in out[0]


# ── hoarder mode: keep everything ────────────────────────────────────────────────

def test_kb_text_uncapped_keeps_everything():
    big = "x" * 20000
    raw = json.dumps({"passages": [{"text": big}]})
    assert len(rw._kb_text(raw, max_chars=0)) >= 20000        # 0 = no truncation
    assert len(rw._kb_text(raw, max_chars=4000)) == 4000      # bounded default still caps


def test_kb_text_max_items_limits_passages():
    raw = json.dumps({"passages": [{"text": f"p{i}"} for i in range(20)]})
    assert rw._kb_text(raw, max_items=3).count("\n") == 2     # 3 passages → 2 newlines
    assert rw._kb_text(raw, max_items=12).count("\n") == 11


def test_web_via_tool_hoard_keeps_more_and_uncapped():
    results = [{"title": f"T{i}", "content": ("y" * 5000), "url": f"u{i}"} for i in range(10)]
    tools = FakeTools({"web_search": json.dumps({"results": results}), "web_fetch": ""})
    text, _ = _run(rw.web_via_tool(tools, "q", fetch_top=False, max_chars=0, max_items=10))
    assert len(text) > 40000                                  # many big snippets, nothing dropped


# ── gather_sources orchestration (all tools) ─────────────────────────────────────

class ComboTools:
    """Both interfaces: call (web_search/web_fetch) and call_raw (kb_*/library_search)."""
    def __init__(self, call_map, raw_map, active=True):
        self.call_map, self.raw_map, self.active = call_map, raw_map, active

    async def call(self, name, args):
        return self.call_map.get(name, "")

    async def call_raw(self, name, args):
        if name not in self.raw_map:
            return {"ok": False, "result": "", "error": "unknown"}
        return {"ok": True, "result": self.raw_map[name]}


class BoomSession:
    """A fake aiohttp session whose .get raises — so wiki_search fails soft to None."""
    def get(self, *a, **k):
        raise RuntimeError("no network in test")


class RecTrace:
    def __init__(self): self.events = []
    def write(self, **k): self.events.append(k)


def test_gather_sources_tries_every_tool():
    tools = ComboTools(
        call_map={"web_search": json.dumps({"results": [
            {"title": "Reg", "content": "clause says X", "url": "https://r"}]}), "web_fetch": ""},
        raw_map={"kb_ask": json.dumps({"items": [{"label": "P", "text": "kb procedure"}]}),
                 "library_search": json.dumps({"results": [{"text": "library doc body"}]})})
    tr = RecTrace()
    # scholarly=False keeps this focused on the local + wiki + web tools (scholarly nets have
    # their own suite); web=True exercises the general-web step.
    parts, first_url = _run(rw.gather_sources(BoomSession(), tools, "meta question",
                                              searxng_url=None, trace=tr,
                                              scholarly=False, web=True))
    labels = [p[0] for p in parts]
    assert "Knowledge base" in labels and "Document library" in labels and "Web search" in labels
    assert not any("Wikipedia" in l for l in labels)          # wiki failed soft, others still ran
    blob = rw._combine(parts)
    assert "kb procedure" in blob and "library doc body" in blob and "clause says X" in blob
    # every tool was declared in the feed (found/not) — 4 source_try events
    tried = [e["tool"] for e in tr.events if e.get("kind") == "source_try"]
    assert tried == ["knowledge base", "document library", "wikipedia", "web search"]


def test_gather_sources_runs_source_pick():
    tools = ComboTools(call_map={}, raw_map={})               # no kb/library/web/Mac-tool hits
    tr = RecTrace()
    parts, _ = _run(rw.gather_sources(BoomSession(), tools, "neural network transformer model",
                                      searxng_url=None, trace=tr, scholarly=True, web=False))
    picked = [e for e in tr.events if e.get("kind") == "source_pick"]
    assert picked and picked[0]["sources"] == ["scholar"]     # STEM question → OpenAlex (fallback, no LM)
    tried = [e["tool"] for e in tr.events if e.get("kind") == "source_try"]
    assert "OpenAlex" in tried and "web search" not in tried  # web disabled


# ── search-query distillation (question → keywords) ──────────────────────────────

def test_search_query_strips_interrogatives():
    # The motivating failure: "How does …" was matching dictionary pages for "does".
    q = rw._search_query("How does dissociation manifest as 'gaps' in consciousness or "
                         "memory in psychiatric disorders?")
    low = q.lower()
    assert "does" not in low.split() and "how" not in low.split()
    for kw in ("dissociation", "consciousness", "memory", "psychiatric", "disorders"):
        assert kw in low, kw


def test_search_query_keeps_quoted_phrase():
    q = rw._search_query('what is the "theory of mind" in autism')
    assert '"theory of mind"' in q and "autism" in q


def test_search_query_caps_terms():
    q = rw._search_query(" ".join(f"term{i}" for i in range(30)), max_terms=5)
    assert len(q.split()) == 5


def test_search_query_falls_back_when_empty():
    assert rw._search_query("how do you do it") != ""      # all-stopword → falls back to original
    assert rw._search_query("") == ""


# ── headline parsing (structured feeds AND prose) ────────────────────────────────

def test_headline_items_parses_structured_json():
    raw = json.dumps({"items": [{"title": "Quake", "link": "u1", "source": "BBC"},
                                 {"headline": "Budget", "url": "u2"}]})
    items = rw._headline_items(raw)
    assert [i.get("title") or i.get("headline") for i in items] == ["Quake", "Budget"]


def test_headline_items_splits_prose_lines():
    prose = "1. Election recount ordered\n- Markets rally on rate cut\n\n* Storm warning issued"
    items = rw._headline_items(prose)
    titles = [i["title"] for i in items]
    assert "Election recount ordered" in titles and "Markets rally on rate cut" in titles
    assert "Storm warning issued" in titles


def test_headline_items_drops_trivial_lines_and_empty():
    assert rw._headline_items("") == []
    # short lines (< 8 chars: dividers, section heads) are noise and dropped
    assert rw._headline_items("- short\n- A real headline here") == [{"title": "A real headline here"}]


def main():
    passed = failed = 0
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
