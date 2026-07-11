"""Tests for the scholarly research sources (arXiv/PubMed/Semantic Scholar/Crossref) + the big-LM
source selection.  Network is faked; the parsing + routing + fallback are what's exercised."""
import asyncio
import sys
import types

try:                                    # these funcs don't use aiohttp; stub it if absent
    import aiohttp
except Exception:
    _stub = types.ModuleType("aiohttp")
    _stub.ClientSession = object
    _stub.ClientTimeout = lambda **k: None
    sys.modules["aiohttp"] = _stub

import research_worker as rw


def _run(c):
    return asyncio.run(c)


class FakeResp:
    def __init__(self, status=200, text="", json_data=None):
        self.status, self._text, self._json = status, text, json_data
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def text(self):
        return self._text
    async def json(self):
        return self._json


class FakeSession:
    """Routes each GET to a canned response by URL substring."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        for sub, resp in self.routes:
            if sub in url:
                return resp
        return FakeResp(status=404)


# ── parsers ──────────────────────────────────────────────────────────────────────

def test_arxiv_parses_atom():
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
           '<id>http://arxiv.org/abs/2301.00001v1</id>'
           '<title>Retrieval-Augmented Generation</title>'
           '<summary>We combine retrieval with generation for LLMs.</summary>'
           '</entry></feed>')
    s = FakeSession([("export.arxiv.org", FakeResp(text=xml))])
    text, url = _run(rw.arxiv_search(s, "RAG"))
    assert "Retrieval-Augmented Generation" in text and "retrieval with generation" in text
    assert url == "http://arxiv.org/abs/2301.00001v1"
    assert s.calls[0][1]["search_query"] == "all:RAG"     # query is namespaced into search_query


def test_pubmed_esearch_then_efetch():
    s = FakeSession([
        ("esearch.fcgi", FakeResp(json_data={"esearchresult": {"idlist": ["111", "222"]}})),
        ("efetch.fcgi", FakeResp(text="1. Dissociative disorders: a review.\nAbstract body…")),
    ])
    text, url = _run(rw.pubmed_search(s, "dissociation"))
    assert "Dissociative disorders" in text and url == "https://pubmed.ncbi.nlm.nih.gov/111/"


def test_pubmed_no_ids_returns_none():
    s = FakeSession([("esearch.fcgi", FakeResp(json_data={"esearchresult": {"idlist": []}}))])
    assert _run(rw.pubmed_search(s, "xyzzy")) is None


def test_semantic_scholar_parses():
    s = FakeSession([("api.semanticscholar.org", FakeResp(json_data={"data": [
        {"title": "Graph memory for agents", "abstract": "A graph-based memory.", "url": "http://ss/1"}]}))])
    text, url = _run(rw.semantic_scholar_search(s, "graph memory"))
    assert "Graph memory for agents" in text and url == "http://ss/1"


def test_crossref_strips_jats_abstract():
    s = FakeSession([("api.crossref.org", FakeResp(json_data={"message": {"items": [
        {"title": ["Vector databases"], "abstract": "<jats:p>Dense retrieval at scale.</jats:p>",
         "DOI": "10.1/abc", "URL": "http://x/doi"}]}}))])
    text, url = _run(rw.crossref_search(s, "vector db"))
    assert "Vector databases" in text and "Dense retrieval at scale." in text
    assert "<jats:p>" not in text and url == "http://x/doi"


def test_scholarly_non_200_is_none():
    s = FakeSession([("export.arxiv.org", FakeResp(status=500))])
    assert _run(rw.arxiv_search(s, "q")) is None


# ── selection: LM pick + deterministic fallback over the Mac research sources ─────
# The Mac tool host's keyless tools (scholar_search/literature_search/qa_search/events_search/…) are
# the primary research egress; the LM routes to them by the question's nature, with a domain-keyword
# fallback.  The arxiv/pubmed/… APIs above remain as the tunnel-down offline fallback.

class FakeMem:
    def __init__(self, result):
        self.result = result
    async def _chat_json(self, url, model, prompt):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_fallback_sources_routes_by_domain():
    assert rw._fallback_sources("How does dissociation manifest in psychiatric patients?") == ["literature"]
    assert rw._fallback_sources("What is a transformer neural network for RAG?") == ["scholar"]
    assert rw._fallback_sources("How do I install and configure nginx?") == ["qa"]
    assert rw._fallback_sources("latest news on the ceasefire") == ["events"]
    assert rw._fallback_sources("clinical trial of a machine learning model for cancer") == ["literature", "scholar"]
    assert rw._fallback_sources("the history of the roman senate") == ["scholar"]


def test_select_sources_uses_lm_pick():
    picks = _run(rw.select_sources("anything", FakeMem({"sources": ["literature", "qa"]}), "u", "m"))
    assert picks == ["literature", "qa"]


def test_select_sources_caps_at_three_and_dedups():
    picks = _run(rw.select_sources("q", FakeMem({"sources": ["scholar", "scholar", "qa", "hn", "events"]}), "u", "m"))
    assert picks == ["scholar", "qa", "hn"]


def test_select_sources_falls_back_on_bad_or_missing_pick():
    picks = _run(rw.select_sources("neural network model", FakeMem({"sources": ["bogus"]}), "u", "m"))
    assert picks == ["scholar"]                          # invalid names → domain heuristic (STEM)
    picks = _run(rw.select_sources("psychiatric disease", FakeMem(RuntimeError("down")), "u", "m"))
    assert picks == ["literature"]                       # LM error → fallback (health)


# ── _mac_text: prose / JSON tolerance + GDELT throttle sentinel ───────────────────

def test_mac_text_prose_and_json():
    assert rw._mac_text("plain readable answer") == "plain readable answer"
    assert "abc" in rw._mac_text('{"items": [{"text": "abc"}]}')
    assert rw._mac_text('{"foo": "bar"}') == '{"foo": "bar"}'   # unknown JSON shape kept as-is
    assert rw._mac_text("") == ""


def test_mac_text_drops_gdelt_ratelimit():
    assert rw._mac_text("rate-limited, try again") == ""


# ── mac_source: Mac tool primary + direct-HTTP offline fallback ───────────────────

class MacTools:
    """Tool host exposing call_raw with the §2 envelope; unknown tool → ok:false."""
    def __init__(self, results, active=True):
        self.results, self.active, self.calls = results, active, []
    async def call_raw(self, name, args):
        self.calls.append((name, args))
        if name not in self.results:
            return {"ok": False, "error": "unknown"}
        return {"ok": True, "result": self.results[name]}


def test_mac_source_uses_mac_tool_first():
    tools = MacTools({"scholar_search": "OpenAlex prose about transformers"})
    out = _run(rw.mac_source(tools, FakeSession([]), "scholar", "transformers", max_items=4))
    assert out == ("OpenAlex prose about transformers", "")
    assert tools.calls[0] == ("scholar_search", {"query": "transformers", "limit": 4})


def test_mac_source_falls_back_to_direct_http_when_tool_absent():
    # Mac host lacks scholar_search → mac_source falls back to semantic_scholar (direct HTTP).
    tools = MacTools({})
    s = FakeSession([("api.semanticscholar.org", FakeResp(json_data={"data": [
        {"title": "Graph memory", "abstract": "A graph.", "url": "http://ss/1"}]}))])
    text, url = _run(rw.mac_source(tools, s, "scholar", "graph memory"))
    assert "Graph memory" in text and url == "http://ss/1"


def test_mac_source_none_for_mac_only_source_when_absent():
    # qa has no direct-HTTP fallback → absent Mac tool yields None (skipped, not an error blob).
    assert _run(rw.mac_source(MacTools({}), FakeSession([]), "qa", "how to sharpen a knife")) is None


def test_mac_source_drops_gdelt_throttle():
    tools = MacTools({"events_search": "rate-limited, try again"})
    assert _run(rw.mac_source(tools, FakeSession([]), "events", "election")) is None


def test_mac_source_drug_uses_name_arg():
    tools = MacTools({"drug_info": "Sugammadex label: dosing…"})
    out = _run(rw.mac_source(tools, FakeSession([]), "drug", "sugammadex"))
    assert out and out[0].startswith("Sugammadex")
    assert tools.calls[0] == ("drug_info", {"name": "sugammadex"})


def test_mac_source_archive_routes_to_archive_search():
    tools = MacTools({"archive_search": "Archived pages about the Vinkona 500…"})
    out = _run(rw.mac_source(tools, FakeSession([]), "archive", "vinkona 500", max_items=5))
    assert out and "Archived pages" in out[0]
    assert tools.calls[0] == ("archive_search", {"query": "vinkona 500"})   # arg is query(+mediatype?), no limit


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
