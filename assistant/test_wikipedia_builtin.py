"""Built-in online Wikipedia lookup — the no-tool-host fallback.

Covers the "auto" offer rule (offered exactly when the tool host is disabled),
the lookup + summary round-trip against stub endpoints, and the failure shapes
(no match, HTTP error) — all without touching the real Wikipedia.

Run:  python3 test_wikipedia_builtin.py     (needs aiohttp, like the cascade)
"""
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import llm_bridge as lb

CHECKS = 0


def ok(label):
    global CHECKS
    CHECKS += 1
    print(f"  ok {CHECKS}  {label}")


class Wiki(BaseHTTPRequestHandler):
    """Stub of the two endpoints wiki_lookup speaks to."""
    broken = False

    def do_GET(self):
        u = urlparse(self.path)
        if Wiki.broken:
            self.send_response(503)
            self.end_headers()
            return
        if u.path == "/w/api.php":
            q = parse_qs(u.query).get("search", [""])[0]
            hits = (["Marie Curie", "Curie family", "Pierre Curie"]
                    if "curie" in q.lower() else [])
            body = json.dumps([q, hits, [], []]).encode()
        elif u.path.startswith("/summary/"):
            body = json.dumps({"title": "Marie Curie",
                               "description": "Polish-French physicist and chemist",
                               "extract": "Marie Curie discovered polonium and radium."}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    # ── the "auto" offer rule ────────────────────────────────────────────
    assert lb.resolve_wikipedia_flag({"enabled": False}) is True
    assert lb.resolve_wikipedia_flag({}) is True
    assert lb.resolve_wikipedia_flag({"enabled": True}) is False
    assert lb.resolve_wikipedia_flag({"enabled": True, "wikipedia": True}) is True
    assert lb.resolve_wikipedia_flag({"enabled": False, "wikipedia": False}) is False
    assert lb.resolve_wikipedia_flag({"enabled": True, "wikipedia": "auto"}) is False
    ok("auto: offered exactly when the tool host is off; true/false override")

    # ── lookup round-trip against the stub ───────────────────────────────
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Wiki)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    out = asyncio.run(lb.wiki_lookup("marie curie",
                                     api_base=base + "/w/api.php",
                                     summary_base=base + "/summary/"))
    assert out.startswith("Wikipedia — Marie Curie"), out
    assert "Polish-French physicist" in out
    assert "discovered polonium" in out
    assert "Other matching articles: Curie family; Pierre Curie" in out
    ok("lookup: title + description + extract + alternatives")

    out = asyncio.run(lb.wiki_lookup("zzz-nothing",
                                     api_base=base + "/w/api.php",
                                     summary_base=base + "/summary/"))
    assert "no article matching" in out
    ok("no match reported gently")

    Wiki.broken = True
    out = asyncio.run(lb.wiki_lookup("marie curie",
                                     api_base=base + "/w/api.php",
                                     summary_base=base + "/summary/"))
    assert "failed: HTTP 503" in out
    ok("endpoint failure surfaces as a message, never a crash")
    httpd.shutdown()

    # ── the tool is catalogued/dispatched only when on ───────────────────
    assert lb.SEARCH_WIKIPEDIA_TOOL["function"]["name"] == "search_wikipedia"
    assert "query" in lb.SEARCH_WIKIPEDIA_TOOL["function"]["parameters"]["properties"]
    ok("tool spec shape")

    print(f"test_wikipedia_builtin: {CHECKS} checks OK")


if __name__ == "__main__":
    sys.exit(main())
