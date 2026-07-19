"""Research export, remote lane (folder = "http://host:8771").

Covers the split-machine hand-off semantics: drops POST to the knowledge
host's /drop route; a mid-run network failure aborts WITHOUT advancing the
rowid watermark (so the next run retries — byte-identical re-posts are host
no-ops); a clean run advances it exactly like the filesystem lane.

Run:  python3 test_research_export_http.py     (stdlib only)
"""
import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import research_export as rexport

CHECKS = 0


def ok(label):
    global CHECKS
    CHECKS += 1
    print(f"  ok {CHECKS}  {label}")


class FakeMemory:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE documents (url TEXT, title TEXT, topic TEXT, "
                        "text TEXT, digest TEXT, card_hint TEXT, fetched_at REAL, kind TEXT)")
        self.state = {}

    def add(self, topic, text):
        self.db.execute("INSERT INTO documents VALUES ('u', 't', ?, ?, '', NULL, 0, 'research')",
                        (topic, text))

    def get_state(self, k):
        return self.state.get(k)

    def set_state(self, k, v):
        self.state[k] = v


class DropHost(BaseHTTPRequestHandler):
    fail = False
    seen = []
    bodies = {}
    gmode = "ok"          # handshake behavior: ok | noaccept | 401 | 404
    inventory = {}
    gaps = []

    def _json(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if DropHost.fail:
            self.send_response(500)
            self.end_headers()
            return
        DropHost.seen.append((req["name"], self.headers.get("Authorization")))
        DropHost.bodies[req["name"]] = req["content"]
        self._json(json.dumps({"ok": True, "changed": True}).encode())

    def do_GET(self):
        if DropHost.gmode in ("401", "404"):
            self.send_response(int(DropHost.gmode))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if DropHost.gmode == "noaccept":
            self._json(json.dumps({"ok": True, "accepts": False,
                                   "reason": "research_solved_dir is not configured"}).encode())
            return
        self._json(json.dumps({"ok": True, "accepts": True,
                               "drops": DropHost.inventory,
                               "gaps": DropHost.gaps}).encode())

    def log_message(self, *_):
        pass


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DropHost)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    mem = FakeMemory()
    mem.add("how to season cast iron", "long answer " * 50)
    mem.add("how to season cast iron", "second doc")
    mem.add("mail", "PERSONAL — never exported")   # excluded by crawl_sources

    # failure first: nothing lands, watermark must NOT move
    DropHost.fail = True
    res = rexport.export_research(mem, url, ["mail"], token="tok")
    assert res["ok"] is False and "drop failed" in res["error"], res
    assert mem.get_state("research.export_watermark") is None, "watermark advanced on failure"
    ok("mid-run failure aborts without advancing the watermark")

    # clean run: one drop (both docs share the question), Bearer sent, watermark set
    DropHost.fail = False
    res = rexport.export_research(mem, url, ["mail"], token="tok")
    assert res["ok"] is True and res["written"] == 1 and res["questions"] == 1, res
    name, auth = DropHost.seen[0]
    assert name == rexport.question_hash("how to season cast iron") + ".md"
    assert auth == "Bearer tok"
    assert int(mem.get_state("research.export_watermark")) >= 2
    ok("clean remote run: one drop per question, Bearer token, watermark advanced")

    # incremental: nothing new -> no posts
    before = len(DropHost.seen)
    res = rexport.export_research(mem, url, ["mail"], token="tok")
    assert res["ok"] is True and len(DropHost.seen) == before, res
    ok("incremental no-op posts nothing")

    # ── smart routing: resolve_export_target ────────────────────────────
    R = rexport.resolve_export_target
    t = R({"research": {"export": {"folder": "http://x:1"}}})
    assert (t["mode"], t["dest"]) == ("http", "http://x:1"), t
    t = R({"research": {"export": {"folder": "/tmp/out"}},
           "knowledge_host": {"enabled": True, "url": "http://box:8771", "token": "kh"}})
    assert (t["mode"], t["dest"], t["token"]) == ("http", "http://box:8771", "kh"), t
    assert t["fallback_folder"] == "/tmp/out"
    t = R({"research": {"export": {"folder": "/tmp/out"}},
           "knowledge_host": {"enabled": True, "url": "http://127.0.0.1:8771"}})
    assert (t["mode"], t["dest"]) == ("folder", "/tmp/out"), t
    t = R({"knowledge_host": {"enabled": True, "url": "http://127.0.0.1:8771", "token": "k"}})
    assert t["mode"] == "http", t
    t = R({"research": {"export": {"folder": "/tmp/out", "transport": "folder"}},
           "knowledge_host": {"enabled": True, "url": "http://box:8771"}})
    assert t["mode"] == "folder", t
    assert R({})["mode"] == "off"
    ok("routing: URL wins; REMOTE host beats a local outbox; pins honored; off when nothing")

    # ── run_export: handshake inventory skips shipped bytes ─────────────
    def kh_cfg(extra_export=None):
        exp = {"transport": "http", "token": "tok"}
        exp.update(extra_export or {})
        return {"research": {"export": exp},
                "knowledge_host": {"enabled": True, "url": url}}

    name = rexport.question_hash("how to season cast iron") + ".md"
    DropHost.inventory = {name: rexport._hash16(DropHost.bodies[name])}
    before = len(DropHost.seen)
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["ok"] is True and res["skipped"] == 1 and len(DropHost.seen) == before, res
    assert res["transport"] == "http", res
    ok("handshake inventory: an already-held drop re-exports with zero bytes shipped")

    DropHost.inventory = {name: "0000000000000000"}          # host holds a STALE copy
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["written"] == 1 and len(DropHost.seen) == before + 1, res
    ok("handshake inventory: a stale copy on the host re-ships")

    DropHost.gmode = "404"                                   # older host: no handshake route
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["ok"] is True and len(DropHost.seen) == before + 2, res
    ok("no handshake route (older host): POSTs blind, back-compatible")

    DropHost.gmode = "401"
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["ok"] is False and "token" in res["error"], res
    ok("handshake denied: clear token error, nothing shipped")

    DropHost.gmode = "noaccept"
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["ok"] is False and "can't store" in res["error"], res
    ok("host can't store (no solved dir): surfaced, watermark untouched")
    DropHost.gmode = "ok"

    # ── the return leg: open gaps ride the handshake, verbatim ──────────
    DropHost.gaps = [{"query": "How do  plasmids replicate?", "count": 3, "intent": "ask"},
                     {"query": "how do  plasmids replicate?", "count": 1},   # dupe (case)
                     {"query": "  "}, "bare string gap"]
    res = rexport.run_export(mem, kh_cfg(), ["mail"], full=True)
    assert res["gaps"] == ["How do  plasmids replicate?", "bare string gap"], res["gaps"]
    ok("gaps: deduped case-insensitively, VERBATIM text preserved (close_gap contract)")

    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        folder_cfg = {"research": {"export": {"folder": td}},
                      "knowledge_host": {"enabled": True, "url": url, "token": "tok"}}
        res = rexport.run_export(mem, folder_cfg, ["mail"], full=True)
        assert res["transport"] == "folder" and os.path.exists(os.path.join(td, name)), res
        assert res["gaps"] == ["How do  plasmids replicate?", "bare string gap"], res
    ok("folder transport still handshakes: local setups get the gap return leg too")
    DropHost.gaps = []

    # ── down host: auto mode falls back to the folder outbox ────────────
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        down_cfg = {"research": {"export": {"folder": td, "token": "tok"}},
                    "knowledge_host": {"enabled": True, "url": "http://box:9"}}
        neg0 = rexport.negotiate_drop
        rexport.negotiate_drop = lambda *a, **k: ("down", None)
        try:
            res = rexport.run_export(mem, down_cfg, ["mail"], full=True)
        finally:
            rexport.negotiate_drop = neg0
        assert res["ok"] is True and res["transport"] == "folder", res
        assert os.path.exists(os.path.join(td, name)), "fallback drop not written"
    ok("remote host down: falls back to the folder outbox (mined when host returns)")

    httpd.shutdown()
    print(f"test_research_export_http: {CHECKS} checks OK")


if __name__ == "__main__":
    sys.exit(main())
