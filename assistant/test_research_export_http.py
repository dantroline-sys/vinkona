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

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if DropHost.fail:
            self.send_response(500)
            self.end_headers()
            return
        DropHost.seen.append((req["name"], self.headers.get("Authorization")))
        body = json.dumps({"ok": True, "changed": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    httpd.shutdown()
    print(f"test_research_export_http: {CHECKS} checks OK")


if __name__ == "__main__":
    sys.exit(main())
