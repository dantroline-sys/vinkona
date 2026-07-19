#!/usr/bin/env python3
"""The launcher's easy-mode CLI verbs (supervisor.py): config-patch /
vinur-probe / preflight --json / logtail.  The desktop launcher is a thin
shell over exactly these, so this is where their behavior is pinned.

Run:  python3 test_easy_cli.py   (stdlib only)
"""
import contextlib
import io
import json
import pathlib
import tempfile

import supervisor as sup

N = [0]


def ok(desc):
    N[0] += 1
    print(f"  ok {N[0]:2d}  {desc}")


def run(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, json.loads(buf.getvalue())


# ── config-patch: deep merge, atomic, creates the file ───────────────────────
with tempfile.TemporaryDirectory() as td:
    dir0 = sup.DIR
    sup.DIR = pathlib.Path(td)
    try:
        rc, res = run(sup.cmd_config_patch,
                      json.dumps({"knowledge_host": {"enabled": True, "url": "http://b:8771"}}))
        assert rc == 0 and res["ok"], res
        cfg = json.load(open(pathlib.Path(td) / "config" / "config.json"))
        assert cfg["knowledge_host"]["url"] == "http://b:8771"
        ok("config-patch: creates config.json with the fragment")

        rc, res = run(sup.cmd_config_patch,
                      json.dumps({"knowledge_host": {"token": "s3"},
                                  "research": {"export": {"enabled": True}}}))
        cfg = json.load(open(pathlib.Path(td) / "config" / "config.json"))
        assert cfg["knowledge_host"] == {"enabled": True, "url": "http://b:8771",
                                         "token": "s3"}, cfg
        assert cfg["research"]["export"]["enabled"] is True
        ok("config-patch: DEEP merge — sibling keys survive, nested added")

        rc, res = run(sup.cmd_config_patch, "not json {")
        assert rc == 1 and not res["ok"] and "bad patch" in res["error"]
        rc, res = run(sup.cmd_config_patch, '["not", "an", "object"]')
        assert rc == 1 and "object" in res["error"]
        cfg = json.load(open(pathlib.Path(td) / "config" / "config.json"))
        assert cfg["knowledge_host"]["token"] == "s3", "bad patch must not touch the file"
        ok("config-patch: malformed input rejected, file untouched")
    finally:
        sup.DIR = dir0

# ── vinur-probe: arbitrary url/token, same states as vinur_link ──────────────
import http.server
import threading


class _KH(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"{}"
        elif self.path == "/drop" and self.headers.get("Authorization") == "Bearer good":
            body = json.dumps({"ok": True, "accepts": True, "count": 2,
                               "drops": {}, "gaps": []}).encode()
        else:
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _KH)
threading.Thread(target=srv.serve_forever, daemon=True).start()
kurl = f"http://127.0.0.1:{srv.server_address[1]}"

assert sup.probe_kb(kurl, "good")["state"] == "up"
assert sup.probe_kb(kurl, "bad")["state"] == "no-auth"
srv.shutdown()
assert sup.probe_kb("http://127.0.0.1:1", "x")["state"] == "down"
ok("vinur-probe core: up / no-auth / down for arbitrary url+token")

# ── logtail ──────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    logs0 = sup.LOGS
    sup.LOGS = pathlib.Path(td)
    try:
        (pathlib.Path(td) / "fetch_models.log").write_text(
            "\n".join(f"line {i}" for i in range(100)))
        rc, res = run(sup.cmd_logtail, ["fetch_models", "3"])
        assert rc == 0 and res["text"] == "line 97\nline 98\nline 99", res
        rc, res = run(sup.cmd_logtail, ["../etc/passwd"])
        assert rc == 1 and not res["ok"]
        rc, res = run(sup.cmd_logtail, ["nope"])
        assert rc == 1
        ok("logtail: last-N lines; traversal and missing logs refused")
    finally:
        sup.LOGS = logs0

print(f"test_easy_cli: {N[0]} checks OK")
