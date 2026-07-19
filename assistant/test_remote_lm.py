#!/usr/bin/env python3
"""Remote LM tiers (big_lm.remote = true — served by another machine, e.g. the
Vinur GPU box's vLLM): no local service, no model preflight, friendly refusal
from llm_server, and both status views list the tier with reachability.

Run:  python3 test_remote_lm.py   (stdlib only, no services needed)
"""
import http.server
import json
import threading
import unittest.mock as mock

import supervisor as sup
import llm_server


N = [0]


def ok(desc):
    N[0] += 1
    print(f"  ok {N[0]:2d}  {desc}")


def svc_names(mode, topo, cfg):
    return [s["name"] for s in sup.services_for(mode, topo, cfg)]


# ── services_for: remote tiers get no local service ──────────────────────────
base = {"big_lm": {"url": "http://kb-box:11438", "model": "big", "remote": True}}

names = svc_names("normal", {}, base)
assert "big_lm" not in names, names
assert "fast_lm" in names and "embed" in names, names
ok("services_for: remote big_lm skipped, other tiers untouched")

names = svc_names("knowledge", {}, base)
assert "big_lm" not in names and "big_lm2" not in names, names
ok("knowledge mode: big_lm skipped; big_lm2 inherits remote from big_lm")

names = svc_names("knowledge", {}, {"big_lm": {"url": "http://kb-box:11438",
                                               "model": "big", "remote": True},
                                    "big_lm2": {"remote": False}})
assert "big_lm2" in names and "big_lm" not in names, names
ok("big_lm2 can override the inherited remote flag")

names = svc_names("normal", {}, {"big_lm": {"url": "http://127.0.0.1:11438"}})
assert "big_lm" in names, names
ok("no remote flag: big_lm launches locally as before")

# ── llm_server: launching a remote tier refuses with the reason ──────────────
cfg = {"big_lm": {"url": "http://kb-box:11438", "model": "big", "remote": True}}
try:
    llm_server.build_command(cfg, "big_lm")
    raise AssertionError("build_command should exit on a remote tier")
except SystemExit as e:
    assert "remote" in str(e) and "kb-box" in str(e), e
ok("llm_server.build_command: remote tier exits with a clear message")

# ── missing_models: a remote tier's model is a name, not a file ──────────────
remote_cfg = {"models_dir": "Models",
              "fast_lm": {"url": None},
              "big_lm": {"url": "http://kb-box:11438",
                         "model": "definitely-not-a-file", "remote": True}}
fake_mod = mock.Mock()
fake_mod.load_config.return_value = remote_cfg
with mock.patch("importlib.util.module_from_spec", return_value=fake_mod), \
     mock.patch.object(fake_mod, "load_config", return_value=remote_cfg):
    with mock.patch("importlib.util.spec_from_file_location") as spec:
        spec.return_value.loader.exec_module = lambda m: None
        miss = sup.missing_models()
assert not [m for m in miss if m[0] == "big_lm"], miss
ok("missing_models: remote tier not treated as a missing GGUF")

# ── remote_tiers: probes /health, drives both status views ───────────────────
class _H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]

cfg = {"big_lm": {"url": f"http://127.0.0.1:{port}", "model": "big", "remote": True}}
tiers = sup.remote_tiers(cfg)
assert tiers == [("big_lm", True, f"http://127.0.0.1:{port}")], tiers
ok("remote_tiers: /health answering -> (tier, True, url)")

srv.shutdown()
cfg_down = {"big_lm": {"url": "http://127.0.0.1:1", "model": "big", "remote": True}}
tiers = sup.remote_tiers(cfg_down)
assert tiers == [("big_lm", False, "http://127.0.0.1:1")], tiers
ok("remote_tiers: unreachable server -> False (status will say so)")

with mock.patch.object(sup, "supervisor_pid", return_value=4242), \
     mock.patch.object(sup, "load_config", return_value=cfg_down), \
     mock.patch.object(sup, "load_topo", return_value={}), \
     mock.patch.object(sup.json, "load", return_value={"services": {}}), \
     mock.patch("builtins.open", mock.mock_open(read_data="{}")):
    payload = sup.status_payload()
rem = [s for s in payload["services"] if s["name"] == "big_lm"]
assert rem and rem[0]["up"] is False and "remote" in rem[0]["detail"], payload
assert rem[0]["pid"] is None
ok("status --json: remote tier listed for the launcher (up=false, detail)")

print(f"test_remote_lm: {N[0]} checks OK")
