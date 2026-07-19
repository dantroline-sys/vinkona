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

# ── resolve_remote_lms: reconcile names with what the server serves ──────────
import config as cfgmod


class _Models(http.server.BaseHTTPRequestHandler):
    served = ["served-big"]
    hits = [0]

    def do_GET(self):
        type(self).hits[0] += 1
        body = json.dumps({"data": [{"id": i} for i in self.served]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


msrv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Models)
threading.Thread(target=msrv.serve_forever, daemon=True).start()
murl = f"http://127.0.0.1:{msrv.server_address[1]}"
logs = []

cfgmod._REMOTE_MODEL_CACHE.clear()
cfgmod._REMOTE_PROBE_FAILED.clear()
c = {"big_lm": {"remote": True, "url": murl, "model": "stale-gguf-name"}}
cfgmod.resolve_remote_lms(c, log=logs.append)
assert c["big_lm"]["model"] == "served-big", c
assert logs and "served-big" in logs[0] and "stale-gguf-name" in logs[0], logs
ok("resolve_remote_lms: single served id adopted (logged with both names)")

hits_before = _Models.hits[0]
c2 = {"big_lm": {"remote": True, "url": murl, "model": "stale-gguf-name"}}
cfgmod.resolve_remote_lms(c2, log=logs.append)
assert c2["big_lm"]["model"] == "served-big" and _Models.hits[0] == hits_before
ok("memoized: per-connection reloads cost zero HTTP after the first")

cfgmod._REMOTE_MODEL_CACHE.clear()
c3 = {"big_lm": {"remote": True, "url": murl, "model": "served-big"}}
cfgmod.resolve_remote_lms(c3, log=logs.append)
assert c3["big_lm"]["model"] == "served-big"
ok("configured name already served: kept")

cfgmod._REMOTE_MODEL_CACHE.clear()
_Models.served = ["model-a", "model-b"]
logs.clear()
c4 = {"big_lm": {"remote": True, "url": murl, "model": "stale-gguf-name"}}
cfgmod.resolve_remote_lms(c4, log=logs.append)
assert c4["big_lm"]["model"] == "stale-gguf-name"
assert logs and "model-a" in logs[0] and "model-b" in logs[0], logs
cfgmod.resolve_remote_lms(dict(c4), log=logs.append)
assert len(logs) == 1, logs
ok("several served ids: name kept, servers listed, warned once")

# TTL re-validation: the box changes what it serves; a stale cache heals.
cfgmod._REMOTE_MODEL_CACHE.clear()
cfgmod._REMOTE_PROBE_FAILED.clear()
_Models.served = ["new-big"]
key = (murl, "stale-gguf-name")
cfgmod._REMOTE_MODEL_CACHE[key] = ("old-big", 0.0)          # long expired
c6 = {"big_lm": {"remote": True, "url": murl, "model": "stale-gguf-name"}}
cfgmod.resolve_remote_lms(c6, log=logs.append)
assert c6["big_lm"]["model"] == "new-big", c6
ok("TTL: expired cache re-probes and adopts the box's new single model")

_Models.served = ["old-big", "other"]
cfgmod._REMOTE_MODEL_CACHE[key] = ("old-big", 0.0)          # adopted, expired
c7 = {"big_lm": {"remote": True, "url": murl, "model": "stale-gguf-name"}}
cfgmod.resolve_remote_lms(c7, log=logs.append)
assert c7["big_lm"]["model"] == "old-big", c7
ok("TTL: adopted name still served among several -> kept, no regression")

msrv.shutdown()
cfgmod._REMOTE_MODEL_CACHE.clear()
cfgmod._REMOTE_PROBE_FAILED.clear()
c5 = {"big_lm": {"remote": True, "url": "http://127.0.0.1:1", "model": "x"},
      "fast_lm": {"url": "http://127.0.0.1:11435", "model": "y"}}
cfgmod.resolve_remote_lms(c5, log=logs.append)
assert c5["big_lm"]["model"] == "x" and c5["fast_lm"]["model"] == "y"
assert "http://127.0.0.1:1" in cfgmod._REMOTE_PROBE_FAILED, "down server backs off"
assert "http://127.0.0.1:11435" not in cfgmod._REMOTE_PROBE_FAILED, \
    "local (non-remote) tier must never be probed"
ok("unreachable server: names untouched, backoff set; local tiers ignored")

print(f"test_remote_lm: {N[0]} checks OK")
