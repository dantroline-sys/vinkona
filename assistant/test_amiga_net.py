#!/usr/bin/env python
"""Vinkona's egress broker (amiga_net) — SAME contract as Vinur's, plus the
async BrokerSession lane its aiohttp research/wikipedia egress needs.

The sync lane runs against a real loopback server; the async lane is driven
with a fake raw session (aiohttp isn't a hard dep of the tests) so the policy
gate + audit are proven without a live aiohttp."""
import asyncio
import json
import os
import sys
import tempfile
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["AMIGA_FETCH_ENGINE"] = "stdlib"

from assistant.amiga_net import audit, broker, policy, status  # noqa: E402

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


BODY = b'{"query": "ok", "items": [1, 2, 3]}'


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
Thread(target=srv.serve_forever, daemon=True).start()
PORT = srv.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"

TD = Path(tempfile.mkdtemp())
POL = TD / "egress.toml"
POL.write_text(f"""
[[rule]]
name = "research"
hosts = ["127.0.0.1"]
port = {PORT}
methods = ["GET"]
purpose = "test research"
ttl_seconds = 60
max_uses = 100
""")
policy.POLICY_PATH = POL
policy.LEASE_DIR = TD / "run"
audit.LOG_PATH = TD / "egress.jsonl"


def verdicts():
    return [e["verdict"] for e in audit.tail(200)]


# ── shipped policy is lease-only, same format as Vinur ──────────────────────
prod = policy.load(Path(__file__).resolve().parent / "egress.toml")
assert prod, "assistant/egress.toml must parse"
assert all(r.leased for r in prod), "Vinkona ships NO standing egress rules"
assert all(r.purpose for r in prod), "every rule carries a plain-language purpose"
names = {r.name for r in prod}
assert {"research", "wikipedia"} <= names, names
ok("shipped egress.toml: parses, lease-only, research + wikipedia rules present")

# ── deny-by-default ──────────────────────────────────────────────────────────
try:
    broker.request("nope", "https://evil.example.com/x")
    raise AssertionError("must deny")
except broker.EgressDenied as e:
    assert "no rule" in str(e)
assert verdicts()[-1] == "DENIED"
ok("deny-by-default: an unlisted destination is refused and audited")

# ── leased rule: nothing until opened ───────────────────────────────────────
try:
    broker.request("no lease", f"{BASE}/x")
    raise AssertionError("must deny without a lease")
except broker.EgressDenied as e:
    assert "lease" in str(e)
with broker.lease("a research turn", "research"):
    assert broker.request("arxiv", f"{BASE}/x") == BODY
v = verdicts()
assert v.count("LEASE_OPEN") == 1 and v.count("LEASE_CLOSE") == 1
assert policy.live_leases(policy.load()) == []
ok("leased rule: grants nothing between operations; open→allowed→closed paired")


# ── the async BrokerSession lane (fake raw session — proves gate + audit) ───
class FakeResp:
    def __init__(self, status=200, n=len(BODY)):
        self.status = status
        self.headers = {"Content-Length": str(n)}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return BODY.decode()


class FakeRaw:
    """Enough of aiohttp.ClientSession for BrokerSession to drive."""
    def __init__(self):
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return FakeResp()


async def _async_case():
    raw = FakeRaw()
    bs = broker.BrokerSession(raw, "research: arxiv")
    # inside a lease, an allowed host returns the response AND audits ALLOWED
    with broker.lease("research turn", "research"):
        async with bs.get(f"{BASE}/api") as r:
            assert r.status == 200
            assert await r.text() == BODY.decode()
    assert raw.calls == [f"{BASE}/api"]
    assert [e for e in audit.tail(5) if e["verdict"] == "ALLOWED"], "async GET audited"
    # a denied host raises before the raw session is ever touched
    raw2 = FakeRaw()
    bs2 = broker.BrokerSession(raw2, "research: evil")
    try:
        bs2.get("https://evil.example.com/x")
        raise AssertionError("denied host must raise at .get()")
    except broker.EgressDenied:
        pass
    assert raw2.calls == [], "a denied request never reaches the network"


asyncio.run(_async_case())
ok("BrokerSession: allowed GET returns + audits under lease; denied host "
   "raises at .get() and never touches the raw session")

# ── kill switch + revoke (same as Vinur) ────────────────────────────────────
policy.set_rule_enabled("research", False, POL)
try:
    with broker.lease("x", "research"):
        pass
    raise AssertionError("disabled rule must refuse a lease")
except broker.EgressDenied as e:
    assert "disabled" in str(e)
policy.set_rule_enabled("research", True, POL)
with broker.lease("re-enabled", "research"):
    assert broker.request("ok", f"{BASE}/x") == BODY
ok("kill switch: disabled rule refuses leases; re-enable restores egress")

# ── traffic rollup ───────────────────────────────────────────────────────────
stats = audit.summarize()
research = next(x for x in stats["rules"] if x["rule"] == "research")
assert research["requests"] > 0 and research["bytes_in"] > 0
assert not any("items" in json.dumps(x) for x in stats["rules"]), "counts, not bodies"
ok("summarize(): per-rule requests/bytes, never content")

# ── audit hygiene: no bodies ─────────────────────────────────────────────────
raw_log = audit.LOG_PATH.read_text()
assert "items" not in raw_log and "query" not in raw_log
ok("the audit log holds no request/response bodies")

# ── status renders ───────────────────────────────────────────────────────────
out = status.render(10)
assert "deny by default" in out and "test research" in out
ok("status: policy in plain language + recent events")

# ── credentials never ride argv (world-readable /proc/*/cmdline) ─────────────
import subprocess as _sp
from pathlib import Path as _P
captured = {}
def _fake_run(cmd, check=True, timeout=None, env=None):
    captured["cmd"] = list(cmd)
    captured["env"] = dict(env) if env else None
    conf = next((c.split("=", 1)[1] for c in cmd if str(c).startswith("--conf-path=")), None)
    captured["conf_text"] = _P(conf).read_text() if conf else ""
    captured["conf_mode"] = (_P(conf).stat().st_mode & 0o777) if conf else None
    if env and env.get("WGETRC"):
        captured["conf_text"] = _P(env["WGETRC"]).read_text()
        captured["conf_mode"] = _P(env["WGETRC"]).stat().st_mode & 0o777
    return None
_real_run = broker.subprocess.run
broker.subprocess.run = _fake_run
try:
    hdrs = {"Authorization": "Bearer hf_SECRET123", "User-Agent": "amiga"}
    broker._dl_aria2c("https://x/y.bin", _P("/tmp/amiga-test-dl.bin"), hdrs, 5)
    assert not any("hf_SECRET123" in str(c) for c in captured["cmd"]), "token on argv!"
    assert any(str(c).startswith("--conf-path=") for c in captured["cmd"])
    assert "hf_SECRET123" in captured["conf_text"] and captured["conf_mode"] == 0o600
    assert any("User-Agent" in str(c) for c in captured["cmd"]), "plain headers stay on argv"
    ok("aria2c: bearer token via 0600 conf file, never argv")
    broker._dl_wget("https://x/y.bin", _P("/tmp/amiga-test-dl.bin"), hdrs, 5)
    assert not any("hf_SECRET123" in str(c) for c in captured["cmd"])
    assert "hf_SECRET123" in captured["conf_text"] and captured["conf_mode"] == 0o600
    ok("wget: bearer token via 0600 WGETRC, never argv")
finally:
    broker.subprocess.run = _real_run

# ── the configured proxy applies to broker traffic (sync + async) ───────────
# A proxy set on the Network tab must steer request()/download() AND the async
# BrokerSession lane — urllib reads env only and aiohttp reads nothing, so the
# config keys were silently ignored and brokered calls went direct.
POL.write_text(POL.read_text() + f"""
[[rule]]
name = "proxytest"
hosts = ["proxytest.example"]
port = 443
methods = ["GET"]
purpose = "proxy injection check"
ttl_seconds = 60
max_uses = 10

[[rule]]
name = "huggingface"
hosts = ["huggingface.co"]
port = 443
methods = ["GET"]
purpose = "weights gate check"
ttl_seconds = 60
max_uses = 10
""")
_real_cfg = broker._config
broker._config = lambda: {"http_proxy": "http://proxy.corp:3128",
                          "https_proxy": "http://proxy.corp:3129",
                          "no_proxy": "corp.internal"}
try:
    env = broker._proxy_env()
    assert env["https_proxy"] == "http://proxy.corp:3129" and env["HTTP_PROXY"]
    assert "127.0.0.1" in env["no_proxy"] and "corp.internal" in env["no_proxy"]
    assert broker._proxy_for("https://proxytest.example/x") == "http://proxy.corp:3129"
    assert broker._proxy_for("http://proxytest.example/x") == "http://proxy.corp:3128"
    assert broker._proxy_for("https://a.corp.internal/x") == ""
    assert broker._proxy_for(f"{BASE}/x") == "", "loopback is always exempt"

    class FakeRawKw(FakeRaw):
        def get(self, url, **kw):
            self.calls.append((url, kw))
            return FakeResp()

    async def _proxy_case():
        raw = FakeRawKw()
        bs = broker.BrokerSession(raw, "proxy check")
        with broker.lease("proxy check", "proxytest"):
            async with bs.get("https://proxytest.example/api") as r:
                assert r.status == 200
        (url, kw), = raw.calls
        assert kw.get("proxy") == "http://proxy.corp:3129", kw

    asyncio.run(_proxy_case())

    captured2 = {}
    _rr = broker.subprocess.run
    broker.subprocess.run = lambda cmd, check=True, timeout=None, env=None: \
        captured2.update(cmd=list(cmd), env=dict(env or {}))
    try:
        broker._dl_aria2c("https://x/y.bin", Path(TD / "p.bin"), {}, 5)
        assert captured2["env"].get("https_proxy") == "http://proxy.corp:3129"
        assert "127.0.0.1" in captured2["env"].get("no_proxy", "")
        broker._dl_wget("https://x/y.bin", Path(TD / "p.bin"), {}, 5)
        assert captured2["env"].get("http_proxy") == "http://proxy.corp:3128"
    finally:
        broker.subprocess.run = _rr
finally:
    broker._config = _real_cfg
ok("proxy: Network-tab mapping steers sync + async + engine lanes; "
   "no_proxy + loopback exempt")

# ── chatterbox weights ride the broker (cached-else-lease, like qwen3) ──────
import contextlib as _ctx
import importlib.util as _ilu
import types as _types

sys.modules.setdefault("amiga_net", sys.modules["assistant.amiga_net"])
_spec = _ilu.spec_from_file_location(
    "tts_chatterbox", Path(__file__).resolve().parent / "tts_chatterbox.py")
_cbx = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cbx)

_hub = _types.ModuleType("huggingface_hub")
_snap_dir = TD / "snap"
_snap_dir.mkdir(exist_ok=True)
_hub.snapshot_download = lambda repo, local_files_only=False: str(_snap_dir)
_prev_hub = sys.modules.get("huggingface_hub")
sys.modules["huggingface_hub"] = _hub
_prev_offline = os.environ.pop("HF_HUB_OFFLINE", None)
try:
    # incomplete cache (files missing) -> a broker lease, audited on entry
    gate = _cbx._weights_gate("ResembleAI/chatterbox")
    assert not isinstance(gate, _ctx.nullcontext), "incomplete cache must lease"
    with gate:
        pass
    assert any(e["verdict"] == "LEASE_OPEN" and "chatterbox" in e.get("purpose", "")
               for e in audit.tail(5)), "the weights lease must be audited"
    # complete cache -> offline load, no lease, HF_HUB_OFFLINE pinned
    for f in _cbx._WEIGHT_FILES:
        (_snap_dir / f).write_text("x")
    gate = _cbx._weights_gate("ResembleAI/chatterbox")
    assert isinstance(gate, _ctx.nullcontext), "complete cache loads offline"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    # disabled rule -> the download is DENIED before any bytes move
    for f in _cbx._WEIGHT_FILES:
        (_snap_dir / f).unlink()
    policy.set_rule_enabled("huggingface", False, POL)
    try:
        with _cbx._weights_gate("ResembleAI/chatterbox"):
            raise AssertionError("disabled rule must deny the weights download")
    except broker.EgressDenied:
        pass
    policy.set_rule_enabled("huggingface", True, POL)
finally:
    if _prev_hub is not None:
        sys.modules["huggingface_hub"] = _prev_hub
    else:
        sys.modules.pop("huggingface_hub", None)
    if _prev_offline is not None:
        os.environ["HF_HUB_OFFLINE"] = _prev_offline
ok("chatterbox weights: complete cache loads offline; else an audited "
   "huggingface lease; a disabled rule DENIES the download")

srv.shutdown()
print(f"test_amiga_net: {OK} checks OK")
