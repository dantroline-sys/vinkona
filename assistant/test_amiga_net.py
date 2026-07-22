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

srv.shutdown()
print(f"test_amiga_net: {OK} checks OK")
