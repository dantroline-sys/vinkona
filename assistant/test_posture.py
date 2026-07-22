#!/usr/bin/env python
"""Vinkona posture (B-18..23): the parsers and each grader's bands, with the
contract's core rule held throughout — UNKNOWN IS NEVER A PASS.  The cascade's
LAN bind is graded by whether its first-frame token auth is on."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import posture  # noqa: E402
from assistant.amiga_net import policy  # noqa: E402

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


TCP = (
    "  sl  local_address rem_address   st ...\n"
    "   0: 0100007F:2322 00000000:0000 0A ...\n"          # 127.0.0.1:8994? loopback
    "   1: 00000000:2326 00000000:0000 0A ...\n"          # 0.0.0.0:8998 wildcard
    "   2: 0100007F:AAAA 00000000:0000 01 ...\n")         # ESTABLISHED, not LISTEN
rows = posture.parse_proc_tcp(TCP)
assert ("loopback", 0x2322) in rows and ("wildcard", 0x2326) in rows
assert not any(p == 0xAAAA for _, p in rows)
ok("parse_proc_tcp: LISTEN only, loopback vs wildcard")

CFG = {"server": {"host": "0.0.0.0", "port": 8998, "auth": {"require_auth": True}},
       "config_server": {"port": 8090}, "tts": {"port": 11436}}
ep = posture.expected_ports(CFG)
assert ep[8998] == ("cascade WSS (phone)", True)
assert ep[8090] == ("config server", False)
assert posture.cascade_auth_on(CFG) is True
assert posture.cascade_auth_on({"server": {"auth": {"require_auth": False}}}) is False
ok("expected_ports: cascade needs_auth, config server loopback-only")

# cascade LAN-bound WITH token = declared/warn; WITHOUT = bad
warn = posture.check_listeners(CFG, [("wildcard", 8998)])
assert warn["state"] == "warn" and "token required" in warn["detail"]
bad = posture.check_listeners(
    {**CFG, "server": {**CFG["server"], "auth": {"require_auth": False}}},
    [("wildcard", 8998)])
assert bad["state"] == "bad" and "without auth" in bad["detail"]
# a loopback-only service exposed = always bad
bad2 = posture.check_listeners(CFG, [("wildcard", 8090)])
assert bad2["state"] == "bad" and "127.0.0.1" in bad2["fix"]
# all loopback = good
good = posture.check_listeners(CFG, [("loopback", 8998), ("loopback", 8090)])
assert good["state"] == "good"
# can't read = unknown, never a pass
assert posture.check_listeners(CFG, None)["state"] == "unknown"
ok("listeners: cascade gated=warn / ungated=bad, loopback service exposed=bad, "
   "all-loopback=good, can't-read=unknown")

# wireguard honesty
assert posture.check_wireguard(None, False)["state"] == "unknown"
assert posture.check_wireguard([], True)["state"] == "warn"
assert posture.check_wireguard([], False)["state"] == "good"
assert posture.check_wireguard([("wg0", "up")], True)["state"] == "good"
ok("wireguard: unknown / absent-but-exposed / up graded honestly")

# policy grading against a patched file
with tempfile.TemporaryDirectory() as td:
    pol = Path(td) / "egress.toml"
    keep = policy.POLICY_PATH
    try:
        policy.POLICY_PATH = pol
        assert posture.check_policy()["state"] == "bad"
        pol.write_text('[[rule]]\nname="a"\nhosts=["x.com"]\npurpose="t"\nttl_seconds=60\n')
        assert posture.check_policy()["state"] == "good"
        pol.write_text('[[rule]]\nname="a"\nhosts=["x.com"]\npurpose="t"\n')
        assert posture.check_policy()["state"] == "warn"
    finally:
        policy.POLICY_PATH = keep
ok("policy: missing=bad, lease-only=good, standing=warn")

# token perms
with tempfile.TemporaryDirectory() as td:
    cp = Path(td) / "config.json"
    cp.write_text("{}")
    os.chmod(cp, 0o644)
    assert posture.check_token(
        {"hf_token": "x", "_config_path": str(cp)})["state"] == "warn"
    os.chmod(cp, 0o600)
    assert posture.check_token(
        {"hf_token": "x", "_config_path": str(cp)})["state"] == "good"
assert posture.check_token({})["state"] == "good"
ok("token: world-readable config with a token = warn + chmod named")

# the whole scan
res = posture.scan(CFG)
assert len(res["checks"]) == 7
assert all(c["state"] in ("good", "warn", "bad", "unknown") for c in res["checks"])
assert all(c["detail"] for c in res["checks"])
assert res["summary"]["overall"] != "good", "installs-unconfined is a standing warn"
ok("scan(): 7 checks, every state legal + explained, overall honest")

print(f"test_posture: {OK} checks OK")
