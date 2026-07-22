"""The egress broker: every outbound request in this codebase goes through
here, is matched against the policy, and is written to the audit log.

    request(purpose, url, ...)   -> bytes      small API calls
    download(purpose, url, dest) -> Path       big files: resumable, verified
    lease(purpose, rule_name)                  context manager for leased rules

Deny-by-default; leased rules grant nothing until an operation opens them and
close themselves (crash-safe: the lease file expires by its own clock).

Transfer engines for download(), in order of preference: aria2c (segmented,
`-c -x4` — the actual fix for snail-pace fetches), wget -c, then a pure-stdlib
single stream with HTTP Range resume — so macOS and Windows work with nothing
installed.  Engines are subprocesses the broker spawns and accounts for.
Force one with AMIGA_FETCH_ENGINE=aria2c|wget|stdlib.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from . import audit, policy


class EgressDenied(Exception):
    """The policy said no.  The message is user-facing: it names the host and
    what would make the request legitimate (a rule, or an open lease)."""


def _dest_of(url: str) -> tuple:
    u = urlsplit(url)
    host = u.hostname or ""
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, port


def _config() -> dict:
    """The merged config, however this package was loaded.  Vinkona is a
    script-tree (no `assistant` package at runtime), so `from ..config` only
    works under test/package context — fall back to the top-level `config`
    module that sits on sys.path beside us."""
    try:
        from ..config import load_config        # package context (tests / vinur-style)
        return load_config()
    except Exception:
        pass
    try:
        import config                            # script-tree top-level (runtime)
        return config.load_config()
    except Exception:
        return {}


def _secret(name: str) -> str:
    """A named secret for rule auth: config first, then the environment."""
    if not name:
        return ""
    val = str(_config().get(name) or "").strip()
    if val:
        return val
    return os.environ.get(name.upper(), "").strip()


def _check(purpose: str, url: str, method: str) -> policy.Rule:
    """Match or refuse — the single decision point.  Every refusal is logged.
    When several rules match, any one that PERMITS wins (a standing rule isn't
    shadowed by a closed lease); denial names the first match."""
    host, port = _dest_of(url)
    rules = policy.load()
    hits = policy.find_all(rules, host, port, method)
    if not hits:
        audit.write("DENIED", purpose=purpose, host=host, port=port,
                    detail="no policy rule matches")
        raise EgressDenied(
            f"egress to {host}:{port} denied — no rule in egress.toml matches. "
            f"Add one deliberately if this destination is legitimate.")
    for rule in hits:
        if rule.enabled and (not rule.leased or policy.lease_state(rule) is not None):
            return rule
    rule = hits[0]
    if not any(r.enabled for r in hits):
        audit.write("DENIED", purpose=purpose, host=host, port=port,
                    rule=rule.name, detail="rule is disabled")
        raise EgressDenied(
            f"egress to {host} denied — rule '{rule.name}' is disabled "
            f"(the Network tab re-enables it).")
    audit.write("DENIED", purpose=purpose, host=host, port=port,
                rule=rule.name, detail="rule is leased and no lease is open")
    raise EgressDenied(
        f"egress to {host} requires an open lease on rule '{rule.name}' — "
        f"leased rules grant nothing between operations (run the operation "
        f"that opens one, e.g. a pull).")


@contextlib.contextmanager
def lease(purpose: str, rule_name: str):
    """Open a lease on a named rule for the duration of one operation.  The
    close is guaranteed (finally) and both ends are audited as a pair; a crash
    leaves a lease file that expires by its own clock."""
    rules = policy.load()
    rule = next((r for r in rules if r.name == rule_name), None)
    if rule is None:
        raise EgressDenied(f"no rule named '{rule_name}' in egress.toml")
    if not rule.enabled:
        raise EgressDenied(f"rule '{rule_name}' is disabled — no lease can open "
                           "on it (the Network tab re-enables it)")
    st = policy.lease_open(rule, purpose)
    audit.write("LEASE_OPEN", purpose=purpose, rule=rule.name,
                detail=f"ttl={int(rule.ttl_seconds or 3600)}s")
    try:
        yield st
    finally:
        policy.lease_close(rule.name)
        audit.write("LEASE_CLOSE", purpose=purpose, rule=rule.name)


def request(purpose: str, url: str, method: str = "GET", data: bytes | None = None,
            headers: dict | None = None, timeout: float = 30.0) -> bytes:
    """A small allowed call; returns the body bytes.  Auth headers come from
    the RULE (its `auth` names a config secret) — components never hold keys."""
    rule = _check(purpose, url, method)
    hdrs = {"User-Agent": "amiga-net", **(headers or {})}
    tok = _secret(rule.auth)
    if tok:
        hdrs.setdefault("Authorization", f"Bearer {tok}")
    req = urllib.request.Request(url, method=method, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            audit.write("AUTH_REJECT", purpose=purpose, host=_dest_of(url)[0],
                        port=_dest_of(url)[1], rule=rule.name, detail=f"HTTP {e.code}")
            raise
        raise
    policy.lease_use(rule)
    audit.write("ALLOWED", purpose=purpose, host=_dest_of(url)[0],
                port=_dest_of(url)[1], rule=rule.name,
                bytes_out=len(data or b""), bytes_in=len(body))
    return body


# ── download engines ─────────────────────────────────────────────────────────

def _engine() -> str:
    forced = os.environ.get("AMIGA_FETCH_ENGINE", "").strip()
    if forced:
        return forced
    pick = str(_config().get("fetch_engine") or "").strip()   # Network tab
    if pick:
        return pick
    for cand in ("aria2c", "wget"):
        if shutil.which(cand):
            return cand
    return "stdlib"


def _dl_aria2c(url: str, dest: Path, headers: dict, timeout: float) -> None:
    # engines run silent (--summary-interval=0, -q): the broker's own watcher
    # prints ONE progress format whichever engine is doing the work
    cmd = ["aria2c", "-c", "-x4", "-s4", "--file-allocation=none",
           "--auto-file-renaming=false", "--allow-overwrite=true",
           "--console-log-level=warn", "--summary-interval=0",
           "-d", str(dest.parent), "-o", dest.name, url]
    for k, v in headers.items():
        cmd[1:1] = [f"--header={k}: {v}"]
    subprocess.run(cmd, check=True, timeout=timeout)


def _dl_wget(url: str, dest: Path, headers: dict, timeout: float) -> None:
    cmd = ["wget", "-c", "-q", "-O", str(dest), url]
    for k, v in headers.items():
        cmd[1:1] = [f"--header={k}: {v}"]
    subprocess.run(cmd, check=True, timeout=timeout)


def _dl_stdlib(url: str, dest: Path, headers: dict, timeout: float) -> None:
    """Single stream with HTTP Range resume — works everywhere, needs nothing."""
    have = dest.stat().st_size if dest.exists() else 0
    hdrs = dict(headers)
    mode = "wb"
    if have:
        hdrs["Range"] = f"bytes={have}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=hdrs)
    try:
        r = urllib.request.urlopen(req, timeout=min(timeout, 120))
    except urllib.error.HTTPError as e:
        if e.code == 416:                     # already complete
            return
        raise
    with r, open(dest, mode) as f:
        if have and getattr(r, "status", 200) == 200:   # server ignored Range:
            f.seek(0)                                    # start over cleanly
            f.truncate()
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)


def _left(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _progress_line(have: int, total: int, rate: float) -> str:
    gb = have / 2**30
    speed = f"{rate / 2**20:.0f} MB/s"
    if total:
        eta = f" · ~{_left((total - have) / rate)} left" if rate > 1e5 and have < total else ""
        return f"      … {gb:.2f} / {total / 2**30:.2f} GB ({100 * have / total:.0f}%) · {speed}{eta}"
    return f"      … {gb:.2f} GB · {speed}"


def download(purpose: str, url: str, dest: Path, *, sha256: str = "",
             size: int = 0, timeout: float = 14400.0, headers: dict | None = None,
             progress=None) -> Path:
    """A big allowed transfer: resumable (a .part file survives interruption),
    engine-accelerated when aria2c/wget exist, sha256-verified when the caller
    knows the digest.  Audited with real byte counts.

    `progress` (a print-like callable) gets one line every AMIGA_PROGRESS_S
    seconds (default 15) — bytes so far, %, speed, ETA when `size` is known —
    measured from the .part file itself, so the format is identical whichever
    engine is transferring."""
    rule = _check(purpose, url, "GET")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    hdrs = {"User-Agent": "amiga-net", **(headers or {})}
    tok = _secret(rule.auth)
    if tok:
        hdrs.setdefault("Authorization", f"Bearer {tok}")

    resumed_from = part.stat().st_size if part.exists() else 0
    eng = _engine()
    stop = None
    if progress is not None:
        interval = float(os.environ.get("AMIGA_PROGRESS_S", "15") or 15)
        if interval > 0:
            stop = threading.Event()

            def _watch(prev=resumed_from, prev_t=time.time()):
                while not stop.wait(interval):
                    try:
                        have = part.stat().st_size if part.exists() else 0
                    except OSError:
                        continue
                    now = time.time()
                    rate = max(0, have - prev) / max(1e-9, now - prev_t)
                    prev, prev_t = have, now
                    try:
                        progress(_progress_line(have, size, rate))
                    except Exception:
                        return                # a dead log sink must not kill the fetch

            threading.Thread(target=_watch, daemon=True).start()
    try:
        if eng == "aria2c":
            _dl_aria2c(url, part, hdrs, timeout)
        elif eng == "wget":
            _dl_wget(url, part, hdrs, timeout)
        else:
            _dl_stdlib(url, part, hdrs, timeout)
    finally:
        if stop is not None:
            stop.set()

    if sha256:
        if progress is not None and size > (1 << 30):
            progress(f"      sha256-verifying {dest.name} …")   # ~1 min/30 GB, not a hang
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != sha256.lower():
            part.unlink(missing_ok=True)      # corrupt — a resume would keep it
            audit.write("DENIED", purpose=purpose, host=_dest_of(url)[0],
                        port=_dest_of(url)[1], rule=rule.name,
                        detail="sha256 mismatch — file discarded")
            raise EgressDenied(f"sha256 mismatch on {dest.name} — expected "
                               f"{sha256[:16]}…, got {h.hexdigest()[:16]}…; "
                               "the partial file was discarded, re-run to refetch")
    size = part.stat().st_size
    os.replace(part, dest)
    policy.lease_use(rule)
    audit.write("ALLOWED", purpose=purpose, host=_dest_of(url)[0],
                port=_dest_of(url)[1], rule=rule.name,
                bytes_in=max(0, size - resumed_from),
                detail=f"engine={eng}" + (" resumed" if resumed_from else ""))
    return dest


# ── the async lane (aiohttp) ─────────────────────────────────────────────────
# Vinkona's research/wikipedia egress is async.  B-12: aiohttp.ClientSession is
# permitted ONLY inside the broker.  BrokerSession is that one session — every
# .get()/.post() runs through the same _check() + audit as the sync lane, so a
# call site keeps its `session.get(url, params=…)` shape but cannot reach a
# destination the policy hasn't allowed.

@contextlib.asynccontextmanager
async def session(purpose: str, rule_name: str = "", timeout: float = 30.0):
    """An async context manager yielding a policy-enforcing session for one
    operation.  When rule_name is given a lease is opened for the whole block
    (research turns); otherwise each request stands on its own rule.  Requires
    aiohttp — the broker is the only place that imports it as a client."""
    import aiohttp
    tmo = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=tmo) as raw:
        bs = BrokerSession(raw, purpose)
        if rule_name:
            with lease(purpose, rule_name):
                yield bs
        else:
            yield bs


class _Resp:
    """Wraps an aiohttp response context manager so the audit row is written
    when the body is actually consumed — bytes_in is real, never guessed."""
    def __init__(self, cm, purpose, rule, url):
        self._cm = cm
        self._purpose, self._rule, self._url = purpose, rule, url
        self._r = None

    async def __aenter__(self):
        self._r = await self._cm.__aenter__()
        host, port = _dest_of(self._url)
        if getattr(self._r, "status", 200) in (401, 403):
            audit.write("AUTH_REJECT", purpose=self._purpose, host=host, port=port,
                        rule=self._rule.name, detail=f"HTTP {self._r.status}")
        else:
            policy.lease_use(self._rule)
            n = int((getattr(self._r, "headers", None) or {}).get("Content-Length") or 0)
            audit.write("ALLOWED", purpose=self._purpose, host=host, port=port,
                        rule=self._rule.name, bytes_in=n)
        return self._r

    async def __aexit__(self, *a):
        return await self._cm.__aexit__(*a)


class BrokerSession:
    """A drop-in for the aiohttp.ClientSession a call site used to create: same
    .get()/.post(url, params=…, headers=…, timeout=…) surface, but each call is
    policy-checked and audited first.  A denied destination raises EgressDenied
    exactly like the sync lane — the call site's own try/except turns that into
    'no result', which is the correct behaviour for a blocked research source."""
    def __init__(self, raw, purpose: str):
        self._raw = raw
        self._purpose = purpose

    def _guard(self, method: str, url: str, kwargs: dict):
        rule = _check(self._purpose, url, method)     # raises EgressDenied on refusal
        tok = _secret(rule.auth)
        if tok:
            hdrs = dict(kwargs.get("headers") or {})
            hdrs.setdefault("Authorization", f"Bearer {tok}")
            kwargs["headers"] = hdrs
        return rule

    def get(self, url, **kwargs):
        rule = self._guard("GET", url, kwargs)
        return _Resp(self._raw.get(url, **kwargs), self._purpose, rule, url)

    def post(self, url, **kwargs):
        rule = self._guard("POST", url, kwargs)
        return _Resp(self._raw.post(url, **kwargs), self._purpose, rule, url)
