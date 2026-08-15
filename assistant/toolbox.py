"""
toolbox.py — Vinkona's OWN tools: small single-file programs she can run inside a
sandbox during a conversation, and (later, P3) write for herself while idle.

The security model is one hard guarantee, stated the way Dan framed it: **she may
read anywhere, but she may only ever WRITE inside her own sandbox store.**  That is
enforced by the OS, not in Python.

Cross-platform by design: the containment lives behind a small BACKEND seam
(``sandbox_backend``) so each OS plugs in its own mechanism without touching the
registry, the catalogue, or the bridge wiring above it.  Backends, present and planned:

  * **Linux — bubblewrap** (implemented).  ``--ro-bind / /`` (read anywhere),
    ``--bind <store> <store>`` laid over it (write only there), ``--tmpfs /tmp``,
    ``--unshare-all`` (no network — egress stays the broker's monopoly — plus
    PID/IPC/UTS/user isolation), ``--die-with-parent``; RLIMIT_FSIZE/CPU via a POSIX
    preexec.
  * **macOS — sandbox-exec** (planned): an SBPL profile denying file-write outside the
    store and denying network; the same argv-building shape.
  * **Windows — AppContainer / restricted token, or a WSL bwrap hop** (planned): a
    write-restricted token whose only writable ACL is the store.

Everything OS-specific is confined to ``sandbox_backend`` + ``run_tool``; ``resource``
and ``preexec_fn`` are POSIX-only and guarded, so the module imports and the registry
works on every platform.  On a platform with NO containment backend yet, ``enabled``
collapses to off with a loud posture note unless the operator explicitly sets
``require_sandbox=false`` (an informed opt-in for a locked-down single-user box).  We
never silently run uncontained.

A tool is a directory ``<tools>/<name>/`` holding:
  * ``tool.py``       reads a JSON args object on stdin, prints a JSON result on stdout
  * ``manifest.json`` {name, description, parameters (JSON-schema), created_at, author…}
  * ``test.json``     {"input": {...}[, "expect_keys": [...]]} — the self-test

``install_tool`` validates the manifest, runs the self-test in a THROWAWAY sandbox (its
writes land in a temp dir, never the live store), and only promotes a tool that passes —
so a tool she writes is proven to at least run before it is ever offered.  The registry
(``Toolbox``) advertises installed tools as OpenAI-style function specs and calls them,
matching the tool-host contract the cascade already speaks (catalogue/call).

Stdlib only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

try:
    import resource                     # POSIX-only (RLIMIT_*); absent on Windows
except ImportError:                     # pragma: no cover - platform-dependent
    resource = None

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
# Names Vinkona's own tools may never take — they would shadow a built-in or a host
# tool in the dispatcher.  (The dispatcher also checks membership explicitly, but a
# collision should be refused at INSTALL, with a clear reason, not silently.)
RESERVED = frozenset({
    "calculate", "search_wikipedia", "queue_research", "remind_me", "news_search",
    "note_person", "deliberate", "revise_self", "mail", "files", "news", "web",
    "calendar", "kb_search", "kb_ask", "weather",
})

_TOOL_STDIN_LIMIT = 256 * 1024          # args JSON handed to a tool (generous; it's ours)


# ── sandbox backends (the platform seam) ─────────────────────────────────────
# A backend turns (store, tool_py, python) into the argv that runs the tool with
# read-anywhere / write-only-in-store containment on ITS OS.  Add a platform by
# adding a backend here; nothing above this line changes.

class _Backend:
    name = "none"

    def available(self) -> bool:
        return False

    def argv(self, store: Path, tool_py: Path, py: str) -> list:
        raise NotImplementedError


class _BwrapBackend(_Backend):
    """Linux: bubblewrap namespaces.  Whole tree read-only, the store laid over it
    writable, no network, PID/IPC/UTS/user isolation, dies with the parent."""
    name = "bwrap"

    def available(self) -> bool:
        return sys.platform.startswith("linux") and bool(shutil.which("bwrap"))

    def argv(self, store: Path, tool_py: Path, py: str) -> list:
        # /tmp stays READ-ONLY (part of the whole-tree bind) so "read anywhere" really
        # means anywhere, /tmp included; writable scratch is TMPDIR inside the store, so
        # well-behaved tempfile use lands in the sandbox and a hard-coded /tmp write fails
        # (correctly — that's outside the sandbox).
        return [
            shutil.which("bwrap"),
            "--ro-bind", "/", "/",              # read anywhere (whole host tree, read-only) …
            "--proc", "/proc",
            "--dev", "/dev",
            "--bind", str(store), str(store),   # … write only here (laid over the ro tree)
            "--unshare-all",                    # no network + PID/IPC/UTS/user isolation
            "--die-with-parent",
            "--new-session",
            "--chdir", str(store),
            "--clearenv",
            "--setenv", "HOME", str(store),
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "TMPDIR", str(store),
            "--", py, str(tool_py),
        ]


# macOS (sandbox-exec, SBPL) and Windows (AppContainer / restricted token / WSL hop)
# backends slot in here with the same .available()/.argv() shape.  Registered
# most-specific-first; the first available one wins.
_BACKENDS: list = [_BwrapBackend()]


def sandbox_backend() -> _Backend | None:
    """The active containment backend for this OS, or None when none is available."""
    for b in _BACKENDS:
        if b.available():
            return b
    return None


def available() -> bool:
    """True when SOME sandbox backend can contain writes on this platform."""
    return sandbox_backend() is not None


# ── the seed tools (materialised on first init) ──────────────────────────────
# Two hand-written exemplars: one proves READ-ANYWHERE, one proves CONTAINED-WRITE.
# They ship as strings so they are versioned here and covered by the test battery;
# P3's idle toolsmith will write new ones through the exact same install path.

_SEED_READ_LINES = '''\
import sys, json
a = json.load(sys.stdin)
path = str(a.get("path") or "")
start = max(1, int(a.get("start", 1)))
count = min(2000, max(1, int(a.get("count", 200))))
if not path:
    print(json.dumps({"error": "no path given"})); sys.exit(0)
lines, total = [], 0
try:
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            total = i
            if i < start:
                continue
            if i >= start + count:
                total = -1  # more remain
                break
            lines.append(line.rstrip("\\n"))
except OSError as e:
    print(json.dumps({"error": str(e), "path": path})); sys.exit(0)
print(json.dumps({"path": path, "start": start, "returned": len(lines),
                  "more": total == -1, "lines": lines}))
'''

_SEED_SAVE_NOTE = '''\
import sys, json, re
a = json.load(sys.stdin)
name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(a.get("name") or "note"))[:64] or "note"
if not name.endswith(".txt"):
    name += ".txt"
text = str(a.get("text") or "")
append = bool(a.get("append"))
try:
    with open(name, "a" if append else "w") as f:
        f.write(text + "\\n")
except OSError as e:
    print(json.dumps({"error": str(e)})); sys.exit(0)
print(json.dumps({"saved": name, "bytes": len(text) + 1, "appended": append}))
'''

SEED_TOOLS = {
    "read_lines": {
        "code": _SEED_READ_LINES,
        "manifest": {
            "description": "Read a slice of any text file on this machine (read-only). "
                           "Give the path and optionally a starting line and how many lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute file path to read"},
                    "start": {"type": "integer", "description": "first line (1-based), default 1"},
                    "count": {"type": "integer", "description": "how many lines, default 200"},
                },
                "required": ["path"],
            },
        },
        "test": {"input": {"path": "/etc/hostname", "count": 5},
                 "expect_keys": ["lines", "path"]},
    },
    "save_note": {
        "code": _SEED_SAVE_NOTE,
        "manifest": {
            "description": "Save a text note into your own sandbox store so you can keep "
                           "it for later. Give it a name and the text; set append to add on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "note file name"},
                    "text": {"type": "string", "description": "the text to store"},
                    "append": {"type": "boolean", "description": "append instead of overwrite"},
                },
                "required": ["name", "text"],
            },
        },
        "test": {"input": {"name": "selftest", "text": "hello from the sandbox"},
                 "expect_keys": ["saved"]},
    },
}


# ── the sandboxed runner ─────────────────────────────────────────────────────

def _rlimits(max_write_mb: int, cpu_s: int):
    """preexec_fn: RLIMIT_FSIZE (no file may exceed the cap — a second, per-file guard
    on top of the single writable dir) and RLIMIT_CPU (a spin can't outlive the wall
    clock).  Inherited across the exec into the sandboxed tool.  POSIX-only — returns
    None on Windows, where subprocess forbids preexec_fn anyway (the Windows backend
    will carry job-object limits instead)."""
    if resource is None:
        return None

    def _apply():
        fsize = max(1, int(max_write_mb)) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        cpu = max(1, int(cpu_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    return _apply


def run_tool(tool_dir: Path, args: dict, *, store: Path, cfg: dict | None = None) -> dict:
    """Run one tool under the sandbox.  `store` is the writable directory to expose (the
    live shared store, or a throwaway dir for a self-test).  Returns a normalised dict:
    {ok, result|error, [truncated], elapsed_s}.  Never raises for a tool-level failure —
    a crash, a timeout, a non-JSON stdout all come back as {ok:false, error:…}."""
    cfg = cfg or {}
    require = bool(cfg.get("require_sandbox", True))
    timeout_s = float(cfg.get("timeout_s", 10))
    max_out = int(cfg.get("max_output_kb", 64)) * 1024
    max_write_mb = int(cfg.get("max_write_mb", 32))
    py = cfg.get("python") or sys.executable or "python3"
    tool_py = Path(tool_dir) / "tool.py"
    if not tool_py.is_file():
        return {"ok": False, "error": f"tool has no tool.py ({tool_dir})"}
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)

    backend = sandbox_backend()
    contained = False
    if backend is not None:
        argv = backend.argv(store, tool_py, py)
        contained = True
    elif not require:
        # Explicit informed opt-out: NO write containment.  cwd=store is a courtesy, not
        # a boundary — the caller accepted this by setting require_sandbox=false.
        argv = [py, str(tool_py)]
    else:
        return {"ok": False, "error": "sandbox unavailable — no write-containment backend "
                "on this platform (Linux needs bubblewrap installed), so the tool was not "
                "run. Install the backend, or set own_tools.require_sandbox=false to accept "
                "the risk on a trusted single-user box."}

    payload = json.dumps(args or {})
    if len(payload) > _TOOL_STDIN_LIMIT:
        return {"ok": False, "error": "arguments too large"}
    # preexec_fn is POSIX-only; a backend that builds its own env/cwd (bwrap --chdir)
    # doesn't need our cwd, the uncontained fallback does.
    kw: dict = {}
    if os.name == "posix":
        kw["preexec_fn"] = _rlimits(max_write_mb, int(timeout_s) + 2)
    if not contained:
        kw["cwd"] = str(store)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            argv, input=payload.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s, **kw)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"tool timed out after {timeout_s:.0f}s",
                "elapsed_s": round(time.monotonic() - t0, 2)}
    except OSError as e:
        return {"ok": False, "error": f"could not launch the sandbox: {e}"}
    elapsed = round(time.monotonic() - t0, 2)

    out = proc.stdout or b""
    truncated = len(out) > max_out
    text = out[:max_out].decode("utf-8", "replace").strip()
    if proc.returncode != 0 and not text:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return {"ok": False, "error": f"tool exited {proc.returncode}: {err[:400]}",
                "elapsed_s": elapsed}
    try:
        result = json.loads(text) if text else None
    except ValueError:
        return {"ok": False, "error": "tool did not print a JSON result",
                "raw": text[:400], "elapsed_s": elapsed}
    res = {"ok": True, "result": result, "elapsed_s": elapsed}
    if truncated:
        res["truncated"] = True
    return res


# ── the registry ─────────────────────────────────────────────────────────────

class Toolbox:
    """Vinkona's installed own-tools: a catalogue for the LM and a call() dispatcher.
    One shared writable `store/` under the tools root is every tool's sandbox home."""

    def __init__(self, root: str | Path, cfg: dict | None = None, *, seed: bool = True):
        self.cfg = cfg or {}
        self.root = Path(root).expanduser()
        self.tools_dir = self.root / "tools"
        self.store = self.root / "store"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.store.mkdir(parents=True, exist_ok=True)
        if seed:
            self._seed()

    # -- discovery --
    def _tool_dir(self, name: str) -> Path:
        return self.tools_dir / name

    def _read_manifest(self, name: str) -> dict | None:
        p = self._tool_dir(name) / "manifest.json"
        try:
            m = json.loads(p.read_text())
            return m if isinstance(m, dict) else None
        except (OSError, ValueError):
            return None

    def names(self) -> list:
        if not self.tools_dir.is_dir():
            return []
        out = []
        for d in sorted(self.tools_dir.iterdir()):
            if d.is_dir() and (d / "tool.py").is_file() and (d / "manifest.json").is_file():
                out.append(d.name)
        return out

    def has(self, name: str) -> bool:
        return name in set(self.names())

    def catalogue(self) -> list:
        """OpenAI-style function specs for every installed tool (the cascade's tool
        contract).  A manifest with no usable parameters still advertises an empty
        object schema so the LM can call it."""
        specs = []
        for name in self.names():
            m = self._read_manifest(name) or {}
            params = m.get("parameters")
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object", "properties": {}}
            specs.append({"type": "function", "function": {
                "name": name,
                "description": str(m.get("description") or name)[:600],
                "parameters": params}})
        return specs

    def describe(self) -> list:
        """A compact roster for a system-prompt line / the panel — [{name, description}]."""
        return [{"name": n, "description": str((self._read_manifest(n) or {}).get(
                 "description") or "")[:200]} for n in self.names()]

    def roster(self) -> list:
        """Fuller per-tool metadata for the config panel's list view."""
        out = []
        for n in self.names():
            m = self._read_manifest(n) or {}
            out.append({"name": n, "description": str(m.get("description") or "")[:300],
                        "author": str(m.get("author") or ""),
                        "created_at": str(m.get("created_at") or "")})
        return out

    def read(self, name: str) -> dict | None:
        """An installed tool's three source files, for the panel's viewer/editor.
        None when the tool doesn't exist."""
        if not self.has(name):
            return None
        d = self._tool_dir(name)

        def _r(fn):
            try:
                return (d / fn).read_text()
            except OSError:
                return ""
        return {"name": name, "code": _r("tool.py"),
                "manifest": _r("manifest.json"), "test": _r("test.json")}

    # -- execution --
    def call(self, name: str, args: dict) -> dict:
        if not self.has(name):
            return {"ok": False, "error": f"no such tool: {name}"}
        return run_tool(self._tool_dir(name), args or {}, store=self.store, cfg=self.cfg)

    # -- installation (validate → self-test in a throwaway sandbox → promote) --
    def install(self, name: str, code: str, manifest: dict, test: dict,
                *, author: str = "vinkona", overwrite: bool = True) -> dict:
        """Install one tool.  Validates the name+manifest, stages the files in a temp
        dir, runs its self-test with a THROWAWAY writable store (never the live one),
        and only on a clean pass moves it into place.  Returns {ok, name|error, test}."""
        name = str(name or "").strip().lower()
        if not _NAME_RE.match(name):
            return {"ok": False, "error": "name must be lower_snake_case, 3-40 chars, "
                    "starting with a letter"}
        if name in RESERVED:
            return {"ok": False, "error": f"'{name}' is a reserved tool name"}
        if self.has(name) and not overwrite:
            return {"ok": False, "error": f"'{name}' already exists"}
        if not isinstance(manifest, dict):
            return {"ok": False, "error": "manifest must be an object"}
        params = manifest.get("parameters")
        if params is not None and (not isinstance(params, dict)
                                   or params.get("type") != "object"):
            return {"ok": False, "error": "manifest.parameters must be a JSON-schema object"}
        if not isinstance(test, dict) or not isinstance(test.get("input"), dict):
            return {"ok": False, "error": "test must be {\"input\": {...}} for the self-test"}

        staging = self.tools_dir.parent / f".staging-{name}-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            (staging / "tool.py").write_text(code)
            man = {"name": name, "author": str(manifest.get("author") or author),
                   "created_at": str(manifest.get("created_at") or _man_time(self.cfg)),
                   "description": str(manifest.get("description") or name),
                   "parameters": params or {"type": "object", "properties": {}}}
            (staging / "manifest.json").write_text(json.dumps(man, indent=2))
            (staging / "test.json").write_text(json.dumps(test, indent=2))

            # self-test: a throwaway writable store, so save-style tools don't touch
            # the live sandbox and a failing tool leaves nothing behind.
            probe_store = self.tools_dir.parent / f".probe-{name}-{os.getpid()}"
            probe_store.mkdir(parents=True, exist_ok=True)
            try:
                res = run_tool(staging, test["input"], store=probe_store, cfg=self.cfg)
            finally:
                shutil.rmtree(probe_store, ignore_errors=True)
            if not res.get("ok"):
                return {"ok": False, "error": f"self-test failed: {res.get('error')}",
                        "test": res}
            want = test.get("expect_keys")
            if want:
                got = res.get("result")
                if not isinstance(got, dict) or not all(k in got for k in want):
                    return {"ok": False, "error": "self-test output missing expected keys "
                            f"{want} (got {list(got) if isinstance(got, dict) else type(got).__name__})",
                            "test": res}

            dest = self._tool_dir(name)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            staging.replace(dest)
            staging = None                      # consumed by replace()
            return {"ok": True, "name": name, "test": res}
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def remove(self, name: str) -> dict:
        if not self.has(name):
            return {"ok": False, "error": f"no such tool: {name}"}
        shutil.rmtree(self._tool_dir(name), ignore_errors=True)
        return {"ok": True, "removed": name}

    def _seed(self) -> None:
        """Materialise the exemplar tools if the box has none yet.  Silent when they are
        already present or when the sandbox is unavailable (they'd fail their self-test)."""
        if self.names() or not (available() or not self.cfg.get("require_sandbox", True)):
            return
        for name, spec in SEED_TOOLS.items():
            try:
                self.install(name, spec["code"], spec["manifest"], spec["test"],
                             author="seed")
            except Exception:
                pass                            # a seed that won't take is not fatal


def _man_time(cfg: dict) -> str:
    # Tests can pin a deterministic clock; else wall time.
    fn = cfg.get("_now")
    if callable(fn):
        return str(fn())
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
