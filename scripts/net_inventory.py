#!/usr/bin/env python3
"""AMIGA-OPS-01 B-1 — the network inventory: every socket bind and every
outbound call site in this repository, found by deterministic search.

    python3 scripts/net_inventory.py > docs/net-inventory.md

This is the audit baseline the whole confinement workstream closes against:
the OPS-01 closing report must account for every row printed here.  It is a
SCANNER, not a judge — it reports what the code does today; dispositions
(broker / loopback / lease / delete) are decided in the contract's later
steps.  Deterministic: AST walk for Python, line scan for shell/config, output
ordered by path and line, no timestamps except the header's commit id.

Four passes:
  P1  network-capable imports, per file (the future gate G-5 baseline);
      client libs and server libs reported separately — aiohttp.web serving
      on loopback is allowed under the amended B-12, ClientSession is not
  P2  bind/listen sites + non-loopback address literals (G-6 baseline)
  P3  outbound call sites: urlopen/Request/ClientSession/ws_connect/… plus
      every http(s)/ws(s) URL literal anywhere in tracked text files
  P4  subprocess-mediated and shell-script egress: curl/wget/git clone/uv/
      hf download in *.sh — the install-time surface the contract must
      either cover or explicitly bound

Stdlib only.  Excludes venvs, caches, model stores and third-party trees.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "var", "bin", "Models", "models", "node_modules", "external",
             "vinkona_env", "neutts_env", "chatterbox_env", "orpheus_env",
             "personaplex_env", ".venv", "deps", "logs", "config", "certs",
             "target", "build", "dist", "__pycache__", "eval", "Vinkona"}

# Modules that can open an OUTBOUND connection (G-5's future prohibition list)
CLIENT_MODS = {"requests", "httpx", "urllib.request", "urllib3", "aiohttp",
               "websockets", "ftplib", "smtplib", "poplib", "imaplib",
               "telnetlib", "http.client", "xmlrpc", "huggingface_hub"}
# Modules that can LISTEN (allowed, but every bind must be loopback)
SERVER_MODS = {"http.server", "socketserver", "asyncio"}
# Both-ways
BOTH_MODS = {"socket"}

OUT_CALLS = {"urlopen", "urlretrieve", "ws_connect", "create_connection",
             "getaddrinfo", "HTTPSConnection", "HTTPConnection", "SMTP",
             "ClientSession", "request_swap"} - {"request_swap"}
BIND_CALLS = {"bind", "listen", "serve_forever", "run_app", "start_server",
              "TCPSite", "HTTPServer", "ThreadingHTTPServer", "TCPServer"}

URL_RE = re.compile(r"(?:https?|wss?)://[^\s'\"<>)\]}]+")
NONLOOP_RE = re.compile(r'"0\.0\.0\.0"|\'0\.0\.0\.0\'|"::"|\'::\'')
SH_EGRESS_RE = re.compile(
    r"\b(curl|wget|git\s+clone|git\s+fetch|git\s+pull|pip\s+install|"
    r"uv\s+sync|uv\s+lock|hf\s+download|huggingface-cli|podman\s+pull|"
    r"docker\s+pull|vk_hf_download|vk_uv)\b")

LOOPBACK_HINTS = ("127.0.0.1", "localhost", "::1")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30).stdout.splitlines()
        files = [ROOT / f for f in out]
    except (subprocess.SubprocessError, OSError):
        files = list(ROOT.rglob("*"))
    keep = []
    for f in files:
        if not f.is_file():
            continue
        parts = set(f.relative_to(ROOT).parts[:-1])
        if parts & SKIP_DIRS:
            continue
        keep.append(f)
    return sorted(keep)


def classify_dest(text: str) -> str:
    if any(h in text for h in LOOPBACK_HINTS):
        return "loopback"
    m = URL_RE.search(text)
    if m:
        u = m.group(0)
        # a format-string URL ({host}, %s) is config-driven, not a literal peer
        if "{" in u or "%s" in u or "%d" in u:
            return "dynamic (format)"
        return "EXTERNAL"
    return "dynamic (config)"


def mod_matches(name: str, mods: set) -> str | None:
    for m in mods:
        if name == m or name.startswith(m + "."):
            return m
    return None


def scan_python(files):
    imports, binds, calls = [], [], []
    for f in files:
        if f.suffix != ".py":
            continue
        try:
            src = f.read_text()
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            # P1 — imports
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                for mods, kind in ((CLIENT_MODS, "client"), (SERVER_MODS, "server"),
                                   (BOTH_MODS, "both")):
                    m = mod_matches(n, mods)
                    if m:
                        guarded = node.col_offset > 0
                        imports.append((rel(f), node.lineno, m, kind, guarded))
                        break
            # P2 + P3 — calls
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else \
                    fn.id if isinstance(fn, ast.Name) else ""
                ctx = lines[node.lineno - 1].strip()[:140] if node.lineno <= len(lines) else ""
                if name in BIND_CALLS:
                    binds.append((rel(f), node.lineno, name, ctx))
                elif name in OUT_CALLS or name == "ClientSession":
                    calls.append((rel(f), node.lineno, name, classify_dest(ctx), ctx))
        # P2b — non-loopback literals + P3b — URL literals (line scan, all lines)
        for i, ln in enumerate(lines, 1):
            if NONLOOP_RE.search(ln):
                binds.append((rel(f), i, "0.0.0.0/:: literal", ln.strip()[:140]))
            for u in URL_RE.findall(ln):
                if "example." in u or "spdx.org" in u or u.startswith("http://www.w3.org"):
                    continue
                calls.append((rel(f), i, "url-literal", classify_dest(u), u[:120]))
    # dedupe imports per (file, module, kind)
    seen, uniq = set(), []
    for row in imports:
        key = (row[0], row[2], row[3])
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    return uniq, sorted(set(binds)), sorted(set(calls))


def scan_text(files):
    """P3b for non-Python text + P4 shell egress."""
    urls, shell = [], []
    for f in files:
        if f.suffix in (".py", ".png", ".jpg", ".gguf", ".zip", ".lock", ".ico"):
            continue
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, ln in enumerate(lines, 1):
            if f.suffix in (".sh", "") and SH_EGRESS_RE.search(ln) \
                    and not ln.strip().startswith("#"):
                shell.append((rel(f), i, ln.strip()[:140]))
            if f.suffix in (".toml", ".json", ".sh", ".md") and f.name != "net-inventory.md":
                for u in URL_RE.findall(ln):
                    if "example." in u:
                        continue
                    urls.append((rel(f), i, u[:120]))
    return sorted(set(urls)), sorted(set(shell))


def head_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out) if rows else "*(none found)*"


def main():
    files = tracked_files()
    imports, binds, calls = scan_python(files)
    urls, shell = scan_text(files)
    py_count = sum(1 for f in files if f.suffix == ".py")

    print(f"""# Network inventory (AMIGA-OPS-01 B-1)

Repo: `{ROOT.name}` @ `{head_commit()}` — generated by `scripts/net_inventory.py`
(deterministic AST + line scan; {len(files)} tracked files, {py_count} Python).

This is the audit BASELINE.  Every row below must be accounted for in the
OPS-01 closing report with a disposition: **broker** (migrate to the egress
broker), **loopback** (stays, bind/target confined to loopback), **lease**
(broker + time-boxed policy rule), or **delete**.

## P1 — network-capable imports ({len(imports)})

`kind=client` can open outbound connections (the future G-5 prohibition,
outside the broker); `server` can listen; `both` is raw sockets.  `guarded` =
imported inside try/def (optional path).

{table(["file", "line", "module", "kind", "guarded"],
       [(f, l, m, k, "yes" if g else "") for f, l, m, k, g in imports])}

## P2 — bind/listen sites + non-loopback literals ({len(binds)})

{table(["file", "line", "what", "context"], binds)}

## P3 — outbound call sites + URL literals in Python ({len(calls)})

{table(["file", "line", "call", "dest class", "context"], calls)}

## P3b — URL literals in shell/config/docs ({len(urls)})

{table(["file", "line", "url"], urls)}

## P4 — shell-script / subprocess egress ({len(shell)})

The install-time surface: package managers, model fetches, git clones.  The
contract either brings these under broker leases or states explicitly that
confinement begins post-install.

{table(["file", "line", "command"], shell)}
""")


if __name__ == "__main__":
    sys.exit(main())
