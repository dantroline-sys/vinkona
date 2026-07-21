#!/usr/bin/env python
"""The dependency ratchet — supply-chain surface, held by a test.

Two invariants, both cheap and both about the same thing (every third-party
package is code someone else ships into this assistant):

  1. The set of packages the cascade HARD-imports (fails without) may not grow
     silently.  New capability should arrive as a lazy/optional import that
     degrades gracefully — the pattern the whole tree already follows — and a
     genuinely new hard dependency is a decision, taken by editing the
     allowlist here in the same commit.

  2. Every dependency declared in pyproject must actually be imported
     somewhere.  Dead declarations are pure attack surface: fastapi, uvicorn
     and pydantic sat in the lock for months, imported by nothing.

Stdlib only; parses the tree with ast, so nothing is executed.
"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

# ── invariant 1: the hard-import allowlist ───────────────────────────────────
# aiohttp    the cascade's one web framework (server + client)
# numpy      audio frames + embeddings
# dateutil   calendar date resolution (python-dateutil)
# soundfile  TTS server audio I/O
HARD_ALLOWED = {"aiohttp", "numpy", "dateutil", "soundfile"}

# pyproject name -> import name, where they differ
IMPORT_NAME = {"faster-whisper": "faster_whisper", "python-dateutil": "dateutil",
               "huggingface-hub": "huggingface_hub"}

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


def scan():
    """(hard, soft): third-party module -> set(files).  hard = imported at
    column 0 (module scope, unguarded); anything indented — inside try/def —
    is soft: the file loads without it."""
    std = set(sys.stdlib_module_names)
    local = {p.stem for p in HERE.glob("*.py")}
    hard, soft = {}, {}
    for f in sorted(HERE.glob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in std or m in local:
                    continue
                (hard if node.col_offset == 0 else soft).setdefault(m, set()).add(f.name)
    return hard, soft


def declared():
    """Dependency names from pyproject.toml — [project].dependencies plus every
    [dependency-groups] list.  Regex, not tomllib: tests run on the system
    python and the 3.9 floor predates tomllib."""
    text = (HERE / "pyproject.toml").read_text()
    names = []
    grab = False
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if re.match(r"^(dependencies\s*=\s*\[|[\w-]+\s*=\s*\[)", s):
            grab = True
        for m in re.findall(r'"([A-Za-z0-9._-]+?)(?:\[[^\]]*\])?(?:[<>=!~;].*)?"', s) if grab else []:
            names.append(m)
        if grab and s.endswith("]"):
            grab = False
    return [n for n in names if n and not n.startswith("-")]


def main():
    hard, soft = scan()
    # runtime surface: what a shipped file needs just to import
    hard_runtime = {m for m, files in hard.items()
                    if any(not f.startswith("test_") for f in files)}

    grown = hard_runtime - HARD_ALLOWED
    check("no NEW hard dependency has crept in", not grown,
          f"{sorted(grown)} — make it lazy/optional (try/except or in-function), "
          "or add it to HARD_ALLOWED here as a deliberate decision")
    shrunk = HARD_ALLOWED - hard_runtime
    check("the allowlist matches reality (no stale entries)", not shrunk,
          f"{sorted(shrunk)} no longer hard-imported — ratchet DOWN: remove from HARD_ALLOWED")

    used = set(hard) | set(soft)
    dead = []
    for dep in declared():
        mod = IMPORT_NAME.get(dep, dep.replace("-", "_"))
        if mod not in used:
            dead.append(dep)
    check("every declared dependency is imported somewhere", not dead,
          f"{dead} declared in pyproject but imported by nothing — "
          "remove it (dead declarations are pure attack surface)")

    # the flip side: a soft import of something never declared is a landmine
    # that only detonates on the machine where it happens to be installed —
    # EXCEPT the deliberately-external engines that carry their own venvs.
    own_venv = {"torch", "chatterbox", "neutts"}     # deps/neutts project
    dec = {IMPORT_NAME.get(d, d.replace("-", "_")) for d in declared()}
    undeclared = {m for m in (set(soft) | set(hard)) - dec - own_venv
                  if not all(f.startswith("test_") for f in (soft.get(m, set()) | hard.get(m, set())))}
    check("every runtime import is declared (or an own-venv engine)", not undeclared,
          f"{sorted(undeclared)} imported but not in pyproject")

    print(f"\nhard runtime surface: {sorted(hard_runtime)}")
    print(f"lazy/optional: {sorted(set(soft) - hard_runtime)}")
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
