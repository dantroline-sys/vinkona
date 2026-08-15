"""localmod — the ONE copy of the sibling-import fallback (was pasted ~27×).

Vinkona's assistant/ is a script tree, not a package: normal runs start
scripts from this directory, so `import safety` works and is preferred.  But
several hosts load a module BY FILE PATH (config_server's tab loaders, tests,
engine venvs) and then this directory may not be on sys.path — the plain
import fails.  Every module used to carry its own try/except spec-load block
for that case; this is that block, once.

use(name) resolves a sibling module in either context and — unlike the old
inline copies and the per-file _load() helpers — registers the result in
sys.modules, so every caller shares ONE module object.  The old fresh-load
copies duplicated module state: caches, module-level memoization and test
monkeypatches lived in parallel universes, one per loading site.

Circular-import-safe: the module is registered before exec and removed again
on failure, the same order the real import machinery uses.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def use(name: str):
    """Import sibling module `name` (a .py file in this directory) in any
    context; returns the module object, shared process-wide."""
    try:
        return importlib.import_module(name)
    except Exception:
        # Not just ImportError — a same-named foreign module earlier on
        # sys.path that fails mid-import must not mask the sibling (the old
        # inline blocks caught Exception for the same reason).
        pass
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no sibling module {name!r} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod
