#!/usr/bin/env python3
"""`./vinkona.sh doctor` — is this machine ready, and if not, what ONE command
fixes it?  A novice's install fails in one of about six ways; this prints all
six checks in one table instead of leaving them to be discovered one crash at
a time.  Stdlib only, no venv needed — it must run BEFORE install has worked.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GOOD, BAD, MEH = "✓", "✗", "–"


def row(mark: str, label: str, detail: str, fix: str = "") -> None:
    print(f"  {mark}  {label:<22} {detail}" + (f"\n        ↳ {fix}" if fix else ""))


def ram_gb() -> int:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            return int(out) // (1 << 30)
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) // (1 << 20)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0


def gpu() -> str:
    if platform.system() == "Darwin":
        return "Apple silicon (Metal)" if platform.machine() == "arm64" else "none detected"
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=8).stdout.strip().splitlines()
            if out:
                return out[0]
        except subprocess.SubprocessError:
            pass
        return "NVIDIA driver present"
    return "none detected"


def cfg_port(key_path: list, default: int) -> int:
    try:
        cfg = json.loads((ROOT / "config" / "config.json").read_text())
        cur = cfg
        for k in key_path:
            cur = cur[k]
        return int(cur)
    except (OSError, ValueError, KeyError, TypeError):
        return default


def main() -> int:
    problems = 0
    print("Vinkona doctor\n")

    u = platform.uname()
    ram = ram_gb()
    try:
        free_gb = shutil.disk_usage(ROOT).free // (1 << 30)
    except OSError:
        free_gb = 0
    row(GOOD, "machine", f"{u.system} {platform.machine()} · {ram} GB RAM · "
                         f"{free_gb} GB free disk")
    if free_gb < 25:
        problems += 1
        row(BAD, "disk", f"{free_gb} GB free — the model set alone is ~5-25 GB",
            "free some space, or MODELS_DIR=/big/disk ./fetch_models.sh")
    row(GOOD, "gpu", gpu())

    if sys.version_info < (3, 9):
        problems += 1
        row(BAD, "python", platform.python_version(), "install Python 3.9+ ")
    else:
        row(GOOD, "python", platform.python_version())

    uv = shutil.which("uv") or (str(ROOT / "bin" / "uv")
                                if (ROOT / "bin" / "uv").exists() else "")
    row(GOOD if uv else MEH, "uv", uv or "not yet — install.sh bootstraps it itself")

    venv = ROOT / "vinkona_env" / "bin" / "python"
    if venv.exists():
        row(GOOD, "python env", "vinkona_env synced")
    else:
        problems += 1
        row(BAD, "python env", "vinkona_env missing", "./install.sh   (or ./vinkona.sh setup)")

    lls = ROOT / "bin" / "llama-server"
    which_ls = shutil.which("llama-server")
    if lls.exists() or which_ls:
        tagf = ROOT / "bin" / "llama-server.commit"
        tag = tagf.read_text().strip() if tagf.exists() else "untracked build"
        row(GOOD, "llama-server", f"{lls if lls.exists() else which_ls}  ({tag})")
    else:
        problems += 1
        row(BAD, "llama-server", "not installed",
            "./install.sh llama   (downloads a prebuilt; compiles only as fallback)")

    models = sorted((ROOT / "Models").glob("*.gguf")) if (ROOT / "Models").is_dir() else []
    if models:
        size = sum(m.stat().st_size for m in models) // (1 << 30)
        row(GOOD, "models", f"{len(models)} GGUF file(s), ~{size} GB in Models/")
    else:
        problems += 1
        row(BAD, "models", "Models/ is empty",
            "./fetch_models.sh   (picks a set sized to this machine's RAM)")

    if (ROOT / "config" / "config.json").exists():
        row(GOOD, "config", "config/config.json present")
    else:
        row(MEH, "config", "not yet — seeded from the example on first install")

    chat = cfg_port(["server", "port"], 8998)
    panel = cfg_port(["config_server", "port"], 8090)
    print()
    if problems:
        print(f"{problems} thing(s) to fix — each line above says the one command.")
        print("Or run everything at once:  ./vinkona.sh setup")
        return 1
    print("Ready.  ./vinkona.sh start   then open:")
    print(f"  chat   http://localhost:{chat}/chat")
    print(f"  panel  http://localhost:{panel}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
