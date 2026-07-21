#!/usr/bin/env python3
"""Fetch a PREBUILT llama-server from llama.cpp's official releases — the
LM-Studio move: no cmake, no gcc, no CUDA-toolkit saga, a novice's install
never compiles anything.

    python3 fetch_llama.py                # detect platform, fetch, install
    python3 fetch_llama.py --plan         # show what would be picked, do nothing
    LLAMA_CPP_REF=b4837 python3 ...       # pin a release tag (else: latest)

What it does:
  * asks the GitHub API for the pinned release (or latest) of
    ggml-org/llama.cpp and picks the asset for this OS/arch — on a box with an
    NVIDIA driver it prefers a CUDA build, then Vulkan (GPU speed with no
    toolkit at all), then plain CPU;
  * verifies the download against the sha256 digest the GitHub API publishes
    for every asset (recorded either way, in bin/llama-server.sha256);
  * unpacks into var/prebuilt/llama-<tag>/ and symlinks bin/llama-server at
    the binary IN PLACE — its bundled shared libraries sit beside it and are
    found via the release's $ORIGIN rpath, so nothing is scattered;
  * records the tag in bin/llama-server.commit, same file the source build
    writes, so `doctor` gives one answer for both paths.

Exit codes: 0 installed · 3 no matching asset (caller should fall back to the
source build) · 1 real failure.  Stdlib only — runs before any venv exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
UA = {"User-Agent": "vinkona-installer", "Accept": "application/vnd.github+json"}


def say(msg: str) -> None:
    print(f"[fetch-llama] {msg}", flush=True)


def detect() -> dict:
    """This machine, in release-asset vocabulary."""
    sysname = platform.system().lower()
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    os_token = {"darwin": "macos", "linux": "ubuntu", "windows": "win"}.get(sysname, sysname)
    nvidia = shutil.which("nvidia-smi") is not None
    return {"os": os_token, "arch": arch, "nvidia": nvidia}


def variant_rank(plat: dict) -> list:
    """Asset-name variant preference, best first.  '' = the plain build (CPU on
    linux/windows, Metal on macOS — Apple's GPU path is in the default binary).
    Vulkan matters: on an NVIDIA box with no CUDA asset it is GPU acceleration
    with zero toolchain — exactly the novice case."""
    if plat["os"] == "macos":
        return [""]
    if plat["nvidia"]:
        return ["cuda", "vulkan", ""]
    return ["", "vulkan"]


def pick(assets: list, plat: dict) -> dict | None:
    """The best-matching asset: right OS + arch, best variant, .zip only."""
    def matches(name: str) -> bool:
        n = name.lower()
        return (n.endswith(".zip") and "bin" in n
                and plat["os"] in n and plat["arch"] in n)
    cands = [a for a in assets if matches(a.get("name", ""))]
    for want in variant_rank(plat):
        for a in cands:
            n = a["name"].lower()
            has_variant = any(v in n for v in ("cuda", "vulkan", "hip", "sycl", "rocm"))
            if want == "" and not has_variant:
                return a
            if want and want in n:
                return a
    return None


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def download(url: str, dest: Path) -> str:
    """Stream to disk with a coarse progress line; returns the sha256 hex."""
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            h.update(block)
            done += len(block)
            if total:
                pct = done * 100 // total
                print(f"\r[fetch-llama] downloading… {pct}% "
                      f"({done // (1 << 20)} / {total // (1 << 20)} MB)",
                      end="", flush=True)
    print(flush=True)
    return h.hexdigest()


def install(zip_path: Path, tag: str) -> Path:
    """Unpack and point bin/llama-server at the binary, in place."""
    dest = ROOT / "var" / "prebuilt" / f"llama-{tag}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    served = None
    for p in dest.rglob("llama-server*"):
        if p.is_file() and not p.name.endswith((".dll", ".lib", ".pdb")):
            served = p
            break
    if served is None:
        raise FileNotFoundError("no llama-server inside the release zip")
    # zip forgets the exec bit — restore it for every binary in that folder
    for p in served.parent.iterdir():
        if p.is_file():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    bindir = ROOT / "bin"
    bindir.mkdir(exist_ok=True)
    link = bindir / "llama-server"
    if link.is_symlink() or link.exists():
        link.unlink()
    # a SYMLINK, not a copy: the bundled shared libs live beside the real
    # binary and resolve via the release's $ORIGIN rpath
    link.symlink_to(served)
    (bindir / "llama-server.commit").write_text(f"prebuilt {tag}\n")
    return served


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="pick, print, don't download")
    ap.add_argument("--assets-json", help="test seam: a file with the release JSON")
    args = ap.parse_args()

    plat = detect()
    say(f"platform: {plat['os']}-{plat['arch']}"
        + (" · NVIDIA driver present" if plat["nvidia"] else ""))

    ref = os.environ.get("LLAMA_CPP_REF", "").strip()
    try:
        if args.assets_json:
            rel = json.loads(Path(args.assets_json).read_text())
        elif ref:
            rel = fetch_json(f"{API}/tags/{ref}")
        else:
            rel = fetch_json(f"{API}/latest")
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"could not reach the release API ({e}) — falling back to source build")
        return 3
    tag = rel.get("tag_name") or ref or "unknown"

    asset = pick(rel.get("assets") or [], plat)
    if asset is None:
        say(f"release {tag} has no prebuilt for {plat['os']}-{plat['arch']} — "
            "falling back to source build")
        return 3
    say(f"picked {asset['name']}  (release {tag})")
    if args.plan:
        return 0

    tmp = ROOT / "var" / "prebuilt"
    tmp.mkdir(parents=True, exist_ok=True)
    zp = tmp / asset["name"]
    try:
        got = download(asset["browser_download_url"], zp)
    except (urllib.error.URLError, OSError) as e:
        say(f"download failed ({e}) — falling back to source build")
        return 3
    want = str(asset.get("digest") or "")            # GitHub publishes sha256:<hex>
    if want.startswith("sha256:") and want.split(":", 1)[1] != got:
        say(f"sha256 MISMATCH on {asset['name']} — refusing to install it "
            f"(expected {want}, got sha256:{got})")
        zp.unlink(missing_ok=True)
        return 1
    served = install(zp, tag)
    (ROOT / "bin" / "llama-server.sha256").write_text(f"{got}  {asset['name']}\n")
    zp.unlink(missing_ok=True)
    say(f"installed bin/llama-server -> {served.relative_to(ROOT)}"
        + ("  (sha256 verified)" if want.startswith("sha256:") else
           "  (sha256 recorded; API published no digest)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
