#!/usr/bin/env python
"""Tests for fetch_llama.py — the prebuilt-binary path that keeps a novice's
install from ever compiling.  The GitHub API is never touched: pick() is pure,
and the end-to-end path runs against a fake release JSON via --assets-json."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

spec = importlib.util.spec_from_file_location("fetch_llama", HERE / "fetch_llama.py")
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def A(*names):
    return [{"name": n, "browser_download_url": f"https://x/{n}"} for n in names]


RELEASE = A("llama-b6100-bin-macos-arm64.zip",
            "llama-b6100-bin-macos-x64.zip",
            "llama-b6100-bin-ubuntu-x64.zip",
            "llama-b6100-bin-ubuntu-vulkan-x64.zip",
            "llama-b6100-bin-win-cpu-x64.zip",
            "llama-b6100-bin-win-cuda-12.4-x64.zip",
            "llama-b6100-source.tar.gz")


def test_pick():
    mac = {"os": "macos", "arch": "arm64", "nvidia": False}
    check("macOS picks the plain arm64 build (Metal is inside it)",
          fl.pick(RELEASE, mac)["name"] == "llama-b6100-bin-macos-arm64.zip")

    lin = {"os": "ubuntu", "arch": "x64", "nvidia": False}
    check("linux without a GPU picks plain CPU, not vulkan",
          fl.pick(RELEASE, lin)["name"] == "llama-b6100-bin-ubuntu-x64.zip")

    nv = {"os": "ubuntu", "arch": "x64", "nvidia": True}
    check("linux + NVIDIA prefers vulkan when no cuda asset exists "
          "(GPU speed with zero toolchain)",
          fl.pick(RELEASE, nv)["name"] == "llama-b6100-bin-ubuntu-vulkan-x64.zip")

    with_cuda = RELEASE + A("llama-b6100-bin-ubuntu-cuda-x64.zip")
    check("…and cuda outranks vulkan when it does",
          fl.pick(with_cuda, nv)["name"] == "llama-b6100-bin-ubuntu-cuda-x64.zip")

    arm_lin = {"os": "ubuntu", "arch": "arm64", "nvidia": False}
    check("no matching asset -> None (caller falls back to source build)",
          fl.pick(RELEASE, arm_lin) is None)

    check("a source tarball is never picked",
          all(fl.pick(A("llama-b6100-source.tar.gz"), p) is None
              for p in (mac, lin, nv)))


def test_end_to_end_plan():
    """--plan --assets-json: full CLI path, no network, no download."""
    rel = {"tag_name": "b6100", "assets": RELEASE}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rel, f)
        fake = f.name
    r = subprocess.run([sys.executable, str(HERE / "fetch_llama.py"),
                        "--plan", "--assets-json", fake],
                       capture_output=True, text=True, timeout=30)
    check("--plan exits 0 when an asset matches this machine", r.returncode == 0)
    check("…and names the asset it picked", "picked llama-b6100-bin-" in r.stdout)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tag_name": "b6100", "assets": []}, f)
        empty = f.name
    r2 = subprocess.run([sys.executable, str(HERE / "fetch_llama.py"),
                         "--plan", "--assets-json", empty],
                        capture_output=True, text=True, timeout=30)
    check("no assets -> exit 3, the fall-back-to-build signal", r2.returncode == 3)
    check("…and it says so in plain words", "falling back to source build" in r2.stdout)


def main():
    test_pick()
    test_end_to_end_plan()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
