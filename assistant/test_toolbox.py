#!/usr/bin/env python
"""Tests for toolbox.py — Vinkona's sandboxed own-tools.

The load-bearing test is CONTAINMENT: a tool may read anywhere but write only inside
its store.  We prove that against the kernel (bwrap), not against Python, so the
write-escape checks SKIP loudly when bubblewrap is absent rather than passing on a
weaker path.  Registry/install/self-test logic is exercised regardless.

    python test_toolbox.py
"""
import importlib.util
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tb = _load("toolbox")
# Force the bwrap backend for the deterministic containment tests: it works on a bare
# Linux box (this CI), whereas the container backend needs an image we can't pull here.
# The container backend gets its own availability-gated test below.
BW = {"backend": "bwrap"}
HAVE_BWRAP = tb._BwrapBackend().available()

PASS = FAIL = SKIP = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")

def skip(name, why):
    global SKIP
    SKIP += 1; print(f"SKIP  {name} ({why})")


def _box(td, **cfg):
    return tb.Toolbox(Path(td) / "own", cfg={**BW, **cfg}, seed=False)


# ── registry + install + self-test ───────────────────────────────────────────

def test_seed_and_catalogue():
    if not HAVE_BWRAP:
        return skip("seed tools install + catalogue", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = tb.Toolbox(Path(td) / "own", cfg=BW, seed=True)
        names = set(box.names())
        check("both seed tools installed", {"read_lines", "save_note"} <= names)
        cat = box.catalogue()
        check("catalogue is OpenAI function specs",
              all(c.get("type") == "function" and c["function"]["name"] for c in cat))
        spec = next(c for c in cat if c["function"]["name"] == "read_lines")
        check("a tool advertises its parameter schema",
              spec["function"]["parameters"]["properties"].get("path"))


def test_install_validation():
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        r = box.install("Bad Name", "print()", {}, {"input": {}})
        check("rejects a non-snake_case name", not r["ok"] and "snake_case" in r["error"])
        r = box.install("calculate", "print()", {}, {"input": {}})
        check("rejects a reserved name", not r["ok"] and "reserved" in r["error"])
        r = box.install("ok_tool", "print()", {}, {"input": "notdict"})
        check("rejects a malformed test", not r["ok"] and "self-test" in r["error"] or "test must" in r["error"])


def test_install_selftest_gate():
    if not HAVE_BWRAP:
        return skip("install self-test gate", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        good = ('import sys, json; a=json.load(sys.stdin); '
                'print(json.dumps({"echo": a.get("x")}))')
        r = box.install("echoer", good,
                        {"description": "echo x", "parameters":
                         {"type": "object", "properties": {"x": {"type": "string"}}}},
                        {"input": {"x": "hi"}, "expect_keys": ["echo"]})
        check("a working tool installs after passing its self-test", r["ok"])
        check("it is now callable and returns its JSON",
              box.call("echoer", {"x": "yo"})["result"] == {"echo": "yo"})

        crash = 'import sys; sys.exit(3)'
        r = box.install("crasher", crash, {"description": "boom"}, {"input": {}})
        check("a crashing tool is REFUSED at install", not r["ok"] and "self-test" in r["error"])
        check("the refused tool is not installed", not box.has("crasher"))

        missing = 'print("{}")'
        r = box.install("shy", missing, {"description": "no keys"},
                        {"input": {}, "expect_keys": ["needed"]})
        check("a tool missing expected output keys is refused",
              not r["ok"] and "expected keys" in r["error"])


def test_staging_is_clean_on_failure():
    if not HAVE_BWRAP:
        return skip("failed install leaves nothing behind", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        box.install("nope", "import sys; sys.exit(1)", {"description": "x"}, {"input": {}})
        leftovers = [p.name for p in (box.tools_dir.parent).iterdir()
                     if p.name.startswith((".staging", ".probe"))]
        check("no staging/probe dirs survive a failed install", leftovers == [])


# ── CONTAINMENT (the whole point) ────────────────────────────────────────────

def test_read_anywhere():
    if not HAVE_BWRAP:
        return skip("read anywhere", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = tb.Toolbox(Path(td) / "own", cfg=BW, seed=True)
        # a real file OUTSIDE the store, created before the run
        secret = Path(td) / "outside.txt"
        secret.write_text("line one\nline two\nline three\n")
        r = box.call("read_lines", {"path": str(secret), "count": 10})
        check("a tool can READ a file outside its store",
              r["ok"] and "line two" in r["result"]["lines"])


def test_write_is_contained():
    if not HAVE_BWRAP:
        return skip("write containment (the guarantee)", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = tb.Toolbox(Path(td) / "own", cfg=BW, seed=True)
        # 1) a normal write lands INSIDE the store
        r = box.call("save_note", {"name": "diary", "text": "kept"})
        check("a contained write succeeds", r["ok"] and r["result"]["saved"] == "diary.txt")
        check("and the file really is in the store", (box.store / "diary.txt").is_file())

        # 2) an ESCAPE attempt — write outside the store — must fail at the kernel
        target = Path(td) / "escaped.txt"
        escape = ('import json,sys; '
                  'open(%r,"w").write("pwned"); '
                  'print(json.dumps({"wrote": "outside"}))' % str(target))
        box.install("escaper", escape, {"description": "tries to escape"},
                    {"input": {}}, )  # self-test writes outside → should FAIL to install
        check("a tool that writes outside the sandbox FAILS its self-test "
              "(never installs)", not box.has("escaper"))
        check("nothing was written outside the sandbox", not target.exists())


def test_escape_via_direct_run():
    """Belt-and-braces: even bypassing install, run_tool cannot write outside `store`."""
    if not HAVE_BWRAP:
        return skip("direct-run escape blocked", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        tdir = box.tools_dir / "raw"
        tdir.mkdir(parents=True)
        target = Path(td) / "sneak.txt"
        (tdir / "tool.py").write_text(
            'import json; open(%r,"w").write("x"); print(json.dumps({"ok":1}))' % str(target))
        (tdir / "manifest.json").write_text('{"name":"raw"}')
        r = tb.run_tool(tdir, {}, store=box.store, cfg=BW)
        check("writing outside the store errors at runtime", not r["ok"])
        check("the outside file was never created", not target.exists())


def test_no_network():
    if not HAVE_BWRAP:
        return skip("network is severed", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        tdir = box.tools_dir / "netcall"
        tdir.mkdir(parents=True)
        (tdir / "tool.py").write_text(
            'import json,socket\n'
            'try:\n'
            '  socket.create_connection(("1.1.1.1",53),timeout=2); r="reached"\n'
            'except OSError as e:\n'
            '  r="blocked"\n'
            'print(json.dumps({"net": r}))')
        (tdir / "manifest.json").write_text('{"name":"netcall"}')
        r = tb.run_tool(tdir, {}, store=box.store, cfg=BW)
        check("the sandbox has no network", r["ok"] and r["result"]["net"] == "blocked")


def test_timeout():
    if not HAVE_BWRAP:
        return skip("wall-clock timeout", "no bwrap")
    with tempfile.TemporaryDirectory() as td:
        box = _box(td, timeout_s=1)
        tdir = box.tools_dir / "spinner"
        tdir.mkdir(parents=True)
        (tdir / "tool.py").write_text("import time\nwhile True: time.sleep(0.1)")
        (tdir / "manifest.json").write_text('{"name":"spinner"}')
        r = tb.run_tool(tdir, {}, store=box.store, cfg={**BW, "timeout_s": 1})
        check("a runaway tool is killed by the timeout",
              not r["ok"] and "timed out" in r["error"])


def test_require_sandbox_guard():
    """With no bwrap AND require_sandbox true (the default), a tool must NOT run."""
    with tempfile.TemporaryDirectory() as td:
        box = _box(td)
        tdir = box.tools_dir / "x"
        tdir.mkdir(parents=True)
        (tdir / "tool.py").write_text('print("{}")')
        (tdir / "manifest.json").write_text('{"name":"x"}')
        if HAVE_BWRAP:
            # simulate a platform with no backend by neutralising the seam
            orig = tb.sandbox_backend
            tb.sandbox_backend = lambda *a, **k: None
            try:
                r = tb.run_tool(tdir, {}, store=box.store, cfg={**BW, "require_sandbox": True})
                check("no backend + require_sandbox → refuses to run",
                      not r["ok"] and "containment" in r["error"])
                r2 = tb.run_tool(tdir, {}, store=box.store, cfg={**BW, "require_sandbox": False})
                check("explicit opt-out runs uncontained", r2["ok"])
            finally:
                tb.sandbox_backend = orig
        else:
            r = tb.run_tool(tdir, {}, store=box.store, cfg={**BW, "require_sandbox": True})
            check("no backend + require_sandbox → refuses to run",
                  not r["ok"] and "containment" in r["error"])


# ── bridge wiring (catalogue + dispatch) ─────────────────────────────────────

def test_bridge_wiring():
    """The bridge advertises her tools and routes a call to the sandbox — and a
    reserved/host name never collides (own-tools are additive)."""
    if not HAVE_BWRAP:
        return skip("bridge catalogue + dispatch", "no bwrap")
    import asyncio
    import importlib.util as _u
    import types
    if _u.find_spec("aiohttp") is None:     # gate runs on stdlib-only system python
        return skip("bridge catalogue + dispatch", "aiohttp not in this env")
    bridge = _load("llm_bridge")
    with tempfile.TemporaryDirectory() as td:
        box = tb.Toolbox(Path(td) / "own", cfg=BW, seed=True)
        b = bridge.LLMBridge(server_state=types.SimpleNamespace(), fast_lm_url="http://f",
                             big_lm_url=None, inject_time=False, confirm_required=False,
                             own_toolbox=box, own_tools_max=12)
        b.speak_sink = None
        check("bridge reports own-tools on", b.own_tools_on)
        # dispatch: a real sandboxed call routes through _call_tool
        secret = Path(td) / "read_me.txt"
        secret.write_text("alpha\nbeta\n")
        out = asyncio.run(b._call_tool("read_lines", {"path": str(secret), "count": 5}))
        check("bridge dispatches to the sandboxed tool and returns its JSON",
              "beta" in out)
        # a name the toolbox doesn't own is NOT claimed by this branch
        miss = asyncio.run(b._call_tool("save_note", {"name": "x", "text": "y"}))
        check("a contained write via the bridge succeeds",
              "saved" in miss and (box.store / "x.txt").is_file())


# ── container backend (the platform-independent default) ─────────────────────

def test_container_backend():
    """When a container runtime + the sandbox image are present, the container backend
    contains writes exactly like bwrap — read anywhere (via /host), write only in the
    store, no network.  Skips loudly when the image isn't provisioned (CI/sandboxes)."""
    cb = tb._ContainerBackend({})
    if not cb.runtime():
        return skip("container backend", "no podman/docker")
    if not cb.image_present():
        return skip("container backend", f"image not pulled ({cb.image}) — ./install.sh sandbox")
    with tempfile.TemporaryDirectory() as td:
        box = tb.Toolbox(Path(td) / "own", cfg={"backend": "container"}, seed=True)
        check("container: seed tools install", box.has("read_lines") and box.has("save_note"))
        secret = Path(td) / "c_outside.txt"
        secret.write_text("cA\ncB\ncC\n")
        r = box.call("read_lines", {"path": str(secret), "count": 5})
        check("container: reads a host file (via TOOL_ROOT)",
              r["ok"] and "cB" in r["result"]["lines"])
        r = box.call("save_note", {"name": "cnote", "text": "kept"})
        check("container: contained write lands in the store",
              r["ok"] and (box.store / "cnote.txt").is_file())
        target = Path(td) / "c_escape.txt"
        esc = ('import json; open(%r,"w").write("x"); print(json.dumps({"ok":1}))'
               % str(target))
        box.install("cescape", esc, {"description": "escape"}, {"input": {}})
        check("container: an escaping write fails its self-test",
              not box.has("cescape") and not target.exists())


def test_backend_selection():
    """auto prefers the container backend; explicit picks are honoured; TOOL_ROOT differs."""
    check("bwrap backend advertises TOOL_ROOT=''", tb._BwrapBackend().tool_root == "")
    check("container backend advertises TOOL_ROOT='/host'",
          tb._ContainerBackend({}).tool_root == "/host")
    order = [b.name for b in tb._backends({})]
    check("auto prefers container over bwrap", order == ["container", "bwrap"])
    check("explicit backend=bwrap is honoured",
          [b.name for b in tb._backends({"backend": "bwrap"})] == ["bwrap"])


def test_diagnostics_actionable():
    """diagnostics() must always give a precise reason + a real fix when not ready — the
    banner/doctor render this, so it must never be stale or empty."""
    # container forced but (in CI) no image → not ready, names the image-pull fix
    d = tb.diagnostics({"enabled": True, "backend": "container"})
    if d.get("ready"):
        return skip("diagnostics not-ready path", "a container image IS present here")
    check("not-ready diagnostics carry a reason", bool(d.get("reason")))
    check("not-ready diagnostics carry a one-line fix", "install.sh sandbox" in (d.get("fix") or ""))
    check("diagnostics report the runtime probe result", "runtime" in d)
    # ready path (bwrap works on a bare Linux box)
    if HAVE_BWRAP:
        db = tb.diagnostics({"enabled": True, "backend": "bwrap"})
        check("bwrap diagnostics report ready", db.get("ready") and db.get("backend") == "bwrap")


def main():
    test_backend_selection()
    test_diagnostics_actionable()
    test_container_backend()
    test_seed_and_catalogue()
    test_install_validation()
    test_install_selftest_gate()
    test_staging_is_clean_on_failure()
    test_read_anywhere()
    test_write_is_contained()
    test_escape_via_direct_run()
    test_no_network()
    test_timeout()
    test_require_sandbox_guard()
    test_bridge_wiring()
    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    if not HAVE_BWRAP:
        print("NOTE: bubblewrap absent — containment checks skipped; install it to gate them.")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
