#!/usr/bin/env python
"""The Models-tab TTS engine swap (config_server.tts_status / tts_select).

The panel lets a user pick a TTS engine and either apply-and-restart or schedule
for next restart; the heavy install self-provisions in serve_tts.sh.  Here we pin
the panel-side contract: the catalogue + per-engine install detection (by
filesystem, since the config process doesn't import the engines), the legacy
alias, writing the choice atomically, and rejecting an unknown engine.

    python assistant/test_tts_select.py
"""
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cs = _load("config_server")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _fake_venv(root: Path, venv: str, module: str):
    sp = root / venv / "lib" / "python3.12" / "site-packages" / module
    sp.mkdir(parents=True, exist_ok=True)


def test_installed_detection_by_filesystem():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        orig = cs.ASSIST_DIR
        cs.ASSIST_DIR = root
        try:
            _fake_venv(root, "neutts_env", "numpy")            # neutts looks installed
            _fake_venv(root, "chatterbox_env", "chatterbox")   # chatterbox too
            # qwen3_env absent → not installed; orpheus needs onnxruntime + a gguf → not installed
            check("neutts detected as installed", cs._tts_installed("neutts") is True)
            check("chatterbox detected as installed", cs._tts_installed("chatterbox") is True)
            check("qwen3 (no venv) → not installed", cs._tts_installed("qwen3") is False)
            check("orpheus_gguf (no onnxruntime/gguf) → not installed",
                  cs._tts_installed("orpheus_gguf") is False)
            # a bare venv dir without the module must NOT count (failed pip)
            (root / "qwen3_env").mkdir()
            check("empty venv dir does not count as installed",
                  cs._tts_installed("qwen3") is False)
        finally:
            cs.ASSIST_DIR = orig


def test_status_lists_current_and_catalogue():
    st = cs.tts_status({"tts": {"engine": "chatterbox"}})
    check("status reports the current engine", st["current"] == "chatterbox")
    keys = {e["key"] for e in st["engines"]}
    check("catalogue lists all four engines",
          keys == {"orpheus_gguf", "neutts", "chatterbox", "qwen3"})
    cur = [e for e in st["engines"] if e["current"]]
    check("exactly the current engine is flagged current",
          len(cur) == 1 and cur[0]["key"] == "chatterbox")
    check("every engine carries a label + footprint + note",
          all(e.get("label") and e.get("footprint") and e.get("note") for e in st["engines"]))


def test_legacy_orpheus_alias():
    st = cs.tts_status({"tts": {"engine": "orpheus"}})
    check("legacy 'orpheus' normalises to orpheus_gguf", st["current"] == "orpheus_gguf")


def test_select_writes_choice_atomically():
    with tempfile.TemporaryDirectory() as d:
        cfgp = Path(d) / "config.json"
        cfgp.write_text(json.dumps({"tts": {"engine": "orpheus_gguf", "default_voice": "tara"}}))
        res = cs.tts_select(str(cfgp), "qwen3")
        check("select reports ok + the engine", res.get("ok") and res.get("engine") == "qwen3")
        saved = json.loads(cfgp.read_text())
        check("config.json now selects qwen3", saved["tts"]["engine"] == "qwen3")
        check("select preserves other tts settings", saved["tts"]["default_voice"] == "tara")


def test_select_rejects_unknown_engine():
    with tempfile.TemporaryDirectory() as d:
        cfgp = Path(d) / "config.json"
        cfgp.write_text(json.dumps({"tts": {"engine": "orpheus_gguf"}}))
        res = cs.tts_select(str(cfgp), "espeak")
        check("unknown engine is refused", res.get("ok") is False and "unknown" in res.get("error", "").lower())
        check("config is untouched on a rejected select",
              json.loads(cfgp.read_text())["tts"]["engine"] == "orpheus_gguf")


def main():
    test_installed_detection_by_filesystem()
    test_status_lists_current_and_catalogue()
    test_legacy_orpheus_alias()
    test_select_writes_choice_atomically()
    test_select_rejects_unknown_engine()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
