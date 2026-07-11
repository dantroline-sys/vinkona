#!/usr/bin/env python
"""
Unit tests for the profiles system (config.py): bootstrap/migration, create,
duplicate (snapshot), switch, delete, stats, name safety, and load_config routing.
Pure stdlib — points the module's profile globals at a temp dir so the real
project is never touched.

    python test_profiles.py
"""

import importlib.util
import sqlite3
import sys
import tempfile
from pathlib import Path

cfg = None
PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _make_db(path: Path, n: int):
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE memories(id TEXT PRIMARY KEY)")
    c.executemany("INSERT INTO memories(id) VALUES (?)", [(f"m{i}",) for i in range(n)])
    c.commit(); c.close()


def main():
    global cfg
    spec = importlib.util.spec_from_file_location("config", Path(__file__).parent / "config.py")
    cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)

    tmp = Path(tempfile.mkdtemp())
    # Redirect every profile global at the temp tree so nothing real is touched.
    cfg._ROOT = tmp
    cfg.PROFILES_DIR = tmp / "config" / "profiles"
    cfg.ACTIVE_PROFILE_FILE = tmp / "config" / "active_profile"
    (tmp / "config").mkdir(parents=True)

    # Legacy (pre-profiles) state to be migrated into profiles/default.
    _make_db(tmp / "config" / "memory.db", 2)
    (tmp / "config" / "personas.json").write_text('{"default":"vinkona","personas":{"vinkona":{}}}')

    # ── bootstrap + migration ──
    cfg.ensure_profiles_bootstrap()
    default = cfg.profile_dir("default")
    check("bootstrap creates profiles/default", default.is_dir())
    check("active pointer is 'default'", cfg.active_profile() == "default")
    check("legacy memory.db migrated into default", (default / "memory.db").exists())
    check("legacy memory.db removed from old location", not (tmp / "config" / "memory.db").exists())
    check("legacy personas.json migrated", (default / "personas.json").exists())
    check("bootstrap is idempotent (no raise on 2nd run)", cfg.ensure_profiles_bootstrap() is None)

    st = cfg.profile_stats("default")
    check("stats report 2 migrated memories", st["memories"] == 2)
    check("stats report 1 persona", st["personas"] == 1)
    check("stats flag default active", st["active"] is True)

    # ── create fresh ──
    cfg.create_profile("work")
    check("create makes the dir", cfg.profile_dir("work").is_dir())
    check("fresh profile seeds personas.json", (cfg.profile_dir("work") / "personas.json").exists())
    check("fresh profile has 0 memories", cfg.profile_stats("work")["memories"] in (-1, 0))
    try:
        cfg.create_profile("work"); dup_ok = False
    except ValueError:
        dup_ok = True
    check("create rejects an existing name", dup_ok)

    # ── duplicate / snapshot ──
    cfg.duplicate_profile("default", "snap")
    check("snapshot copies the memory DB", (cfg.profile_dir("snap") / "memory.db").exists())
    check("snapshot preserves the 2 memories", cfg.profile_stats("snap")["memories"] == 2)

    # ── switch ──
    cfg.set_active_profile("work")
    check("switch updates the active pointer", cfg.active_profile() == "work")
    check("default is no longer active", cfg.profile_stats("default")["active"] is False)

    # ── delete (refuses active, removes others) ──
    try:
        cfg.delete_profile("work"); del_active_ok = False
    except ValueError:
        del_active_ok = True
    check("delete refuses the active profile", del_active_ok)
    cfg.delete_profile("snap")
    check("delete removes a non-active profile", "snap" not in cfg.list_profiles())

    # ── name safety ──
    for bad in ("../etc", "a/b", "..", "", "x" * 65, "bad name"):
        try:
            cfg._safe_profile_name(bad); ok = False
        except ValueError:
            ok = True
        check(f"rejects unsafe name {bad!r}", ok)
    check("accepts a good name", cfg._safe_profile_name("good-1.2_x") == "good-1.2_x")

    # ── load_config routes paths through the active profile ──
    full = cfg.load_config(None)
    check("load_config exposes active profile", full["profile"]["active"] == "work")
    check("db_path points into the active profile",
          full["memory"]["db_path"].endswith(str(Path("profiles") / "work" / "memory.db")))
    check("personas_path points into the active profile",
          full["personas_path"].endswith(str(Path("profiles") / "work" / "personas.json")))
    check("available profiles listed", set(full["profile"]["available"]) >= {"default", "work"})

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
