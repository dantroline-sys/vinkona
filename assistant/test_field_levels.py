#!/usr/bin/env python
"""Settings-form audience tiers (config.FIELD_LEVELS).

The tiers are the single curation point that lets the Settings UI default to a small "Basic"
view.  Their one failure mode is silent drift: a knob gets renamed in DEFAULTS and its tier
entry keeps pointing at the dead path, so it quietly falls back to 'advanced' and the curation
rots.  These checks make that fail LOUDLY — every exact path and every glob prefix must still
resolve against DEFAULTS — and pin the resolver's precedence.

    python assistant/test_field_levels.py
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cfg = _load("config")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _leaves(d, p=""):
    for k, v in d.items():
        dp = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            yield from _leaves(v, dp)
        else:
            yield dp


ALL = set(_leaves(cfg.DEFAULTS))


def test_every_exact_path_exists():
    dead = [k for k in cfg.FIELD_LEVELS if not k.endswith(".*") and k not in ALL]
    check(f"every exact tier path exists in DEFAULTS (offenders: {dead})", not dead)


def test_every_glob_matches_something():
    dead = [k for k in cfg.FIELD_LEVELS
            if k.endswith(".*") and not any(l == k[:-2] or l.startswith(k[:-2] + ".") for l in ALL)]
    check(f"every glob prefix matches at least one leaf (offenders: {dead})", not dead)


def test_levels_are_valid():
    bad = [(k, v) for k, v in cfg.FIELD_LEVELS.items() if v not in cfg.LEVEL_ORDER]
    check(f"every tier value is basic/advanced/expert (offenders: {bad})", not bad)
    check("LEVEL_DEFAULT is a valid level", cfg.LEVEL_DEFAULT in cfg.LEVEL_ORDER)


def test_resolver_precedence():
    # exact beats the enclosing glob
    check("exact path wins over a .* glob",
          cfg.field_level("memory.working_graph.enabled") == "basic"
          and "memory.working_graph.*" in cfg.FIELD_LEVELS)
    # a nested exact promotion under an expert glob
    check("a promoted knob overrides its expert subtree",
          cfg.field_level("memory.working_graph.brief_max_chars") == "advanced")
    # glob applies where no exact entry exists
    check("a .* glob tiers the rest of its subtree",
          cfg.field_level("memory.working_graph.tau_s") == "expert")
    # unlisted paths fall back to the default
    check("an unlisted knob resolves to the default tier",
          cfg.field_level("research.max_topics_per_session") == cfg.LEVEL_DEFAULT)
    # longest prefix wins when two globs could match
    check("longest matching prefix wins",
          cfg.field_level("tts.orpheus_gguf.temperature") == "expert")


def test_resolved_map_is_non_default_only():
    res = cfg.resolved_field_levels()
    check("resolved map excludes default-tier fields",
          all(v != cfg.LEVEL_DEFAULT for v in res.values()))
    check("resolved map keys are all real leaf paths",
          all(k in ALL for k in res))
    # there is a real, non-trivial Basic surface, and it isn't everything
    basics = [k for k, v in res.items() if v == "basic"]
    check(f"there is a curated Basic surface ({len(basics)} knobs)", 10 <= len(basics) <= 80)
    check("Basic is a small fraction of all knobs (the point of tiering)",
          len(basics) < len(ALL) // 3)


def test_key_features_have_a_basic_toggle():
    # the sections a newcomer actually reasons about must each expose at least their on/off in Basic
    for path in ("memory.working_graph.enabled", "memory.mind_graph.enabled", "research.enabled",
                 "tts.engine", "tools.enabled", "notifications.enabled", "calendar_sync.enabled"):
        check(f"'{path}' is Basic", cfg.field_level(path) == "basic")


def main():
    test_every_exact_path_exists()
    test_every_glob_matches_something()
    test_levels_are_valid()
    test_resolver_precedence()
    test_resolved_map_is_non_default_only()
    test_key_features_have_a_basic_toggle()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
