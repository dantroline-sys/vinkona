#!/usr/bin/env python
"""Basic-view feature recipes (config.FEATURE_RECIPES).

Recipes drive the preview-then-confirm step on Basic on/off toggles: plain-English copy plus the
companion knobs a feature needs to behave coherently.  Their failure mode is silent drift — a
companion/requires path gets renamed in DEFAULTS and the recipe keeps setting a dead key.  These
checks make that fail LOUDLY: every path a recipe touches must exist in DEFAULTS, every companion
value must match its option's type, and every recipe must actually carry the copy the UI renders.

    python assistant/test_feature_recipes.py
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


_MISSING = object()


def _leaf(path):
    """The DEFAULTS value at a dotted path, or _MISSING if the path isn't a real leaf."""
    o = cfg.DEFAULTS
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return _MISSING
        o = o[k]
    return _MISSING if isinstance(o, dict) else o


def test_recipe_keys_are_real_bool_toggles():
    bad = []
    for k in cfg.FEATURE_RECIPES:
        v = _leaf(k)
        if v is _MISSING or not isinstance(v, bool):
            bad.append(k)
    check(f"every recipe key is a real boolean toggle in DEFAULTS (offenders: {bad})", not bad)


def test_recipes_have_required_copy():
    bad = []
    for k, rec in cfg.FEATURE_RECIPES.items():
        en = rec.get("enable") or {}
        if not rec.get("title") or not en.get("summary"):
            bad.append(k)
    check(f"every recipe has a title and an enable summary (offenders: {bad})", not bad)


def test_all_change_paths_exist_and_typecheck():
    bad_path, bad_type, bad_label = [], [], []
    for k, rec in cfg.FEATURE_RECIPES.items():
        for blockname in ("enable", "disable"):
            for c in (rec.get(blockname) or {}).get("changes", []):
                p, val, label = c.get("path"), c.get("value"), c.get("label")
                cur = _leaf(p) if p else _MISSING
                if cur is _MISSING:
                    bad_path.append(f"{k}/{blockname}:{p}")
                elif cur is not None and type(cur) is not type(val):
                    bad_type.append(f"{k}/{blockname}:{p} ({type(val).__name__} vs {type(cur).__name__})")
                if not (isinstance(label, str) and label.strip()):
                    bad_label.append(f"{k}/{blockname}:{p}")
    check(f"every companion change path exists in DEFAULTS (offenders: {bad_path})", not bad_path)
    check(f"every companion value matches the option's type (offenders: {bad_type})", not bad_type)
    check(f"every companion change has a non-empty plain-English label (offenders: {bad_label})", not bad_label)


def test_all_requires_paths_exist():
    bad = []
    for k, rec in cfg.FEATURE_RECIPES.items():
        for blockname in ("enable", "disable"):
            for r in (rec.get(blockname) or {}).get("requires", []):
                if _leaf(r.get("path")) is _MISSING:
                    bad.append(f"{k}/{blockname}:{r.get('path')}")
                if not (isinstance(r.get("label"), str) and r["label"].strip()):
                    bad.append(f"{k}/{blockname}:label")
    check(f"every 'requires' path exists and is labelled (offenders: {bad})", not bad)


def test_recipe_toggles_are_basic_tier():
    # a recipe only helps where the user meets the toggle — the Basic view.  Every recipe key must
    # be a Basic-tier field, else it's authored for a toggle a Basic user never sees.
    bad = [k for k in cfg.FEATURE_RECIPES if cfg.field_level(k) != "basic"]
    check(f"every recipe toggle is Basic-tier (offenders: {bad})", not bad)


def test_no_change_targets_its_own_toggle():
    bad = []
    for k, rec in cfg.FEATURE_RECIPES.items():
        for blockname in ("enable", "disable"):
            for c in (rec.get(blockname) or {}).get("changes", []):
                if c.get("path") == k:
                    bad.append(f"{k}/{blockname}")
    check(f"a recipe never lists its own toggle as a companion change (offenders: {bad})", not bad)


def main():
    test_recipe_keys_are_real_bool_toggles()
    test_recipes_have_required_copy()
    test_all_change_paths_exist_and_typecheck()
    test_all_requires_paths_exist()
    test_recipe_toggles_are_basic_tier()
    test_no_change_targets_its_own_toggle()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
