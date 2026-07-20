#!/usr/bin/env python
"""Tests for pronouns.py — the persona's declared sex decides how the app refers
to it, in prompts and in the panel."""
import importlib.util
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


pr = _load("pronouns")
people = _load("people")

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def test_sets():
    check("three sets, no more", set(pr.CHOICES) == {"female", "male", "none"})
    check("female is she/her/hers/herself",
          [pr.SETS["female"][k] for k in ("subj", "obj", "poss", "possp", "refl")]
          == ["she", "her", "her", "hers", "herself"])
    check("male is he/him/his/himself",
          [pr.SETS["male"][k] for k in ("subj", "obj", "poss", "possp", "refl")]
          == ["he", "him", "his", "his", "himself"])
    check("none is it/its/itself",
          [pr.SETS["none"][k] for k in ("subj", "obj", "poss", "possp", "refl")]
          == ["it", "it", "its", "its", "itself"])


def test_resolution():
    check("a declared sex decides it",
          pr.for_identity({"sex": "male"})["subj"] == "he")
    check("the declared sex beats a stale free-text pronoun field",
          pr.for_identity({"sex": "male", "pronouns": "she/her"})["subj"] == "he")
    check("an existing persona with only the free-text field still works",
          pr.for_identity({"pronouns": "he/him"})["subj"] == "he")
    check("'it' parses to the neuter set", pr.for_identity({"pronouns": "it"})["subj"] == "it")
    check("nothing declared keeps the shipped character (no silent switch on upgrade)",
          pr.for_identity({})["subj"] == "she")
    check("an unrecognised sex falls back rather than raising",
          pr.for_identity({"sex": "wombat"})["subj"] == "she")
    check("a persona block resolves through identity",
          pr.for_persona({"identity": {"sex": "none"}})["label"] == "it/its")


def test_render():
    t = "{Subj} said it was {poss} choice; ask {obj} {refl}."
    check("female renders", pr.render(t, pr.SETS["female"])
          == "She said it was her choice; ask her herself.")
    check("male renders", pr.render(t, pr.SETS["male"])
          == "He said it was his choice; ask him himself.")
    check("neuter renders", pr.render(t, pr.SETS["none"])
          == "It said it was its choice; ask it itself.")
    check("untokenised text is untouched", pr.render("plain text", pr.SETS["male"])
          == "plain text")


def test_seeded_store():
    """The live pronoun set comes from the self record the persona seeded, so
    every prompt and panel reads the same answer."""
    c = sqlite3.connect(tempfile.mktemp(suffix=".db")); c.row_factory = sqlite3.Row
    ps = people.PeopleStore(c)
    check("an unseeded store still answers", ps.pronoun_set()["subj"] == "she")

    ps.seed_self(name="Aleks", sex="male", summary="a dry, exact companion")
    check("seeding with sex='male' sets the store's pronouns",
          ps.pronoun_set()["subj"] == "he")
    check("…and it is stored on the person row, not recomputed each time",
          ps.get_person("self")["pronouns"] == "he/him")

    c2 = sqlite3.connect(tempfile.mktemp(suffix=".db")); c2.row_factory = sqlite3.Row
    ps2 = people.PeopleStore(c2)
    ps2.seed_self(name="Ada", pronouns="she/her")
    check("a persona with only the legacy field is unchanged",
          ps2.pronoun_set()["label"] == "she/her")

    c3 = sqlite3.connect(tempfile.mktemp(suffix=".db")); c3.row_factory = sqlite3.Row
    ps3 = people.PeopleStore(c3)
    ps3.seed_self(name="Box", sex="none")
    check("a persona with no sex assigned is it/its", ps3.pronoun_set()["subj"] == "it")
    # the guard refusals are spoken in the persona's own voice
    ps3.set_attribute("self", "values", "honesty", "says what it means", layer="core")
    try:
        ps3.adapt("self", "honesty", "softer", context="when he's tired",
                  derived_from="honesty")
        check("an unadaptable facet is refused", False)
    except ValueError as e:
        check("the refusal uses the persona's pronoun, not a default 'she'",
              " it " in f" {e} " or "it " in str(e))


def main():
    test_sets()
    test_resolution()
    test_render()
    test_seeded_store()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
