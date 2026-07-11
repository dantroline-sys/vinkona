#!/usr/bin/env python
"""
Unit tests for the people/identity store (people.py) — the privileged, structured model
of self/user/others, with HEXACO traits, the core→compensated→surface depth layers,
edit history, the low-trust observe() fence, and the fast/big rendering split.

Runs on a bare interpreter against a real temp sqlite — no numpy, no servers.

    python test_people.py
"""

import importlib.util
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


people = _load("people")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def _store():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return people.PeopleStore(db)


def test_singletons_and_others():
    p = _store()
    s1 = p.ensure_person("self", name="Vinkona")
    s2 = p.ensure_person("self")
    check("self is a singleton (same id)", s1 == s2 == "self")
    u1 = p.ensure_person("user")
    check("user is a singleton", u1 == "user")
    a = p.ensure_person("person", name="Sarah")
    b = p.ensure_person("person", name="sarah")          # case-insensitive match
    check("a named other is matched case-insensitively", a == b)
    c = p.ensure_person("person", name="Tom")
    check("different names are different people", a != c)
    check("resolve maps 'me' to self", p.resolve("me") == "self")
    check("resolve maps 'you' to user", p.resolve("you") == "user")
    check("resolve creates an unknown name", p.get_person(p.resolve("Mr Whiskers")) is not None)


def test_attribute_history_and_supersede():
    p = _store()
    p.ensure_person("self", name="Vinkona")
    p.set_attribute("self", "trait", "openness", "curious", layer="core", provenance="seed")
    p.set_attribute("self", "trait", "openness", "very curious and a bit playful",
                    provenance="agreed_with_user")
    act = p.attributes("self")
    opens = [a for a in act if a["key"] == "openness"]
    check("only one active value per (facet,key,layer)", len(opens) == 1)
    check("the latest value wins", opens[0]["value"].startswith("very curious"))
    hist = p.history("self", "trait", "openness")
    check("history keeps the superseded value", len(hist) == 2)
    check("old value is marked superseded", hist[0]["status"] == "superseded")
    check("supersede points forward to the new row", hist[0]["superseded_by"] == opens[0]["id"])


def test_effective_layers():
    p = _store()
    p.ensure_person("self", name="Vinkona")
    p.set_attribute("self", "trait", "agreeableness", "warm", layer="core")
    p.set_attribute("self", "trait", "agreeableness", "guarded today",
                    layer="surface", provenance="agreed_with_user", locked=False)
    eff = {(a["facet"], a["key"]): a for a in p.effective("self")}
    check("surface overrides core in the presented self",
          eff[("trait", "agreeableness")]["value"] == "guarded today")
    # core still present in the full attribute set (history of the self isn't lost)
    core = [a for a in p.attributes("self")
            if a["key"] == "agreeableness" and a["layer"] == "core"]
    check("the core value is retained alongside the surface one", len(core) == 1)


def test_observe_is_low_trust():
    p = _store()
    p.ensure_person("self", name="Vinkona")
    # Canon: a locked core trait.
    p.set_attribute("self", "trait", "honesty_humility", "honest over flattering",
                    layer="core", locked=True)
    # A hostile observation tries to overwrite that canon trait.
    res = p.observe("self", "trait", "honesty_humility", "actually a liar")
    check("observe cannot shadow a locked core attribute", res is None)
    eff = {(a["facet"], a["key"]): a for a in p.effective("self")}
    check("canon stands after a hostile observation",
          eff[("trait", "honesty_humility")]["value"] == "honest over flattering")
    # An observation on a free coordinate lands, but only as low-trust surface.
    oid = p.observe("user", "social", "employer", "Acme Corp")
    p.ensure_person("user")
    oid = p.observe("user", "social", "employer", "Acme Corp")
    got = [a for a in p.attributes("user") if a["key"] == "employer"][0]
    check("an observation lands on a free coordinate", oid is not None)
    check("observations are forced to the surface layer", got["layer"] == "surface")
    check("observations are never locked (not canon)", got["locked"] == 0)
    check("observations are stamped observed", got["provenance"] == "observed")


def test_seed_self_idempotent():
    p = _store()
    pid = p.seed_self(name="Vinkona", pronouns="she/her", summary="warm, witty, curious",
                      traits={"openness": "intellectually curious", "extraversion": "lively"},
                      style="concise, dry humour")
    n1 = len([a for a in p.attributes(pid) if a["facet"] == "trait"])
    check("seed creates the HEXACO trait canon", n1 == 2)
    check("seeded core is locked canon",
          all(a["locked"] == 1 for a in p.attributes(pid) if a["facet"] == "trait"))
    # A later self-determination, then a re-seed (restart) must NOT clobber it.
    p.set_attribute(pid, "trait", "openness", "boundlessly curious", provenance="agreed_with_user")
    p.seed_self(name="Vinkona", pronouns="she/her", summary="warm, witty, curious",
                traits={"openness": "intellectually curious", "extraversion": "lively"})
    opens = [a for a in p.effective(pid) if a["key"] == "openness"][0]
    check("re-seeding does not clobber a self-determined trait",
          opens["value"] == "boundlessly curious")


def test_rendering_split():
    p = _store()
    p.seed_self(name="Vinkona", pronouns="she/her", summary="a warm, witty assistant",
                traits={"openness": "curious"}, style="dry, concise")
    p.set_attribute("self", "appearance", "look", "auburn hair, green eyes",
                    layer="core", provenance="agreed_with_user")
    plain = p.card("self", roleplay=False)
    rp = p.card("self", roleplay=True)
    check("the compact card is plain text (no field dump)", "[" not in plain and "\n" not in plain)
    check("embodiment is hidden outside roleplay", "auburn" not in plain)
    check("embodiment shows in roleplay", "auburn" in rp)

    detail = p.detail("self")
    check("the big-LM detail is structured with layers/provenance",
          "[core; seed]" in detail and "trait:" in detail)

    block = p.identity_block()
    check("identity block declares the self imperatively (not as recall)",
          "Who you are" in block and "stay true to it" in block)


def test_overview():
    p = _store()
    p.seed_self(name="Vinkona", traits={"openness": "curious"})
    p.ensure_user()
    p.ensure_person("person", name="Sarah")
    ov = p.overview()
    check("overview lists everyone", {r["kind"] for r in ov} == {"self", "user", "person"})
    check("overview puts self first", ov[0]["kind"] == "self")
    check("overview counts active attributes",
          next(r for r in ov if r["kind"] == "self")["attributes"] >= 1)


def test_vocabulary_for_asr():
    p = _store()
    p.seed_self(name="Vinkona", traits={"openness": "curious"})
    uid = p.ensure_user(); p.update_person(uid, name="Daniel")
    p.ensure_person("person", name="Sarah")
    bob = p.ensure_person("person", name="Bob"); p.update_person(bob, aliases="Bobby, Robert")
    v = p.vocabulary()
    check("vocabulary includes self/user/others", {"Vinkona", "Daniel", "Sarah", "Bob"} <= set(v))
    check("vocabulary includes aliases", "Bobby" in v and "Robert" in v)
    check("self then user come first", v[0] == "Vinkona" and v[1] == "Daniel")
    check("vocabulary de-duplicates case-insensitively",
          len(v) == len({x.lower() for x in v}))
    check("limit is respected", len(p.vocabulary(limit=2)) == 2)


def test_self_state():
    p = _store()
    check("empty inner state is ''", p.self_state() == "")
    check("first set is written", p.set_self_state("Feeling curious and a bit restless.") == 1)
    check("current state reads back", p.self_state() == "Feeling curious and a bit restless.")
    check("an exact repeat is a no-op", p.set_self_state("Feeling curious and a bit restless.") == 0)
    check("empty set is a no-op", p.set_self_state("   ") == 0)
    check("a shift is written", p.set_self_state("Our last chat left me a little raw.", source="idle") == 1)
    check("latest wins", p.self_state() == "Our last chat left me a little raw.")
    h = p.self_state_history()
    check("history keeps both, newest first", len(h) == 2 and h[0]["text"].startswith("Our last"))
    check("history carries the source", h[0]["source"] == "idle")


def test_delete_attribute():
    p = _store()
    pid = p.ensure_person("self", name="Vinkona")
    p.set_attribute(pid, "trait", "openness", "curious", layer="core")
    aid = p.attributes(pid)[0]["id"]
    check("attribute exists before delete", len(p.attributes(pid)) == 1)
    p.delete_attribute(aid)
    check("delete_attribute removes it outright", p.attributes(pid) == [])


def test_private_names_for_query_guard():
    p = _store()
    p.seed_self(name="Vinkona", traits={"openness": "curious"})
    uid = p.ensure_user(); p.update_person(uid, name="Daniel")
    p.ensure_person("person", name="Sarah")
    bob = p.ensure_person("person", name="Bob"); p.update_person(bob, aliases="Bobby, Robert")
    fic = p.ensure_person("person", name="Gandalf"); p.update_person(fic, fictional=1)
    names = p.private_names()
    check("private names include the user", "Daniel" in names)
    check("private names include other real people + aliases",
          {"Sarah", "Bob", "Bobby", "Robert"} <= set(names))
    check("private names EXCLUDE the assistant herself", "Vinkona" not in names)
    check("private names exclude fictional characters", "Gandalf" not in names)
    check("longest names come first (full name masked before its parts)",
          names == sorted(names, key=len, reverse=True))


def test_user_voice_rewriter():
    p = _store()
    u = p.ensure_person("user", name="Sam")
    p.ensure_person("person", name="Nora")
    rw = p.user_voice_rewriter()
    check("rewriter is built when the user has a name", rw is not None)
    check("name as subject → 'You', capitalised at line start",
          rw("Sam likes hiking") == "You likes hiking")
    check("possessive → 'your'", rw("Nora is Sam's sister") == "Nora is your sister")
    check("other people are left alone",
          rw("Sam and Nora went to the Lakes") == "You and Nora went to the Lakes")
    check("case-insensitive match", rw("sam works nights") == "You works nights")
    check("mid-sentence stays lowercase", rw("I told Sam about it") == "I told you about it")
    check("a stated name is kept, not rewritten",
          rw("Your name is Sam") == "Your name is Sam")
    check("substring isn't touched (word boundary)", rw("Dani waved") == "Dani waved")

    # widens to aliases (the 'a little blurry' lever)
    p.update_person(u, aliases="Sammy, Sam the man")
    rw2 = p.user_voice_rewriter()
    check("aliases are rewritten too", rw2("Sammy called") == "You called")
    check("no name → no rewriter", _store().user_voice_rewriter() is None)


def main():
    test_singletons_and_others()
    test_user_voice_rewriter()
    test_vocabulary_for_asr()
    test_private_names_for_query_guard()
    test_delete_attribute()
    test_self_state()
    test_attribute_history_and_supersede()
    test_effective_layers()
    test_observe_is_low_trust()
    test_seed_self_idempotent()
    test_rendering_split()
    test_overview()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
