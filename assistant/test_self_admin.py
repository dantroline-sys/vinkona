#!/usr/bin/env python
"""Tests for config_server.SelfAdmin — the Self tab's read/edit of Vinkona's self-authored
identity (people store) + self-memories.  Real temp sqlite; no servers."""
import importlib.util
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cs = _load("config_server")
people = _load("people")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _cfg():
    tmp = tempfile.mkdtemp()
    return {"memory": {"db_path": str(Path(tmp) / "m.db")}, "embed_lm": {"url": "x", "model": "m"}}


def _seed_self_memory(path):
    c = sqlite3.connect(path); c.row_factory = sqlite3.Row
    people.PeopleStore(c)               # ensure people tables
    c.execute("""CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY, triggers TEXT, context_tags TEXT, payload TEXT,
        priority INTEGER, recency REAL, last_used REAL, created_at REAL,
        category TEXT, expiry REAL, source TEXT, cooldown_until REAL, embedding BLOB, doc_id TEXT)""")
    c.execute("INSERT INTO memories(id,triggers,context_tags,payload,priority,category,source) "
              "VALUES ('m1','[]','[]','I keep answers short with this user.',5,'self','reflection')")
    c.execute("INSERT INTO memories(id,triggers,context_tags,payload,priority,category,source) "
              "VALUES ('m2','[]','[]','You like cycling.',4,'profile','reflection')")
    c.commit(); c.close()


def test_view_set_delete():
    cfg = _cfg()
    sa = cs.SelfAdmin(cfg)
    check("view on a missing db is empty, not an error",
          sa.view() == {"self": None, "attributes": [], "self_memories": []})

    # set_attribute creates the self person on first write
    sa.set_attribute({"facet": "trait", "key": "openness", "value": "curious", "layer": "core"})
    v = sa.view()
    check("self person now exists", v["self"] and v["self"]["kind"] == "self")
    check("attribute is stored", any(a["key"] == "openness" and a["value"] == "curious"
                                     for a in v["attributes"]))
    check("core attribute is locked canon by default",
          [a for a in v["attributes"] if a["key"] == "openness"][0]["locked"] == 1)
    check("owner edit is provenance-tagged",
          [a for a in v["attributes"] if a["key"] == "openness"][0]["provenance"] == "user_edit")

    # editing supersedes (same coords), so still one active value
    sa.set_attribute({"facet": "trait", "key": "openness", "value": "very curious", "layer": "core"})
    act = [a for a in sa.view()["attributes"] if a["key"] == "openness"]
    check("editing keeps a single active value", len(act) == 1 and act[0]["value"] == "very curious")

    # self-memories surface (and only category 'self')
    _seed_self_memory(cfg["memory"]["db_path"])
    mems = sa.view()["self_memories"]
    check("self-memories are listed", any(m["id"] == "m1" for m in mems))
    check("non-self memories are excluded", not any(m["id"] == "m2" for m in mems))

    # delete removes the attribute outright
    aid = act[0]["id"]
    sa.delete_attribute({"id": aid})
    check("delete_attribute removes it", not any(a["id"] == aid for a in sa.view()["attributes"]))

    # inner state round-trips through the Self tab
    check("inner state starts empty", sa.view()["state"] == "")
    sa.set_state({"text": "Content, and quietly curious where this goes."})
    v2 = sa.view()
    check("inner state is stored and read back", v2["state"].startswith("Content"))
    check("inner-state history is exposed", any("Content" in h["text"] for h in v2["state_history"]))


def main():
    test_view_set_delete()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
