#!/usr/bin/env python
"""Panel-side mind-graph controls (config_server.MemoryAdmin).

The Memory tab reads the durable graph's size + backlog and queues an on-demand distill for the
research worker.  Those read the mind_graph tables the cascade owns — kg_nodes / kg_edges and the
kg_state checkpoint (columns k/v, not key/value).  If that schema drifts these queries fail
silently and the panel shows zeros forever, so pin the contract here.

    python assistant/test_mind_graph_admin.py
"""
import asyncio
import importlib.util
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


mg = _load("mind_graph")
cs = _load("config_server")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _build_db():
    path = tempfile.mktemp(suffix=".db")
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE chat_logs(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "session_id TEXT, ts REAL, role TEXT, text TEXT);")
    for role, text in [("user", "My sister Mara lives in Bristol."), ("assistant", "nice"),
                       ("user", "I work with Sam."), ("assistant", "ok"),
                       ("user", "A turn not distilled yet.")]:
        db.execute("INSERT INTO chat_logs(session_id,ts,role,text) VALUES('s',0,?,?)", (role, text))
    db.commit()
    g = mg.MindGraph(db, {"distill_batch_turns": 2, "distill_max_batches": 1})   # drain 1 batch = 2 user turns
    stub = lambda _p: {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
                       "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of",
                                  "quote": "My sister Mara", "fact": "Mara is the user's sister"},
                                 {"src": "Mara", "dst": "Bristol", "rel": "lives_in",
                                  "quote": "lives in Bristol", "fact": "Mara lives in Bristol"}]}
    asyncio.run(g.catch_up(stub))
    db.commit(); db.close()
    return path


def _adm(path, enabled=True):
    return cs.MemoryAdmin({"memory": {"db_path": path, "mind_graph": {"enabled": enabled}},
                           "embed_lm": {}})


def test_stats_reads_graph_and_backlog():
    path = _build_db()
    st = _adm(path).mind_graph_stats()
    check("stats counts entities, excluding the 'you' anchor (Mara + Bristol = 2)", st["nodes"] == 2)
    check("stats reports the folded edges", st["edges"] == 2)
    check("backlog counts the undistilled user turn (kg_state k/v checkpoint read correctly)",
          st["backlog"] == 1)
    check("total counts all user turns (3 user, 2 assistant)", st["total"] == 3)
    check("processed = total − backlog (2 of 3 distilled)", st["processed"] == 2)
    check("stats mirrors the enabled flag", st["enabled"] is True)


def test_request_sets_worker_flag():
    path = _build_db()
    r = _adm(path).request_mind_graph_distill()
    check("a distill request is queued", r.get("ok") and r.get("queued"))
    c = sqlite3.connect(path)
    row = c.execute("SELECT value FROM worker_state WHERE key='mind_graph_request'").fetchone()
    c.close()
    check("the worker flag mind_graph_request lands in worker_state", row is not None and row[0])


def test_rebuild_clears_and_requeues():
    path = _build_db()                                   # 2 entities, 2 of 3 turns distilled
    before = _adm(path).mind_graph_stats()
    check("precondition: some entities + processed turns exist",
          before["nodes"] >= 1 and before["processed"] >= 1)
    r = _adm(path).rebuild_mind_graph()
    check("rebuild reports ok", r.get("ok") and r.get("rebuilt"))
    after = _adm(path).mind_graph_stats()
    check("rebuild clears every entity and relation", after["nodes"] == 0 and after["edges"] == 0)
    check("rebuild rewinds so the whole transcript is backlog again",
          after["processed"] == 0 and after["backlog"] == after["total"])
    c = sqlite3.connect(path)
    anchor = c.execute("SELECT COUNT(*) FROM kg_nodes WHERE id='user:self'").fetchone()[0]
    req = c.execute("SELECT value FROM worker_state WHERE key='mind_graph_request'").fetchone()
    c.close()
    check("rebuild keeps the locked user anchor (edge labels still resolve)", anchor == 1)
    check("rebuild queues a re-distill for the worker", req is not None and req[0])


def test_request_guarded_when_off():
    path = _build_db()
    r = _adm(path, enabled=False).request_mind_graph_distill()
    check("a distill request is refused (with a plain message) when the feature is off",
          r.get("ok") is False and "off" in (r.get("error") or "").lower())


def test_snapshot_renders_grounded_relations():
    path = _build_db()
    snap = _adm(path).mind_graph_snapshot()
    labels = {n["label"] for n in snap["nodes"]}
    check("snapshot lists the entities", "Mara" in labels and "Bristol" in labels)
    check("snapshot excludes the user anchor from the entity list",
          all(n.get("id") != "user:self" for n in snap["nodes"]))
    check("every entity carries a type", all(n.get("type") for n in snap["nodes"]))
    # the user→Mara edge resolves the anchor to 'you' and keeps its grounding quote
    ue = [e for e in snap["edges"] if e["obj"] == "Mara" and e["subj"] == "you"]
    check("an edge to the user resolves to 'you' with its relation",
          len(ue) == 1 and ue[0]["rel"] == "sibling of")
    check("relations keep the verbatim grounding quote (viewable as knowledge)",
          ue and "My sister Mara" in ue[0]["quote"])
    check("snapshot surfaces the clean paraphrased fact alongside the quote",
          ue and ue[0].get("fact") == "Mara is the user's sister")


def test_snapshot_empty_db_is_empty_not_error():
    empty = tempfile.mktemp(suffix=".db")
    sqlite3.connect(empty).close()
    snap = _adm(empty).mind_graph_snapshot()
    check("snapshot on a graph-less db is empty lists, not an error",
          snap["nodes"] == [] and snap["edges"] == [])


def test_stats_empty_db_is_zero_not_error():
    empty = tempfile.mktemp(suffix=".db")
    sqlite3.connect(empty).close()                         # exists but has no tables
    st = _adm(empty).mind_graph_stats()
    check("stats on a graph-less db returns zeros, not an error",
          st["nodes"] == 0 and st["edges"] == 0 and st["backlog"] == 0)


def main():
    test_stats_reads_graph_and_backlog()
    test_request_sets_worker_flag()
    test_rebuild_clears_and_requeues()
    test_request_guarded_when_off()
    test_snapshot_renders_grounded_relations()
    test_snapshot_empty_db_is_empty_not_error()
    test_stats_empty_db_is_zero_not_error()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
