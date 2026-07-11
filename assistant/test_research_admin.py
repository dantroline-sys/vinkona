"""Tests for config_server MemoryAdmin research/plan reset helpers (dead-end retry + per-plan)."""
import importlib.util
import sqlite3
import tempfile
import types
import os

_spec = importlib.util.spec_from_file_location("config_server", "config_server.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _adm():
    db = tempfile.mktemp(suffix=".db")
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE learning_plans (id INTEGER PRIMARY KEY, topic TEXT, status TEXT, completed_at REAL);
        CREATE TABLE plan_questions (id INTEGER PRIMARY KEY, plan_id INTEGER, question TEXT,
                                     kind TEXT, status TEXT, answer TEXT, updated_at REAL);
        INSERT INTO learning_plans(id,topic,status) VALUES (1,'KPIs','done'),(2,'regs','open');
        INSERT INTO plan_questions(id,plan_id,question,kind,status,answer) VALUES
         (10,1,'r1','research','answered','(no source found)'),
         (11,1,'r2','research','answered','The provided source does not answer this question.'),
         (12,1,'r3','research','answered','A real answer.'),
         (13,1,'a1','ask_user','asked',NULL),
         (20,2,'r4','research','answered','(no source found)'),
         (21,2,'p1','research','skipped','(withheld — holds private data)');
    """)
    c.commit(); c.close()
    return cs.MemoryAdmin({"memory": {"db_path": db}, "embed_lm": {}}), db


def test_deadend_counts():
    adm, db = _adm()
    assert adm.research_deadend_counts() == {"no_source": 2, "no_answer": 1}  # 10,20 / 11
    os.unlink(db)


def test_reopen_no_source_scopes_and_reopens_plan():
    adm, db = _adm()
    r = adm.reopen_research_questions("no_source")
    assert r["reopened"] == 2 and r["plans_reopened"] == 1        # plan 1 was done → reopened; 2 already open
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    st = {row["id"]: (row["status"], row["answer"]) for row in con.execute("SELECT * FROM plan_questions")}
    assert st[10][0] == "open" and st[10][1] is None
    assert st[12][0] == "answered"                                # a real answer is untouched
    assert st[13][0] == "asked"                                   # ask_user untouched
    assert st[21][0] == "skipped"                                 # withheld untouched
    con.close(); os.unlink(db)


def test_reopen_no_answer_only_matches_verdict():
    adm, db = _adm()
    r = adm.reopen_research_questions("no_answer")
    assert r["reopened"] == 1                                     # only q11
    os.unlink(db)


def test_reopen_plan_research_only_leaves_ask_user():
    adm, db = _adm()
    r = adm.reopen_plan(1, "research")
    assert r["reopened"] == 3                                     # r1,r2,r3 (not the ask_user)
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    st = {row["id"]: row["status"] for row in con.execute("SELECT * FROM plan_questions WHERE plan_id=1")}
    assert st[10] == st[11] == st[12] == "open" and st[13] == "asked"
    assert con.execute("SELECT status FROM learning_plans WHERE id=1").fetchone()[0] == "open"
    con.close(); os.unlink(db)


def test_reopen_plan_all_includes_ask_user():
    adm, db = _adm()
    assert adm.reopen_plan(1, "all")["reopened"] == 4            # r1,r2,r3 + a1
    os.unlink(db)


def test_reopen_plan_bad_id():
    adm, db = _adm()
    assert adm.reopen_plan("nope")["ok"] is False
    os.unlink(db)


def main():
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn(); passed += 1; print(f"  ok  {name}")
            except Exception as e:
                failed += 1; print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
