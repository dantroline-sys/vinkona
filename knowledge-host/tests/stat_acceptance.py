"""VINKONA-STAT-01 §11 acceptance tests — byte-exact conformance.

Builds the §11.1 test corpus (K_COOC=4, MIN_COOC=2), runs ``vinkona-stat build``, and
verifies: the byte-exact §11.4 stats_report.json INCLUDING the exact stats_version
(§7 is fully specified, so unlike CONF-01 the hash itself must reproduce), the §11.3
drop ledger, the materialized live tables, atomic-swap idempotence, determinism, and
the §9 error contract.

Run:  python tests/stat_acceptance.py     (from the repo root; stdlib only)
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.conflict import ensure_schema
from knowledgehost.corpstats import StatError, build, compute, stats_version

EXPECTED_VERSION = "ff9ecad3acf1d421ccef87b90fba7d8d273e549d05a99d33ca699043439f4af2"

EXPECTED_REPORT = '{"stats_version":"ff9ecad3acf1d421ccef87b90fba7d8d273e549d05a99d33ca699043439f4af2","algo_version":"VINKONA-STAT-01/1.0","params":{"K_COOC":4,"MIN_COOC":2},"corpus":{"N":10},"salience":[{"card_id":"card:1","concept_id":"n:appendicectomy","weight":0.951465},{"card_id":"card:1","concept_id":"n:appendicitis","weight":0.968084},{"card_id":"card:1","concept_id":"n:hub","weight":1.0},{"card_id":"card:1","concept_id":"n:nausea","weight":0.461297},{"card_id":"card:10","concept_id":"n:analgesia","weight":0.70953},{"card_id":"card:10","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:10","concept_id":"n:appendicitis","weight":0.897647},{"card_id":"card:10","concept_id":"n:nausea","weight":0.721923},{"card_id":"card:10","concept_id":"n:rare","weight":0.59991},{"card_id":"card:2","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:2","concept_id":"n:appendicitis","weight":0.820886},{"card_id":"card:2","concept_id":"n:hub","weight":0.961134},{"card_id":"card:2","concept_id":"n:nausea","weight":0.484828},{"card_id":"card:3","concept_id":"n:appendicectomy","weight":0.698736},{"card_id":"card:3","concept_id":"n:appendicitis","weight":0.971159},{"card_id":"card:3","concept_id":"n:hub","weight":1.0},{"card_id":"card:4","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:4","concept_id":"n:hub","weight":0.775437},{"card_id":"card:4","concept_id":"n:nausea","weight":0.662285},{"card_id":"card:5","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:5","concept_id":"n:hub","weight":0.845264},{"card_id":"card:5","concept_id":"n:nausea","weight":0.820886},{"card_id":"card:6","concept_id":"n:analgesia","weight":0.866065},{"card_id":"card:6","concept_id":"n:appendicitis","weight":0.71094},{"card_id":"card:6","concept_id":"n:hub","weight":1.0},{"card_id":"card:7","concept_id":"n:hub","weight":1.0},{"card_id":"card:7","concept_id":"n:nausea","weight":0.431171},{"card_id":"card:8","concept_id":"n:appendicitis","weight":0.57342},{"card_id":"card:8","concept_id":"n:hub","weight":0.806565},{"card_id":"card:8","concept_id":"n:rare","weight":1.0},{"card_id":"card:9","concept_id":"n:analgesia","weight":1.0},{"card_id":"card:9","concept_id":"n:appendicectomy","weight":1.0},{"card_id":"card:9","concept_id":"n:appendicitis","weight":0.820886},{"card_id":"card:9","concept_id":"n:hub","weight":0.681954},{"card_id":"card:9","concept_id":"n:nausea","weight":0.820886}],"cooccurrence":[{"concept_a":"n:analgesia","concept_b":"n:nausea","ppmi":0.133531,"cooc_count":4},{"concept_a":"n:appendicectomy","concept_b":"n:appendicitis","ppmi":0.356675,"cooc_count":5},{"concept_a":"n:appendicectomy","concept_b":"n:nausea","ppmi":0.133531,"cooc_count":4}]}'

EXPECTED_FLOOR = [
    {"concept_a": "n:appendicitis", "concept_b": "n:rare", "cooc_count": 1},
    {"concept_a": "n:hub", "concept_b": "n:rare", "cooc_count": 1},
]
EXPECTED_CLAMP = [
    {"concept_a": "n:analgesia", "concept_b": "n:appendicectomy", "pmi": -0.223144, "cooc_count": 2},
    {"concept_a": "n:analgesia", "concept_b": "n:appendicitis", "pmi": -0.154151, "cooc_count": 3},
    {"concept_a": "n:analgesia", "concept_b": "n:hub", "pmi": -0.287682, "cooc_count": 3},
    {"concept_a": "n:appendicectomy", "concept_b": "n:hub", "pmi": -0.287682, "cooc_count": 3},
    {"concept_a": "n:appendicitis", "concept_b": "n:hub", "pmi": -0.113329, "cooc_count": 5},
    {"concept_a": "n:appendicitis", "concept_b": "n:nausea", "pmi": -0.202941, "cooc_count": 4},
    {"concept_a": "n:hub", "concept_b": "n:nausea", "pmi": -0.113329, "cooc_count": 5},
]

CORPUS = {
    "card:1": [("n:appendicitis", 3), ("n:appendicectomy", 2), ("n:nausea", 1), ("n:hub", 5)],
    "card:2": [("n:appendicitis", 2), ("n:appendicectomy", 2), ("n:nausea", 1), ("n:hub", 4)],
    "card:3": [("n:appendicitis", 2), ("n:appendicectomy", 1), ("n:hub", 3)],
    "card:4": [("n:analgesia", 3), ("n:nausea", 2), ("n:hub", 4)],
    "card:5": [("n:analgesia", 2), ("n:nausea", 2), ("n:hub", 3)],
    "card:6": [("n:appendicitis", 1), ("n:analgesia", 1), ("n:hub", 2)],
    "card:7": [("n:nausea", 1), ("n:hub", 6)],
    "card:8": [("n:rare", 1), ("n:appendicitis", 1), ("n:hub", 2)],
    "card:9": [("n:appendicitis", 1), ("n:appendicectomy", 1), ("n:nausea", 1),
               ("n:analgesia", 1), ("n:hub", 1)],
    "card:10": [("n:appendicitis", 5), ("n:appendicectomy", 4), ("n:nausea", 3),
                ("n:analgesia", 2), ("n:rare", 1)],
}
CONCEPTS = ["n:analgesia", "n:appendicectomy", "n:appendicitis", "n:hub", "n:nausea", "n:rare"]


def make_db(path):
    conn = sqlite3.connect(path)
    ensure_schema(conn)                      # canon registry (conflict_node) hosts the concepts
    conn.executemany("INSERT INTO conflict_node(node_id,label,kind) VALUES(?,?,'concept')",
                     [(c, c.split(":", 1)[1]) for c in CONCEPTS])
    conn.execute("""CREATE TABLE posting (
        card_id TEXT NOT NULL, concept_id TEXT NOT NULL,
        local_count INTEGER NOT NULL CHECK (local_count >= 1),
        PRIMARY KEY (card_id, concept_id))""")
    conn.executemany("INSERT INTO posting(card_id,concept_id,local_count) VALUES(?,?,?)",
                     [(card, c, n) for card, rows in CORPUS.items() for c, n in rows])
    conn.commit()
    conn.close()


def diff_bytes(name, got, want):
    if got != want:
        print(f"FAIL {name}")
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                print(f"  first divergence at byte {i}: got {got[i:i+70]!r} want {want[i:i+70]!r}")
                break
        else:
            print(f"  length differs: got {len(got)} want {len(want)}")
        sys.exit(1)
    print(f"PASS {name}  ({len(got)} bytes, byte-exact)")


def main():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "stat.db")
        out = os.path.join(td, "out")
        make_db(db)

        res = build(db, out, k_cooc=4, min_cooc=2)

        assert re.fullmatch(r"[0-9a-f]{64}", res["stats_version"])
        if res["stats_version"] != EXPECTED_VERSION:
            print(f"FAIL stats_version: got {res['stats_version']} want {EXPECTED_VERSION}")
            sys.exit(1)
        print(f"PASS stats_version == spec value ({EXPECTED_VERSION[:16]}…)")

        with open(os.path.join(out, "stats_report.json"), encoding="utf-8") as fh:
            diff_bytes("stats_report.json (§11.4)", fh.read(), EXPECTED_REPORT)

        assert res["drops"]["floor"] == EXPECTED_FLOOR, f"floor ledger: {res['drops']['floor']}"
        assert res["drops"]["ppmi_clamp"] == EXPECTED_CLAMP, f"clamp ledger: {res['drops']['ppmi_clamp']}"
        print("PASS §11.3 drop ledger (2 floor, 7 PPMI-clamp — every hub pair discarded)")

        # live tables materialized + meta coherent
        conn = sqlite3.connect(db)
        ns = conn.execute("SELECT COUNT(*) FROM card_concept_salience").fetchone()[0]
        nc = conn.execute("SELECT COUNT(*) FROM concept_cooccurrence").fetchone()[0]
        meta = dict(conn.execute("SELECT k,v FROM stats_meta"))
        w = conn.execute("SELECT weight FROM card_concept_salience WHERE card_id='card:9' "
                         "AND concept_id='n:hub'").fetchone()[0]
        p = conn.execute("SELECT ppmi,cooc_count FROM concept_cooccurrence WHERE "
                         "concept_a='n:appendicectomy' AND concept_b='n:appendicitis'").fetchone()
        conn.close()
        assert (ns, nc) == (35, 3), f"live table counts: {(ns, nc)}"
        assert meta["stats_version"] == EXPECTED_VERSION and meta["N"] == "10"
        assert w == 0.681954 and tuple(p) == (0.356675, 5)
        print("PASS live tables (35 salience rows, 3 co_occurs_with edges, stats_meta coherent)")

        # determinism + swap idempotence: a second full build over the live tables
        res2 = build(db, out, k_cooc=4, min_cooc=2)
        assert res2["report"] == res["report"] == EXPECTED_REPORT
        print("PASS determinism (rebuild byte-identical; §6.2 swap idempotent)")

        # ── §9 error contract ──
        for kw, code in [({"k_cooc": 1}, "E_BAD_PARAM"), ({"min_cooc": 0}, "E_BAD_PARAM"),
                         ({"k_cooc": 4.0}, "E_BAD_PARAM")]:
            try:
                build(db, None, **{"k_cooc": 4, "min_cooc": 2, **kw})
                print(f"FAIL: expected {code} for {kw}"); sys.exit(1)
            except StatError as e:
                assert e.code == code, f"{kw}: got {e.code}"
        try:
            compute([("c", "x", 1), ("c", "x", 2)], k_cooc=4, min_cooc=2)
            print("FAIL: duplicate posting accepted"); sys.exit(1)
        except StatError as e:
            assert e.code == "E_POSTINGS_MALFORMED"
        try:
            compute([("c", "x", 0)], k_cooc=4, min_cooc=2)
            print("FAIL: local_count 0 accepted"); sys.exit(1)
        except StatError as e:
            assert e.code == "E_POSTINGS_MALFORMED"
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM posting WHERE card_id='card:1' AND concept_id='n:hub'")
        conn.execute("INSERT INTO posting VALUES('card:1','n:ghost',1)")
        conn.commit()
        try:
            build(conn, None, k_cooc=4, min_cooc=2)
            print("FAIL: unknown concept accepted"); sys.exit(1)
        except StatError as e:
            assert e.code == "E_UNKNOWN_CONCEPT" and "n:ghost" in e.message
        conn.close()
        print("PASS §9 error contract (E_BAD_PARAM / E_POSTINGS_MALFORMED / E_UNKNOWN_CONCEPT)")

        # ── §G0: N = 0 is a valid empty build, not an error ──
        db0 = os.path.join(td, "empty.db")
        conn = sqlite3.connect(db0)
        ensure_schema(conn)
        conn.execute("CREATE TABLE posting (card_id TEXT NOT NULL, concept_id TEXT NOT NULL,"
                     " local_count INTEGER NOT NULL, PRIMARY KEY (card_id, concept_id))")
        conn.commit()
        r0 = build(conn, None, k_cooc=4, min_cooc=2)
        assert r0["N"] == 0 and r0["salience_rows"] == 0 and r0["cooccurrence_edges"] == 0
        assert '"corpus":{"N":0},"salience":[],"cooccurrence":[]' in r0["report"]
        conn.close()
        print("PASS N=0 empty build (valid result set, no error)")

        print("\nALL ACCEPTANCE TESTS PASS — VINKONA-STAT-01 §11 conformant")


if __name__ == "__main__":
    main()
