#!/usr/bin/env python
"""VIN-WM-01 phase 0 — per-conversation working-memory activation in recall().

Unit-tests the WorkingActivation dynamics and the recall() integration:
  * decay is exponential wall-clock (boost to 1.0, wait tau -> 1/e)
  * floor prunes dead nodes; cap bounds the set deterministically; boost clamps at 1.0
  * activation RE-RANKS candidates — flip which memory is activated, flip the winner
  * wm=None (and an empty working set) is bit-identical to baseline  [G-ACCEL]
  * activation does NOT override cooldown — a hot but just-used low-priority memory
    stays suppressed  [the activation-vs-cooldown balance, §4.3]

numpy is stubbed (embeddings off -> trigger match only), over a real temp SQLite store.

    python test_working_memory_phase0.py
"""

import asyncio
import importlib.util
import json
import math
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.modules.setdefault("numpy", types.ModuleType("numpy"))   # embeddings off -> trigger match only

spec = importlib.util.spec_from_file_location("memory", HERE / "memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)
WA = memory.WorkingActivation

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _store(cooldown=0, cap=1024):
    tmp = tempfile.mkdtemp()
    cfg = {
        "memory": {"db_path": str(Path(tmp) / "m.db"), "recall_top_k": 5,
                   "recency_halflife_s": 1209600, "default_cooldown_s": cooldown, "min_score": 0.5,
                   "weights": {"priority": 0.5, "trigger": 2, "semantic": 1.5, "recency": 0.3,
                               "tag": 0.5, "cooldown_override_priority": 8, "activation": 0.6},
                   "neighbours": 0, "neighbour_min_sim": 0.65, "garden": {},
                   "working_memory": {"enabled": True, "tau_s": 900, "boost_surface": 0.5,
                                      "boost_mention": 0.35, "floor": 0.02, "cap": cap}},
        "embed_lm": {"url": "http://x", "model": "m"},
    }
    return memory.MemoryStore(cfg)


def _insert(m, mid, trigger, priority=2):
    m.db.execute(
        "INSERT INTO memories(id,triggers,context_tags,payload,priority,recency,last_used,"
        "created_at,category,expiry,source,cooldown_until,embedding,doc_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, json.dumps([trigger]), "[]", f"payload-{mid}", priority, 0.0, 0.0, 0.0,
         "user", None, "user", 0.0, None, None))
    m.db.commit()


def ids(entries):
    return [e["id"] for e in entries]


# ── WorkingActivation unit dynamics ──────────────────────────────────────────
def test_dynamics():
    wa = WA({"tau_s": 900, "floor": 0.02, "cap": 3})
    wa.boost(["a"], 1.0)
    check("boost sets activation", wa.get("a") == 1.0)
    wa.boost(["a"], 0.5)
    check("boost clamps at 1.0", wa.get("a") == 1.0)

    # exp wall-clock decay: 1.0 after exactly tau -> 1/e  (G7-style)
    wa.last_ts = 0.0
    wa.decay(900.0)
    check("decay tau -> 1/e within 1e-9", abs(wa.get("a") - math.exp(-1)) < 1e-9)

    # first decay only seeds the clock (no over-decay from a None last_ts)
    fresh = WA({"tau_s": 900})
    fresh.boost(["x"], 0.4)
    fresh.decay(1000.0)                       # last_ts was None -> just sets it
    check("first decay does not fade (clock seed only)", fresh.get("x") == 0.4)

    # floor prunes a node decayed below the floor
    wa2 = WA({"tau_s": 900, "floor": 0.02})
    wa2.boost(["b"], 0.03)
    wa2.last_ts = 0.0
    wa2.decay(10000.0)                        # 0.03 * exp(-11.1) << floor -> gone
    check("floor prunes dead nodes", "b" not in wa2.act)

    # cap keeps the hottest, deterministic tie-break by id
    wa3 = WA({"cap": 2, "floor": 0.0})
    wa3.boost(["a"], 0.9)
    wa3.boost(["b"], 0.5)
    wa3.boost(["c"], 0.5)                      # over cap -> drop the coldest; b,c tie -> keep by id
    check("cap bounds the set", len(wa3.act) == 2)
    check("cap keeps hottest + deterministic tie-break", set(wa3.act) == {"a", "b"})


# ── recall() integration ─────────────────────────────────────────────────────
def test_reranks_candidates():
    m = _store(cooldown=0)
    _insert(m, "A", "alpha")
    _insert(m, "B", "beta")
    m.reload()

    # Flip which memory carries prior activation -> flip the winner, everything else equal.
    wmA = m.new_working_set(); wmA.boost(["A"], 0.8)
    resA = asyncio.run(m.recall("alpha beta", set(), wm=wmA))
    check("activated A ranks first among tied candidates", ids(resA)[0] == "A")

    wmB = m.new_working_set(); wmB.boost(["B"], 0.8)
    resB = asyncio.run(m.recall("alpha beta", set(), wm=wmB))
    check("activated B ranks first (winner flips with activation)", ids(resB)[0] == "B")


def test_accel_off_is_identical():
    m = _store(cooldown=0)
    _insert(m, "A", "alpha")
    _insert(m, "B", "beta")
    m.reload()

    base = ids(asyncio.run(m.recall("alpha beta", set(), wm=None)))
    empty = ids(asyncio.run(m.recall("alpha beta", set(), wm=m.new_working_set())))
    check("G-ACCEL: wm=None and an EMPTY working set give identical recall",
          base == empty and len(base) == 2)

    dis = _store(cooldown=0)
    dis.wm_cfg = {"enabled": False}
    check("disabled config -> new_working_set() is None (term stays zero)",
          dis.new_working_set() is None)


def test_activation_does_not_override_cooldown():
    m = _store(cooldown=600)                  # cooldown ON
    _insert(m, "A", "alpha", priority=2)       # low priority (< cooldown_override_priority 8)
    m.reload()
    wm = m.new_working_set()

    t1 = asyncio.run(m.recall("alpha", set(), wm=wm))
    check("turn 1 surfaces A (and activates it)", ids(t1) == ["A"] and wm.get("A") > 0)

    t2 = asyncio.run(m.recall("alpha", set(), wm=wm))   # immediately again
    check("turn 2: A is on cooldown -> suppressed DESPITE high activation",
          "A" not in ids(t2) and wm.get("A") > 0)


if __name__ == "__main__":
    test_dynamics()
    test_reranks_candidates()
    test_accel_off_is_identical()
    test_activation_does_not_override_cooldown()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        sys.exit(1)
    print(f"ALL OK ({PASS} checks)")
