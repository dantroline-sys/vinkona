#!/usr/bin/env python
"""VIN-WM-02 phase 1a — the deterministic conversational working-memory graph.
Runs on a bare interpreter (no model, no deps): extraction, thread persistence via
decay, boundedness, grounding, the fenced briefing, and exact replay determinism.

    python assistant/test_working_graph.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


wg = _load("working_graph")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def test_keyphrases():
    ph = dict(wg.keyphrases("the quick brown fox"))
    check("stopword 'the' dropped, content phrase kept", "quick brown fox" in ph and "the" not in ph)

    ph = dict(wg.keyphrases("machine learning models are useful"))
    check("multiword phrase stays together across spaces", "machine learning models" in ph)

    ph = dict(wg.keyphrases("cats, dogs and birds"))
    check("punctuation splits phrases", "cats" in ph and "dogs" in ph and "birds" in ph
          and "cats dogs" not in ph)

    # Deterministic: same input → identical ranked output, twice.
    a = wg.keyphrases("server migration blocks the quarterly budget review")
    b = wg.keyphrases("server migration blocks the quarterly budget review")
    check("extraction is deterministic (identical twice)", a == b)
    check("nothing extracted from a pure-stopword string", wg.keyphrases("the and of to it is") == [])


def test_ingest_and_frame():
    g = wg.WorkingGraph()
    g.ingest("we need to finish the server migration", now=0.0)
    check("a content phrase becomes a node", "p:server migration" in g.nodes)
    check("boosted node carries the supporting turn (grounded)", g.nodes["p:server migration"]["turns"] == [1])
    check("a fresh boost sets activation to b_direct",
          abs(g.nodes["p:server migration"]["activation"] - 0.4) < 1e-9)


def test_thread_persists_and_one_offs_fade():
    """The core payoff: a recurring thread stays in the frame across many turns, while a
    phrase mentioned once and never again decays out on its own."""
    g = wg.WorkingGraph()
    g.ingest("the quarterly budget is the main concern", now=0.0)          # turn 1: budget thread
    g.ingest("random unrelated chatter about lunch plans", now=100.0)       # turn 2: one-off
    g.ingest("back to the quarterly budget and its risks", now=200.0)       # turn 3: budget again
    g.ingest("more quarterly budget discussion of numbers", now=300.0)      # turn 4: budget again
    check("recurring thread is in the frame", "p:quarterly budget" in g.frame)
    check("a one-off from turn 2 has faded far below the recurring thread",
          g.nodes["p:quarterly budget"]["activation"] > g.nodes.get("p:lunch plans", {"activation": 0}).get("activation", 0) + 0.3)

    # A single early mention, never repeated, is gone once enough time elapses (evicted < floor).
    g2 = wg.WorkingGraph()
    g2.ingest("passing mention of parsley", now=0.0)                        # 'parsley' → 0.4
    g2.ingest("", now=2800.0)   # 0.4*exp(-2800/900) ≈ 0.0185 < floor 0.02 → evicted
    check("an unrepeated phrase fades out and is evicted", "p:parsley" not in g2.nodes)


def test_decay_math():
    g = wg.WorkingGraph()
    g.ingest("alpha beta", now=0.0)                    # 'alpha beta' at 0.4
    g.ingest("", now=900.0)                            # decay one tau, no new phrase
    check("decay matches exp(-dt/tau) to 1e-6",
          abs(g.nodes["p:alpha beta"]["activation"] - 0.4 / 2.718281828459045) < 1e-6)


def test_bounded():
    g = wg.WorkingGraph({"cap": 5, "edge_cap": 5, "top_k": 50, "link_top": 50})
    g.ingest("apple banana cherry date elder fig grape kiwi lemon mango melon nectarine", now=0.0)
    # every word is its own short phrase (single content words), far more than the cap
    check("node count is hard-capped", len(g.nodes) <= 5)
    check("edge count is hard-capped", len(g.edges) <= 5)


def test_grounded():
    g = wg.WorkingGraph()
    g.ingest("deploy pipeline touches the staging cluster", now=0.0)
    g.ingest("the staging cluster needs a config change", now=50.0)
    check("every node records ≥1 supporting turn", all(n["turns"] for n in g.nodes.values()))
    check("every edge records ≥1 supporting turn (no ungrounded edge)",
          all(e["turns"] for e in g.edges.values()))
    check("edges only connect resident nodes",
          all(a in g.nodes and b in g.nodes for (a, b) in g.edges))


def test_briefing():
    check("cold graph yields an empty briefing (prompt unchanged when off)", wg.WorkingGraph().briefing() == "")
    g = wg.WorkingGraph()
    g.ingest("the database index is corrupt", now=0.0)
    g.ingest("the database index rebuild is slow", now=30.0)
    b = g.briefing()
    check("briefing is fenced as fallible working notes", "may be wrong" in b and b.startswith("Working notes"))
    check("briefing surfaces the hot thread", "database index" in b)
    g3 = wg.WorkingGraph({"brief_max_chars": 40})
    g3.ingest("alpha bravo charlie delta echo foxtrot", now=0.0)
    check("briefing is hard length-capped", len(g3.briefing()) <= 40)


def test_replay_determinism():
    """G-8: same turns + same clock → bit-identical graph (frame, nodes, briefing)."""
    turns = [("the release plan slipped again", 0.0),
             ("marketing wants the release plan sooner", 40.0),
             ("engineering pushes back on the release plan", 95.0),
             ("unrelated note about coffee", 130.0)]
    def run():
        g = wg.WorkingGraph()
        for t, n in turns:
            g.ingest(t, now=n)
        return g.stats(), g.briefing(), {k: round(v["activation"], 12) for k, v in g.nodes.items()}
    check("replay is bit-identical (stats + briefing + activations)", run() == run())


def test_metrics():
    g = wg.WorkingGraph()
    g.ingest("the database index is corrupt", now=0.0)
    m = g.last_metrics
    check("metrics carry the expected fields",
          {"turns", "nodes", "edges", "entropy", "frame_churn", "frame_drift",
           "ungrounded_edges", "frame"} <= set(m))
    check("ungrounded_edges is zero (grounding invariant, live)", m["ungrounded_edges"] == 0)
    check("frame lists the phrase labels (inspection window)", "database index" in m["frame"])

    # Re-ingesting the same turn leaves the frame unchanged → zero churn.
    g.ingest("the database index is corrupt", now=5.0)
    check("identical frame → frame_churn 0", g.last_metrics["frame_churn"] == 0.0)
    # A topic shift moves the frame → drift climbs from the seed.
    for i, t in enumerate(["switch to the billing subsystem entirely",
                           "billing subsystem invoices and refunds", "billing retries and dunning"]):
        g.ingest(t, now=20.0 + i * 10)
    check("a topic shift raises frame_drift above zero", g.last_metrics["frame_drift"] > 0.0)

    # Entropy: one dominant node ≈ 0; several balanced nodes > 0.
    g1 = wg.WorkingGraph(); g1.ingest("solo", now=0.0)
    check("a single node has ~zero activation entropy", g1.last_metrics["entropy"] < 1e-9)
    g2 = wg.WorkingGraph(); g2.ingest("apple, banana, cherry, date", now=0.0)
    check("several balanced nodes have positive entropy", g2.last_metrics["entropy"] > 1.0)


def test_stall_guard():
    # Repetitive frame with a full-enough frame trips the guard; it only flags, never mutates.
    g = wg.WorkingGraph({"k_frame": 2, "k_lock": 2, "e_lock": 0.1})
    before = None
    for i in range(5):
        g.ingest("alpha, beta", now=float(i * 10))
        if i == 0:
            before = dict(g.nodes["p:alpha"])
    check("a frozen frame trips the stall guard", g.stalled is not None)
    check("the stall guard only flags — activations still just decayed, not reset",
          g.nodes["p:alpha"]["activation"] > 0 and before is not None)

    # A varied conversation keeps the frame moving → no stall.
    g2 = wg.WorkingGraph({"k_frame": 2, "k_lock": 2, "e_lock": 0.1})
    for i, t in enumerate(["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel", "india juliet"]):
        g2.ingest(t, now=float(i * 10))
    check("a varied conversation does not trip the stall guard", g2.stalled is None)

    # Determinism extends to the instrument: same turns → identical metrics.
    def run():
        gg = wg.WorkingGraph()
        for i, t in enumerate(["release plan slipped", "release plan again", "coffee break", "release plan risks"]):
            gg.ingest(t, now=float(i * 30))
        return gg.last_metrics
    check("metrics are deterministic on replay", run() == run())


def test_snapshot():
    g = wg.WorkingGraph()
    g.ingest("the database index rebuild is slow and risky", now=0.0)
    g.ingest("the database index rebuild keeps failing", now=20.0)
    snap = g.snapshot(max_nodes=5, max_edges=5)
    check("snapshot carries nodes/edges/metrics/turns/stalled",
          {"nodes", "edges", "metrics", "turns", "stalled"} <= set(snap))
    check("snapshot nodes are capped + carry id/label/activation/frame",
          len(snap["nodes"]) <= 5 and all({"id", "label", "activation", "frame"} <= set(n) for n in snap["nodes"]))
    ids = {n["id"] for n in snap["nodes"]}
    check("snapshot edges are capped and only connect shown nodes",
          len(snap["edges"]) <= 5 and all(e["a"] in ids and e["b"] in ids for e in snap["edges"]))
    check("snapshot nodes are hottest-first",
          all(snap["nodes"][i]["activation"] >= snap["nodes"][i + 1]["activation"]
              for i in range(len(snap["nodes"]) - 1)))
    check("snapshot is JSON-serialisable", isinstance(json.dumps(snap), str))


def test_persistence():
    g = wg.WorkingGraph()
    g.ingest("we spent ages on the server migration and the database index", now=0.0)
    g.ingest("the server migration is the big risk this week", now=30.0)
    blob = g.to_dict()
    check("to_dict is JSON-serialisable", isinstance(json.dumps(blob), str))
    saved = blob["nodes"]["p:server migration"]["activation"]
    check("to_dict carries nodes with last_ts + edges with last_ts",
          "last_ts" in blob["nodes"]["p:server migration"] and blob["edges"]
          and "last_ts" in blob["edges"][0])

    # Reload a day later into a fresh graph as dormant associations.
    g2 = wg.WorkingGraph({"persist_tau_s": 604800.0, "carry_factor": 0.6})
    carried = g2.load_persisted(blob, now=86400.0)                 # +1 day
    check("nodes are carried over", carried > 0 and "p:server migration" in g2.nodes)
    check("carried nodes are dormant (primed)", g2.nodes["p:server migration"]["primed"] is True)
    check("dormant carried nodes are held OUT of the frame (no bleed)", g2.frame == [])
    check("carried briefing is empty until re-mentioned", g2.briefing() == "")
    check("carried activation is demoted below its saved value",
          g2.nodes["p:server migration"]["activation"] < saved * 0.8)

    # Re-mention lifts dormancy → it wakes into the frame.  (Sentence ends on the phrase so
    # the extractor yields exactly "server migration", not a merged longer phrase.)
    g2.ingest("what about the server migration", now=86400.0 + 10)
    check("re-mention lifts dormancy", g2.nodes["p:server migration"]["primed"] is False)
    check("re-mentioned carried node enters the frame", "p:server migration" in g2.frame)
    check("its briefing now surfaces it", "server migration" in g2.briefing())

    # Load is deterministic (same blob + now → identical graph state).
    a = wg.WorkingGraph({"persist_tau_s": 604800.0}); a.load_persisted(blob, now=86400.0)
    b = wg.WorkingGraph({"persist_tau_s": 604800.0}); b.load_persisted(blob, now=86400.0)
    check("load is deterministic",
          {k: round(v["activation"], 12) for k, v in a.nodes.items()}
          == {k: round(v["activation"], 12) for k, v in b.nodes.items()})


def test_cross_session_decay_and_dual_clock():
    g = wg.WorkingGraph(); g.ingest("quarterly budget", now=0.0)
    blob = g.to_dict()

    near = wg.WorkingGraph({"persist_tau_s": 604800.0, "carry_factor": 0.6})
    near.load_persisted(blob, now=3600.0)                          # +1h
    far = wg.WorkingGraph({"persist_tau_s": 604800.0, "carry_factor": 0.6})
    far.load_persisted(blob, now=3600.0 + 60 * 604800)            # +60 weeks
    check("a recently-touched association survives the gap", "p:quarterly budget" in near.nodes)
    check("a long-unused association wanes away entirely", "p:quarterly budget" not in far.nodes)

    # Dual clock: a dormant carried node decays on the SLOW tau within a session (it lingers),
    # not on the fast attention tau it would if it were active.
    g3 = wg.WorkingGraph({"persist_tau_s": 604800.0, "carry_factor": 0.6, "tau_s": 900})
    g3.load_persisted(blob, now=0.0)
    a0 = g3.nodes["p:quarterly budget"]["activation"]
    g3.ingest("something completely unrelated here", now=900.0)   # one fast-tau elapsed
    a1 = g3.nodes["p:quarterly budget"]["activation"]
    check("a dormant node barely fades over one fast-tau (slow clock, not attention clock)",
          a1 > a0 * 0.98)


def main():
    test_keyphrases()
    test_ingest_and_frame()
    test_thread_persists_and_one_offs_fade()
    test_decay_math()
    test_bounded()
    test_grounded()
    test_briefing()
    test_replay_determinism()
    test_metrics()
    test_stall_guard()
    test_snapshot()
    test_persistence()
    test_cross_session_decay_and_dual_clock()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
