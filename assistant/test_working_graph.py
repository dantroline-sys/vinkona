#!/usr/bin/env python
"""VIN-WM-02 phase 1a — the deterministic conversational working-memory graph.
Runs on a bare interpreter (no model, no deps): extraction, thread persistence via
decay, boundedness, grounding, the fenced briefing, and exact replay determinism.

    python assistant/test_working_graph.py
"""
import importlib.util
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


def main():
    test_keyphrases()
    test_ingest_and_frame()
    test_thread_persists_and_one_offs_fade()
    test_decay_math()
    test_bounded()
    test_grounded()
    test_briefing()
    test_replay_determinism()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
