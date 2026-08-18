#!/usr/bin/env python
"""VIN-TOOL-01 graph store (graphstore.py) + exclusion sampling — TG5.

The §11 criteria pinned here: a forced regression during probation
auto-disables the graph and yields a well-formed Gap Report; exclusion
sampling surfaces rejected items; a changed block refuses to run under an
old deployment (§7).

    python assistant/test_graphstore.py
"""
import tempfile
from pathlib import Path

import blocks
import palette  # noqa: F401
import graphrun
import toolgraph
from blocks import ReplayCtx
from graphstore import GraphStore, GraphStoreError

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


_XML = ("<rss><channel>"
        "<item><title>Rain over Alpha</title><link>http://n/1</link>"
        "<description>Heavy rain tonight.</description></item>"
        "<item><title>Cup final</title><link>http://n/2</link>"
        "<description>Beta won.</description></item>"
        "</channel></rss>")


def pinned():
    g = {"name": "rain_news", "goal": "Rain headlines from my feed.",
         "steps": [
             {"id": "s1", "block": "rss_fetch",
              "inputs": {"feed": {"url": "http://n/feed"}}},
             {"id": "s2", "block": "filter_predicate",
              "params": {"terms": ["rain"]},
              "inputs": {"docs": "$s1.docs"}},
             {"id": "s3", "block": "sort_rank",
              "inputs": {"docs": "$s2.docs", "query": {"text": "rain"}}},
             {"id": "s4", "block": "digest_render",
              "inputs": {"ranked": "$s3.ranked"}}],
         "outputs": {"result": "$s4.digest"}}
    v = toolgraph.validate(g)
    assert v.ok, v.errors
    return v.pinned


def _ok_run():
    return graphrun.run_graph(pinned(), ReplayCtx({"net": {"http://n/feed": _XML}}))


def test_exclusion_sampling():
    res = _ok_run()
    row = res["steps"][1]                        # the filter
    check("a dropping step records what it rejected",
          row.get("excluded") == 1
          and row["excluded_sample"] == [{"title": "Cup final",
                                          "url": "http://n/2"}])
    check("a non-dropping step records nothing",
          "excluded" not in res["steps"][2])


def test_deploy_and_probation():
    with tempfile.TemporaryDirectory() as td:
        store = GraphStore(td, probation_runs=3)
        doc = store.deploy(pinned(), recorded={"net": {"http://n/feed": _XML}},
                           appraisal="matches the goal")
        check("a deploy starts on probation with the capability sentence",
              doc["state"] == "probation" and doc["probation_left"] == 3
              and doc["capability_summary"] == "This tool can reach the internet.")
        check("the deploy snapshots the hashes of the blocks it uses",
              set(doc["manifest"]) == {"rss_fetch@1.0.0",
                                       "filter_predicate@1.0.0",
                                       "sort_rank@1.0.0", "digest_render@1.0.0"})

        for i in range(3):
            out = store.record_run("rain_news", _ok_run())
        check("N clean runs promote probation to active",
              out["doc"]["state"] == "active"
              and out["doc"]["probation_left"] == 0 and out["gap"] is None)
        check("run summaries keep the exclusion sample",
              out["doc"]["runs"][-1]["steps"][1]["excluded_sample"][0]["title"]
              == "Cup final")

        # an active graph failing is recorded, not disabled
        bad = graphrun.run_graph(pinned(), ReplayCtx({}))
        out = store.record_run("rain_news", bad)
        check("an active graph surviving a bad day stays active",
              out["doc"]["state"] == "active" and out["gap"] is None
              and not out["doc"]["runs"][-1]["ok"])


def test_probation_regression_files_a_gap():
    with tempfile.TemporaryDirectory() as td:
        store = GraphStore(td, probation_runs=5)
        store.deploy(pinned())
        store.record_run("rain_news", _ok_run())
        out = store.record_run("rain_news", graphrun.run_graph(pinned(), ReplayCtx({})))
        doc, gap = out["doc"], out["gap"]
        check("a probation regression auto-disables the graph",
              doc["state"] == "disabled" and "regressed" in doc["disabled_reason"])
        check("…and yields a well-formed §8 Gap Report",
              gap is not None
              and {"kind", "goal", "attempted_graph", "missing",
                   "closest_existing", "evidence", "urgency"} <= set(gap)
              and gap["kind"] == "defect" and gap["attempted_graph"] == "rain_news"
              and "replay miss" in gap["evidence"])
        try:
            store.load_runnable("rain_news")
            check("a disabled graph refuses to load", False)
        except GraphStoreError as e:
            check("a disabled graph refuses to load", "disabled" in str(e))


def test_manifest_drift_refuses():
    with tempfile.TemporaryDirectory() as td:
        store = GraphStore(td)
        doc = store.deploy(pinned())
        g, rec = store.load_runnable("rain_news")
        check("a fresh deployment loads with its recording",
              g["name"] == "rain_news" and rec is None)
        # simulate a block whose source changed since deploy
        doc["manifest"]["rss_fetch@1.0.0"] = "0" * 64
        store._write(doc)
        try:
            store.load_runnable("rain_news")
            check("a changed block refuses to run under an old deploy", False)
        except GraphStoreError as e:
            check("a changed block refuses to run under an old deploy",
                  "rss_fetch@1.0.0" in str(e) and "redeploy" in str(e))


def test_hand_controls():
    with tempfile.TemporaryDirectory() as td:
        store = GraphStore(td)
        store.deploy(pinned())
        check("disable by hand records the reason",
              store.set_state("rain_news", "disabled", "taking a look")
              ["disabled_reason"] == "taking a look")
        check("re-probation resets the countdown",
              store.set_state("rain_news", "probation")["probation_left"] == 5)
        check("a schedule can be set", store.set_schedule("rain_news", 3600)
              ["schedule_s"] == 3600)
        rows = store.list()
        check("list rows carry what the panel shows",
              len(rows) == 1 and rows[0]["name"] == "rain_news"
              and rows[0]["capability_summary"]
              and rows[0]["schedule_s"] == 3600)
        check("redeploying a name resets probation",
              store.deploy(pinned())["probation_left"] == 5)
        check("remove removes", store.remove("rain_news") is True
              and store.get("rain_news") is None)


def main():
    test_exclusion_sampling()
    test_deploy_and_probation()
    test_probation_regression_files_a_gap()
    test_manifest_drift_refuses()
    test_hand_controls()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
