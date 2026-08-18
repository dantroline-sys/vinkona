#!/usr/bin/env python
"""VIN-TOOL-01 executor + self-test protocol (graphrun.py) — the TG4 stage.

The §11 criteria pinned here: a mutate/fs-write step in dry-run leaves the
real store untouched and produces a valid snapshot; runtime edges are
shape-checked; provenance answers "which step ate everything"; recorded live
traffic replays deterministically for the next shadow pass.

    python assistant/test_graphrun.py
"""
import tempfile
from pathlib import Path

import blocks
import palette  # noqa: F401 — populates the registry
import graphrun
import toolgraph
from blocks import Block, BlockError, ReplayCtx
from graphrun import LiveCtx, RecordingCtx, run_graph, selftest

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


def news_pinned(terms=("rain",)):
    g = {"name": "rain_news", "goal": "Rain headlines from my feed.",
         "steps": [
             {"id": "s1", "block": "rss_fetch",
              "inputs": {"feed": {"url": "http://n/feed"}}},
             {"id": "s2", "block": "filter_predicate",
              "params": {"terms": list(terms)},
              "inputs": {"docs": "$s1.docs"}},
             {"id": "s3", "block": "sort_rank",
              "inputs": {"docs": "$s2.docs", "query": {"text": "rain"}}},
             {"id": "s4", "block": "digest_render",
              "inputs": {"ranked": "$s3.ranked"}}],
         "outputs": {"result": "$s4.digest"}}
    v = toolgraph.validate(g)
    assert v.ok, v.errors
    return v.pinned


def test_executor():
    res = run_graph(news_pinned(), ReplayCtx({"net": {"http://n/feed": _XML}}))
    check("the news graph runs end to end", res["ok"], res["error"])
    check("the digest carries the matching headline",
          "Rain over Alpha" in res["outputs"]["result"]["text"]
          and res["outputs"]["result"]["count"] == 1)
    check("provenance has one row per step, with counts",
          [r["id"] for r in res["steps"]] == ["s1", "s2", "s3", "s4"]
          and res["steps"][0]["out"] == {"docs": 2}
          and res["steps"][1]["out"] == {"docs": 1})

    miss = run_graph(news_pinned(), ReplayCtx({}))
    check("a replay miss fails loudly at the fetching step",
          not miss["ok"] and "s1" in miss["error"] and "replay miss" in miss["error"])
    check("…and the trail still shows the failed step",
          miss["steps"][0]["ok"] is False)


def test_runtime_shape_enforcement():
    blocks.register(Block(
        name="dummy_badshape", version="1.0.0", summary="returns a non-Query",
        ports_in={"q": "Query"}, ports_out={"q": "Query"}, params={},
        capabilities=frozenset(), failure_modes=("always wrong",),
        fixtures=({"name": "a", "inputs": {"q": {"text": "x"}}},
                  {"name": "b", "inputs": {"q": {"text": "y"}}}),
        fn=lambda i, p, c: {"q": {"nope": 1}}))
    g = {"name": "bad", "goal": "g",
         "steps": [{"id": "s1", "block": "dummy_badshape",
                    "inputs": {"q": {"text": "hi"}}}],
         "outputs": {"result": "$s1.q"}}
    v = toolgraph.validate(g)
    res = run_graph(v.pinned, ReplayCtx({}))
    check("a wrong-shaped output stops the run at that edge",
          not res["ok"] and "not a valid Query" in res["error"])


def test_write_containment_and_rollback():
    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "store"
        store.mkdir()
        (store / "digests").mkdir()
        (store / "digests" / "latest.txt").write_text("OLD")

        live = LiveCtx(store_dir=store)
        p = live.write("digests/latest.txt", "NEW")
        check("a live write lands inside the store",
              Path(p).read_text() == "NEW" and Path(p).parent == store / "digests")
        check("the overwrite left a valid snapshot",
              (store / "digests" / "latest.txt.snap").read_text() == "OLD")
        live.write("digests/fresh.txt", "CREATED")
        live.rollback()
        check("rollback restores the old content and removes created files",
              (store / "digests" / "latest.txt").read_text() == "OLD"
              and not (store / "digests" / "fresh.txt").exists())

        try:
            live.write("../escape.txt", "x")
            check("a write escaping the store is refused", False)
        except BlockError as e:
            check("a write escaping the store is refused", "escapes" in str(e))

        scratch = Path(td) / "scratch"
        scratch.mkdir()
        dry = LiveCtx(store_dir=store, dry=True, scratch_dir=scratch)
        dry.write("digests/latest.txt", "DRY")
        check("dry-mode writes go to scratch; the store is untouched",
              (scratch / "digests" / "latest.txt").read_text() == "DRY"
              and (store / "digests" / "latest.txt").read_text() == "OLD")
        try:
            LiveCtx(store_dir=store, dry=True).write("a.txt", "x")
            check("dry mode without a scratch dir refuses to write", False)
        except BlockError:
            check("dry mode without a scratch dir refuses to write", True)


def test_recording_replays_deterministically():
    inner = ReplayCtx({"net": {"http://n/feed": _XML}})
    rec = RecordingCtx(inner)
    first = run_graph(news_pinned(), rec)
    check("recording captured the live traffic",
          rec.recorded["net"] == {"http://n/feed": _XML})
    again = run_graph(news_pinned(), ReplayCtx(rec.recorded))
    check("the recorded traffic replays to identical outputs",
          first["ok"] and again["outputs"] == first["outputs"])


def test_selftest_protocol():
    # static catch: a tampered pin fails before anything runs
    broken = news_pinned()
    broken["steps"][0]["block_version"] = "9.9.9"
    st = selftest(broken)
    check("self-test stage 1 catches a bad version pin",
          not st["passed"] and not st["stages"]["static"]["ok"])

    # stages 1–2 without a live ctx: cheap emission-time gate
    st = selftest(news_pinned())
    check("without a live ctx the dry-run is explicitly skipped",
          st["passed"] and st["stages"]["dry_run"]["skipped"] is True)

    # full protocol with a live ctx; her appraisal hook sees the evidence
    seen = {}

    def appraise(evidence):
        seen["evidence"] = evidence
        return "The rain headline came through; this matches the goal."

    live = LiveCtx(net=lambda url: _XML)
    st = selftest(news_pinned(), live, appraise=appraise)
    check("the full self-test passes on a working graph", st["passed"],
          str(st["stages"]))
    check("the dry-run recorded traffic for future shadow passes",
          st["recorded"] and "http://n/feed" in st["recorded"]["net"])
    check("her appraisal hook received the provenance evidence",
          "s2" in seen["evidence"] and st["appraisal"].startswith("The rain"))

    # the recorded traffic powers the NEXT self-test's shadow compose
    st2 = selftest(news_pinned(), LiveCtx(net=lambda url: _XML),
                   recorded=st["recorded"])
    check("a later self-test composes end-to-end from the recording",
          st2["passed"] and st2["stages"]["fixtures"]["shadow_composed"] is True)

    # the classic invisible failure: a filter that eats everything is NOTED
    st3 = selftest(news_pinned(terms=("blizzard",)), LiveCtx(net=lambda url: _XML))
    check("a step that ate everything is noted, not hidden",
          st3["passed"] and any("s2" in n and "produced none" in n
                                for n in st3["notes"]))


def test_fswrite_dry_run():
    # §11: a mutate/fs-write step in dry-run leaves the real store untouched.
    g = {"name": "save_rain", "goal": "Save the rain digest.",
         "steps": [
             {"id": "s1", "block": "rss_fetch",
              "inputs": {"feed": {"url": "http://n/feed"}}},
             {"id": "s2", "block": "sort_rank",
              "inputs": {"docs": "$s1.docs", "query": {"text": "rain"}}},
             {"id": "s3", "block": "digest_render",
              "inputs": {"ranked": "$s2.ranked"}},
             {"id": "s4", "block": "store_write",
              "inputs": {"digest": "$s3.digest"}}],
         "outputs": {"result": "$s4.file"}}
    v = toolgraph.validate(g)
    assert v.ok, v.errors
    check("the graph's capability set includes fs-write",
          v.capabilities == {"net", "fs-write"})
    check("net+fs-write together trip the approval gate",
          toolgraph.needs_approval(v))

    with tempfile.TemporaryDirectory() as td:
        store, scratch = Path(td) / "store", Path(td) / "scratch"
        store.mkdir(); scratch.mkdir()

        st = selftest(v.pinned, LiveCtx(net=lambda url: _XML, store_dir=store))
        check("an fs-write dry-run without scratch is refused",
              not st["passed"]
              and "scratch" in st["stages"]["dry_run"]["errors"][0])

        st = selftest(v.pinned, LiveCtx(net=lambda url: _XML, store_dir=store,
                                        scratch_dir=scratch))
        check("with scratch, the dry-run passes and the store stays untouched",
              st["passed"] and not list(store.rglob("*"))
              and (scratch / "digests" / "latest.txt").exists())

        live = LiveCtx(net=lambda url: _XML, store_dir=store)
        res = run_graph(v.pinned, live)
        check("the live run writes the digest into the store",
              res["ok"]
              and "Rain over Alpha" in (store / "digests" / "latest.txt").read_text())


def main():
    test_executor()
    test_runtime_shape_enforcement()
    test_write_containment_and_rollback()
    test_recording_replays_deterministically()
    test_selftest_protocol()
    test_fswrite_dry_run()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
