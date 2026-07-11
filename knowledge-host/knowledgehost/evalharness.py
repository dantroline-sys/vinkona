"""Retrieval eval harness (retrieval contract §8) — BUILD FIRST.

The rule the whole v2 effort is gated on: *no threshold tuning, no "improvement", is
accepted without this harness reporting.*  It scores a retriever (via the
``retrieval.retrieve`` seam) against a graded gold set and reports the headline
**abstention-quality** metrics, not just ranking:

  * false-confidence rate  — Tier-1 answers whose top card is graded 0–1.  **≤ 2%.**
  * out-of-graph correctness — genuinely-unanswerable queries returned Tier 3/4. **≥ 90%.**
  * abstention overreach — answerable queries wrongly abstained.  report; target ≤ 25%.
  * ranking: recall@5, recall@20, nDCG@10 on in-graph answerable queries.
  * latency p50/p95 per caller.

The metric functions are pure (list of (gold, result) pairs → numbers) so they unit-test
without the model stack; only ``run()`` needs a live KB + embedder.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.eval")

OUT_OF_GRAPH = "out_of_graph"

# Acceptance gates (contract §8.2 / §8.4).  A run that violates these "fails" — used by
# the regression gate; ranking gains never buy back a false-confidence regression.
ACCEPT_FALSE_CONFIDENCE_MAX = 0.02
ACCEPT_OUT_OF_GRAPH_MIN = 0.90


# ── gold set ─────────────────────────────────────────────────────────────────
def load_gold(path: str) -> list:
    """Load JSONL gold fixtures.  Each line:
      {"id","query","category","judgments":{card_id:0..3},"caller"?,"note"?}
    out-of-graph queries carry category="out_of_graph" and (usually) empty judgments."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}:{i}: bad JSON ({e})")
            if not r.get("query"):
                raise ValueError(f"{path}:{i}: record missing 'query'")
            r.setdefault("id", f"q{i:04d}")
            r.setdefault("category", "lookup_fact")
            r.setdefault("judgments", {})
            rows.append(r)
    return rows


def relevant_cards(rec: dict, min_grade: int = 2) -> set:
    return {cid for cid, g in (rec.get("judgments") or {}).items() if (g or 0) >= min_grade}


def is_out_of_graph(rec: dict) -> bool:
    return rec.get("category") == OUT_OF_GRAPH


def is_answerable(rec: dict) -> bool:
    """In-graph AND has at least one directly/partially relevant card to find."""
    return not is_out_of_graph(rec) and bool(relevant_cards(rec))


# ── metrics (pure: pairs = [(gold_record, RetrievalResult)]) ─────────────────
def _ranked_ids(result) -> list:
    return [c.card_id for c in (result.candidates or []) if c.card_id]


def recall_at_k(rec, result, k: int) -> float | None:
    rel = relevant_cards(rec)
    if not rel:
        return None
    got = set(_ranked_ids(result)[:k])
    return len(rel & got) / len(rel)


def ndcg_at_k(rec, result, k: int = 10) -> float | None:
    judg = rec.get("judgments") or {}
    if not any((g or 0) >= 2 for g in judg.values()):
        return None
    ranked = _ranked_ids(result)[:k]
    dcg = sum((2 ** (judg.get(cid, 0)) - 1) / math.log2(i + 2)
              for i, cid in enumerate(ranked))
    ideal = sorted((g or 0 for g in judg.values()), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return (dcg / idcg) if idcg else 0.0


def _top_grade(rec, result) -> int:
    ranked = _ranked_ids(result)
    if not ranked:
        return 0
    return int((rec.get("judgments") or {}).get(ranked[0], 0))


def compute_metrics(pairs: list) -> dict:
    """pairs = [(gold_record, RetrievalResult)].  Returns the full report dict."""
    from .retrieval import Tier
    n = len(pairs)
    answerable = [(r, res) for r, res in pairs if is_answerable(r)]
    oog = [(r, res) for r, res in pairs if is_out_of_graph(r)]

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.fmean(xs), 4) if xs else None

    # ranking (answerable only)
    ranking = {
        "recall@5": _mean(recall_at_k(r, res, 5) for r, res in answerable),
        "recall@20": _mean(recall_at_k(r, res, 20) for r, res in answerable),
        "ndcg@10": _mean(ndcg_at_k(r, res, 10) for r, res in answerable),
        "n_answerable": len(answerable),
    }

    # false confidence: of Tier-1 results, fraction whose top card is graded 0–1
    tier1 = [(r, res) for r, res in pairs if res.tier == Tier.CONFIDENT]
    false_conf = [1 for r, res in tier1 if _top_grade(r, res) <= 1]
    ranking_false_confidence = (len(false_conf) / len(tier1)) if tier1 else 0.0

    # out-of-graph correctness: fraction of OOG returned Tier 3/4
    oog_correct = [1 for r, res in oog if res.tier in (Tier.ASSOCIATIVE, Tier.ABSTAIN)]
    oog_rate = (len(oog_correct) / len(oog)) if oog else None

    # abstention overreach: answerable queries wrongly returned Tier 3/4
    over = [1 for r, res in answerable if res.tier in (Tier.ASSOCIATIVE, Tier.ABSTAIN)]
    overreach = (len(over) / len(answerable)) if answerable else None

    # latency per caller
    lat: dict = {}
    by_caller: dict = {}
    for _r, res in pairs:
        by_caller.setdefault(res.caller or "?", []).append(res.latency_ms)
    for caller, xs in by_caller.items():
        xs = sorted(xs)
        lat[caller] = {"p50": round(_pct(xs, 50), 1), "p95": round(_pct(xs, 95), 1),
                       "n": len(xs)}

    tier_hist: dict = {}
    for _r, res in pairs:
        tier_hist[res.tier] = tier_hist.get(res.tier, 0) + 1

    metrics = {
        "n_queries": n,
        "ranking": ranking,
        "abstention": {
            "false_confidence_rate": round(ranking_false_confidence, 4),
            "false_confidence_n": f"{len(false_conf)}/{len(tier1)}",
            "out_of_graph_correctness": None if oog_rate is None else round(oog_rate, 4),
            "out_of_graph_n": f"{len(oog_correct)}/{len(oog)}",
            "abstention_overreach": None if overreach is None else round(overreach, 4),
        },
        "tiers": {str(k): tier_hist.get(k, 0) for k in (1, 2, 3, 4)},
        "latency_ms": lat,
    }
    metrics["accept"] = _acceptance(metrics)
    return metrics


def _pct(sorted_xs: list, p: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = (p / 100) * (len(sorted_xs) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (idx - lo)


def _acceptance(metrics: dict) -> dict:
    ab = metrics["abstention"]
    fc = ab["false_confidence_rate"]
    oog = ab["out_of_graph_correctness"]
    fc_ok = fc is not None and fc <= ACCEPT_FALSE_CONFIDENCE_MAX
    oog_ok = oog is None or oog >= ACCEPT_OUT_OF_GRAPH_MIN     # None = no OOG golds yet
    return {"false_confidence_ok": bool(fc_ok), "out_of_graph_ok": bool(oog_ok),
            "passed": bool(fc_ok and oog_ok)}


# ── report formatting ────────────────────────────────────────────────────────
def format_report(metrics: dict, *, retriever: str = "?", run_id: str = "") -> str:
    r = metrics["ranking"]
    a = metrics["abstention"]
    acc = metrics["accept"]
    L = []
    L.append(f"── retrieval eval · retriever={retriever} · run={run_id} ──")
    L.append(f"queries: {metrics['n_queries']}   tiers: "
             + " ".join(f"T{k}={metrics['tiers'][k]}" for k in ("1", "2", "3", "4")))
    L.append(f"ranking (answerable={r['n_answerable']}): "
             f"recall@5={r['recall@5']}  recall@20={r['recall@20']}  ndcg@10={r['ndcg@10']}")
    fc_mark = "✓" if acc["false_confidence_ok"] else "✗"
    oog_mark = "✓" if acc["out_of_graph_ok"] else "✗"
    L.append(f"[{fc_mark}] false-confidence: {a['false_confidence_rate']} "
             f"({a['false_confidence_n']})   (accept ≤ {ACCEPT_FALSE_CONFIDENCE_MAX})")
    L.append(f"[{oog_mark}] out-of-graph correct: {a['out_of_graph_correctness']} "
             f"({a['out_of_graph_n']})   (accept ≥ {ACCEPT_OUT_OF_GRAPH_MIN})")
    L.append(f"    abstention overreach: {a['abstention_overreach']}  (target ≤ 0.25)")
    for caller, s in (metrics["latency_ms"] or {}).items():
        L.append(f"    latency[{caller}]: p50={s['p50']}ms p95={s['p95']}ms (n={s['n']})")
    L.append("RESULT: " + ("PASS ✓" if acc["passed"] else "FAIL ✗ — see gates above"))
    return "\n".join(L)


# ── runner (needs a live KB + embedder) ──────────────────────────────────────
def run(kb, embedder, cfg, *, gold_path: str, retriever: str = "current_path",
        caller: str | None = None, trace: bool = False) -> dict:
    """Score a retriever over the gold set.  Returns the metrics dict and writes a
    JSON run artifact.  Set trace=True for per-query debug logging."""
    from . import retrieval
    if trace:
        logging.getLogger("knowledgehost.retrieval.trace").setLevel(logging.DEBUG)
    gold = load_gold(gold_path)
    log.info("eval: %d gold queries · retriever=%s · gold=%s", len(gold), retriever, gold_path)
    pairs = []
    for rec in gold:
        ctx = retrieval.RetrievalContext(
            caller=caller or rec.get("caller") or "reflection",
            mode=rec.get("mode"), rigor=rec.get("rigor"),
            context_features=rec.get("context_features"),
            k=int(rec.get("k", cfg.get("default_k", 6))))
        try:
            res = retrieval.retrieve(rec["query"], ctx, kb=kb, embedder=embedder,
                                     cfg=cfg, retriever=retriever)
        except Exception as e:                       # a retriever must never crash the harness
            log.error("query %s failed: %s: %s", rec.get("id"), type(e).__name__, e)
            res = retrieval.RetrievalResult(query=rec["query"], tier=retrieval.Tier.ABSTAIN,
                                            caller=ctx.caller, debug={"error": str(e)})
        pairs.append((rec, res))
        if trace:
            log.info("· %s [%s] → T%d(%s) top=%s", rec.get("id"), rec.get("category"),
                     res.tier, res.tier_name,
                     (res.candidates[0].card_id if res.candidates else "—"))
    metrics = compute_metrics(pairs)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + retriever
    metrics["run_id"] = run_id
    metrics["retriever"] = retriever
    metrics["gold_path"] = gold_path
    _write_artifact(cfg, run_id, metrics, pairs)
    print(format_report(metrics, retriever=retriever, run_id=run_id))
    return metrics


def _write_artifact(cfg, run_id, metrics, pairs):
    out = cfg.get("eval_out_dir") or ""
    d = Path(out).expanduser() if out else (Path(cfg.get("control_dir") or ".") / "eval-runs")
    try:
        d.mkdir(parents=True, exist_ok=True)
        per_query = [{"id": r.get("id"), "category": r.get("category"),
                      "tier": res.tier, "top": (res.candidates[0].card_id if res.candidates else None),
                      "top_score": res.top_score, "band": res.band,
                      "latency_ms": round(res.latency_ms, 1), "debug": res.debug}
                     for r, res in pairs]
        (d / f"{run_id}.json").write_text(json.dumps(
            {"metrics": metrics, "per_query": per_query}, indent=2))
        log.info("eval artifact: %s", d / f"{run_id}.json")
    except OSError as e:                              # pragma: no cover - best effort
        log.warning("could not write eval artifact: %s", e)
