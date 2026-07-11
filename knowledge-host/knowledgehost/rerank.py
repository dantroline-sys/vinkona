"""Fuse the two retrieval arms, then rerank the shortlist.

1. **Reciprocal Rank Fusion** of the dense and sparse candidate lists — the
   single biggest quality lever at encyclopedia scale, and it needs no score
   calibration between the two arms (it fuses *ranks*, not raw scores).

2. **Rerank** the fused shortlist.  This is the natural — and per KNOWLEDGE.md,
   the *only correct* — home for the **intent-conditioned ("favoritism")**
   signal: the caller's optional `intent` shapes what "relevant" means, scored
   **locally**, and is **never added to any outbound query**.  A real cross-
   encoder is the planned drop-in; until then `heuristic` is a transparent
   lexical reranker (term overlap + title/section/intent boosts).  The top
   rerank score doubles as a **confidence signal** for the web-fallback gate.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request

log = logging.getLogger("knowledgehost.rerank")


def _terms(text: str) -> set:
    return {t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2}


def rrf_fuse(*arms, rrf_k: int = 60) -> list:
    """Combine any number of ranked lists by id with RRF: score = Σ 1/(rrf_k + rank).
    Arms (dense, sparse, wikipedia, …) are fused by their shared ``id``."""
    by_id: dict = {}
    fused: dict = {}
    for arm in arms:
        for rank, item in enumerate(arm or []):
            cid = item["id"]
            by_id.setdefault(cid, item)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    out = []
    for cid, s in fused.items():
        item = dict(by_id[cid])
        item["fused"] = s
        out.append(item)
    out.sort(key=lambda d: d["fused"], reverse=True)
    return out


def heuristic_rerank(query: str, intent: str | None, candidates: list) -> list:
    """Lexical reranker with an intent boost.  Returns candidates sorted by a
    0..1-ish `score`, the top of which is the confidence signal.  Cheap,
    explainable, and a stand-in for the planned cross-encoder."""
    q = _terms(query)
    intent_terms = _terms(intent) if intent else set()
    if not q and not intent_terms:
        for c in candidates:
            c["score"] = c.get("fused", 0.0)
        return candidates
    scored = []
    for c in candidates:
        body = _terms(c.get("text"))
        title = _terms(c.get("title")) | _terms(c.get("section"))
        overlap = len(q & body) / (len(q) + 1e-6)
        title_hit = len(q & title) / (len(q) + 1e-6)
        intent_hit = len(intent_terms & body) / (len(intent_terms) + 1e-6) \
            if intent_terms else 0.0
        # blend lexical evidence with the fusion prior (rank-based, already good)
        prior = 1.0 - 1.0 / (1.0 + 50.0 * c.get("fused", 0.0))
        score = (0.55 * overlap + 0.20 * title_hit
                 + 0.15 * intent_hit + 0.30 * prior)
        c2 = dict(c)
        c2["score"] = round(min(1.0, score), 4)
        scored.append(c2)
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored


def rerank(query, intent, candidates, mode="heuristic"):
    if mode == "none" or not candidates:
        for c in candidates:
            c.setdefault("score", c.get("fused", 0.0))
        return candidates
    return heuristic_rerank(query, intent, candidates)


def _intent_boost(intent_terms: set, c: dict) -> float:
    """Small, local-only 'favoritism' nudge layered on the cross-encoder score:
    how much the caller's intent overlaps the passage.  Never leaves the box."""
    if not intent_terms:
        return 0.0
    body = _terms(c.get("text")) | _terms(c.get("title")) | _terms(c.get("section"))
    return 0.10 * (len(intent_terms & body) / (len(intent_terms) + 1e-6))


class CrossEncoderReranker:
    """Cross-encoder rerank via a local reranker endpoint (llama.cpp --reranking,
    Jina/Cohere-style ``/rerank``).  Stdlib-only (urllib), and if the endpoint is
    unreachable it degrades to the transparent lexical `heuristic` reranker so a
    query never fails just because the reranker is down.

    The intent ('favoritism') signal stays **local**: it is applied here as a
    small additive boost on the relevance score, never added to the text the
    reranker (or any model) sees."""

    def __init__(self, cfg: dict):
        self.url = cfg["rerank_url"].rstrip("/")
        self.model = cfg["rerank_model"]
        self.timeout = cfg["rerank_timeout_s"]
        self._warned = False

    def _post(self, query: str, docs: list[str]):
        payload = {"model": self.model, "query": query, "documents": docs}
        req = urllib.request.Request(
            f"{self.url}/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _scores(self, query: str, docs: list[str]):
        """Relevance per doc aligned with `docs`, or None if the endpoint is down."""
        try:
            data = self._post(query, docs)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            if not self._warned:
                log.warning("reranker unreachable (%s) — falling back to heuristic", e)
                self._warned = True
            return None
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            return None
        out: list = [None] * len(docs)
        for item in results:
            i = item.get("index", 0)
            s = item.get("relevance_score", item.get("score"))
            if 0 <= i < len(out) and s is not None:
                out[i] = float(s)
        return out

    def rerank(self, query, intent, candidates):
        if not candidates:
            return candidates
        scores = self._scores(query, [c.get("text") or "" for c in candidates])
        if scores is None:
            return heuristic_rerank(query, intent, candidates)
        # Some reranker builds return raw logits; squash to 0..1 only if needed so
        # the top score stays meaningful as the web-fallback confidence signal.
        vals = [s for s in scores if s is not None]
        squash = bool(vals) and (min(vals) < 0.0 or max(vals) > 1.0)
        intent_terms = _terms(intent) if intent else set()
        out = []
        for c, s in zip(candidates, scores):
            base = c.get("fused", 0.0) if s is None else \
                (1.0 / (1.0 + math.exp(-s)) if squash else s)
            c2 = dict(c)
            c2["score"] = round(min(1.0, base + _intent_boost(intent_terms, c)), 4)
            out.append(c2)
        out.sort(key=lambda d: d["score"], reverse=True)
        return out


class _LocalReranker:
    """Wraps the in-process `rerank()` (heuristic | none) in the same interface."""

    def __init__(self, mode: str):
        self.mode = mode

    def rerank(self, query, intent, candidates):
        return rerank(query, intent, candidates, self.mode)


def make_reranker(cfg: dict):
    """One reranker object for the server's lifetime (keeps the endpoint warning
    state and avoids rebuilding per query)."""
    if cfg.get("rerank") == "cross-encoder":
        return CrossEncoderReranker(cfg)
    return _LocalReranker(cfg.get("rerank", "heuristic"))
