"""Retrieval subsystem — the stable v2 boundary (retrieval contract §1).

This is the seam the whole read-path effort hangs off:

    retrieve(query, context) -> RetrievalResult

Everything above it (Vinkona, the eval harness, the reflection loop) speaks only in
``RetrievalContext`` / ``RetrievalResult`` / ``AnswerTier`` — so we can swap the
implementation underneath (add card-BM25, a cross-encoder, a *trained* query→card
linker, LM-hypothesised gap reports) and compare them with the SAME harness, without
touching a single caller.  A retriever is just a callable registered by name; the
harness runs whichever one you name.

**Augment, don't replace.**  The first registered retriever (`current_path`) is a thin
adapter over the existing ``query.answer`` — so the harness measures what we ship today
as the baseline, and new retrievers are added beside it, never in place of the tested
fit-gate / firewall / provenance-banding path.

Design intent (contract §0): honesty over helpfulness.  A result is one of four
explicit tiers; the subsystem must never dress a low-confidence result as Tier 1.
Every result carries a ``debug`` trace so a bad answer can be located fast.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict

log = logging.getLogger("knowledgehost.retrieval")
trace = logging.getLogger("knowledgehost.retrieval.trace")   # per-query debug channel


# ── tiers (contract §6.1) ────────────────────────────────────────────────────
class Tier:
    CONFIDENT = 1     # top card strong + agreed + margin      → answer
    PARTIAL = 2       # plausible but a Tier-1 condition failed → hedged answer
    ASSOCIATIVE = 3   # no fit, but entities/related exist      → "unsure; possibly related …"
    ABSTAIN = 4       # nothing linked and nothing fit          → "I don't know"

    NAMES = {1: "confident", 2: "partial", 3: "associative", 4: "abstain"}


@dataclass
class RetrievalContext:
    """Explicit per-call context (no globals — keeps the harness pure, contract §1)."""
    caller: str = "reflection"            # realtime | reflection | researcher
    topic_entities: tuple = ()            # conversation entities (surfaces or node ids); may be empty
    mode: str | None = None               # epistemic mode filter (science/…); None = general
    rigor: str | None = None              # low | high | None(auto)
    context_features: object = None       # structured discriminators for the fit-gate
    facets: object = None                 # additive facet filter {axis: value(s)} (facets.py)
    k: int = 6                            # answer breadth
    pool: int = 50                        # recall depth before ranking


@dataclass
class Candidate:
    card_id: str
    kind: str = "card"                    # card | node | edge
    score: float | None = None
    label: str = ""
    channels: tuple = ()                  # which recall channels surfaced it (provenance, §4)


@dataclass
class GapReport:
    """Machine-readable abstention → the researcher/reflection write path (contract §6.3).
    ``hypothesis`` is rule-derived here; the large-LM refinement is a later step."""
    query: str
    tier: int
    unlinked_entities: list = field(default_factory=list)
    best_rejected: list = field(default_factory=list)     # [{card_id, score, channels}]
    linked_but_sparse: list = field(default_factory=list) # [{node_id, card_count}]
    hypothesis: str = "ambiguous_query"   # missing_card | missing_node | missing_edge | ambiguous_query
    structured_query: dict | None = None


@dataclass
class RetrievalResult:
    query: str
    tier: int
    candidates: list = field(default_factory=list)        # ranked Candidate list (may be [] at tier 4)
    top_score: float | None = None
    band: str | None = None               # underlying grounding band (debug bridge to today's path)
    gap: GapReport | None = None
    latency_ms: float = 0.0
    caller: str = "reflection"
    debug: dict = field(default_factory=dict)             # per-stage trace (explicit logging)

    @property
    def tier_name(self) -> str:
        return Tier.NAMES.get(self.tier, "?")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier_name"] = self.tier_name
        return d


# ── band → tier bridge (measuring today's path against the 4-tier model) ─────
# The current read path emits a 5-value grounding band; map it onto the contract's
# tiers so the harness can score today's behaviour.  This mapping is the ONE place the
# old and new vocabularies meet — when real fitted thresholds (contract §8.3) land, the
# tiering moves into a dedicated stage and this bridge is retired.
def band_to_tier(band: str | None, *, has_candidates: bool, has_related: bool) -> int:
    if band == "high":
        return Tier.CONFIDENT
    if band in ("medium", "low", "contra"):
        return Tier.PARTIAL
    # band == "none" (or unknown): abstain — associative only if we can still name relatives.
    return Tier.ASSOCIATIVE if has_related else Tier.ABSTAIN


# ── retriever registry (the pluggable seam) ──────────────────────────────────
_REGISTRY: dict = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_retriever(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown retriever {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list:
    return sorted(_REGISTRY)


# A cross-encoder reranker is built once and reused (keeps the endpoint-warn state,
# avoids per-query rebuilds).  Keyed by rerank endpoint so a config change rebuilds it.
_RERANKER: dict = {}


def _get_reranker(cfg):
    key = (cfg.get("rerank"), cfg.get("rerank_url"), cfg.get("rerank_model"))
    if key not in _RERANKER:
        from . import rerank
        _RERANKER[key] = rerank.make_reranker(cfg)
    return _RERANKER[key]


def _bundle_to_result(query, ctx, bundle, latency, *, channel="current") -> RetrievalResult:
    """Normalise a ``query.answer`` bundle into a RetrievalResult (shared by every
    query.answer-backed retriever, so they tier + gap-report identically)."""
    items = bundle.get("items") or []
    related = bundle.get("related") or []
    band = bundle.get("grounding")
    cands = [Candidate(card_id=it.get("id"), kind=it.get("kind") or "card",
                       score=it.get("score"), label=it.get("label") or "",
                       channels=(channel,)) for it in items if it.get("id")]
    tier = band_to_tier(band, has_candidates=bool(cands), has_related=bool(related))
    gap = None
    if tier >= Tier.ASSOCIATIVE:
        gap = GapReport(
            query=query, tier=tier,
            best_rejected=[{"card_id": c.card_id, "score": c.score,
                            "channels": list(c.channels)} for c in cands[:3]],
            hypothesis="missing_card" if cands else "ambiguous_query")
    res = RetrievalResult(
        query=query, tier=tier, candidates=cands,
        top_score=(cands[0].score if cands else None),
        band=band, gap=gap, latency_ms=latency, caller=ctx.caller,
        debug={"intent": bundle.get("intent"), "confidence": bundle.get("confidence"),
               "fit": bundle.get("fit"), "n_items": len(items),
               "n_related": len(related), "abstain": bundle.get("abstain"),
               "channel": channel})
    trace.debug("q=%r [%s] tier=%d(%s) band=%s top=%.4f items=%d related=%d %.1fms",
                query, channel, tier, res.tier_name, band,
                (res.top_score or 0.0), len(items), len(related), latency)
    return res


def _answer_kwargs(ctx, cfg) -> dict:
    return dict(rigor=ctx.rigor, k=ctx.k, mode=ctx.mode,
                context_features=ctx.context_features, facets=ctx.facets, pool=ctx.pool,
                prior_penalty=float(cfg.get("ask_prior_penalty", 0.5)),
                vinkona_penalty=float(cfg.get("ask_vinkona_penalty", 0.85)),
                fit_gate=bool(cfg.get("ask_fit_gate", True)),
                use_spacy=bool(cfg.get("use_spacy", False)),
                spacy_model=cfg.get("spacy_model", "en_core_web_sm"))


# ── retriever #1: the shipping read path (the baseline) ──────────────────────
@register("current_path")
def current_path(query: str, ctx: RetrievalContext, *, kb, embedder, cfg) -> RetrievalResult:
    """Baseline: the existing dense + graph-walk path, exactly as shipped.  Every later
    retriever is scored against this."""
    from . import query as query_mod
    t0 = time.perf_counter()
    bundle = query_mod.answer(kb, embedder, query, **_answer_kwargs(ctx, cfg))
    return _bundle_to_result(query, ctx, bundle, (time.perf_counter() - t0) * 1e3,
                             channel="current")


# ── retriever #2: card-BM25 + cross-encoder rerank (contract step 2) ─────────
@register("card_hybrid")
def card_hybrid(query: str, ctx: RetrievalContext, *, kb, embedder, cfg) -> RetrievalResult:
    """Adds the lexical BM25 card arm and folds the cross-encoder reranker into the SAME
    path (fit-gate / grounding / structure / provenance all preserved).  Everything above
    is unchanged — this just gives the pool a lexical channel and one comparable scale."""
    from . import query as query_mod
    t0 = time.perf_counter()
    bundle = query_mod.answer(kb, embedder, query, bm25=True, reranker=_get_reranker(cfg),
                              rerank_pool=int(cfg.get("rerank_pool", 40)),
                              **_answer_kwargs(ctx, cfg))
    return _bundle_to_result(query, ctx, bundle, (time.perf_counter() - t0) * 1e3,
                             channel="hybrid")


def retrieve(query: str, ctx: RetrievalContext, *, kb, embedder, cfg,
             retriever: str = "current_path") -> RetrievalResult:
    """Dispatch to a named retriever.  Callers and the harness go through here."""
    return get_retriever(retriever)(query, ctx, kb=kb, embedder=embedder, cfg=cfg)
