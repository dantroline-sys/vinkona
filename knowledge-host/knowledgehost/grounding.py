"""Read-time machinery for the query path (spec §10-11).

Nothing here is stored: the write path preserved *evidence* (the support set);
this module computes the *verdict* on demand, and only as far as the query's
**rigor** asks for it (§10):

  * low  (default) — attach the provenance bundle, let the model weigh it.
  * high (stakes)  — apply the firewall filter (empirical-only) and compute the
    recency/trust/independence-cleaned ``strength`` (§9.3) to adjudicate.

Plus intent classification (which substrate to consult) and the grounding
confidence band that drives presentation / abstention (§11).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

_HIGH_STAKES = re.compile(
    r"\b(dose|dosage|dosing|drug|interaction|contraindicat|overdose|mg|"
    r"medication|toxic|lethal|allerg|surger|diagnos|treat|symptom|legal|tax)\w*",
    re.I)

_WORDISH = re.compile(r"[a-z0-9]+")


def _terms(s: str) -> set:
    return {t for t in _WORDISH.findall((s or "").lower()) if len(t) > 2}


def classify_intent(q: str) -> str:
    """Cheap intent router (§10).  Supervisor-LLM fallback is a later drop-in."""
    ql = q.lower().strip()
    if ql.startswith(("how do", "how to", "how can", "how should", "steps to")):
        return "how"
    if "what if" in ql or ql.startswith("what would happen"):
        return "what_if"
    # diagnostic why — abduction over candidate causes (§10) — before mechanistic why.
    if (ql.startswith(("what causes", "what could cause", "what might cause",
                       "why might", "why would", "what could explain", "what's causing"))
            or "causes of" in ql or "differential" in ql):
        return "why_diag"
    if ql.startswith("why") or ql.startswith("how does") or ql.startswith("how is"):
        return "why_mech"
    if ql.startswith("who"):
        return "who"
    if ql.startswith("where"):
        return "where"
    if "what kind" in ql or "what type" in ql or "is a kind of" in ql:
        return "taxonomy"
    if ql.startswith("which"):
        return "which"
    return "what"


def default_rigor(q: str) -> str:
    """Stakes decide rigor, not regime (§10).  A cautious heuristic stand-in for
    the supervisor tier: high for safety/▲-consequence wording, else low."""
    return "high" if _HIGH_STAKES.search(q or "") else "low"


# A few natural-language phrasings that map onto vocabulary values the bare token
# match would miss ("both eyes" → bilateral).  Kept small and unambiguous.
_FEATURE_SYNONYMS = {
    "both eyes": "laterality:bilateral", "both sides": "laterality:bilateral",
    "one eye": "laterality:unilateral", "one side": "laterality:unilateral",
    "all over": "laterality:diffuse", "out of nowhere": "onset:sudden",
    "came on slowly": "onset:gradual", "comes and goes": "timing:episodic",
}


@lru_cache(maxsize=1)
def _vocab_matcher():
    """Compile the closed vocabulary ONCE into (value→feature lexicon, one alternation
    regex) instead of re-importing FEATURE_VOCAB and running a regex per value on every
    call.  Values are ordered longest-first so a value that is a prefix of another can't
    shadow it in the alternation.  Rebuilt only if the process reloads the module."""
    from .distill import FEATURE_VOCAB
    lex = {v: feat for feat, vals in FEATURE_VOCAB.items() for v in vals}
    ordered = sorted(lex, key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(re.escape(v) for v in ordered) + r")\w*\b") \
        if ordered else None
    return lex, rx


@lru_cache(maxsize=1024)
def extract_context_features(query: str) -> frozenset:
    """Query-side feature detector (companion spec §7, deterministic form): spot
    shared-vocabulary VALUES in the question (onset 'sudden', laterality 'bilateral',
    quality 'burning', …) so the differential is scored on the SAME features the
    extractor tagged causes with — a clean overlap, not fuzzy text match.  Matches a
    value plus any suffix ('gradually' → gradual) and a few phrase synonyms.

    Memoised (this is called once per candidate item on the hot ask path, always with
    the same query) and returned as an immutable frozenset so the cached value is safe
    to share.  (An LM extractor is the later upgrade.)"""
    ql = (query or "").lower()
    lex, rx = _vocab_matcher()
    feats = set()
    if rx is not None:
        for val in rx.findall(ql):
            feat = lex.get(val)
            if feat:
                feats.add(f"{feat}:{val}")
    for phrase, fv in _FEATURE_SYNONYMS.items():
        if phrase in ql:
            feats.add(fv)
    return frozenset(feats)


def _by_feature(featset) -> dict:
    m: dict = {}
    for f in featset:
        k, _, v = f.partition(":")
        m.setdefault(k, set()).add(v)
    return m


def _pairs(items) -> set:
    """{feature:value} objects → a set of 'feature:value' strings."""
    out = set()
    for d in items or []:
        if isinstance(d, dict) and d.get("feature") and d.get("value"):
            out.add(f"{str(d['feature']).lower()}:{str(d['value']).lower()}")
    return out


def item_features(item: dict) -> set:
    """The structured features an item carries, as 'feature:value' strings the query's
    context_features overlap directly.  For an edge/procedure card that's its
    `discriminators`; a **criteria card** also contributes its `required` + `supportive`
    features (the must-have / may-have of a diagnosis), so matching a presentation against
    a diagnostic definition uses the very same gate."""
    out = _pairs(item.get("discriminators"))
    crit = item.get("criteria")
    if isinstance(crit, dict):
        out |= _pairs(crit.get("required"))
        out |= _pairs(crit.get("supportive"))
    return out


def exclusion_features(item: dict) -> set:
    """A criteria card's must-NOT-have features — presence of any of these in the query
    RULES THE CARD OUT (a differential exclusion), the diagnostic form of abstention."""
    crit = item.get("criteria")
    return _pairs(crit.get("exclusion")) if isinstance(crit, dict) else set()


def _feature_overlap(qf: set, ef: set) -> float:
    """Signed feature agreement in [0,1] from the caller/query features `qf` against the
    item's discriminators `ef`: an exact feature:value match supports it, a same-feature/
    different-value CLASH (caesarean vs labour, spinal vs epidural) counts against it.
    0.0 means a net clash — the item describes a *different* situation than the one asked
    about, the signal that turns 'topically near' into 'wrong answer, abstain'."""
    qg, eg = _by_feature(qf), _by_feature(ef)
    matches = mismatches = 0
    for feat, vals in qg.items():
        if feat in eg:
            if vals & eg[feat]:
                matches += 1
            else:
                mismatches += 1
    return max(0.0, (matches - 0.5 * mismatches) / (len(qg) + 1e-6))


def answer_fit(item: dict, query: str, qfeats=None) -> float:
    """How well an item actually answers THIS question — structured discriminator overlap
    first (on caller-supplied context_features plus any query-detected vocabulary values),
    falling back to term overlap when neither side carries features.  This is the relevance
    signal that proximity (cosine) can't give: 'orange ≈ orange' but a recipe doesn't fit
    'how do I cut one'.  Returns [0,1]."""
    q = _terms(query)
    text = (_terms(item.get("text")) | _terms(item.get("label"))
            | _terms(item.get("mechanism")) | _terms(item.get("conditions")))
    text_score = (len(q & text) / (len(q) + 1e-6)) if q else 0.0
    qf = set(qfeats or set()) | extract_context_features(query)
    # A must-not-have criterion present in the asked-about context rules this card out —
    # e.g. querying 'thunderclap onset' against a card that excludes thunderclap.  This is
    # the diagnostic exclusion: a confident 0, not a weak match.
    if qf and (qf & exclusion_features(item)):
        return 0.0
    ef = item_features(item)
    if not qf or not ef:
        return min(1.0, text_score)                    # no structured signal → text only
    feat_score = _feature_overlap(qf, ef)
    return max(0.0, min(1.0, 0.7 * feat_score + 0.3 * text_score))


def context_fit(query: str, edge: dict, qfeats=None) -> float:
    """Diagnostic abduction scoring (§10): how well a candidate cause matches the query's
    context — structured discriminator overlap, else mechanism/condition term overlap.
    Thin wrapper over answer_fit so the differential and the confidence gate score by the
    exact same rule."""
    if not _terms(query):
        return 0.0
    return answer_fit(edge, query, qfeats)


def read_time_strength(support: list, contra_pressure: float = 0.0,
                       k: float = 2.0, lam: float = 0.5) -> float:
    """Empirical strength (§9.3), computed on demand from the support SET:
    independent (copy-discounted) trust mass, saturating, minus contradiction
    pressure.  Repetition counts only when sources are independent."""
    clusters: dict = {}
    for s in support or []:
        key = s.get("evidence_cluster") or s.get("doc_id")
        clusters[key] = max(clusters.get(key, 0.0), float(s.get("trust_weight") or 0.0))
    raw = sum(clusters.values())                      # independent, copy-discounted
    base = 1.0 - math.exp(-raw / k)                   # diminishing returns
    # recency: dates are frequently absent in this corpus; full decay/supersession
    # is deferred — treat present-but-undated support as current.
    recency = 1.0
    return max(0.0, min(1.0, base * recency - lam * contra_pressure))


def grounding(items: list, top_fit: float = 1.0, *, gated: bool = False,
              clash: bool = False) -> tuple:
    """Confidence in [0,1] and a band (§11), from the top item's evidence — gated by
    relevance when we have a structured signal to judge with.

    Proximity is not relevance: a high-cosine item that doesn't fit the query's
    context_features must not present as confident.  When `gated` (the caller supplied
    context_features, or the query carried vocabulary values), confidence is scaled by
    `top_fit`, and a net feature `clash` — the item describes a different situation than
    the one asked about — abstains outright.  Ungated queries behave exactly as before,
    so nothing regresses for plain look-ups."""
    if not items:
        return 0.0, "none"
    top = items[0]
    n_support = len(top.get("support") or [])
    structured = 0.2 if top.get("kind") in ("node", "edge", "card") else 0.0
    conf = min(1.0, 0.5 * float(top.get("score") or 0.0)
               + 0.3 * min(1.0, n_support / 2.0) + structured)
    if gated:                                          # scale by relevance, never above 1
        conf = conf * (0.5 + 0.5 * max(0.0, min(1.0, top_fit)))
    if top.get("contradictions"):
        band = "contra"
    elif gated and clash:                              # wrong situation → abstain, don't guess
        band = "none"
    elif conf >= 0.66:
        band = "high"
    elif conf >= 0.40:
        band = "medium"
    elif conf > 0.0:
        band = "low"
    else:
        band = "none"
    return round(conf, 4), band


_NOTE = {
    "high": "Well grounded — answer directly.",
    "medium": "Answer, but note any single-/older-source caveat.",
    "low": "Weak grounding — hedge.",
    "contra": "Sources disagree — present both with provenance and which is more trusted/recent.",
    "none": "No grounded answer in the knowledge base.",
}


def note_for(band: str) -> str:
    return _NOTE.get(band, "")
