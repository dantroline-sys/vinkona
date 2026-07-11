"""Deterministic query understanding (retrieval contract Stage A — mechanical form).

The core idea: an utterance's **illocutionary force** — what the speaker is *doing*
(asking, checking permission, narrating a past event, hypothesising, reporting a present
action) — sets **which way to traverse the graph** from the same anchor.  "phenylephrine"
is one node, but "what is it", "can we give it", "we gave it", and "what if we hadn't"
each want a different walk.

So we split the job the way the calendar/retrieval contracts do — *classify, then the
code resolves*:

  1. **speech act**   — a closed set, detected by surface morphology (modality, tense,
     negation, question-form, deixis).  µs, CPU-only, offline.  The LM (Stage A proper)
     is a later drop-in for the ambiguous tail, emitting this same enum.
  2. **traversal plan** — a fixed table: speech act → which direction/edges/cards to
     gather.  Deterministic and inspectable; the part that MUST be programmatic.
  3. ambiguity        — not guessed.  Low confidence → `broaden` (present the map) or ask
     the one clarifying question (the caller's belief-state loop).

This module is layers 1–2.  It has no model dependency and is fully unit-testable.
"""
from __future__ import annotations

import re

SPEECH_ACTS = ("factual", "feasibility", "hypothetical", "counterfactual_omit",
               "observation", "in_progress", "diagnostic", "comparison", "ambiguous")

# speech act → traversal plan over the EXISTING graph ops.  `intent` picks the primary
# walk (None ⇒ defer to the intent classifier); the flags request additive role-pulls.
TRAVERSAL_PLANS: dict = {
    "factual":            {"intent": None},
    "feasibility":        {"intent": "how", "safety": True, "requirements": True},
    "hypothetical":       {"intent": "what_if"},
    "counterfactual_omit": {"intent": "what_if", "counterfactual": True},
    "observation":        {"intent": "what_if", "criteria": True, "cards": True},
    "in_progress":        {"intent": "how", "mid_procedure": True},
    "diagnostic":         {"intent": "why_diag", "criteria": True},
    "comparison":         {"intent": "what", "alternatives": True},
    "ambiguous":          {"intent": None, "broaden": True},
}

# ── surface patterns (ordered: most specific first) ──────────────────────────
_WH = re.compile(r"\b(what|how|which|who|whom|whose|where|when|why)\b", re.I)
_COMPARISON = re.compile(
    r"\b(vs|versus)\b|\bwhich (?:is|one|would).{0,30}\b(better|best|prefer|choose|rather)\b"
    r"|\bcompare\b|\b(better|preferable) than\b|\b\w+ or \w+\s*\?", re.I)
_CF_OMIT = re.compile(
    r"\bwhat if\b.{0,40}\b(did ?n'?t|do ?n'?t|does ?n'?t|had ?n'?t|have ?n'?t|were ?n'?t|"
    r"not|without|skip|skipp?ed|omit|fail(?:ed)? to|stop(?:ped)?|no longer|never)\b"
    r"|\bif (?:i|we|you|he|she|they) (?:did ?n'?t|had ?n'?t|do ?n'?t|stop|skip|never)\b"
    r"|\bwithout (?:giving|doing|using|taking)\b", re.I)
_HYPOTHETICAL = re.compile(
    r"\bwhat if\b|\bwhat happens if\b|\bwhat would happen\b|\bsuppose\b|\bif i (?:do|give|"
    r"take|use|add|try|apply|start)\b|\bwhat about (?:if|when)\b", re.I)
_FEASIBILITY = re.compile(
    r"\b(can|could|may|might|should|shall) (?:i|we|you|one)\b"
    r"|\bis it (?:safe|ok|okay|wise|advisable|appropriate|acceptable|possible|alright|"
    r"fine|reasonable) to\b"
    r"|\bare (?:we|you) (?:able|allowed|supposed|meant|ok) to\b"
    r"|\bam i (?:allowed|able|supposed|meant|ok) to\b"
    r"|\bdo (?:i|we) (?:need|have) to\b|\bis it contraindicated\b", re.I)
_IN_PROGRESS = re.compile(
    r"\b(?:now|currently|right now)\b.{0,12}\bi'?m?\b|\bi'?m (?:about to|currently|now|"
    r"in the (?:middle|process) of|going to|trying to)\b"
    r"|\b(?:i'?m|we'?re|i am|we are) \w+ing\b"
    r"|\band now (?:i|we)\b|\bat this (?:point|stage)\b", re.I)
_DIAGNOSTIC = re.compile(
    r"\bwhy\b|\bwhat'?s causing\b|\bwhat is causing\b|\b(?:the )?(?:cause|reason|aetiology|"
    r"etiology) (?:of|for|behind)\b|\bwhat'?s (?:behind|driving|going on)\b"
    r"|\bwhat could (?:be )?(?:cause|explain)\b|\bdifferential\b", re.I)
# narration: a subject leading a past/state clause, no question
_OBSERVE_SUBJ = re.compile(
    r"^\s*(?:we|he|she|they|it|i|the patient|the pt|patient|my \w+|the \w+)\b", re.I)
_OBSERVE_VERB = re.compile(
    r"\b(had|has|have|was|were|is|are|did|gave|took|developed|presented|showed|shows|"
    r"presents|became|got|felt|started|noticed|complained|reported|underwent|received|"
    r"has been|turned)\b", re.I)


def _has(pat, q):
    return bool(pat.search(q))


def classify_speech_act(query: str) -> tuple:
    """(speech_act, confidence).  Deterministic, order-sensitive: the most specific
    illocutionary cue wins.  Confidence is a coarse rules-certainty for the belief-state."""
    q = " ".join((query or "").split())
    if not q:
        return "ambiguous", 0.0
    ql = q.lower()
    is_question = q.endswith("?") or bool(_WH.match(q))

    if _has(_COMPARISON, ql):
        return "comparison", 0.8
    if _has(_CF_OMIT, ql):
        return "counterfactual_omit", 0.85
    if _has(_FEASIBILITY, ql):
        return "feasibility", 0.85
    if _has(_HYPOTHETICAL, ql):
        return "hypothetical", 0.8
    if _has(_DIAGNOSTIC, ql):
        return "diagnostic", 0.8
    if _has(_IN_PROGRESS, ql):
        return "in_progress", 0.75
    # narration: subject-led past/state clause that is NOT a wh-question
    if not is_question and _has(_OBSERVE_SUBJ, ql) and _has(_OBSERVE_VERB, ql) \
            and not _WH.search(ql):
        return "observation", 0.7
    if is_question or _has(_WH, ql):
        return "factual", 0.6
    return "ambiguous", 0.3


def analyze(query: str, intent_override: str | None = None,
            intent_classifier=None, *, use_spacy=False,
            spacy_model="en_core_web_sm") -> dict:
    """Full mechanical understanding: the speech act, the resolved primary `intent` for
    the graph walk, the additive `flags` (safety / criteria / alternatives / counterfactual
    / broaden …), and a confidence.  `intent_override` (an explicit caller intent) wins;
    else the plan's intent; else the passed intent_classifier (grounding.classify_intent).

    When `use_spacy` and the parser is installed, it is enriched with structural fields —
    `focus` (the head concept, so recall can search on it not the whole diluted string),
    `entities` (candidate anchor spans) and `negations` — WITHOUT changing the speech-act
    label (the regex layer still decides).  Absent spaCy, those fields are simply omitted."""
    act, conf = classify_speech_act(query)
    plan = TRAVERSAL_PLANS.get(act, {"intent": None})
    intent = intent_override or plan.get("intent")
    if not intent:
        intent = (intent_classifier(query) if intent_classifier else "what")
    flags = {k: v for k, v in plan.items() if k != "intent"}
    out = {"speech_act": act, "intent": intent, "flags": flags,
           "confidence": round(conf, 2)}
    if use_spacy:
        struct = parse_structure(query, spacy_model)
        for key in ("focus", "entities", "negations"):
            if struct.get(key):
                out[key] = struct[key]
    return out


# ── optional spaCy structure extraction (guarded; degrades to nothing) ───────
# spaCy sits between regex (µs) and an LLM (seconds): a short-sentence parse is ~1-5 ms on
# CPU — the "structure without the LM tax" option.  It's an OPTIONAL dependency: absent, the
# regex path above stands alone.  It adds NO tunable knobs — its output only contributes
# extra candidates to the existing recall pool, which the reranker + fit-gate already sort.
_NLP = None
_NLP_TRIED = False
_LEADING_DET = re.compile(r"^(the|a|an|my|his|her|their|our|your|its|this|that|these|those)\s+",
                          re.I)


def _get_nlp(model: str):
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy
        # parser + tagger only — we don't need NER/lemmatizer, so this loads/runs leaner.
        _NLP = spacy.load(model, disable=["ner", "lemmatizer"])
    except Exception as e:                          # not installed / model absent
        import logging
        logging.getLogger("knowledgehost.understand").info(
            "spaCy requested but unavailable (%s) — using the regex-only path "
            "(pip install spacy && python -m spacy download %s to enable)", e, model)
        _NLP = None
    return _NLP


def _clean_np(text: str) -> str:
    return _LEADING_DET.sub("", (text or "").strip()).strip()


def parse_structure(query: str, model: str = "en_core_web_sm") -> dict:
    """Best-effort structural read of a query via spaCy.  Returns {} when spaCy/the model
    is unavailable, so callers degrade to whole-query behaviour.  No knobs, no thresholds —
    just the head `focus`, candidate `entities`, and negation heads."""
    nlp = _get_nlp(model)
    if nlp is None or not (query or "").strip():
        return {}
    try:
        doc = nlp(query)
    except Exception:                               # pragma: no cover - defensive
        return {}
    chunks = list(doc.noun_chunks)
    entities = []
    for nc in chunks:
        t = _clean_np(nc.text)
        if len(t) > 1 and t.lower() not in (e.lower() for e in entities):
            entities.append(t)
    # focus = what the main predicate acts on (object), else its subject, else the last
    # content noun phrase (queries tend to trail their head concept).
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    focus = None
    if root is not None:
        for deps in (("dobj", "attr", "pobj", "nsubjpass", "acomp"), ("nsubj",)):
            for child in root.children:
                if child.dep_ in deps:
                    span = doc[child.left_edge.i: child.right_edge.i + 1]
                    focus = _clean_np(span.text) or _clean_np(child.text)
                    break
            if focus:
                break
    if not focus and entities:
        focus = entities[-1]
    negations = sorted({t.head.text.lower() for t in doc if t.dep_ == "neg"})
    out = {"entities": entities}
    if focus:
        out["focus"] = focus
    if negations:
        out["negations"] = negations
    return out
