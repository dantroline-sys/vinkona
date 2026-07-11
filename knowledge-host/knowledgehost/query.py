"""The online query path (spec §10): intent → retrieve structured items →
grounding → bundle, with the rigor knob deciding how much read-time work to do.

Retrieval produces a grounded candidate set *with a confidence signal*; it never
diagnoses on its own — the inference model reasons, hedges, or abstains using the
bundle.  Low rigor just attaches provenance; high rigor switches on the
empirical-only firewall filter and the recency/trust/strength adjudication (§9.3),
then re-ranks by strength.
"""
from __future__ import annotations

import logging
import time

from . import facets as facets_mod
from . import grounding
from . import understand
from .reconcile_import import COMMONSENSE_DOCS

log = logging.getLogger("knowledgehost.query")


def _is_prior(item) -> bool:
    """True if EVERY backing source is a bulk commonsense import (ConceptNet/ATOMIC/
    GLUCOSE/CauseNet) — an ungrounded background prior, not something distilled from your
    documents.  An item attested by even one real document counts as grounded."""
    sup = item.get("support") or []
    if not sup:
        return False
    return all(s.get("doc_id") in COMMONSENSE_DOCS for s in sup)


def _is_vinkona(item) -> bool:
    """True if EVERY backing source is one of Vinkona's own research drops (provenance
    'vinkona') — distilled and surfaced, but BANDED below curated cards so a textbook
    always wins a tie (research_loop_spec §6 trust posture).  One curated source in the
    support set is enough to lift it out of the vinkona band."""
    sup = item.get("support") or []
    if not sup:
        return False
    return all(s.get("provenance") == "vinkona" for s in sup)
# Dedicated perf logger: one compact line per ask with the stage breakdown, so the
# end-to-end pathway Vinkona hits can be measured without enabling global DEBUG.  Filter
# with `logging.getLogger("knowledgehost.perf")`.
perf = logging.getLogger("knowledgehost.perf")

# Fallback mode table (cfg['modes'] overrides).  None/general/unknown => keep everything.
_DEFAULT_MODES = {
    "science":    {"empirical", "historical"},
    "scholarly":  {"empirical", "historical", "interpretive", "conventional"},
    "humanities": {"interpretive", "conventional", "historical", "empirical"},
    "fiction":    {"fictional", "interpretive", "conventional"},
}


def _allowed_regimes(mode, modes):
    """The regime allow-set for a mode, or None to keep everything."""
    if not mode or mode == "general":
        return None
    table = modes or _DEFAULT_MODES
    allowed = table.get(mode)
    return set(allowed) if allowed else None


def _mode_keep(item, allowed, strict):
    """Two filtering axes (the user's 'best of both worlds'):
      * lenient (default) — judge by the CLAIM's regime, so a real-world technique
        distilled from a novel (claim=empirical) survives 'science' mode;
      * strict — judge by source ORIGIN, so anything from the fiction/essay folder is
        excluded wholesale regardless of how the claim was tagged.
    Strict falls back to the claim regime for pre-origin (old) support."""
    if strict:
        origins = {s.get("origin") for s in (item.get("support") or []) if s.get("origin")}
        if origins:                            # keep if ANY backing source is allowed
            return bool(origins & allowed)
    return (item.get("regime") or "empirical") in allowed


def _features_to_set(cf) -> set:
    """Normalise caller context_features into a 'feature:value' set.  Accepts a dict
    ({"trigger": "spinal anaesthesia", "sign": "hypotension"}, values may be lists) or an
    already-flat list of "feature:value" strings.  Lower-cased so it overlaps the item's
    discriminators regardless of casing."""
    out: set = set()
    if isinstance(cf, dict):
        for feat, val in cf.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            for v in vals:
                if v not in (None, ""):
                    out.add(f"{str(feat).strip().lower()}:{str(v).strip().lower()}")
    elif isinstance(cf, (list, tuple, set)):
        for s in cf:
            s = str(s).strip().lower()
            if ":" in s:
                out.add(s)
    return out


def answer(kb, embedder, query: str, *, rigor=None, k=6, mode=None, modes=None,
           strict=False, prior_penalty=0.5, pool=50, context_features=None,
           intent=None, fit_gate=True, vinkona_penalty=0.85, facets=None,
           bm25=False, reranker=None, rerank_pool=40,
           use_spacy=False, spacy_model="en_core_web_sm") -> dict:
    # Query understanding (understand.py): the utterance's illocutionary form (question /
    # feasibility check / hypothetical / counterfactual / narration / present-action) steers
    # WHICH WAY we walk the graph.  Deterministic + offline (µs); it resolves the primary
    # `intent` and requests additive role-pulls (safety / alternatives / counterfactual …).
    u = understand.analyze(query, intent, grounding.classify_intent,
                           use_spacy=use_spacy, spacy_model=spacy_model)
    intent = u["intent"]
    uflags = u["flags"]
    rigor = rigor or grounding.default_rigor(query)
    high = (rigor == "high")
    allowed = _allowed_regimes(mode, modes)
    qfeats = _features_to_set(context_features)
    # the relevance gate engages only when there IS a structured signal to judge by —
    # caller-supplied context_features, or vocabulary values detected in the query.  Plain
    # look-ups (no features) keep the old embedding-ranked behaviour exactly.
    gate = bool(fit_gate) and (bool(qfeats) or bool(grounding.extract_context_features(query)))

    def _pf(it):                       # ranking factor: demote ungrounded bulk priors,
        f = prior_penalty if _is_prior(it) else 1.0     # and band vinkona below curated
        if vinkona_penalty < 1.0 and _is_vinkona(it):
            f *= vinkona_penalty
        return f

    _t0 = time.perf_counter()
    qvec = embedder.embed_one(query, "query") if embedder else None
    _t_embed = time.perf_counter()
    # Over-fetch a pool, then prefer grounded (document) items: the ~1M commonsense priors
    # otherwise crowd your distilled documents out of the top-k.  The graph-walk below then
    # expands the top GROUNDED nodes, not whatever ConceptNet word happened to match.
    nodes = kb.search(qvec, max(k, pool), empirical_only=high) if qvec is not None else []
    nodes.sort(key=lambda it: (it.get("score") or 0.0) * _pf(it), reverse=True)
    _t_search = time.perf_counter()

    # Route the matched concepts by intent (§10): walk the typed-edge graph or the
    # cards, rather than relying on the relation being independently embedded.
    items = list(nodes)
    for it in items:                           # the matched concepts are focus candidates
        it["_role"] = "concept"
    node_hits = [it for it in nodes if it["kind"] == "node"]
    # each intent's walk yields items in a known structural ROLE — tag it so the answer
    # can be composed around the graph (focus + caused_by/leads_to/managed_by/…) instead
    # of flattened into a score-sorted mush (§13 presentation).
    _INTENT_ROLE = {"why_mech": "effect", "what_if": "effect", "why_diag": "cause",
                    "how": "management", "taxonomy": "is_a", "who": "attribution",
                    "where": "location"}
    for nit in node_hits[:2]:
        nid, base = nit["id"], (nit.get("score") or 0.0)
        if intent in ("why_mech", "what_if"):
            ex = kb.edges_from(nid, families=["causal"], direction="out", empirical_only=high)
        elif intent == "why_diag":
            ex = kb.edges_from(nid, families=["causal"], direction="in", empirical_only=high)
        elif intent == "how":
            ex = kb.cards_for(nid)
        elif intent == "taxonomy":
            ex = kb.edges_from(nid, families=["taxonomic"], empirical_only=high)
        elif intent in ("who", "where"):
            ex = kb.edges_from(nid, families=[{"who": "epistemic"}.get(intent, "spatial")],
                               empirical_only=high)
        else:
            ex = []
        role = _INTENT_ROLE.get(intent, "related")
        for e in ex:
            if not e:
                continue
            if intent in ("why_diag", "how"):  # rank the differential / cards by context fit
                e["score"] = round(base * (0.6 + 0.4 * grounding.answer_fit(e, query, qfeats)), 4)
            else:
                e["score"] = round(base * 0.95, 4)
            e["_role"] = role
            items.append(e)
    # A management/diagnostic question (signalled by context_features) wants the structured
    # CARD for the concept, not the bare node summary — surface cards for the top grounded
    # node on the non-'how' intents too, scored by how well they fit the asked-about context.
    if qfeats and intent != "how":
        seen_cards = {it["id"] for it in items if it.get("kind") == "card"}
        for nit in node_hits[:1]:
            nid, base = nit["id"], (nit.get("score") or 0.0)
            for c in kb.cards_for(nid):
                if c and c["id"] not in seen_cards:
                    c["score"] = round(base * (0.85 + 0.25 * grounding.answer_fit(c, query, qfeats)), 4)
                    c["_role"] = "management"
                    items.append(c)

    # Speech-act role-pulls (understand.py): the utterance's form asks for extra structural
    # material beyond the primary walk — a feasibility check wants cautions, a counterfactual
    # wants what the omitted action prevents, a comparison wants the alternatives.  Additive
    # and GUARDED by flags, so a plain factual query (no flags) is byte-for-byte unchanged.
    if uflags and node_hits:
        nid, base = node_hits[0]["id"], (node_hits[0].get("score") or 0.0)
        if uflags.get("safety"):            # 'can we / is it safe to' → contraindications, harms
            for e in kb.edges_from(nid, families=["causal"], direction="out",
                                   empirical_only=high):
                if e and (e.get("polarity") == "negative" or e.get("type") in
                          ("contraindicated_in", "exacerbates", "worsens", "harms")):
                    e["score"] = round(base * 0.9, 4)
                    e["_role"] = "caution"
                    items.append(e)
        if uflags.get("counterfactual"):    # 'what if we didn't X' → what X prevents/controls
            for e in kb.edges_from(nid, families=["causal"], direction="out",
                                   empirical_only=high):
                if e and e.get("type") in ("prevents", "reduces", "protects_against",
                                           "controls"):
                    e["score"] = round(base * 0.9, 4)
                    e["_role"] = "risk_if_omitted"
                    items.append(e)
        if uflags.get("alternatives"):      # 'X or Y?' → the sideways alternative_to walk
            for a in kb.alternatives(nid):
                a["score"] = round(base * 0.9, 4)
                items.append(a)

    # Lexical (BM25) card recall channel (contract §3.1) — the arm dense misses on exact
    # terms (drug names, sign names, the query→card matching that's currently imprecise).
    # Appended AFTER the dense/walk items so dedup keeps the dense hit when both surface it.
    if bm25:
        items.extend(kb.search_cards_bm25(query, pool))
        # spaCy focus (understand.py): also search on the HEAD concept, so the context
        # words ("in a pre-eclamptic patient") don't dilute the card match.  Purely additive
        # — extra pool candidates the reranker/fit-gate sort; no weight to tune.
        focus = u.get("focus")
        if focus and focus.strip().lower() != query.strip().lower():
            items.extend(kb.search_cards_bm25(focus, max(pool // 2, 8)))

    _t_walk = time.perf_counter()
    seen, dedup = set(), []                    # dedup, keep first (highest-scored) hit
    for it in items:
        key = (it["kind"], it["id"])
        if key not in seen:
            seen.add(key)
            dedup.append(it)
    items = dedup

    if allowed is not None:                    # mode filter (claim regime, or origin if strict)
        items = [it for it in items if _mode_keep(it, allowed, strict)]

    # Additive facet read-filter (facets.py): drop an item only if it HAS a value on a
    # required axis that clashes; a missing value is kept (never over-exclude).  This is
    # a separate axis-set from the epistemic firewall above — the firewall is untouched,
    # this composes on top of it (domain/time_frame/trust_tier/epistemic).
    req_facets = facets_mod.normalize_filter(facets)
    if req_facets and items:
        fmap = kb.facets_for([(it["kind"], it["id"]) for it in items])
        items = [it for it in items
                 if facets_mod.matches(fmap.get((it["kind"], it["id"]), {}), req_facets)]

    # Cross-encoder rerank (contract §5): put the whole heterogeneous pool on ONE
    # comparable scale — the equaliser that cures incomparable dense/BM25/walk scores and
    # fixes 'context wrong' by scoring (query, item text).  Degrades to heuristic if the
    # endpoint is down; absent (reranker=None) the path is byte-for-byte as before.
    if reranker is not None and items:
        for it in items:                            # a criteria card's goal may be empty
            if not it.get("text"):
                it["text"] = it.get("label") or ""
        n_r = min(len(items), int(rerank_pool))
        items = reranker.rerank(query, intent, items[:n_r]) + items[n_r:]

    # When the caller gave context_features, the structured items (cards/edges that CAN be
    # feature-judged) are the answer they're after — float them above bare nodes so the gate
    # adjudicates the real candidate, not a node summary that carries no discriminators.
    feat_first = (lambda x: 1 if (gate and x.get("kind") in ("card", "edge")
                                  and grounding.item_features(x)) else 0)
    if high:                                   # read-time verdict, only when needed
        for it in items:
            it["strength"] = round(
                grounding.read_time_strength(it.get("support") or [],
                                             kb.contra_pressure(it)), 4)
        items.sort(key=lambda x: (feat_first(x), (x.get("strength") or 0.0) * _pf(x),
                                  (x.get("score") or 0.0) * _pf(x)), reverse=True)
    else:
        items.sort(key=lambda x: (feat_first(x), (x.get("score") or 0.0) * _pf(x)),
                   reverse=True)
    items = items[:max(k, 8)]

    _t_rank = time.perf_counter()
    # Relevance gate (§11): score the presented item against the asked-about context, and
    # detect a feature CLASH (item carries discriminators that contradict the query) → abstain.
    top = items[0] if items else None
    top_fit = grounding.answer_fit(top, query, qfeats) if top else 0.0
    top_ef = grounding.item_features(top) if top else set()
    all_qf = qfeats | grounding.extract_context_features(query)
    top_clash = bool(gate and top_ef and all_qf
                     and grounding._feature_overlap(all_qf, top_ef) <= 0.0)
    conf, band = grounding.grounding(items, top_fit, gated=gate, clash=top_clash)
    bundle = {
        "query": query, "intent": intent, "rigor": rigor,
        "speech_act": u["speech_act"],             # how the utterance was read (steers the walk)
        "mode": mode or "general", "strict": bool(strict),
        "confidence": conf, "grounding": band,
        "abstain": band == "none",
        "note": grounding.note_for(band),
        "items": [_present(it) for it in items],
    }
    if u["confidence"] < 0.5 or uflags.get("broaden"):   # ambiguous → lead with the map, not a guess
        bundle["broaden"] = True
    if u.get("focus"):                          # spaCy structure (observability)
        bundle["focus_term"] = u["focus"]
    if u.get("entities"):
        bundle["entities"] = u["entities"]
    if gate:                                   # surface why a near-miss was demoted
        bundle["fit"] = round(top_fit, 4)
        bundle["context_features"] = sorted(qfeats)
    if req_facets:                             # record the facet filter that was applied
        bundle["facets_filter"] = {a: sorted(v) for a, v in req_facets.items()}
    # Phase-1 linkage payoff: when we have a grounded answer, offer the connected concepts
    # (prerequisites, parents, alternatives) Vinkona can pull next — a navigation aid, kept
    # OUT of `items` so it never affects confidence/grounding.
    if band != "none" and top is not None:
        anchor = top.get("node_id") or (top["id"] if top.get("kind") == "node" else None)
        if anchor:
            rel = kb.neighbours(anchor)
            if rel:
                bundle["related"] = rel
    # Compose content + STRUCTURE (chosen design): organise the walked items around the
    # matched concept (`focus`) and group the connecting edges by role (`structure`), so a
    # caller gets a map — what this is, what causes/leads-to/manages it, the differential,
    # what's related — not a flat snippet list.  `items` is kept unchanged (back-compat).
    if band != "none":
        focus, structure = _compose_frame(items, bundle.get("related"), intent)
        if focus:
            bundle["focus"] = focus
        if structure:
            bundle["structure"] = structure
    # Rights signal (§16.4): the answer's shippable-under licence — the most-restrictive
    # intersection of every surfaced source's licence, plus to whom those rights belong.
    # So Vinkona can gate redistribution / cite the licensor without reading prose.
    if band != "none":
        allsup = [s for it in items for s in (it.get("support") or [])]
        if allsup and hasattr(kb, "license_for_support"):
            bundle["rights"] = kb.license_for_support(allsup)
    if band == "none":
        kb.log_gap(query, intent)
    _t_end = time.perf_counter()
    perf.info("ask embed=%.1f search=%.1f walk=%.1f rank=%.1f ground=%.1f core=%.1fms "
              "items=%d band=%s", (_t_embed - _t0) * 1e3, (_t_search - _t_embed) * 1e3,
              (_t_walk - _t_search) * 1e3, (_t_rank - _t_walk) * 1e3,
              (_t_end - _t_rank) * 1e3, (_t_end - _t0) * 1e3, len(bundle["items"]), band)
    return bundle


# Render by regime so the model never presents fiction/opinion as fact (§11).
_FRAME = {
    "empirical": "{t}",
    "conventional": "By convention: {t}",
    "fictional": "In the work: {t}",
    "interpretive": "An argued position: {t}",
    "historical": "As understood at the time: {t}",
}


def _present(it: dict) -> dict:
    regime = it.get("regime") or "empirical"
    sup = it.get("support") or []
    out = {
        "id": it.get("id"),                        # the exact card/node/edge id (citation + eval)
        "kind": it["kind"], "regime": regime, "score": it.get("score"),
        "text": _FRAME.get(regime, "{t}").format(t=it.get("text") or ""),
        "label": it.get("label"),
        "support": [s.get("doc_id") for s in sup],
        # citeable provenance: doc + page-level locator ("S3.2 p.41") when known.
        "sources": [{"doc_id": s.get("doc_id"), "locator": s.get("locator") or ""}
                    for s in sup],
        "contradictions": it.get("contradictions") or [],
    }
    if it.get("strength") is not None:
        out["strength"] = it["strength"]
    # Flag Vinkona-sourced items so the assistant knows this came from its own research
    # (low-trust, subordinate) rather than a curated source (research §6).
    if sup and all(s.get("provenance") == "vinkona" for s in sup):
        out["provenance"] = "vinkona"
    # structured card payload — only the fields actually present, so a thin card stays thin.
    for fld in ("steps", "red_flags", "discriminators", "escalation", "safety", "conditions",
                # diagnostic-criteria / staging / recommendation-grade / empirical-finding
                "criteria", "grade", "finding"):
        if it.get(fld):
            out[fld] = it[fld]
    ct = it.get("card_type")
    if ct and ct != "procedure":
        out["card_type"] = ct
    return out


# ── answer composition: content + structure (§13 presentation) ───────────────
# Each walked item played a structural ROLE (tagged during the intent walk); we map those
# roles onto the sections of an answer frame so the graph's own edges organise the answer.
_ROLE_TO_SECTION = {"cause": "caused_by", "effect": "leads_to", "management": "managed_by",
                    "is_a": "is_a", "attribution": "who", "location": "where",
                    "caution": "cautions", "risk_if_omitted": "if_omitted",
                    "alternative": "alternatives"}
_SECTION_LABEL = {"is_a": "is a kind of", "caused_by": "caused by",
                  "differential": "differential (looks like, told apart by)",
                  "leads_to": "leads to", "managed_by": "managed by",
                  "cautions": "cautions / contraindications", "alternatives": "alternatives",
                  "if_omitted": "risk if omitted",
                  "who": "who", "where": "where", "related": "related — pull next"}
_SECTION_ORDER = ["is_a", "caused_by", "differential", "leads_to", "managed_by",
                  "cautions", "if_omitted", "alternatives", "who", "where", "related"]


def _compose_frame(items: list, related, intent: str):
    """Organise the RAW walked `items` (they carry `_role`) around the matched concept:
    a `focus` (what the question is about) and a `structure` grouped by graph role — the
    edges that logically join the question to the answer.  Returns (focus, structure)."""
    focus = None
    for it in items:                               # prefer the top concept node as focus
        if it.get("kind") == "node":
            focus = {"id": it.get("id"), "label": it.get("label"),
                     "kind": it.get("node_kind") or "concept",
                     "summary": it.get("text") or ""}
            break
    if focus is None and items:                    # else the top item's own concept
        it = items[0]
        focus = {"id": it.get("id"), "label": it.get("label") or "",
                 "kind": it.get("kind"), "summary": it.get("text") or ""}
    focus_id = focus["id"] if focus else None

    sections: dict = {}
    for it in items:
        if it.get("id") == focus_id:
            continue
        sec = _ROLE_TO_SECTION.get(it.get("_role"))
        if not sec:
            continue
        if intent == "why_diag" and sec == "caused_by":
            sec = "differential"                   # a diagnostic's causes ARE the differential
        entry = {"id": it.get("id"), "kind": it.get("kind"),
                 "label": it.get("label") or (it.get("text") or "")[:60],
                 "relation": it.get("type") or "", "text": (it.get("text") or "")[:200],
                 "score": it.get("score")}
        bucket = sections.setdefault(sec, [])
        if len(bucket) < 6 and not any(x["id"] == entry["id"] for x in bucket):
            bucket.append(entry)
    if related:                                    # neighbours are structure too (pull-next)
        sections["related"] = [{"id": n.get("node_id"), "label": n.get("label"),
                                "relation": n.get("relation"), "kind": "node",
                                "has_card": n.get("has_card")} for n in related[:8]]

    structure = [{"role": s, "label": _SECTION_LABEL.get(s, s), "entries": sections[s]}
                 for s in _SECTION_ORDER if sections.get(s)]
    return focus, structure
