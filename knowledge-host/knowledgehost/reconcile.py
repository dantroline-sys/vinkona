"""Reasoned reconciliation of a candidate claim against the live KB (§9.1-9.2),
parameterised by epistemic regime (§8, §9.0).

Banding keeps the LM off the easy cases (§9.1):
  1. ``edge_hash`` already present              -> corroborate (no LM)
  2. no existing edge shares the (src,dst) pair -> insert      (no LM)
  3. else                                       -> the 5-way LM decides, against
     **same-regime** comparables only (the write-time firewall §8 — a novel's
     herbal "cure" can never corroborate or contradict a clinical claim).

The five-way (§9.2) never clobbers: contradictions are recorded as live
``meta/disagrees_with`` edges between two active claims.  The regime gate (§9.0)
swaps the verdict where the truth model differs — for a *convention*, a
cross-context "conflict" is a ``context_variant_of`` variant, not a disagreement;
*interpretive* claims are attributed, never adjudicated.
"""
from __future__ import annotations

import logging

log = logging.getLogger("knowledgehost.reconcile")

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string",
                     "enum": ["duplicate", "refinement", "novel_distinct",
                              "contradiction", "no_match"]},
        "target_index": {"type": "integer"},
        "conditions": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "target_index"],
}

_SYSTEM = (
    "You reconcile a NEW claim against EXISTING claims about the SAME pair of "
    "concepts, all from the same kind of source. Classify how the new claim "
    "relates to the single closest existing one (by index):\n"
    "- duplicate: same claim, mechanism and conditions;\n"
    "- refinement: same direction and mechanism, but adds or narrows a condition "
    "or population;\n"
    "- novel_distinct: same concepts but a DIFFERENT mechanism or condition regime;\n"
    "- contradiction: opposite or incompatible claim;\n"
    "- no_match: unrelated.\n"
    "Return JSON only."
)


def _claim_str(c: dict) -> str:
    bits = [f"{c.get('type', '')}: {c.get('src_label', c.get('src_id'))} -> "
            f"{c.get('dst_label', c.get('dst_id'))}"]
    if c.get("polarity"):
        bits.append(f"polarity={c['polarity']}")
    if c.get("mechanism"):
        bits.append(f"mechanism={c['mechanism']}")
    mods = c.get("modifiers") or {}
    if isinstance(mods, dict) and mods.get("conditions"):
        bits.append(f"conditions={mods['conditions']}")
    discs = mods.get("discriminators") if isinstance(mods, dict) else None
    if discs:
        bits.append("features=" + ",".join(f"{d.get('feature')}:{d.get('value')}"
                                           for d in discs if isinstance(d, dict)))
    return "; ".join(bits)


def _insert_kwargs(cand: dict) -> dict:
    return dict(family=cand["family"], type=cand["type"],
                mechanism=cand.get("mechanism", ""),
                mechanism_basis=cand.get("mechanism_basis") or "stated",
                polarity=cand.get("polarity", ""),
                modifiers=cand.get("modifiers") or {}, regime=cand["regime"],
                scope=cand.get("scope") or {}, doc_id=cand.get("doc_id"),
                evidence=cand.get("evidence", ""),
                finding=cand.get("finding"))    # empirical study finding (effect/design/n/…)


def _classify(lm, cand: dict, comps: list):
    """The 5-way LM step.  None/garbage falls back to no_match → a plain insert,
    which is safe (never clobbers; at worst leaves a near-duplicate to merge later)."""
    listing = "\n".join(f"[{i}] {_claim_str(c)}" for i, c in enumerate(comps))
    user = (f"NEW claim:\n{_claim_str(cand)}\n\nEXISTING claims:\n{listing}\n\n"
            "Which existing claim is closest, and how does the new one relate?")
    out = lm.chat_json(_SYSTEM, user, DECISION_SCHEMA, max_tokens=200)
    if not out or "decision" not in out:
        return {"decision": "no_match", "target_index": 0}
    return out


def reconcile_edge(kb, lm, cand: dict) -> str:
    """Reconcile one edge candidate into the KB; returns the action taken."""
    regime = cand["regime"]
    world = (cand.get("scope") or {}).get("world", "") or ""
    conditions = (cand.get("modifiers") or {}).get("conditions", "") or ""
    # band 1 — exact claim already present (same regime/world/conditions) → corroborate.
    ex = kb.edge_by_hash(cand["src_id"], cand["dst_id"], cand["family"],
                         cand["type"], cand.get("polarity", ""), cand.get("mechanism", ""),
                         regime, world, conditions)
    if ex:
        kb.corroborate_edge(ex["id"], cand.get("doc_id"), cand.get("evidence", ""))
        return "corroborate"

    # interpretive (§8): never a duplicate/contradiction to resolve — attribute and
    # keep every position.  Just insert.
    if regime == "interpretive":
        kb.add_edge(cand["src_id"], cand["dst_id"], **_insert_kwargs(cand))
        return "insert_attributed"

    # band 2/3 — comparables share the pair AND the regime (firewall §8); fictional
    # also matches on world scope.
    cmp_world = world if regime == "fictional" else None
    comps = kb.comparable_edges(cand["src_id"], cand["dst_id"], regime, scope_world=cmp_world)
    if not comps:
        kb.add_edge(cand["src_id"], cand["dst_id"], **_insert_kwargs(cand))
        return "insert"

    # band 3 needs the big LM.  When it is unavailable (lm=None, e.g. the 3090 is leased
    # to Vinkona) skip adjudication and just insert — never clobbers, at worst leaves a
    # near-duplicate to merge later; keeps the writer moving without the big LM.
    if lm is None:
        kb.add_edge(cand["src_id"], cand["dst_id"], **_insert_kwargs(cand))
        return "insert_unadjudicated"

    # band 3 — the reasoning step.
    d = _classify(lm, cand, comps)
    ti = d.get("target_index", 0)
    target = comps[ti] if isinstance(ti, int) and 0 <= ti < len(comps) else comps[0]
    decision = d.get("decision", "no_match")

    if decision == "duplicate":
        kb.corroborate_edge(target["id"], cand.get("doc_id"), cand.get("evidence", ""))
        return "duplicate"
    if decision == "refinement":
        cond = d.get("conditions") or (cand.get("modifiers") or {}).get("conditions")
        kb.enrich_edge(target["id"], {"conditions": cond},
                       cand.get("doc_id"), cand.get("evidence", ""))
        return "refinement"

    # novel_distinct / contradiction / no_match → insert the new claim, keep both.
    new_id, _ = kb.add_edge(cand["src_id"], cand["dst_id"], **_insert_kwargs(cand))
    if decision == "novel_distinct":
        mtype = "context_variant_of" if regime == "conventional" else "alternative_to"
        kb.link_meta(new_id, target["id"], mtype)
        return f"novel_distinct/{mtype}"
    if decision == "contradiction":
        # convention (§8): a cross-context "conflict" is a variant, not a disagreement.
        mtype = "context_variant_of" if regime == "conventional" else "disagrees_with"
        kb.link_meta(new_id, target["id"], mtype)
        return f"contradiction/{mtype}"
    return "insert"
