"""Facets — the multi-axis classification layer (option B).

One claim has several *independent* coordinates, and cramming them into the single
epistemic ``regime`` column conflated dimensions that want to move separately.  A facet
is ``(target_kind, target_id, axis, value)`` in a small side table — add an axis by
inserting rows, never a migration.  The axes:

  * ``epistemic``  — fact / convention / fiction / opinion / historical.  This mirrors the
    existing ``regime`` column and is the ONLY axis that gates the corroboration firewall.
  * ``time_frame`` — current / historical / superseded / timeless.  (This is the retrieval
    contract's "temporal regime", renamed so it never collides with epistemic ``regime``.)
  * ``trust_tier`` — high / medium / low.  A reliability rating, even for historical works.
  * ``domain``     — coarse subject bucket (clinical, cooking, …).  Swappable vocabulary;
    can grow toward Dewey later without touching this code.

**Safety invariant:** facets other than ``epistemic`` are *additive read-filters only*.
The firewall (which claim may corroborate which) is unchanged and still keyed on the
epistemic regime.  So populating/filtering facets can never alter what corroborates what.

Derivation is cheap because three of the four axes are *already implied by data we store*
(support trust_weights, pub dates, supersession, source bundle) — this module just reads
those out.  Everything here is pure (takes already-fetched rows) so it unit-tests without
a live KB; ``kb.facetize`` feeds it.
"""
from __future__ import annotations

# the axis names are a closed set; add here to introduce a new slice
AXES = ("epistemic", "time_frame", "trust_tier", "domain")

EPISTEMIC_VALUES = ("empirical", "conventional", "fictional", "interpretive", "historical")
TIME_FRAME_VALUES = ("current", "historical", "superseded", "timeless")
TRUST_TIER_VALUES = ("high", "medium", "low")

# trust bands mirror the grounding thresholds so the two speak the same language
_TRUST_HIGH = 0.66
_TRUST_MED = 0.40

# bundles that are organisational, not subjects — they don't imply a domain
_NON_DOMAIN_BUNDLES = {"base", "vinkona", "overlay", ""}


def trust_tier(support) -> str | None:
    """Reliability band from the strongest independent source backing a claim."""
    weights = [s.get("trust_weight") for s in (support or [])
               if isinstance(s, dict) and s.get("trust_weight") is not None]
    if not weights:
        return None
    top = max(float(w) for w in weights)
    if top >= _TRUST_HIGH:
        return "high"
    if top >= _TRUST_MED:
        return "medium"
    return "low"


def epistemic(regime, support) -> str | None:
    """The epistemic axis mirrors the item's regime (or the dominant support regime)."""
    r = (regime or "").strip().lower()
    if r in EPISTEMIC_VALUES:
        return r
    tally: dict = {}
    for s in (support or []):
        v = (s.get("regime") or "").strip().lower() if isinstance(s, dict) else ""
        if v in EPISTEMIC_VALUES:
            tally[v] = tally.get(v, 0) + 1
    return max(tally, key=tally.get) if tally else None


def domain(support, source_lookup) -> list:
    """Coarse subject bucket(s), seeded from the source bundle(s) a claim rests on.
    A source in a topical bundle (e.g. 'clinical') lends that domain; organisational
    bundles (base/vinkona/overlay) lend none.  Multi-valued — a claim spanning two
    domains carries both.  Empty ⇒ leave undomained (so a domain filter won't drop it)."""
    doms = []
    for s in (support or []):
        if not isinstance(s, dict) or not s.get("doc_id"):
            continue
        src = source_lookup(s["doc_id"]) or {}
        b = (src.get("bundle") or "").strip().lower()
        if b and b not in _NON_DOMAIN_BUNDLES and b not in doms:
            doms.append(b)
    return doms


def time_frame(regime, status, support, *, superseded=False, volatility=None) -> str | None:
    """current / historical / superseded / timeless from signals we already keep:
    an explicit supersession, the historical regime, or near-zero volatility (settled
    physiology/mechanism = timeless).  Defaults to 'current' when a claim has support."""
    if (status or "").strip().lower() == "superseded" or superseded:
        return "superseded"
    r = (regime or "").strip().lower()
    if r == "historical":
        return "historical"
    if volatility is not None:
        try:
            if float(volatility) <= 1e-6:
                return "timeless"
        except (TypeError, ValueError):
            pass
    return "current" if support else None


def derive(kind: str, row: dict, support: list, source_lookup,
           *, superseded=False) -> dict:
    """All facet values for one node/card/edge → {axis: [values]} (axes with no value
    are omitted, so nothing is asserted we don't actually know)."""
    out: dict = {}
    ep = epistemic(row.get("regime"), support)
    if ep:
        out["epistemic"] = [ep]
    tt = trust_tier(support)
    if tt:
        out["trust_tier"] = [tt]
    dom = domain(support, source_lookup)
    if dom:
        out["domain"] = dom
    tf = time_frame(row.get("regime"), row.get("status"), support,
                    superseded=superseded, volatility=row.get("volatility"))
    if tf:
        out["time_frame"] = [tf]
    return out


# ── read-time filtering (additive; NEVER over-excludes) ──────────────────────
def matches(item_facets: dict, required: dict) -> bool:
    """Keep an item under a facet filter unless it has a value on a required axis that
    is NOT in the requested set.  A *missing* value on an axis is KEPT (unconstrained) —
    same 'never over-exclude' rule the epistemic mode filter uses, so the feature is safe
    to switch on before every item is backfilled."""
    for axis, want in (required or {}).items():
        if not want:
            continue
        want_set = {want} if isinstance(want, str) else set(want)
        have = item_facets.get(axis)
        if have and not (set(have) & want_set):
            return False
    return True


def normalize_filter(cf) -> dict:
    """Coerce a caller's facet filter into {axis: set(values)}; drop unknown axes."""
    out: dict = {}
    if not cf:
        return out
    for axis, val in dict(cf).items():
        if axis not in AXES or val in (None, "", []):
            continue
        out[axis] = {val} if isinstance(val, str) else set(val)
    return out
