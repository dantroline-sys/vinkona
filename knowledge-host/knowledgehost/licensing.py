"""Licence capture, detection and roll-up (spec §16.4).

Three jobs:
1. **Structured, not prose.**  A source's rights are stored as an SPDX id + parsed
   permission *flags* (redistribute / derivatives / commercial / attribution /
   share_alike) + the licensor ("to whom the licence belongs") + URL, with the
   verbatim statement kept only as a fallback.  So "is this redistributable?" is a
   query, not a read-through.
2. **Detect if it exists.**  ``detect()`` is a best-effort scan of a document for an
   SPDX tag, a Creative-Commons URL/badge, a public-domain/all-rights line, and a
   copyright holder — fills what it finds, leaves the rest unknown, always
   manually overridable.
3. **Roll up.**  A card/answer's *shippable-under* licence is the **most-restrictive
   intersection** of its support-set sources (``combine()``): a permission is granted
   only if every source grants it; a *condition* (attribution/share_alike) applies if
   any source imposes it; unknown never silently becomes "yes".

Pure-python (stdlib re/json) — no model stack, unit-testable in isolation.
"""
from __future__ import annotations

import json
import re

# ── SPDX-ish permission table ────────────────────────────────────────────────
# flags: redistribute, derivatives, commercial  = PERMISSIONS (may I…?)
#        attribution, share_alike                = CONDITIONS  (if I do, I must…)
# True = yes/required, False = no/forbidden, None = unknown.
def _f(redistribute, derivatives, commercial, attribution, share_alike):
    return {"redistribute": redistribute, "derivatives": derivatives,
            "commercial": commercial, "attribution": attribution,
            "share_alike": share_alike}

_ALL_YES = _f(True, True, True, False, False)
_ALL_NO = _f(False, False, False, False, False)

SPDX: dict = {
    "CC0-1.0":         _ALL_YES,
    "public-domain":   _ALL_YES,
    "CC-BY-4.0":       _f(True, True, True, True, False),
    "CC-BY-SA-4.0":    _f(True, True, True, True, True),
    "CC-BY-ND-4.0":    _f(True, False, True, True, False),
    "CC-BY-NC-4.0":    _f(True, True, False, True, False),
    "CC-BY-NC-SA-4.0": _f(True, True, False, True, True),
    "CC-BY-NC-ND-4.0": _f(True, False, False, True, False),
    "MIT":             _f(True, True, True, True, False),
    "Apache-2.0":      _f(True, True, True, True, False),
    "BSD-3-Clause":    _f(True, True, True, True, False),
    "GPL-3.0-only":    _f(True, True, True, True, True),
    "OGL-3.0":         _f(True, True, True, True, False),   # UK Open Government Licence
    "proprietary":     _ALL_NO,
    "all-rights-reserved": _ALL_NO,
    "unknown":         _f(None, None, None, None, None),
}

# Normalise loose aliases to a canonical key (version-agnostic CC etc.).
_ALIASES = {
    "cc0": "CC0-1.0", "cc-0": "CC0-1.0", "publicdomain": "public-domain",
    "pd": "public-domain", "cc-by": "CC-BY-4.0", "cc-by-sa": "CC-BY-SA-4.0",
    "cc-by-nd": "CC-BY-ND-4.0", "cc-by-nc": "CC-BY-NC-4.0",
    "cc-by-nc-sa": "CC-BY-NC-SA-4.0", "cc-by-nc-nd": "CC-BY-NC-ND-4.0",
    "apache": "Apache-2.0", "apache-2": "Apache-2.0", "gpl": "GPL-3.0-only",
    "gpl-3.0": "GPL-3.0-only", "gplv3": "GPL-3.0-only", "bsd": "BSD-3-Clause",
    "arr": "all-rights-reserved", "copyright": "all-rights-reserved",
    "proprietary": "proprietary", "none": "unknown", "": "unknown",
}

_CC_URL = re.compile(
    r"creativecommons\.org/(?:licenses/(by(?:-nc)?(?:-nd|-sa)?)/(\d\.\d)"
    r"|publicdomain/(zero|mark))", re.I)


def canonical(spdx: str | None) -> str:
    """Best-effort map any licence string to a known SPDX key (else 'unknown')."""
    if not spdx:
        return "unknown"
    s = spdx.strip()
    if s in SPDX:
        return s
    low = s.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    # bare CC form e.g. "cc-by-nc-4.0"
    m = re.match(r"cc-(by(?:-nc)?(?:-nd|-sa)?)-?(\d\.\d)?", low)
    if m:
        key = "CC-" + m.group(1).upper() + "-" + (m.group(2) or "4.0")
        if key in SPDX:
            return key
        base = "CC-" + m.group(1).upper() + "-4.0"
        if base in SPDX:
            return base
    return "unknown"


def flags_for(spdx: str | None) -> dict:
    return dict(SPDX.get(canonical(spdx), SPDX["unknown"]))


def is_known(spdx: str | None) -> bool:
    return canonical(spdx) != "unknown"


# ── detection ────────────────────────────────────────────────────────────────
_SPDX_TAG = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)")
_ARR = re.compile(r"\ball rights reserved\b", re.I)
_PD = re.compile(r"\bpublic domain\b", re.I)
_COPYRIGHT = re.compile(
    r"(?:©|\(c\)|copyright)\s*(?:©\s*)?(?:(\d{4})(?:\s*[-–]\s*\d{4})?)?\s*"
    r"(?:by\s+)?([A-Z][A-Za-z0-9.,&'’\- ]{2,70})", re.I)


def _snippet(text: str, at: int, span: int = 160) -> str:
    lo = max(0, at - 20)
    return " ".join(text[lo:lo + span].split())


def detect(text: str, *, url: str | None = None) -> dict:
    """Best-effort licence extraction from a document's text.  Returns
    {license, license_holder, license_url, license_text} with any field None when
    not found.  Conservative: only reports what it actually saw."""
    out = {"license": None, "license_holder": None,
           "license_url": url, "license_text": None}
    if not text:
        return out
    head = text[:20000]                             # licences live near the top/edges

    m = _SPDX_TAG.search(head) or _SPDX_TAG.search(text[-4000:])
    if m:
        out["license"] = canonical(m.group(1))
        out["license_text"] = _snippet(head, m.start())

    if out["license"] is None:
        cc = _CC_URL.search(text)
        if cc:
            if cc.group(3):                          # publicdomain/zero|mark
                out["license"] = "CC0-1.0" if cc.group(3).lower() == "zero" \
                    else "public-domain"
            else:
                out["license"] = canonical("CC-" + cc.group(1).upper() + "-"
                                           + cc.group(2))
            out["license_url"] = out["license_url"] or ("https://" + cc.group(0))
            out["license_text"] = out["license_text"] or _snippet(text, cc.start())

    if out["license"] is None and _PD.search(head):
        out["license"] = "public-domain"
        out["license_text"] = _snippet(head, _PD.search(head).start())

    cr = _COPYRIGHT.search(head)
    if cr:
        holder = cr.group(2).strip(" .,-")
        # trim trailing noise the greedy class may have grabbed: a sentence break,
        # or a licence/rights clause that follows the holder name.
        holder = re.split(r"(?:\.\s|\s+(?:all rights|is\s+licensed|licensed|under|"
                          r"released|published|\d{4}))", holder,
                          maxsplit=1, flags=re.I)[0].strip(" .,-")
        if len(holder) >= 2:
            out["license_holder"] = holder
        out["license_text"] = out["license_text"] or _snippet(head, cr.start())
        if out["license"] is None and _ARR.search(head):
            out["license"] = "all-rights-reserved"

    if out["license"] is None and _ARR.search(head):
        out["license"] = "all-rights-reserved"
        out["license_text"] = out["license_text"] or _snippet(head, _ARR.search(head).start())
    return out


# ── roll-up: most-restrictive intersection over a support set ────────────────
def _and_perm(a, b):
    """Permission granted only if BOTH grant it; a hard No wins; unknown poisons
    to unknown (never silently 'yes')."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _or_cond(a, b):
    """A condition (attribution/share_alike) applies if EITHER imposes it."""
    if a is True or b is True:
        return True
    if a is None and b is None:
        return None
    return bool(a or b)


def combine(licenses) -> dict:
    """Roll a list of per-source licence records up to the shippable-under licence
    of the thing they support (a card / an answer).  Each item may be an SPDX id
    string or a dict with a 'license'/'flags'/'license_holder' key.  Returns the
    intersected flags, the distinct licences, the holders, and a plain
    'redistributable' verdict for quick gating."""
    perms = {"redistribute": True, "derivatives": True, "commercial": True}
    conds = {"attribution": None, "share_alike": None}
    ids, holders = [], []
    seen = False
    for item in licenses or []:
        seen = True
        if isinstance(item, str):
            spdx, holder, fl = canonical(item), None, flags_for(item)
        else:
            spdx = canonical(item.get("license"))
            holder = item.get("license_holder")
            fl = item.get("flags") or flags_for(spdx)
        ids.append(spdx)
        if holder:
            holders.append(holder)
        for k in perms:
            perms[k] = _and_perm(perms[k], fl.get(k))
        for k in conds:
            conds[k] = _or_cond(conds[k], fl.get(k))
    if not seen:
        return {"license": "unknown", "flags": flags_for("unknown"),
                "licenses": [], "holders": [], "redistributable": None}
    flags = {**perms, **conds}
    uniq = sorted(set(ids))
    label = uniq[0] if len(uniq) == 1 else ("intersection: " + ", ".join(uniq))
    return {"license": label, "flags": flags, "licenses": uniq,
            "holders": sorted(set(holders)),
            "redistributable": flags["redistribute"]}


def summary(record: dict) -> str:
    """One-line human summary of a source's licence record."""
    lic = record.get("license") or "unknown"
    who = record.get("license_holder")
    return f"{lic}" + (f" — © {who}" if who else "")
