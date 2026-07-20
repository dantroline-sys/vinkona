"""Which pronouns the persona is referred to by.

The assistant's character is chosen, so its pronouns are part of the choice: a
persona configured as a man should not be described as "she" by its own prompts
or by the panel that edits it.  A persona declares `identity.sex` —

    "female" -> she/her/hers/herself
    "male"   -> he/him/his/himself
    "none"   -> it/its/itself

— and every string that refers to the persona in the third person is written
with tokens ({subj}, {obj}, {poss}, {possp}, {refl}, and their capitalised
forms) and rendered through `render()` against the live set.

`sex` is optional.  Unset falls back to the free-text `identity.pronouns`
already in existing persona files, and failing that to female — the shipped
Vinkona persona, so nobody's assistant changes character on upgrade.

Only the app's own references are affected.  What a person says about
themselves is theirs, and nothing here touches how the user is addressed.
"""
from __future__ import annotations

SETS = {
    "female": {"sex": "female", "subj": "she", "obj": "her", "poss": "her",
               "possp": "hers", "refl": "herself", "label": "she/her"},
    "male": {"sex": "male", "subj": "he", "obj": "him", "poss": "his",
             "possp": "his", "refl": "himself", "label": "he/him"},
    "none": {"sex": "none", "subj": "it", "obj": "it", "poss": "its",
             "possp": "its", "refl": "itself", "label": "it/its"},
}
CHOICES = tuple(SETS)
DEFAULT_SEX = "female"           # the shipped persona; upgrades change nothing

_TOKENS = ("subj", "obj", "poss", "possp", "refl")


def parse(text: str | None) -> str | None:
    """Read a sex from a free-text pronoun field ('she/her', 'he/him', 'it')."""
    t = (text or "").strip().lower().replace("\\", "/")
    if not t:
        return None
    first = t.split("/")[0].strip()
    for sex, s in SETS.items():
        if first in (s["subj"], s["obj"], s["poss"], s["possp"]):
            return sex
    return None


def normalise(sex: str | None) -> str | None:
    """A declared sex, or None when it isn't one of the three."""
    s = (sex or "").strip().lower()
    if s in SETS:
        return s
    if s in ("f", "feminine", "woman"):
        return "female"
    if s in ("m", "masculine", "man"):
        return "male"
    if s in ("neutral", "neuter", "object", "no", "n/a", "unset"):
        return "none"
    return None


def for_identity(identity: dict | None, default: str = DEFAULT_SEX) -> dict:
    """The pronoun set for a persona's `identity` block: declared `sex` wins,
    then the legacy free-text `pronouns` field, then the default."""
    ident = identity or {}
    sex = normalise(ident.get("sex")) or parse(ident.get("pronouns")) \
        or normalise(default) or DEFAULT_SEX
    return SETS[sex]


def for_persona(persona: dict | None, default: str = DEFAULT_SEX) -> dict:
    return for_identity((persona or {}).get("identity"), default)


def render(text: str, pset: dict | None) -> str:
    """Substitute pronoun tokens.  Capitalised token -> capitalised pronoun, so
    a sentence can open with {Subj}."""
    if not text:
        return text
    s = pset or SETS[DEFAULT_SEX]
    for tok in _TOKENS:
        word = s.get(tok, "")
        text = text.replace("{" + tok + "}", word)
        text = text.replace("{" + tok.capitalize() + "}", word.capitalize())
    return text


def label(pset: dict | None) -> str:
    return (pset or SETS[DEFAULT_SEX])["label"]
