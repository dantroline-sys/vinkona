"""Unit tests for confighelp — comment extraction rules on a synthetic DEFAULTS."""
import tempfile

import confighelp


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


SRC = '''
X = 1
DEFAULTS: dict = {
    # ── Section banner ── group docs live here ──────────────
    "alpha": {
        # leading doc for a
        "a": 1,
        "b": 2,                              # trailing doc for b
        "c": 3,                              # trailing that continues
                                             # onto the next line
        # leading doc for d
        "d": 4,                              # plus trailing for d
        "e": 5,
    },
    "beta": {                                # trailing doc for the beta group
        "f": "x#not-a-comment",
    },
    # multi-line leading:
    # second line of it
    "gamma": True,
}
'''


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SRC)
        path = f.name
    h = confighelp.extract(path)

    check("group gets the banner text", "group docs live here" in h.get("alpha", ""))
    check("banner decoration stripped", "──" not in h.get("alpha", ""))
    check("leading doc", h.get("alpha.a") == "leading doc for a")
    check("trailing doc", h.get("alpha.b") == "trailing doc for b")
    check("far-right continuation joins previous key",
          h.get("alpha.c") == "trailing that continues onto the next line")
    check("continuation not stolen by next key",
          h.get("alpha.d") == "leading doc for d plus trailing for d")
    check("uncommented key has no entry", "alpha.e" not in h)
    check("dict key takes only its own line's comment",
          h.get("beta") == "trailing doc for the beta group")
    check("hash inside a string is not a comment", "beta.f" not in h)
    check("multi-line leading joins", h.get("gamma") == "multi-line leading: second line of it")

    # the real config.py: substantial coverage and a few known anchors
    real = confighelp.extract("config.py")
    check("real config.py: 250+ entries", len(real) >= 250)
    check("real: tts_lm group text", "Orpheus TTS backbone" in real.get("tts_lm", ""))
    check("real: nested leaf", "chat template" in real.get("fast_lm.jinja", ""))

    print(f"\\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
