"""VINKONA-LEX-01 — alias lexicon & deterministic span matcher.

Stage 0 (lexicon compile) and Stage 1 (span detection) of the deterministic
utterance→graph matching pipeline.  The matcher emits **unresolved candidate sets** —
it MUST NOT choose between nodes sharing a surface form (sense resolution is Stage 2);
the gap/failure logger consumes ``unmatched_token_indices`` and ``flags``.

Runtime: pure Python stdlib — no ML, GPU, or network.  Token-level Aho–Corasick over
``tok_id`` sequences (character-level AC MUST NOT be used) plus a SymSpell deletion
index (max edit 2, prefix 7) under OSA Damerau–Levenshtein.  ``fuzzy_allowed = 0``
declares an alias safety-critical for spelling (drug names, look-alike/sound-alike):
spelling correction MUST NEVER create a match for such an alias — a near-miss surfaces
as a ``fuzzy_suppressed`` flag instead of a silent correction (§7 M3/M4).

Determinism (§8.2): fixed (text, lexicon_version, matcher_version) ⇒ byte-identical
``match_json()`` across runs/platforms/threads.  All offsets are code-point offsets.

Interpretation notes (where the spec leaves latitude):
  * Artifact binary formats are implementation-defined (§6.4): we persist the alias
    rows + vocabulary as ``lexicon.json`` and reconstruct the AC automaton and the
    SymSpell index deterministically at ``load()`` (cold-load budget §10 is ample).
  * Empty-norm tokens (e.g. ``'s``) and OOV tokens **block** multi-token adjacency:
    "MUST NOT participate in matching" is read conservatively — an alias cannot span
    across them.
  * C3 node existence is validated against the canon registry (``conflict_node``) by
    default; ``node_table``/``node_ids`` parameterize this for VINKONA-ING-01.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

MATCHER_VERSION = "1.0.0"
NORM_VERSION = "NORM-1"
TOK_VERSION = "TOK-1"
_US = ""                       # norm_seq joiner (U+001F UNIT SEPARATOR)
_MAX_INPUT = 4096
_F = {1: 0.8, 2: 0.6}                # §M6 edit-distance factors; empty product = 1

EN_STOP_V1 = frozenset("""a an and are as at be been but by for from had has have he if
in is it no not of off on or out over she so than that the then they this to under up
was we were with you""".split())

_WORD_CATS = ("Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Nl", "No")
_APOSTROPHES = ("'", "’")


class LexError(Exception):
    """§9: E_INVALID_INPUT | E_INPUT_TOO_LONG | E_ARTIFACT_MISSING | E_ARTIFACT_MISMATCH."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ── NORM-1 (§3) ────────────────────────────────────────────────────────────────────────
def norm_token(s: str) -> str:
    """NFKC → full case fold → strip trailing possessive ('s / ’s) → drop apostrophes.
    May legitimately return "" (the token is retained but never matches)."""
    t = unicodedata.normalize("NFKC", s).casefold()
    if len(t) >= 2 and t[-1] == "s" and t[-2] in _APOSTROPHES:
        t = t[:-2]
    return t.replace("'", "").replace("’", "")


# ── TOK-1 (§4) ─────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    """Maximal runs of WORD code points over the ORIGINAL string (offsets exact).
    Hyphens/dashes/slashes are separators BY DESIGN (post-op ≡ post op ≡ post/op);
    '.'/',' are word chars only between two digits (0.5 and 1,000 stay whole)."""
    def word(i: int) -> bool:
        c = text[i]
        if c in _APOSTROPHES:
            return True
        if unicodedata.category(c) in _WORD_CATS:
            return True
        if c in (".", ",") and i - 1 >= 0 and i + 1 < len(text):
            return (unicodedata.category(text[i - 1]) == "Nd"
                    and unicodedata.category(text[i + 1]) == "Nd")
        return False

    out, i, n = [], 0, len(text)
    while i < n:
        if word(i):
            j = i + 1
            while j < n and word(j):
                j += 1
            out.append({"surface": text[i:j], "char_start": i, "char_end": j})
            i = j
        else:
            i += 1
    return out


def _osa(a: str, b: str) -> int:
    """Damerau–Levenshtein, optimal-string-alignment variant (unit costs) — the one
    distance metric everywhere in this spec."""
    la, lb = len(a), len(b)
    prev2, prev = None, list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def _has_letter(s: str) -> bool:
    return any(unicodedata.category(c).startswith("L") for c in s)


# ── compiler — `vinkona-lex compile` (§6) ────────────────────────────────────────────────
def _lexicon_version(rows: list) -> str:
    """§6.5: sha256 over active rows sorted by alias_id, each a JSON array with the
    weight rendered as a %.4f string, joined by newlines."""
    lines = []
    for r in sorted(rows, key=lambda r: r["alias_id"]):
        lines.append(json.dumps(
            [r["alias_id"], r["node_id"], r["surface"], r["norm_seq"], r["n_tokens"],
             r["alias_type"], "%.4f" % r["weight"], r["case_mode"], r["fuzzy_allowed"],
             r["origin"], r["derived_from"], r["status"]],
            ensure_ascii=False, separators=(",", ":")))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def compile_lexicon(db, out, *, node_table: str = "conflict_node", node_ids=None) -> dict:
    """Validate ALL active alias rows (collect every finding), then — only if no ERROR —
    write the artifacts.  The compiler validates rows; it never creates or reweights
    them (curation, including inflection generation, is upstream — VINKONA-LEX-02)."""
    own = isinstance(db, (str, Path))
    conn = sqlite3.connect(str(db)) if own else db
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT alias_id,node_id,surface,norm_seq,n_tokens,alias_type,weight,"
            "case_mode,fuzzy_allowed,origin,derived_from,status FROM alias "
            "WHERE status='active' ORDER BY alias_id")]
        known = (set(node_ids) if node_ids is not None else
                 {r[0] for r in conn.execute(f"SELECT node_id FROM {node_table}")})
    finally:
        if own:
            conn.close()

    findings = []
    err = lambda code, aid, msg: findings.append(
        {"level": "ERROR", "code": code, "alias_id": aid, "message": msg})
    warn = lambda code, aid, msg: findings.append(
        {"level": "WARN", "code": code, "alias_id": aid, "message": msg})

    for r in rows:
        toks = tokenize(r["surface"])
        norms = [norm_token(t["surface"]) for t in toks]
        if any(n == "" for n in norms):
            err("EMPTY_NORM", r["alias_id"],
                f"surface {r['surface']!r} yields an empty norm token")
        calc_seq = _US.join(norms)
        if calc_seq != r["norm_seq"] or len(toks) != r["n_tokens"]:
            err("C4", r["alias_id"],
                f"stored norm_seq/n_tokens differ from recomputation of {r['surface']!r}")
        if len(norms) == 1:
            if len(norms[0]) <= 2 and not (
                    r["case_mode"] in ("exact", "caps") and r["fuzzy_allowed"] == 0):
                err("C1", r["alias_id"],
                    f"short alias {norms[0]!r} requires case_mode exact/caps and fuzzy_allowed=0")
            if norms[0] in EN_STOP_V1 and r["case_mode"] not in ("exact", "caps"):
                err("C2", r["alias_id"],
                    f"stopword alias {norms[0]!r} requires case_mode exact/caps")
        if r["node_id"] not in known:
            err("C3", r["alias_id"], f"node_id not in node table: {r['node_id']}")

    by_seq: dict = {}
    for r in rows:
        by_seq.setdefault(r["norm_seq"], []).append(r)
    for seq, group in sorted(by_seq.items()):
        nodes = sorted({g["node_id"] for g in group})
        if len(nodes) > 4:
            warn("AMBIGUOUS_NORM", None,
                 f"norm_seq {seq.replace(_US, ' ')!r} maps to {len(nodes)} nodes")
        pub_nodes = {g["node_id"] for g in group if g["origin"] == "pub"}
        for g in group:
            if g["origin"] != "pub" and pub_nodes - {g["node_id"]}:
                warn("ORIGIN_SHADOW", g["alias_id"],
                     f"{g['origin']} alias shares norm_seq with a pub alias of a different node")

    n_err = sum(1 for f in findings if f["level"] == "ERROR")
    n_warn = len(findings) - n_err
    report = {"findings": findings,
              "counts": {"aliases": len(rows), "errors": n_err, "warnings": n_warn,
                         "vocab_size": 0},
              "ok": n_err == 0, "lexicon_version": None}

    vocab: dict = {}
    if n_err == 0:
        toks_sorted = sorted({t for r in rows for t in r["norm_seq"].split(_US)},
                             key=lambda s: s.encode("utf-8"))
        for tid, tok in enumerate(toks_sorted):
            freq = sum(1 for r in rows if tok in r["norm_seq"].split(_US))
            fuzzy1 = any(tok in r["norm_seq"].split(_US) and r["fuzzy_allowed"] == 1
                         for r in rows)
            fuzzy0 = any(tok in r["norm_seq"].split(_US) and r["fuzzy_allowed"] == 0
                         for r in rows)
            eligible = len(tok) >= 4 and _has_letter(tok)
            if fuzzy1 and eligible:
                cls = "targetable"
            elif fuzzy0 and eligible:
                cls = "suppressed"
            else:
                cls = "none"
            vocab[tok] = {"id": tid, "freq": freq, "fuzzy_class": cls}
        report["counts"]["vocab_size"] = len(vocab)
        report["lexicon_version"] = _lexicon_version(rows)

    if out is not None:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        (p / "compile_report.json").write_text(
            json.dumps({"findings": findings, "counts": report["counts"]},
                       ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if n_err == 0:                          # fail ⇒ nothing but the report is written
            (p / "lexicon.json").write_text(
                json.dumps({"aliases": rows, "vocab": vocab,
                            "symspell": {"max_edit": 2, "prefix": 7}},
                           ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            import time
            (p / "lexicon.meta.json").write_text(
                json.dumps({"lexicon_version": report["lexicon_version"],
                            "norm_version": NORM_VERSION, "tok_version": TOK_VERSION,
                            "alias_count": len(rows), "vocab_size": len(vocab),
                            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                           ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return report


# ── token-level Aho–Corasick (character-level MUST NOT be used) ────────────────────────
class _AC:
    def __init__(self):
        self.goto = [{}]
        self.out = [[]]
        self.fail = [0]

    def add(self, seq: list, alias_id: int):
        s = 0
        for t in seq:
            nxt = self.goto[s].get(t)
            if nxt is None:
                self.goto.append({})
                self.out.append([])
                nxt = len(self.goto) - 1
                self.goto[s][t] = nxt
            s = nxt
        self.out[s].append((alias_id, len(seq)))

    def build(self):
        from collections import deque
        self.fail = [0] * len(self.goto)
        q = deque()
        for t in sorted(self.goto[0]):
            q.append(self.goto[0][t])
        while q:
            s = q.popleft()
            for t in sorted(self.goto[s]):
                u = self.goto[s][t]
                q.append(u)
                f = self.fail[s]
                while f and t not in self.goto[f]:
                    f = self.fail[f]
                self.fail[u] = self.goto[f].get(t, 0) if self.goto[f].get(t, 0) != u else 0
                self.out[u] = self.out[u] + self.out[self.fail[u]]

    def feed(self, ids: list):
        """Yield (tok_start, tok_end, alias_id).  A None id (OOV / empty norm) resets to
        the root — such tokens block matching rather than being skippable."""
        s = 0
        for pos, t in enumerate(ids):
            if t is None:
                s = 0
                continue
            while s and t not in self.goto[s]:
                s = self.fail[s]
            s = self.goto[s].get(t, 0)
            for alias_id, length in self.out[s]:
                yield (pos - length + 1, pos + 1, alias_id)


# ── matcher — `vinkona-lex match` (§7) ───────────────────────────────────────────────────
class Matcher:
    """Immutable and thread-safe after ``load``; ``match`` is reentrant and MUST NOT
    raise for any valid input string."""

    @classmethod
    def load(cls, artifact_dir: Path) -> "Matcher":
        d = Path(artifact_dir)
        try:
            lex = json.loads((d / "lexicon.json").read_text(encoding="utf-8"))
            meta = json.loads((d / "lexicon.meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise LexError("E_ARTIFACT_MISSING", f"artifact dir incomplete/unreadable: {e}")
        if meta.get("norm_version") != NORM_VERSION or meta.get("tok_version") != TOK_VERSION:
            raise LexError("E_ARTIFACT_MISMATCH",
                           f"artifact versions {meta.get('norm_version')}/{meta.get('tok_version')} "
                           f"!= {NORM_VERSION}/{TOK_VERSION}")
        self = cls()
        self.lexicon_version = meta["lexicon_version"]
        self.vocab = lex["vocab"]                       # norm token -> {id, freq, fuzzy_class}
        self.aliases = {}
        self._ac = _AC()
        for r in sorted(lex["aliases"], key=lambda r: r["alias_id"]):
            r = dict(r)
            r["norm_tokens"] = r["norm_seq"].split(_US)
            r["surface_tokens"] = [t["surface"] for t in tokenize(r["surface"])]
            self.aliases[r["alias_id"]] = r
            self._ac.add([self.vocab[t]["id"] for t in r["norm_tokens"]], r["alias_id"])
        self._ac.build()
        self._by_id = {v["id"]: t for t, v in self.vocab.items()}
        # SymSpell deletion index (max edit 2, prefix 7) over fuzzy_class != none
        self._deletes: dict = {}
        self._fuzzy_tokens = sorted(t for t, v in self.vocab.items()
                                    if v["fuzzy_class"] != "none")
        for tok in self._fuzzy_tokens:
            for v in self._delete_variants(tok[:7]):
                self._deletes.setdefault(v, set()).add(tok)
        return self

    @staticmethod
    def _delete_variants(prefix: str) -> set:
        res, frontier = {prefix}, {prefix}
        for _ in range(2):                              # max edit distance 2
            nxt = set()
            for w in frontier:
                for i in range(len(w)):
                    nxt.add(w[:i] + w[i + 1:])
            res |= nxt
            frontier = nxt
        return res

    def _lookup(self, q: str, d_max: int) -> list:
        """SymSpell candidates verified with OSA; hits (token, distance <= d_max)."""
        cands: set = set()
        for v in self._delete_variants(q[:7]):
            cands |= self._deletes.get(v, set())
        hits = []
        for t in sorted(cands):
            d = _osa(q, t)
            if d <= d_max:
                hits.append((t, d))
        return hits

    def _case_ok(self, alias: dict, start: int, tokens: list) -> bool:
        mode = alias["case_mode"]
        if mode == "fold":
            return True
        for k in range(alias["n_tokens"]):
            ut = tokens[start + k]["surface"]
            if mode == "exact":
                if (unicodedata.normalize("NFKC", ut)
                        != unicodedata.normalize("NFKC", alias["surface_tokens"][k])):
                    return False
            else:                                       # caps: every cased letter is Lu
                for c in ut:
                    if unicodedata.category(c) in ("Ll", "Lt"):
                        return False
        return True

    # ── the six stages ──
    def match(self, text: str) -> dict:
        # M0
        if not isinstance(text, str) or any("\ud800" <= c <= "\udfff" for c in text):
            raise LexError("E_INVALID_INPUT", "input must be a str with no lone surrogates")
        if len(text) > _MAX_INPUT:
            raise LexError("E_INPUT_TOO_LONG", f"{len(text)} code points > {_MAX_INPUT}")

        # M1
        tokens = []
        ids = []
        for i, t in enumerate(tokenize(text)):
            norm = norm_token(t["surface"])
            tokens.append({"i": i, "surface": t["surface"], "norm": norm,
                           "char_start": t["char_start"], "char_end": t["char_end"],
                           "corrected_from": None, "edit_distance": 0})
            v = self.vocab.get(norm) if norm else None
            ids.append(v["id"] if v else None)

        # M2 — exact pass + case_mode filtering (always against the ORIGINAL text)
        exact_pairs = set()
        for s, e, aid in self._ac.feed(ids):
            if self._case_ok(self.aliases[aid], s, tokens):
                exact_pairs.add((s, e, aid))
        exact_cover = {p for s, e, _ in exact_pairs for p in range(s, e)}

        # M3 — spelling correction on eligible OOV tokens, ascending index
        flags = []
        corrected: dict = {}
        for i, tok in enumerate(tokens):
            norm = tok["norm"]
            if (not norm or norm in self.vocab or i in exact_cover
                    or len(norm) < 4 or not _has_letter(norm)):
                continue
            d_max = 1 if len(norm) <= 6 else 2
            hits = self._lookup(norm, d_max)
            if not hits:
                continue
            dmin = min(d for _, d in hits)
            best = [t for t, d in hits if d == dmin]
            suppressed = [t for t in best if self.vocab[t]["fuzzy_class"] == "suppressed"]
            if suppressed:                              # never correct toward safety-critical
                nearest = sorted(suppressed,
                                 key=lambda t: (-self.vocab[t]["freq"], t))[0]
                flags.append({"type": "fuzzy_suppressed", "stage": "token",
                              "token_index": i, "nearest": nearest, "distance": dmin})
                continue
            choice = sorted(best, key=lambda t: (-self.vocab[t]["freq"], t))[0]
            tok["corrected_from"] = norm
            tok["norm"] = choice
            tok["edit_distance"] = dmin
            ids[i] = self.vocab[choice]["id"]
            corrected[i] = dmin

        # M4 — fuzzy pass: union both passes, re-filter, purge fuzzy_allowed=0 spans
        pairs = set(exact_pairs)
        if corrected:
            for s, e, aid in self._ac.feed(ids):
                if self._case_ok(self.aliases[aid], s, tokens):
                    pairs.add((s, e, aid))
        spans: dict = {}
        for s, e, aid in pairs:
            spans.setdefault((s, e), set()).add(aid)
        for (s, e), aids in sorted(spans.items()):
            cov_corr = [p for p in range(s, e) if p in corrected]
            if not cov_corr:
                continue
            removed = sorted(a for a in aids if self.aliases[a]["fuzzy_allowed"] == 0)
            aids -= set(removed)
            if removed and not aids:
                alias = self.aliases[removed[0]]
                flags.append({"type": "fuzzy_suppressed", "stage": "span",
                              "token_index": min(cov_corr),
                              "nearest": " ".join(alias["norm_tokens"]),
                              "distance": sum(corrected[p] for p in cov_corr)})
        spans = {k: v for k, v in spans.items() if v}

        # M6 scoring first (M5 selection needs top-candidate scores)
        scored: dict = {}
        for (s, e), aids in spans.items():
            cands = []
            for aid in aids:
                a = self.aliases[aid]
                score = a["weight"]
                for p in range(s, e):
                    if p in corrected:
                        score *= _F[corrected[p]]
                cands.append({"alias_id": aid, "node_id": a["node_id"],
                              "alias_type": a["alias_type"],
                              "weight": round(a["weight"], 4),
                              "score": round(score, 4)})
            cands.sort(key=lambda c: (-c["score"], c["alias_id"]))
            scored[(s, e)] = cands

        # M5 — leftmost-longest selection; nested and crossing matches are dropped
        remaining = list(scored)
        emitted = []
        while remaining:
            best = min(remaining, key=lambda se: (
                se[0], -se[1], -scored[se][0]["score"], scored[se][0]["alias_id"]))
            emitted.append(best)
            remaining = [se for se in remaining
                         if se[1] <= best[0] or se[0] >= best[1]]
        emitted.sort(key=lambda se: se[0])

        out_spans = []
        covered = set()
        for span_id, (s, e) in enumerate(emitted):
            covered.update(range(s, e))
            cs, ce = tokens[s]["char_start"], tokens[e - 1]["char_end"]
            out_spans.append({
                "span_id": span_id, "tok_start": s, "tok_end": e,
                "char_start": cs, "char_end": ce,
                "surface_original": text[cs:ce],
                "matched_norm": " ".join(tokens[p]["norm"] for p in range(s, e)),
                "fuzzy": any(p in corrected for p in range(s, e)),
                "candidates": scored[(s, e)]})

        flags.sort(key=lambda f: (f["token_index"], 0 if f["stage"] == "token" else 1))
        return {
            "matcher_version": MATCHER_VERSION,
            "lexicon_version": self.lexicon_version,
            "norm_version": NORM_VERSION,
            "tok_version": TOK_VERSION,
            "text": text,
            "tokens": tokens,
            "spans": out_spans,
            "unmatched_token_indices": [t["i"] for t in tokens
                                        if t["norm"] and t["i"] not in covered],
            "flags": flags,
        }

    def match_json(self, text: str) -> bytes:
        """§8.3 canonical serialization — weight/score with exactly 4 decimals, fixed key
        orders, compact separators, non-ASCII raw."""
        r = self.match(text)
        s = json.dumps                                   # proper string escaping
        toks = ",".join(
            '{"i":%d,"surface":%s,"norm":%s,"char_start":%d,"char_end":%d,'
            '"corrected_from":%s,"edit_distance":%d}'
            % (t["i"], s(t["surface"], ensure_ascii=False), s(t["norm"], ensure_ascii=False),
               t["char_start"], t["char_end"],
               "null" if t["corrected_from"] is None else s(t["corrected_from"], ensure_ascii=False),
               t["edit_distance"]) for t in r["tokens"])
        spans = ",".join(
            '{"span_id":%d,"tok_start":%d,"tok_end":%d,"char_start":%d,"char_end":%d,'
            '"surface_original":%s,"matched_norm":%s,"fuzzy":%s,"candidates":[%s]}'
            % (sp["span_id"], sp["tok_start"], sp["tok_end"], sp["char_start"],
               sp["char_end"], s(sp["surface_original"], ensure_ascii=False),
               s(sp["matched_norm"], ensure_ascii=False),
               "true" if sp["fuzzy"] else "false",
               ",".join('{"alias_id":%d,"node_id":%s,"alias_type":%s,"weight":%.4f,"score":%.4f}'
                        % (c["alias_id"], s(c["node_id"], ensure_ascii=False),
                           s(c["alias_type"], ensure_ascii=False), c["weight"], c["score"])
                        for c in sp["candidates"])) for sp in r["spans"])
        flags = ",".join(
            '{"type":"fuzzy_suppressed","stage":%s,"token_index":%d,"nearest":%s,"distance":%d}'
            % (s(f["stage"], ensure_ascii=False), f["token_index"],
               s(f["nearest"], ensure_ascii=False), f["distance"]) for f in r["flags"])
        doc = ('{"matcher_version":%s,"lexicon_version":%s,"norm_version":"NORM-1",'
               '"tok_version":"TOK-1","text":%s,"tokens":[%s],"spans":[%s],'
               '"unmatched_token_indices":[%s],"flags":[%s]}'
               % (s(MATCHER_VERSION), s(r["lexicon_version"]),
                  s(r["text"], ensure_ascii=False), toks, spans,
                  ",".join(str(i) for i in r["unmatched_token_indices"]), flags))
        return doc.encode("utf-8")
