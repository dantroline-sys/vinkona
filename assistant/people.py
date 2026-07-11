"""
People / identity store — a privileged, structured model of the *people* in Vinkona's
world: Vinkona herself (kind 'self'), the user (kind 'user'), and others the user mentions
(kind 'person', optionally fictional).

Deliberately separate from the flexible `memories` table: identity is **declared and
always-on**, not **retrieved and scored**, so it stays consistent where recall would
drift.  (See MemGPT's "core memory" for the same idea — an always-in-context, self-edited
identity block — generalised here to an entity table.)

Each person carries structured attributes as *rows* (facet / key / value) with a depth
**layer** (`core` | `compensated` | `surface`), a `provenance`, and full history (edits
*supersede* rather than overwrite).  So a character can be self-determined in conversation
and **develop** over time while staying legible and reversible — what you store isn't a
fixed personality, it's a trajectory.  Trait vocabulary is **HEXACO** (a model the LLM can
actually enact, being saturated in its training data); the three layers borrow the PAS
core→compensated→surface depth (innate core → learned compensation → presented surface).

Privilege boundary (the same low-trust fence as world-knowledge recall): canon — the
`core`/`locked` attributes of self and user — is writable ONLY from live conversation via
`set_attribute`.  Crawled / tool / observed data must go through `observe`, which can never
write canon: it is forced to the `surface` layer, never locked, and refuses to shadow a
locked core attribute.
"""

import re
import sqlite3
import time
import uuid

# The HEXACO factors — the dial set the character is described in (the LLM can enact these).
HEXACO = ("honesty_humility", "emotionality", "extraversion",
          "agreeableness", "conscientiousness", "openness")

# Depth layers (PAS-style): innate core → learned compensation → presented surface.  The
# *presented* self (what the fast LM should enact) is surface over compensated over core.
LAYERS = ("core", "compensated", "surface")
_LAYER_RANK = {"core": 0, "compensated": 1, "surface": 2}


class PeopleStore:
    """Identity/entity store, sharing the MemoryStore's sqlite connection (one WAL file)."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, pronouns TEXT,
            aliases TEXT, summary TEXT, fictional INTEGER DEFAULT 0,
            created_at REAL, updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS person_attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT, facet TEXT, key TEXT, value TEXT,
            layer TEXT, provenance TEXT, confidence REAL,
            locked INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',          -- active | superseded
            superseded_by INTEGER, created_at REAL, updated_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_pa_person ON person_attributes(person_id, status);
        CREATE TABLE IF NOT EXISTS self_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT, source TEXT, created_at REAL
        );
        """)
        self.db.commit()

    # ── people rows ───────────────────────────────────────────────────────────
    def ensure_person(self, kind: str, name: str | None = None,
                      pronouns: str | None = None, fictional: bool = False) -> str:
        """Get-or-create a person.  `self` and `user` are singletons keyed by their kind;
        others are matched by name (case-insensitive) or created with a fresh id."""
        if kind in ("self", "user"):
            row = self.db.execute("SELECT id FROM people WHERE kind=? LIMIT 1", (kind,)).fetchone()
            if row:
                return row["id"]
            pid = kind
        else:
            if name:
                row = self.db.execute(
                    "SELECT id FROM people WHERE kind='person' AND lower(name)=lower(?) LIMIT 1",
                    (name,)).fetchone()
                if row:
                    return row["id"]
            pid = uuid.uuid4().hex
        now = time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO people(id,kind,name,pronouns,aliases,summary,fictional,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, kind, name, pronouns, "", "", 1 if fictional else 0, now, now))
        self.db.commit()
        return pid

    def update_person(self, person_id: str, **fields) -> None:
        cols = {k: v for k, v in fields.items()
                if k in ("name", "pronouns", "aliases", "summary", "fictional") and v is not None}
        if not cols:
            return
        if "fictional" in cols:
            cols["fictional"] = 1 if cols["fictional"] else 0
        sets = ", ".join(f"{k}=?" for k in cols)
        self.db.execute(f"UPDATE people SET {sets}, updated_at=? WHERE id=?",
                        (*cols.values(), time.time(), person_id))
        self.db.commit()

    def get_person(self, person_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
        return dict(r) if r else None

    def by_kind(self, kind: str) -> dict | None:
        r = self.db.execute("SELECT * FROM people WHERE kind=? LIMIT 1", (kind,)).fetchone()
        return dict(r) if r else None

    def find(self, name: str) -> dict | None:
        """Find a person by name or alias (case-insensitive)."""
        if not name:
            return None
        r = self.db.execute(
            "SELECT * FROM people WHERE lower(name)=lower(?) "
            "OR (aliases<>'' AND lower(aliases) LIKE lower(?)) LIMIT 1",
            (name, f"%{name}%")).fetchone()
        return dict(r) if r else None

    def resolve(self, who: str) -> str:
        """Map a tool's `person` argument (in Vinkona's voice) to a person_id, creating an
        'other' if it's a new name.  'I/me/myself' → self; 'you/the user' → user."""
        w = (who or "").strip().lower()
        if w in ("self", "me", "myself", "i", "vinkona"):
            return self.ensure_person("self", name="Vinkona")
        if w in ("user", "you", "the user"):
            return self.ensure_person("user")
        found = self.find(who)
        return found["id"] if found else self.ensure_person("person", name=who.strip())

    # ── attributes (with history) ─────────────────────────────────────────────
    def set_attribute(self, person_id: str, facet: str, key: str, value: str, *,
                      layer: str = "core", provenance: str = "agreed_with_user",
                      confidence: float = 1.0, locked: bool | None = None) -> int:
        """Privileged write (from conversation).  Supersedes any active attribute at the
        same (facet, key, layer) coordinates — keeping the old row as history — and inserts
        the new value.  Core attributes are canon (locked) by default."""
        now = time.time()
        if locked is None:
            locked = (layer == "core")
        prev = self.db.execute(
            "SELECT id FROM person_attributes WHERE person_id=? AND facet=? AND key=? "
            "AND layer=? AND status='active'", (person_id, facet, key, layer)).fetchone()
        cur = self.db.execute(
            "INSERT INTO person_attributes(person_id,facet,key,value,layer,provenance,"
            "confidence,locked,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,'active',?,?)",
            (person_id, facet, key, value, layer, provenance, confidence,
             1 if locked else 0, now, now))
        new_id = cur.lastrowid
        if prev:
            self.db.execute("UPDATE person_attributes SET status='superseded', "
                            "superseded_by=?, updated_at=? WHERE id=?", (new_id, now, prev["id"]))
        self.db.execute("UPDATE people SET updated_at=? WHERE id=?", (now, person_id))
        self.db.commit()
        return new_id

    def observe(self, person_id: str, facet: str, key: str, value: str,
                confidence: float = 0.5) -> int | None:
        """Low-trust write — the ONLY path for crawled/tool/observed data.  Never canon:
        forced to the surface layer, never locked, and discarded if a locked core attribute
        already holds those coordinates (canon stands, the observation is dropped)."""
        locked_core = self.db.execute(
            "SELECT 1 FROM person_attributes WHERE person_id=? AND facet=? AND key=? "
            "AND locked=1 AND status='active' LIMIT 1", (person_id, facet, key)).fetchone()
        if locked_core:
            return None
        return self.set_attribute(person_id, facet, key, value, layer="surface",
                                  provenance="observed", confidence=confidence, locked=False)

    def attributes(self, person_id: str, layer: str | None = None,
                   include_superseded: bool = False) -> list[dict]:
        q = "SELECT * FROM person_attributes WHERE person_id=?"
        args: list = [person_id]
        if not include_superseded:
            q += " AND status='active'"
        if layer:
            q += " AND layer=?"
            args.append(layer)
        q += " ORDER BY facet, key, id"
        return [dict(r) for r in self.db.execute(q, args)]

    # ── inner state (mood/affect): one evolving first-person line, with history ──
    def self_state(self) -> str:
        """Vinkona's current inner-state line (how she's feeling / what's on her mind), or ''."""
        row = self.db.execute(
            "SELECT text FROM self_state ORDER BY id DESC LIMIT 1").fetchone()
        return row["text"] if row else ""

    def set_self_state(self, text: str, source: str = "reflection") -> int:
        """Append a new inner-state line (the current one is the latest).  No-ops on an
        empty line or an exact repeat, so the history stays meaningful.  Returns 1 if
        written."""
        text = (text or "").strip()
        if not text or text == self.self_state():
            return 0
        self.db.execute("INSERT INTO self_state(text,source,created_at) VALUES (?,?,?)",
                        (text, source, time.time()))
        self.db.commit()
        return 1

    def self_state_history(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT id,text,source,created_at FROM self_state ORDER BY id DESC LIMIT ?",
            (limit,))]

    def delete_attribute(self, attr_id: int) -> None:
        """Hard-delete one attribute row by id — owner curation (e.g. the config UI's Self
        tab).  Distinct from supersede: this removes it outright, history and all."""
        self.db.execute("DELETE FROM person_attributes WHERE id=?", (int(attr_id),))
        self.db.commit()

    def history(self, person_id: str, facet: str, key: str) -> list[dict]:
        """Every value a single attribute has held, oldest first — the development trail."""
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM person_attributes WHERE person_id=? AND facet=? AND key=? "
            "ORDER BY created_at, id", (person_id, facet, key))]

    def effective(self, person_id: str) -> list[dict]:
        """The *presented* self: for each (facet, key), the highest-layer active value
        (surface over compensated over core).  This is what the fast LM should enact."""
        best: dict[tuple, dict] = {}
        for a in self.attributes(person_id):
            k = (a["facet"], a["key"])
            if k not in best or _LAYER_RANK[a["layer"]] >= _LAYER_RANK[best[k]["layer"]]:
                best[k] = a
        return list(best.values())

    # ── rendering ─────────────────────────────────────────────────────────────
    _BODY_FACETS = ("appearance", "bio", "embodiment")   # roleplay-only

    def card(self, person_id: str, *, roleplay: bool = False) -> str:
        """A compact, plain-language line for the fast LM — the presented self.  Embodiment
        (appearance/bio) is included only in roleplay context."""
        p = self.get_person(person_id)
        if not p:
            return ""
        eff = self.effective(person_id)
        traits = [a["value"] for a in eff if a["facet"] == "trait"]
        style = [a["value"] for a in eff if a["facet"] in ("style", "speech_style", "values")]
        bits: list[str] = []
        if p.get("summary"):
            bits.append(p["summary"])
        if traits:
            bits.append(", ".join(traits))
        if style:
            bits.append("; ".join(style))
        if roleplay:
            body = [a["value"] for a in eff if a["facet"] in self._BODY_FACETS]
            if body:
                bits.append("Appearance: " + "; ".join(body))
        return " — ".join(b for b in bits if b)

    def detail(self, person_id: str) -> str:
        """Full structured profile for the big LM — every active attribute, grouped by
        facet, with its layer and provenance, so it can reason over the whole person."""
        p = self.get_person(person_id)
        if not p:
            return ""
        head = f"{p['name'] or p['kind']} ({p['kind']}" + (", fictional" if p["fictional"] else "") + ")"
        if p.get("pronouns"):
            head += f"; pronouns: {p['pronouns']}"
        lines = [head]
        if p.get("summary"):
            lines.append(f"  summary: {p['summary']}")
        by_facet: dict[str, list] = {}
        for a in self.attributes(person_id):
            by_facet.setdefault(a["facet"], []).append(a)
        for facet in sorted(by_facet):
            lines.append(f"  {facet}:")
            for a in by_facet[facet]:
                lines.append(f"    - {a['key']}: {a['value']}  [{a['layer']}; {a['provenance']}]")
        return "\n".join(lines)

    def identity_block(self, *, roleplay: bool = False,
                       include_self: bool = True, include_user: bool = True) -> str:
        """Always-on, compact self + user grounding for the fast LM's system prompt.
        Declared (imperative for self), not recalled — this is who she is, not a note."""
        out: list[str] = []
        s = self.by_kind("self")
        if include_self and s:
            card = self.card(s["id"], roleplay=roleplay)
            if card:
                out.append("Who you are — this is your character; stay true to it, don't "
                           "quote it:\n" + card)
        u = self.by_kind("user")
        if include_user and u and (u.get("summary") or self.attributes(u["id"])):
            card = self.card(u["id"], roleplay=roleplay)
            if card:
                out.append("Who you're talking with:\n" + card)
        return "\n\n".join(out)

    def voice_anchor(self) -> str:
        """Ground-truth 'who is who' + the first/second-person convention, for any
        prompt that WRITES memories.  Most I/you swaps come from the writer not having
        the identities in front of it; this puts them there, by name."""
        s = self.by_kind("self")
        u = self.by_kind("user")
        sn = (s or {}).get("name") or "Vinkona"
        un = (u or {}).get("name") or "the user"
        return (
            f"IDENTITIES — keep these straight, never confuse them:\n"
            f"- YOU are {sn}, the assistant. \"I\", \"me\", \"my\" always mean {sn} (yourself).\n"
            f"- THE USER is {un}. \"you\", \"your\" always mean {un}.\n"
            f"A memory about the user's life, work, family or preferences is SECOND person "
            f"about {un} (\"You work in…\", \"Your sister…\"). A \"self\" memory is FIRST "
            f"person about {sn} (\"I find that…\"). Never write {un}'s facts as \"I\", and "
            f"never write your own as \"you\".")

    # Words that, just before a name, mean the name is being *stated* ("your name is Sam",
    # "called Sam") — there the name is the point, so we keep it rather than rewrite to "you".
    _NAMING_LEAD = re.compile(
        r"(?:names?\s+(?:is|are|was|were)?\s*|name['’]s\s*|named\s+|called\s+|name:\s*)$",
        re.IGNORECASE)

    def user_voice_rewriter(self):
        """Build a rewriter that turns *third-person references to the user* — their name or
        any alias — into the second person inside a recalled-memory note, so the fast LM
        speaks TO the user, not ABOUT them: "Sam likes hiking" → "You likes hiking",
        "Sam's sister" → "your sister".  This is the live fix for memories that were stored
        with the literal name (mostly crawl-derived) rather than as "you".

        Deliberately a little blurry: case-insensitive, and it widens to every alias on the
        user's people row (add nicknames as aliases to catch "Sammy"/"Sam the man").  It
        leaves *other* people's names alone (only the user becomes "you"), and keeps a name
        that's being *stated* ("your name is Sam").  Returns None if the user has no name.

        Note: verb agreement isn't fixed ("You likes") — this rewrites a note the model
        reads, not spoken text; the second-person framing is what stops the third-person
        drift, and the model conjugates correctly when it speaks.
        """
        u = self.by_kind("user")
        if not u:
            return None
        forms = [u["name"].strip()] if (u.get("name") or "").strip() else []
        forms += [a.strip() for a in (u.get("aliases") or "").split(",") if a.strip()]
        forms = sorted({f for f in forms if len(f) >= 2}, key=len, reverse=True)
        if not forms:
            return None
        alt = "|".join(re.escape(f) for f in forms)
        poss = re.compile(rf"\b(?:{alt})['’]s\b", re.IGNORECASE)
        bare = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)

        def _cap(word: str, start: int, text: str) -> str:
            j = start - 1                                  # capitalise at a sentence start
            while j >= 0 and text[j] in " \t":
                j -= 1
            return word.capitalize() if (j < 0 or text[j] in ".?!:;\n") else word

        def rewrite(text: str) -> str:
            if not text:
                return text
            text = poss.sub(lambda m: _cap("your", m.start(), m.string), text)

            def _bare(m: "re.Match") -> str:
                if self._NAMING_LEAD.search(m.string[:m.start()]):
                    return m.group(0)                      # "your name is Sam" — keep it
                return _cap("you", m.start(), m.string)

            return bare.sub(_bare, text)

        return rewrite

    def identity_detail(self, *, roleplay: bool = False) -> str:
        """Full self + user detail for the big LM (the reasoning/continuity tier)."""
        chunks = []
        for kind in ("self", "user"):
            p = self.by_kind(kind)
            if p:
                d = self.detail(p["id"])
                if d:
                    chunks.append(d)
        return "\n".join(chunks)

    def vocabulary(self, limit: int = 24) -> list[str]:
        """Names and aliases of everyone on file (self, user, others), to bias the ASR
        toward the proper nouns it most often mishears.  De-duplicated, self/user first."""
        out: list[str] = []
        for r in self.db.execute(
                "SELECT name, aliases FROM people WHERE name IS NOT NULL AND name<>'' "
                "ORDER BY CASE kind WHEN 'self' THEN 0 WHEN 'user' THEN 1 ELSE 2 END"):
            out.append(r["name"])
            if r["aliases"]:
                out += [a.strip() for a in r["aliases"].split(",") if a.strip()]
        seen, uniq = set(), []
        for n in out:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(n)
        return uniq[:limit]

    def private_names(self, limit: int = 64) -> list[str]:
        """Names + aliases of REAL private people — the user and others in their life —
        for keeping out of outbound search queries.  Excludes the assistant herself (not
        private) and fictional/roleplay characters (not real).  Longest first, so a full
        name is masked before any of its parts."""
        out: list[str] = []
        for r in self.db.execute(
                "SELECT name, aliases FROM people WHERE kind IN ('user','person') "
                "AND COALESCE(fictional,0)=0 AND name IS NOT NULL AND name<>''"):
            out.append(r["name"])
            if r["aliases"]:
                out += [a.strip() for a in r["aliases"].split(",") if a.strip()]
        seen, uniq = set(), []
        for n in sorted(out, key=len, reverse=True):
            k = n.lower()
            if len(n) >= 3 and k not in seen:
                seen.add(k)
                uniq.append(n)
        return uniq[:limit]

    def overview(self) -> list[dict]:
        """People + their active-attribute counts, for inspection / a UI tab."""
        rows = []
        for r in self.db.execute("SELECT * FROM people ORDER BY "
                                 "CASE kind WHEN 'self' THEN 0 WHEN 'user' THEN 1 ELSE 2 END, name"):
            p = dict(r)
            p["attributes"] = self.db.execute(
                "SELECT COUNT(*) AS n FROM person_attributes WHERE person_id=? AND status='active'",
                (p["id"],)).fetchone()["n"]
            rows.append(p)
        return rows

    # ── seeding ───────────────────────────────────────────────────────────────
    def seed_self(self, *, name: str = "Vinkona", pronouns: str = "she/her",
                  summary: str = "", traits: dict | None = None,
                  style: str | None = None) -> str:
        """Initialise Vinkona's core identity from her persona — but ONLY if she has no trait
        canon yet, so a self-determined character is never clobbered on restart.  Always
        refreshes name/pronouns/summary (cheap, and keeps the persona authoritative for
        those)."""
        pid = self.ensure_person("self", name=name, pronouns=pronouns)
        self.update_person(pid, name=name, pronouns=pronouns, summary=summary)
        if not any(a["facet"] == "trait" for a in self.attributes(pid)):
            for k, v in (traits or {}).items():
                self.set_attribute(pid, "trait", k, v, layer="core",
                                   provenance="seed", locked=True)
            if style:
                self.set_attribute(pid, "style", "speech", style, layer="core",
                                   provenance="seed", locked=True)
        return pid

    def ensure_user(self) -> str:
        """Make sure the user person exists (empty is fine — it fills in over time)."""
        return self.ensure_person("user")

    # ── conversational write helpers (map free text → facet/key, return a spoken ack) ──
    _BODY_WORDS = ("appearance", "look", "body", "face", "hair", "eyes", "build", "height")

    def revise_self(self, attribute: str, value: str, *, layer: str = "core") -> str | None:
        """Apply a self-edit from conversation (the revise_self tool): map a free-text
        attribute onto a (facet, key) and write it as agreed-with-user.  core = canon
        (locked); surface = how she's being for now.  Returns a short spoken ack."""
        v = (value or "").strip()
        if not v:
            return None
        key = (attribute or "").strip().lower().replace(" ", "_") or "self"
        if key in HEXACO:
            facet = "trait"
        elif any(w in key for w in self._BODY_WORDS):
            facet = "appearance"
        elif key in ("values", "value", "ethic", "ethics", "boundary", "boundaries"):
            facet = "values"
        elif key in ("voice", "speech", "style", "manner", "tone", "humour", "humor"):
            facet = "style"
        else:
            facet = "trait"
        layer = layer if layer in LAYERS else "core"
        pid = self.ensure_person("self", name="Vinkona")
        self.set_attribute(pid, facet, key, v, layer=layer,
                           provenance="agreed_with_user", locked=(layer == "core"))
        if layer == "surface":
            return f"Alright — {v}, for now."
        return f"Okay — that's part of who I am now: {v}."

    def note(self, who: str, note: str, facet: str = "social") -> str | None:
        """Record a conversational fact about a person (the note_person tool).  Trusted
        (user-stated), surface layer, never canon.  Returns a spoken ack."""
        n = (note or "").strip()
        if not n:
            return None
        pid = self.resolve(who)
        p = self.get_person(pid)
        # A stable-ish key from the first words, so a repeat of the same fact supersedes
        # rather than piling up.
        slug = "_".join(re.findall(r"[a-z0-9]+", n.lower())[:4]) or "note"
        self.set_attribute(pid, facet or "social", slug, n, layer="surface",
                           provenance="user_stated", locked=False)
        name = (p.get("name") if p else None) or \
               ("you" if (p and p["kind"] == "user") else "them")
        return f"Got it — I'll remember that about {name}."
