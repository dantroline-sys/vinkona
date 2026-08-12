"""
Vinkona's durable knowledge graph — the long-term, self-grown counterpart to the volatile
working_graph.  Where working_graph.py is her *attention* over the last few turns (deterministic,
throw-away), this is her *memory of the user's world*: entities (people / places / things / orgs /
events) and the typed relations between them, distilled by the big LM from the USER's own chat
turns during the dreaming / idle phase, then persisted and consulted at recall time.

Design (see [[personal-graph]], the working-memory spec, and the 2026-08-09 chat-only pivot):
  * SOURCE = the conversation, nothing else.  Facts are grounded in what the USER said — their
    turns are the curated ground truth; Vinkona's own turns are context, never a fact source, so
    her own hallucinations can't launder themselves into durable "facts".  No email / file /
    research content ever reaches this graph, so there is NO untrusted-source poisoning vector.
  * GROUNDED: every edge cites the chat_logs turn and a verbatim quote that supports it.  The
    quote must actually occur in a user turn — an edge whose quote can't be found is REFUSED.  That
    single check also catches the LM inventing a relation that was never said.
  * SUPERSEDING: cardinality-one relations (one current home, one current employer) are time-
    versioned — a new value closes the old (sets valid_to); it does not silently overwrite, and the
    history stays for audit.
  * LOCKED ANCHOR: the user's own identity node is locked.  Idle extraction can add relations
    AROUND the user, but can never rename the user or assert who the user IS (the "you are Jane
    Smith" class of edge is refused outright).
  * REVERSIBLE: retraction is a soft status flip, never a delete — always auditable, always
    roll-back-able.
  * ADDITIVE: its own kg_* tables in memory.db; it never touches the memories store, which keeps
    doing its own, different job (world knowledge, research, reflections).

The LM call is INJECTED (`distill` takes an `extract_fn`), so the fold is deterministic and this
whole module builds + tests on a bare interpreter with no model and no network.
"""

import json
import re
import time
import typing as tp

# The user's own node — a fixed anchor.  Relations hang off it; it is never renamed or
# re-identified by extraction.
USER_ID = "user:self"

NODE_TYPES = ("person", "place", "org", "thing", "event", "user")

# Relations that hold at most ONE current value per subject: a new one supersedes (closes) the old.
# Everything else accumulates (you can know many people, own many things, work on many projects).
_CARD_ONE = frozenset({
    "lives_in", "resides_in", "based_in", "located_in", "home_is",
    "works_at", "employed_by", "current_employer", "current_role", "title_is",
    "born_in", "birthday_is", "age_is",
})

# Edges that would (re)assert the USER's own identity — refused when the subject is the anchor,
# so a misread turn can never make her think the user is someone else.
_IDENTITY_RELS = frozenset({
    "is_named", "named", "name_is", "aka", "identity_is", "real_name_is",
    "is_a_person_named", "goes_by", "called",
})

# Ways the user refers to themselves in a turn → the anchor.
_SELF_WORDS = frozenset({"user", "me", "i", "myself", "my", "we", "us"})

DEFAULTS: dict = {
    "enabled": False,             # opt-in, like working_graph — built only when turned on
    "distill_batch_turns": 40,    # user turns folded per LM call (one extraction batch)
    "distill_max_batches": 6,     # batches drained per dreaming pass — lets a backlog catch up
                                  #   (6*40 = 240 user turns/pass) without an unbounded cold run
    "max_context_nodes": 6,       # entities surfaced into a recall context block
    "max_context_edges": 12,      # …and relations among them
    "min_quote_len": 6,           # a grounding quote must be at least this long to count
    "context_max_chars": 700,     # hard cap on a rendered recall block
}

_WS = re.compile(r"\s+")
_NORM_STRIP = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the matching key for a label."""
    return _WS.sub(" ", _NORM_STRIP.sub(" ", (s or "").lower())).strip()


def _rel_norm(s: str) -> str:
    """A relation slug: lowercased, non-alnum → underscore, collapsed."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (s or "").lower())).strip("_")


def _edge_key(src: str, rel: str, dst: str) -> str:
    return f"{src}{rel}{dst}"


class MindGraph:
    """A durable entity/relation graph over the conversation, owned by one memory.db.

    Construct with an open sqlite3 connection (memory's) — it creates its own kg_* tables and
    never reads or writes the memories tables.  `distill(extract_fn)` folds new user turns in;
    `context_for(text)` renders a grounded block for recall."""

    def __init__(self, db, cfg: tp.Optional[dict] = None, *, user_label: str = "you"):
        self.db = db
        self.c = {**DEFAULTS, **(cfg or {})}
        self._ensure_schema()
        self._ensure_anchor(user_label)

    # -- schema ------------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS kg_nodes (
            id TEXT PRIMARY KEY, type TEXT, label TEXT, norm TEXT,
            aliases TEXT, mentions INTEGER DEFAULT 1,
            first_ts REAL, last_ts REAL, status TEXT DEFAULT 'active', locked INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_kg_nodes_norm ON kg_nodes(type, norm);
        CREATE TABLE IF NOT EXISTS kg_edges (
            id TEXT PRIMARY KEY, src TEXT, dst TEXT, rel TEXT,
            mentions INTEGER DEFAULT 1, first_ts REAL, last_ts REAL,
            valid_from REAL, valid_to REAL, source_turn INTEGER, quote TEXT,
            fact TEXT, status TEXT DEFAULT 'active'
        );
        CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src, rel);
        CREATE TABLE IF NOT EXISTS kg_state (k TEXT PRIMARY KEY, v TEXT);
        """)
        # Migration for graphs created before `fact` existed: the clean paraphrase the LM writes
        # per relation, which is what recall surfaces (the quote is only grounding evidence).
        try:
            self.db.execute("ALTER TABLE kg_edges ADD COLUMN fact TEXT")
        except Exception:
            pass                                            # column already present
        self.db.commit()

    def _ensure_anchor(self, user_label: str) -> None:
        row = self.db.execute("SELECT id FROM kg_nodes WHERE id=?", (USER_ID,)).fetchone()
        if not row:
            now = time.time()
            self.db.execute(
                "INSERT INTO kg_nodes(id,type,label,norm,aliases,mentions,first_ts,last_ts,status,locked)"
                " VALUES (?,?,?,?,?,?,?,?,?,1)",
                (USER_ID, "user", user_label, _norm(user_label), json.dumps([]), 1, now, now, "active"))
            self.db.commit()

    # -- checkpoint ---------------------------------------------------------------------
    def _last_id(self) -> int:
        r = self.db.execute("SELECT v FROM kg_state WHERE k='distill_last_id'").fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    def _set_last_id(self, v: int) -> None:
        self.db.execute("INSERT INTO kg_state(k,v) VALUES('distill_last_id',?) "
                        "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(int(v)),))

    def _new_user_turns(self, limit: int) -> list[dict]:
        """Unprocessed USER turns (their words are the fact source; assistant turns are context)."""
        rows = self.db.execute(
            "SELECT id, text FROM chat_logs WHERE role='user' AND id > ? ORDER BY id LIMIT ?",
            (self._last_id(), int(limit))).fetchall()
        return [{"id": int(r[0]), "text": r[1] or ""} for r in rows]

    def backlog(self) -> int:
        """USER turns not yet folded — how far behind the transcript the graph is.  Drops to 0
        once caught up; the number a dreaming pass drives down."""
        r = self.db.execute("SELECT COUNT(*) FROM chat_logs WHERE role='user' AND id > ?",
                            (self._last_id(),)).fetchone()
        return int(r[0]) if r else 0

    # -- distillation (the LM slow lane) ------------------------------------------------
    # A worked example, kept in the prompt so ANY instruction-following model (not just Qwen) sees
    # the exact output shape it must produce.  Weaker / differently-tuned models (Gemma, Llama,
    # Mistral) infer the schema poorly from a description but copy it reliably from an example —
    # this is what makes the extractor model-agnostic.
    _EXAMPLE_IN = ("[41] My sister Mara just moved to Bristol and started at the museum there. "
                   "Honestly traffic lights are awesome, and they")
    _EXAMPLE_OUT = (
        '{"nodes":[{"type":"person","label":"Mara","aliases":[]},'
        '{"type":"place","label":"Bristol","aliases":[]},'
        '{"type":"org","label":"the museum","aliases":[]},'
        '{"type":"thing","label":"traffic lights","aliases":[]}],'
        '"edges":[{"src":"user","dst":"Mara","rel":"sibling_of","quote":"My sister Mara",'
        '"fact":"Mara is the user\'s sister"},'
        '{"src":"Mara","dst":"Bristol","rel":"lives_in","quote":"moved to Bristol",'
        '"fact":"Mara lives in Bristol"},'
        '{"src":"Mara","dst":"the museum","rel":"works_at","quote":"started at the museum",'
        '"fact":"Mara works at the museum"},'
        '{"src":"user","dst":"traffic lights","rel":"likes","quote":"traffic lights are awesome",'
        '"fact":"The user likes traffic lights"}]}')

    def build_prompt(self, turns: list[dict]) -> str:
        """The extraction prompt.  Numbered so the model can't drift on which turn a fact came
        from; strict about modality so hypotheticals / questions never become asserted facts; and
        carries ONE worked example so the output shape is copied, not guessed — the difference
        between Qwen (infers it) and most other models (need to see it)."""
        body = "\n".join(f"[{t['id']}] {t['text']}" for t in turns)
        return (
            "You extract a small knowledge graph of the USER's world from their own words.\n"
            "Read the user's turns and extract ONLY facts the user actually ASSERTS about their "
            "world — people, places, organisations, things, events, and the relations between them.\n"
            "STRICT RULES:\n"
            "- Only asserted statements. Skip questions, hypotheticals ('if I…', 'imagine…'), "
            "jokes, and anything speculative or negated.\n"
            "- Refer to the user as \"user\". NEVER assert who the user is or the user's name — do "
            "not output an is_named/identity relation with src \"user\".\n"
            "- For every relation include a short verbatim \"quote\" copied EXACTLY from the turn "
            "that states it (this is how the fact is grounded — no quote, no edge).\n"
            "- For every relation ALSO write a \"fact\": ONE clean, self-contained, third-person "
            "sentence that states what was meant in plain, internally-consistent language — NOT the "
            "raw words. Resolve fragments and casual phrasing into a proper statement (e.g. the "
            "quote 'traffic lights are awesome, and they' becomes the fact 'The user likes traffic "
            "lights'). The fact is what gets remembered and read back later; the quote is only the "
            "evidence it rests on.\n"
            "- Use short snake_case relation names (lives_in, works_at, sibling_of, friend_of, "
            "owns, working_on, located_in, part_of, happened_on).\n"
            "- Labels in edges must match node labels (or \"user\").\n"
            "- Output ONLY the JSON object — no prose, no markdown fences, no explanation. Output "
            "exactly {} if nothing is asserted.\n"
            'Schema: {"nodes":[{"type":"person|place|org|thing|event","label":"..","aliases":[".."]}],'
            '"edges":[{"src":"..","dst":"..","rel":"..","quote":"..","fact":".."}]}\n\n'
            "EXAMPLE\n"
            f"User turns:\n{self._EXAMPLE_IN}\n"
            f"Output:\n{self._EXAMPLE_OUT}\n\n"
            "NOW YOUR TURN — same output shape, only for what is asserted below.\n"
            f"User turns:\n{body}\nOutput:")

    async def distill(self, extract_fn: tp.Callable, *, now: tp.Optional[float] = None) -> dict:
        """Fold new user turns into the graph.  `extract_fn(prompt)` performs the LM call and
        returns the parsed JSON dict (or None on a FAILED call) — injected so this is deterministic
        + testable.  Returns a stats dict; a no-op (advances nothing) when there is nothing new.

        CRITICAL: the checkpoint is advanced ONLY when the extraction call SUCCEEDS (returns a
        dict — even an empty one, meaning the LM genuinely found nothing to assert).  If the call
        FAILS (None: big LM down / non-JSON / timeout) the checkpoint is LEFT WHERE IT IS, so a
        spell of big-LM trouble can never silently 'process' turns to nothing and strand them
        beyond reach — the backlog stays and the next pass retries them."""
        now = time.time() if now is None else now
        turns = self._new_user_turns(int(self.c["distill_batch_turns"]))
        if not turns:
            return {"turns": 0, "nodes": 0, "edges": 0, "refused": 0}
        data = extract_fn(self.build_prompt(turns))
        if hasattr(data, "__await__"):
            data = await data
        if data is None:                                  # the LM call FAILED — do not advance
            return {"turns": len(turns), "nodes": 0, "edges": 0, "refused": 0, "failed": True}
        stats = self._fold(data, turns, now)
        self._set_last_id(max(t["id"] for t in turns))    # only on a successful extraction
        self.db.commit()
        stats["turns"] = len(turns)
        return stats

    async def catch_up(self, extract_fn: tp.Callable, *, max_batches: tp.Optional[int] = None,
                       now: tp.Optional[float] = None) -> dict:
        """One dreaming pass: drain the backlog of undistilled USER turns, oldest-first, over up
        to `max_batches` successive batches (default from config `distill_max_batches`), stopping
        early the moment we're caught up.

        Oldest-first is deliberate, not incidental: cardinality-one relations supersede in
        PROCESSING order, so replaying the user's turns in the order they were said means the
        newest value (their current home / employer) correctly wins.  Each batch checkpoints as it
        lands (`distill` commits + advances `distill_last_id`), so a crash mid-drain re-folds
        nothing and loses nothing — the next pass just resumes from the checkpoint.  The cap keeps
        a cold backlog from turning one dreaming pass into an unbounded LM run; the leftover simply
        drains over the following passes, which `backlog()` makes visible."""
        cap = int(self.c["distill_max_batches"] if max_batches is None else max_batches)
        total = {"turns": 0, "nodes": 0, "edges": 0, "refused": 0, "batches": 0}
        for _ in range(max(1, cap)):
            st = await self.distill(extract_fn, now=now)
            if st.get("failed"):                       # LM call failed — stop; backlog is preserved
                total["failed"] = True
                break
            if not st.get("turns"):
                break                                  # caught up — nothing new to fold
            for k in ("turns", "nodes", "edges", "refused"):
                total[k] += int(st.get(k, 0))
            total["batches"] += 1
        total["backlog"] = self.backlog()
        return total

    async def catch_up_all(self, extract_fn: tp.Callable, *,
                           should_yield: tp.Optional[tp.Callable] = None,
                           now: tp.Optional[float] = None) -> dict:
        """Drain the ENTIRE backlog — the forced 'Redistil everything' lane.  Successive
        catch_up passes until nothing is left, the LM fails, or `should_yield()` asks us
        to stand down for the user.  The per-pass caps still bound each dreaming pass (and
        each LM call); this only removes the ONE-PASS CEILING that made a full rebuild stop
        after batch_turns×max_batches turns and leave the rest to trickle at dream cadence.

        Returns catch_up's totals plus ``done``: True only when the backlog reached zero.
        A caller that sees done=False (a yield or a failure) should leave its request
        PENDING — the checkpoint means the next attempt resumes exactly where this one
        stopped, re-folding nothing."""
        total = {"turns": 0, "nodes": 0, "edges": 0, "refused": 0, "batches": 0,
                 "done": False}
        while True:
            st = await self.catch_up(extract_fn, now=now)
            for k in ("turns", "nodes", "edges", "refused", "batches"):
                total[k] += int(st.get(k, 0))
            total["backlog"] = int(st.get("backlog") or 0)
            if st.get("failed"):
                total["failed"] = True
                return total
            if not total["backlog"]:
                total["done"] = True
                return total
            if not st.get("turns"):                    # backlog but no progress — never spin
                total["failed"] = True
                return total
            if should_yield and should_yield():        # the user needs the box — stand down
                return total

    def _fold(self, data: dict, turns: list[dict], now: float) -> dict:
        """Deterministic fold of one extraction into the store.  The LM output is untrusted: every
        edge must ground to a real quote in a real user turn, or it is refused."""
        low = {t["id"]: (t["text"] or "").lower() for t in turns}
        # label -> node id, seeded with the ways the user refers to themselves
        label2id = {w: USER_ID for w in _SELF_WORDS}
        n_nodes = n_edges = refused = 0
        for nd in (data.get("nodes") or []):
            typ = (nd.get("type") or "thing").strip().lower()
            if typ == "user" or typ not in NODE_TYPES:
                typ = "thing"                          # only the locked anchor is a 'user' node
            label = (nd.get("label") or "").strip()
            if not label or _norm(label) in _SELF_WORDS:
                continue
            nid = self._resolve_node(typ, label, nd.get("aliases") or [], now)
            label2id[_norm(label)] = nid
            for a in (nd.get("aliases") or []):
                label2id.setdefault(_norm(a), nid)
            n_nodes += 1
        for ed in (data.get("edges") or []):
            rel = _rel_norm(ed.get("rel") or "")
            quote = (ed.get("quote") or "").strip()
            fact = _WS.sub(" ", (ed.get("fact") or "").strip())    # the clean paraphrase (surfaced)
            src = self._ref_to_id(ed.get("src"), label2id, now)
            dst = self._ref_to_id(ed.get("dst"), label2id, now)
            if not (rel and src and dst) or src == dst:
                refused += 1
                continue
            # Firewall: never let extraction (re)assert who the USER is.
            if src == USER_ID and rel in _IDENTITY_RELS:
                refused += 1
                continue
            turn_id = self._ground(quote, low)          # quote must occur in a real user turn
            if turn_id is None:
                refused += 1
                continue
            if self._add_edge(src, dst, rel, quote, fact, turn_id, now):
                n_edges += 1
        return {"nodes": n_nodes, "edges": n_edges, "refused": refused}

    def _ref_to_id(self, ref: tp.Optional[str], label2id: dict, now: float) -> tp.Optional[str]:
        """Resolve an edge endpoint (a label the LM emitted) to a node id.  The user's self-words
        map to the anchor; a known label maps to its node; an unknown label becomes a bare thing
        node (so a relation to something not listed as a node still lands, grounded)."""
        if ref is None:
            return None
        key = _norm(ref)
        if not key:
            return None
        if key in label2id:
            return label2id[key]
        if key in _SELF_WORDS:
            return USER_ID
        # An entity named in an edge but not re-listed as a node this batch is very often one we
        # already know — reuse the existing node (a real type beats a bare 'thing') so relations
        # accrete on ONE node across sessions instead of splintering.
        row = self.db.execute(
            "SELECT id FROM kg_nodes WHERE norm=? AND status='active' "
            "ORDER BY (type='thing'), mentions DESC, id LIMIT 1", (key,)).fetchone()
        if row:
            label2id[key] = row[0]
            return row[0]
        nid = self._resolve_node("thing", ref.strip(), [], now)
        label2id[key] = nid
        return nid

    def _ground(self, quote: str, low: dict) -> tp.Optional[int]:
        """The turn id whose text contains this quote (case-insensitive).  None ⇒ ungrounded
        (the quote was never said) ⇒ the edge is refused.  Also rejects trivially short quotes."""
        q = _WS.sub(" ", (quote or "").strip().lower())
        if len(q) < int(self.c["min_quote_len"]):
            return None
        for tid, text in low.items():
            if q in _WS.sub(" ", text):
                return tid
        return None

    def _resolve_node(self, typ: str, label: str, aliases: list, now: float) -> str:
        """Find-or-create a node by (type, normalised label); merge aliases and bump mentions.
        A locked node is never relabelled."""
        norm = _norm(label)
        nid = f"{typ}:{norm}"
        row = self.db.execute("SELECT aliases, mentions, locked FROM kg_nodes WHERE id=?", (nid,)).fetchone()
        if row:
            existing = set(json.loads(row[0] or "[]"))
            existing.update(a for a in aliases if a)
            self.db.execute("UPDATE kg_nodes SET aliases=?, mentions=mentions+1, last_ts=?, "
                            "status='active' WHERE id=?",
                            (json.dumps(sorted(existing)), now, nid))
            return nid
        self.db.execute(
            "INSERT INTO kg_nodes(id,type,label,norm,aliases,mentions,first_ts,last_ts,status,locked)"
            " VALUES (?,?,?,?,?,1,?,?,'active',0)",
            (nid, typ, label.strip(), norm, json.dumps(sorted({a for a in aliases if a})), now, now))
        return nid

    def _add_edge(self, src: str, dst: str, rel: str, quote: str, fact: str,
                  turn_id: int, now: float) -> bool:
        """Add or reinforce an edge.  Cardinality-one relations from the same subject supersede an
        existing (different) value: the old edge is closed (valid_to set), not deleted.  `fact` is
        the clean paraphrase surfaced at recall; on corroboration the latest non-empty one wins."""
        eid = _edge_key(src, rel, dst)
        row = self.db.execute("SELECT mentions FROM kg_edges WHERE id=? AND status='active'", (eid,)).fetchone()
        if row:                                        # same fact again → corroborate
            self.db.execute("UPDATE kg_edges SET mentions=mentions+1, last_ts=?, valid_to=NULL, "
                            "source_turn=?, quote=?, fact=COALESCE(NULLIF(?,''), fact) WHERE id=?",
                            (now, turn_id, quote, fact, eid))
            return True
        if rel in _CARD_ONE:                           # one current value → close the others
            self.db.execute("UPDATE kg_edges SET valid_to=?, status='superseded' "
                            "WHERE src=? AND rel=? AND status='active' AND valid_to IS NULL",
                            (now, src, rel))
        self.db.execute(
            "INSERT INTO kg_edges(id,src,dst,rel,mentions,first_ts,last_ts,valid_from,valid_to,"
            "source_turn,quote,fact,status) VALUES (?,?,?,?,1,?,?,?,NULL,?,?,?,'active')",
            (eid, src, dst, rel, now, now, now, turn_id, quote, fact))
        return True

    # -- audit / reversibility ----------------------------------------------------------
    def retract_edge(self, eid: str) -> bool:
        cur = self.db.execute("UPDATE kg_edges SET status='retracted' WHERE id=? AND status='active'", (eid,))
        self.db.commit()
        return cur.rowcount > 0

    def retract_node(self, nid: str) -> bool:
        if nid == USER_ID:
            return False                               # the anchor can't be retracted
        self.db.execute("UPDATE kg_nodes SET status='retracted' WHERE id=?", (nid,))
        self.db.execute("UPDATE kg_edges SET status='retracted' WHERE src=? OR dst=?", (nid, nid))
        self.db.commit()
        return True

    # -- recall consumption -------------------------------------------------------------
    def context_for(self, text: str, *, now: tp.Optional[float] = None) -> str:
        """A compact, grounded block for the reply/recall path: the entities MENTIONED in `text`
        that the graph knows, plus their current relations.  Empty when nothing matches (so an
        empty graph leaves the prompt unchanged).  Deterministic, no LM."""
        low = " " + _norm(text) + " "
        hits = []
        for nid, typ, label, norm in self.db.execute(
                "SELECT id, type, label, norm FROM kg_nodes WHERE status='active' AND id!=? "
                "ORDER BY mentions DESC, id", (USER_ID,)).fetchall():
            if norm and (" " + norm + " ") in low:
                hits.append((nid, label))
            if len(hits) >= int(self.c["max_context_nodes"]):
                break
        if not hits:
            return ""
        lines = ["What I already know about this (from earlier conversations):"]
        seen = 0
        seen_facts: set = set()
        cap = int(self.c["max_context_edges"])
        for nid, label in hits:
            rels = self.db.execute(
                "SELECT src, dst, rel, fact FROM kg_edges WHERE status='active' AND valid_to IS NULL "
                "AND (src=? OR dst=?) ORDER BY mentions DESC, id", (nid, nid)).fetchall()
            for src, dst, rel, fact in rels:
                # Surface the LM's clean paraphrase; fall back to a plain triple for pre-`fact`
                # edges (so old graphs still read, just less naturally).
                if fact and fact.strip():
                    line = fact.strip()
                else:
                    other = dst if src == nid else src
                    subj = label if src == nid else self._label(other)
                    obj = self._label(dst) if src == nid else label
                    line = f"{subj} {rel.replace('_', ' ')} {obj}"
                key = line.lower()
                if key in seen_facts:                  # a shared edge hit from both endpoints
                    continue
                seen_facts.add(key)
                lines.append(f"- {line}")
                seen += 1
                if seen >= cap:
                    break
            if seen >= cap:
                break
        return "\n".join(lines)[: int(self.c["context_max_chars"])]

    def _label(self, nid: str) -> str:
        if nid == USER_ID:
            return "you"
        r = self.db.execute("SELECT label FROM kg_nodes WHERE id=?", (nid,)).fetchone()
        return r[0] if r else nid

    # -- observability ------------------------------------------------------------------
    def snapshot(self, max_nodes: int = 60, max_edges: int = 90) -> dict:
        nodes = [{"id": r[0], "type": r[1], "label": r[2], "mentions": r[3],
                  "locked": bool(r[4])}
                 for r in self.db.execute(
                     "SELECT id,type,label,mentions,locked FROM kg_nodes WHERE status='active' "
                     "ORDER BY mentions DESC, id LIMIT ?", (max_nodes,)).fetchall()]
        ids = {n["id"] for n in nodes}
        edges = [{"src": r[0], "dst": r[1], "rel": r[2], "mentions": r[3], "quote": r[4], "fact": r[5]}
                 for r in self.db.execute(
                     "SELECT src,dst,rel,mentions,quote,fact FROM kg_edges WHERE status='active' "
                     "AND valid_to IS NULL ORDER BY mentions DESC, id LIMIT ?", (max_edges,)).fetchall()
                 if r[0] in ids and r[1] in ids]
        return {"nodes": nodes, "edges": edges, "counts": self.stats()}

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM kg_nodes WHERE status='active'").fetchone()[0]
        e = self.db.execute("SELECT COUNT(*) FROM kg_edges WHERE status='active' AND valid_to IS NULL").fetchone()[0]
        return {"nodes": int(n), "edges": int(e), "last_id": self._last_id(),
                "backlog": self.backlog()}
