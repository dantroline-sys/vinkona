"""
VIN-WM-02 phase 1a — the deterministic (no-LM) conversational working-memory graph.

A throw-away, session-scoped graph that keeps the *broad thread* of a long
conversation alive without a long context window.  Each turn is folded in
deterministically:

  extract candidate key phrases  →  boost their nodes  →  link phrases mentioned
  together  →  decay everything on a wall-clock constant  →  evict the faded  →
  recompute the frame.

Extraction is a RAKE-style method (Rapid Automatic Keyword Extraction): split on
stopwords + punctuation into candidate phrases, score each word by degree/frequency,
sum per phrase.  Pure Python — no model, no corpus, no network — so it runs in a
few microseconds on a couple of sentences and works with the box's VRAM vacated
(minimal mode).  A phrase mentioned once and dropped fades out on its own; a thread
that keeps coming up stays hot.  `briefing()` renders the hot part to a compact,
fenced "working notes" block for the next turn's prompt.

Scope (see working_memory_graph_spec.md):
  * This is phase 1a — the cheap, deterministic half.  Nodes + *untyped* co-occurrence
    edges only.
  * Typed relations (causes/blocks/…) and synonym merging are the LM slow lane, phase
    1b — NOT here.  The instrument + drift guard is phase 1c — NOT here.

Invariants honoured here:
  * Deterministic (G-8): fixed iteration order (ascending id), math.exp, no randomness.
    The same turns replay to a bit-identical graph.
  * Volatile (G-2): held in memory, discarded with the conversation.  No persistence.
  * Bounded (G-7): node and edge counts are hard-capped.
  * Grounded (G-6): every node and edge records the turn indices that support it.
  * Fenced (G-5): the briefing is "working notes, may be wrong", never spoken, never canon.
"""

import math
import re
import typing as tp
from collections import Counter

# ── Stopwords: function words, pronouns, auxiliaries, fillers, interrogatives.  A phrase
# boundary falls on any of these (RAKE's delimiter set), so what survives is content. ──
_STOP: frozenset = frozenset("""
a an the this that these those such some any each every all both few more most other another
i me my we us our you your he him his she her it its they them their who whom whose
is am are was were be been being do does did doing have has had having will would shall should
can could may might must ought need dare used
and or but nor so yet for as if then than because while although though unless until whether
of in on at by to from up down over under out off into onto upon with within without about
above below between among through during before after against around
not no nor only just also too very much many lot lots really quite rather somewhat
here there where when why how what which
i'm you're we're they're it's i've you've we've i'd i'll we'll don't doesn't didn't can't won't
he's she's that's there's what's let's
please thanks thank ok okay yeah yes no maybe well um uh like get got getting go going gone
one two three thing things stuff way ways kind sort lot bit
""".split())

_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\sA-Za-z0-9]")   # a word OR a single punctuation mark

DEFAULTS: dict = {
    "enabled": False,           # phase 1a is opt-in until the held-out eval says it helps
    "tau_s": 900.0,             # activation decay constant (s) — ~15 min to 1/e
    "b_direct": 0.4,            # boost for a phrase mentioned this turn
    "floor": 0.02,              # evict a node below this activation
    "cap": 256,                 # hard node cap (bounded working set)
    "edge_cap": 512,            # hard edge cap
    "k_frame": 12,              # frame size (top nodes shown)
    "min_word_len": 3,          # ignore words shorter than this
    "max_phrase_words": 4,      # cap candidate-phrase length
    "top_k": 12,                # keep at most this many phrases per turn
    "link_top": 6,              # …and lay co-occurrence edges among the best this many
    "brief_max_chars": 700,     # hard cap on the rendered briefing
    "brief_threads": 5,         # hottest edges shown as "threads"
    "keep_turns": 8,            # bound the per-node / per-edge supporting-turn lists
    # Instrument / drift guard (VIN-WM-02 1c).  Deterministic metrics per event + a
    # stall detector: if the frame stops moving or activation entropy collapses, the LM
    # graph has ossified (the "memory trap" this is meant to prevent).  It only FLAGS —
    # never mutates activation — and only once the frame is full (≥ k_frame nodes).
    "e_lock": 1.0,              # activation-entropy floor (nats)
    "k_lock": 4,                # consecutive low-signal events before flagging a stall
    # Cross-session persistence (VIN-WM-02 open-Q1, opted in via memory.working_graph.persist).
    # Carried-over nodes are DORMANT (primed): kept + decaying + shown, but held out of the
    # frame until re-mentioned, so an old conversation can't bleed into a new one.  They decay
    # on the SLOW clock (persist_tau_s — days) instead of the fast attention clock, so unused
    # associations wane gradually; a re-mention lifts dormancy and they rejoin attention.
    "persist_tau_s": 604800.0,  # ~7 days: association half-life for dormant carried nodes
    "carry_factor": 0.6,        # scale carried activation on load (starts primed-low, never hot)
}

_HEADER = ("Working notes — my running read of this conversation so far; may be wrong, "
           "for continuity only (do not quote it):")


# ── RAKE-style extraction ──────────────────────────────────────────────────────────────
def _candidates(text: str, min_word_len: int, max_phrase_words: int) -> list[list[str]]:
    """Split text into candidate phrases: maximal runs of content words, broken by a
    stopword, a short word, or punctuation.  Whitespace joins words within a phrase, so
    'computer science' stays one candidate; 'science, art' splits on the comma."""
    phrases: list[list[str]] = []
    cur: list[str] = []
    for m in _TOK.finditer(text or ""):
        t = m.group(0)
        if not t[0].isalnum():                 # punctuation → phrase boundary
            if cur:
                phrases.append(cur); cur = []
            continue
        w = t.lower().strip("'-")
        if len(w) < min_word_len or w in _STOP or not any(c.isalpha() for c in w):
            if cur:
                phrases.append(cur); cur = []
            continue
        cur.append(w)
        if len(cur) >= max_phrase_words:
            phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    return phrases


def keyphrases(text: str, *, min_word_len: int = 3, max_phrase_words: int = 4,
               top_k: int = 12) -> list[tuple[str, float]]:
    """Deterministic RAKE key phrases: candidate phrases scored by summed word
    degree/frequency.  Returns [(phrase, score)] deduped, highest score first, ties broken
    by phrase text so the order is stable."""
    phrases = _candidates(text, min_word_len, max_phrase_words)
    if not phrases:
        return []
    freq: Counter = Counter()
    degree: Counter = Counter()
    for ph in phrases:
        n = len(ph)
        for w in ph:
            freq[w] += 1
            degree[w] += n                     # classic RAKE: degree includes the word itself
    word_score = {w: degree[w] / freq[w] for w in freq}
    best: dict[str, float] = {}
    for ph in phrases:
        norm = " ".join(ph)
        s = sum(word_score[w] for w in ph)
        if s > best.get(norm, 0.0):
            best[norm] = s
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


# ── The graph ────────────────────────────────────────────────────────────────────────
class WorkingGraph:
    """One conversation's volatile phrase graph.  Owned by a session, discarded with it."""

    def __init__(self, cfg: tp.Optional[dict] = None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.nodes: dict[str, dict] = {}       # id -> {label, activation, first_seen_turn,
                                               #        last_boost_turn, last_decay_ts, turns:[..]}
        self.edges: dict[tuple, dict] = {}     # (a,b) sorted -> {weight, turns:[..], last_seen_turn}
        self._turn = 0
        self.frame: list[str] = []
        # instrument state (VIN-WM-02 1c)
        self._prev_frame: set = set()          # frame before the current event (for churn)
        self._seed_frame: tp.Optional[set] = None   # first full frame (for drift)
        self._stall_run = 0                    # consecutive low-signal events
        self.stalled: tp.Optional[str] = None  # non-None when the graph looks ossified
        self.last_metrics: dict = {}           # metrics from the most recent ingest

    # -- lifecycle ----------------------------------------------------------------------
    def ingest(self, text: str, *, now: float) -> int:
        """Fold one turn (either speaker) into the graph.  Deterministic; `now` is an epoch
        float supplied by the caller so replay is exact.  Returns the turn index."""
        self._turn += 1
        turn = self._turn
        self._decay(now)
        phrases = keyphrases(text, min_word_len=self.c["min_word_len"],
                             max_phrase_words=self.c["max_phrase_words"], top_k=self.c["top_k"])
        mentioned: list[str] = []
        for ph, _score in phrases:
            nid = "p:" + ph
            self._boost(nid, ph, turn, now)
            mentioned.append(nid)
        self._link(mentioned[: self.c["link_top"]], turn, now)
        self._prev_frame = set(self.frame)                # frame before this event's reframe
        self._evict()
        self._reframe()
        if self._seed_frame is None and self.frame:
            self._seed_frame = set(self.frame)            # first full frame = drift reference
        self.last_metrics = self.metrics()
        self._update_stall(self.last_metrics)
        return turn

    def _decay(self, now: float) -> None:
        # Dual clock: active nodes decay on the fast attention tau; DORMANT (primed) carried
        # nodes decay on the slow association tau, so they wane over days, not minutes.
        short = float(self.c["tau_s"])
        slow = float(self.c.get("persist_tau_s", short))
        for nid in sorted(self.nodes):                    # fixed order (determinism)
            nd = self.nodes[nid]
            dt = now - nd["last_decay_ts"]
            if dt > 0:
                nd["activation"] *= math.exp(-dt / (slow if nd.get("primed") else short))
                nd["last_decay_ts"] = now

    def _boost(self, nid: str, label: str, turn: int, now: float) -> None:
        nd = self.nodes.get(nid)
        if nd is None:
            nd = self.nodes[nid] = {"label": label, "activation": 0.0,
                                    "first_seen_turn": turn, "last_boost_turn": turn,
                                    "last_decay_ts": now, "turns": [], "primed": False}
        elif nd.get("primed"):
            nd["primed"] = False                          # re-mention lifts a carried node out of dormancy
        nd["activation"] = min(1.0, nd["activation"] + float(self.c["b_direct"]))
        nd["last_boost_turn"] = turn
        if not nd["turns"] or nd["turns"][-1] != turn:
            nd["turns"].append(turn)
            nd["turns"] = nd["turns"][-int(self.c["keep_turns"]):]

    def _link(self, ids: list[str], turn: int, now: float) -> None:
        uniq = sorted(set(ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                key = (uniq[i], uniq[j])
                ed = self.edges.get(key)
                if ed is None:
                    ed = self.edges[key] = {"weight": 0.0, "turns": [], "last_seen_turn": turn,
                                            "last_ts": now}
                ed["weight"] += 1.0
                ed["last_seen_turn"] = turn
                ed["last_ts"] = now                        # wall-clock, for cross-session edge decay
                if not ed["turns"] or ed["turns"][-1] != turn:
                    ed["turns"].append(turn)
                    ed["turns"] = ed["turns"][-int(self.c["keep_turns"]):]

    def _evict(self) -> None:
        floor = float(self.c["floor"])
        for nid in [n for n, d in self.nodes.items() if d["activation"] < floor]:
            del self.nodes[nid]
        cap = int(self.c["cap"])
        if len(self.nodes) > cap:
            # keep the hottest; drop lowest activation, tie-break by descending id (deterministic)
            ordered = sorted(self.nodes.items(), key=lambda kv: (kv[1]["activation"], _neg_id(kv[0])))
            for nid, _ in ordered[: len(self.nodes) - cap]:
                del self.nodes[nid]
        # edges only exist while both endpoints do (bounded + grounded)
        for key in [k for k in self.edges if k[0] not in self.nodes or k[1] not in self.nodes]:
            del self.edges[key]
        ecap = int(self.c["edge_cap"])
        if len(self.edges) > ecap:
            ordered = sorted(self.edges.items(), key=lambda kv: (kv[1]["weight"], _neg_key(kv[0])))
            for key, _ in ordered[: len(self.edges) - ecap]:
                del self.edges[key]

    def _reframe(self) -> None:
        # Dormant (primed) carried nodes are held OUT of the frame — an old conversation
        # can't surface in a new one until something re-mentions it (lifts dormancy).
        self.frame = [nid for nid, _ in sorted(
            ((n, d) for n, d in self.nodes.items() if not d.get("primed")),
            key=lambda kv: (-kv[1]["activation"], kv[0]))][: int(self.c["k_frame"])]

    # -- output -------------------------------------------------------------------------
    def briefing(self) -> str:
        """The compact, fenced 'working notes' block for the reply prompt.  Empty when the
        graph is cold (so with the graph off/empty the prompt is unchanged — G-1/G-ACCEL)."""
        if not self.frame:
            return ""
        lines = [_HEADER]
        for nid in self.frame:
            lines.append(f"- {self.nodes[nid]['label']}")
        threads = sorted(self.edges.items(),
                         key=lambda kv: (-kv[1]["weight"], _neg_key(kv[0])))[: int(self.c["brief_threads"])]
        thr = [f"{self.nodes[a]['label']} · {self.nodes[b]['label']}"
               for (a, b), _ in threads if a in self.nodes and b in self.nodes]
        if thr:
            lines.append("Threads: " + "; ".join(thr))
        return "\n".join(lines)[: int(self.c["brief_max_chars"])]

    def stats(self) -> dict:
        """A few cheap counters."""
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "frame": list(self.frame), "turns": self._turn}

    # -- instrument (VIN-WM-02 1c) ------------------------------------------------------
    def metrics(self) -> dict:
        """Deterministic per-event metrics for observability + drift detection.  Pure — a
        function of the current graph and the frame captured before this event's reframe.
        `frame` lists the phrase labels currently in front of her (the inspection window)."""
        # Entropy + the stall signal are over the ACTIVE working set (attention), not the
        # dormant carried nodes — otherwise a big dim LTM would always read as high-entropy.
        acts = [d["activation"] for d in self.nodes.values() if not d.get("primed")]
        primed = len(self.nodes) - len(acts)
        total = sum(acts)
        entropy = -sum((a / total) * math.log(a / total) for a in acts if a > 0) if total > 0 else 0.0
        fnow = set(self.frame)
        return {
            "turns": self._turn,
            "nodes": len(self.nodes),
            "active": len(acts),                                 # non-dormant (attention) nodes
            "primed": primed,                                    # dormant carried associations
            "edges": len(self.edges),
            "entropy": round(entropy, 4),                        # primary lock signal (nats)
            "frame_churn": round(1.0 - _jaccard(fnow, self._prev_frame), 4),
            "frame_drift": round(1.0 - _jaccard(fnow, self._seed_frame or set()), 4),
            "ungrounded_edges": sum(1 for e in self.edges.values() if not e["turns"]),  # must stay 0
            "frame": [self.nodes[n]["label"] for n in self.frame],
        }

    def _update_stall(self, m: dict) -> None:
        """Flag (never mutate) when the graph has ossified: entropy collapsed or the frame
        stopped moving for k_lock consecutive events.  Only judged once the frame is full,
        so a warming-up conversation with few nodes never trips it."""
        if m.get("active", m["nodes"]) < int(self.c["k_frame"]):
            self._stall_run = 0
            self.stalled = None
            return
        low = m["entropy"] < float(self.c["e_lock"]) or m["frame_churn"] == 0.0
        self._stall_run = self._stall_run + 1 if low else 0
        self.stalled = (f"working graph may be ossified — entropy {m['entropy']} nats, "
                        f"frame_churn {m['frame_churn']} for {self._stall_run} events"
                        if self._stall_run >= int(self.c["k_lock"]) else None)

    def snapshot(self, max_nodes: int = 40, max_edges: int = 60) -> dict:
        """A compact, JSON-able view for the config-screen inspector: the hottest nodes
        and the edges among them, plus the latest metrics.  Bounded so the payload and the
        drawing both stay legible.  Read-only observability — never fed back into the graph
        (WM-2 stays: the graph is not rebuilt from this)."""
        top = sorted(self.nodes.items(),
                     key=lambda kv: (-kv[1]["activation"], kv[0]))[:max_nodes]
        ids = {nid for nid, _ in top}
        fr = set(self.frame)
        nodes = [{"id": nid, "label": nd["label"], "activation": round(nd["activation"], 4),
                  "frame": nid in fr, "primed": bool(nd.get("primed"))} for nid, nd in top]
        edges = [{"a": a, "b": b, "weight": ed["weight"]}
                 for (a, b), ed in sorted(self.edges.items(),
                                          key=lambda kv: (-kv[1]["weight"], _neg_key(kv[0])))
                 if a in ids and b in ids][:max_edges]
        return {"turns": self._turn, "stalled": self.stalled,
                "metrics": self.last_metrics, "nodes": nodes, "edges": edges}

    # -- cross-session persistence (VIN-WM-02 open-Q1) ----------------------------------
    def to_dict(self) -> dict:
        """Full serialisable state for the persistent associative layer.  Each node/edge
        carries its wall-clock last-touch so a later load can decay it for the real elapsed
        time.  Bounded (the graph is already capped)."""
        return {
            "nodes": {nid: {"label": nd["label"], "activation": round(nd["activation"], 6),
                            "last_ts": nd["last_decay_ts"], "turns": list(nd["turns"])}
                      for nid, nd in self.nodes.items()},
            "edges": [{"a": a, "b": b, "weight": round(ed["weight"], 4),
                       "last_ts": ed.get("last_ts", 0.0),   # 0 ⇒ ancient ⇒ decays out on load
                       "turns": list(ed["turns"])} for (a, b), ed in self.edges.items()],
        }

    def load_persisted(self, data: dict, now: float) -> int:
        """Seed this (fresh) graph from a persisted snapshot as DORMANT associations: decay
        every node/edge for the real time elapsed since it was last touched (slow tau — days),
        scale by carry_factor so nothing starts front-of-mind, and mark carried nodes primed
        (held out of the frame until re-mentioned — the anti-bleed gate).  Drops anything that
        has waned below floor.  Returns the number of nodes carried."""
        slow = float(self.c.get("persist_tau_s", self.c["tau_s"]))
        carry = float(self.c.get("carry_factor", 0.6))
        floor = float(self.c["floor"])
        cap = int(self.c["cap"])
        keep = int(self.c["keep_turns"])
        for nid, nd in sorted((data.get("nodes") or {}).items()):    # fixed order (determinism)
            last = float(nd.get("last_ts", now))
            act = float(nd.get("activation", 0.0)) * math.exp(-max(0.0, now - last) / slow) * carry
            if act < floor:
                continue
            turns = [int(t) for t in (nd.get("turns") or [])][-keep:] or [0]   # keep grounding non-empty
            self.nodes[nid] = {"label": nd.get("label", nid), "activation": act,
                               "first_seen_turn": 0, "last_boost_turn": 0, "last_decay_ts": now,
                               "turns": turns, "primed": True}
        for e in (data.get("edges") or []):
            a, b = e.get("a"), e.get("b")
            if a not in self.nodes or b not in self.nodes:
                continue
            last = float(e.get("last_ts", now))
            w = float(e.get("weight", 0.0)) * math.exp(-max(0.0, now - last) / slow)
            if w < 0.1:
                continue
            key = (a, b) if a < b else (b, a)
            self.edges[key] = {"weight": w, "turns": [int(t) for t in (e.get("turns") or [0])][-keep:] or [0],
                               "last_seen_turn": 0, "last_ts": now}
        self._evict()                                    # honour the cap on load
        self._reframe()
        return len(self.nodes)


def _jaccard(a: set, b: set) -> float:
    """|a∩b| / |a∪b|; two empty sets count as identical (no change)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _neg_id(s: str) -> tuple:
    """Sort key that orders ids descending, for deterministic tie-breaking on eviction."""
    return tuple(-ord(c) for c in s)


def _neg_key(k: tuple) -> tuple:
    return _neg_id(k[0]) + _neg_id(k[1])
