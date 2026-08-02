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
        self._link(mentioned[: self.c["link_top"]], turn)
        self._evict()
        self._reframe()
        return turn

    def _decay(self, now: float) -> None:
        tau = float(self.c["tau_s"])
        for nid in sorted(self.nodes):                    # fixed order (determinism)
            nd = self.nodes[nid]
            dt = now - nd["last_decay_ts"]
            if dt > 0:
                nd["activation"] *= math.exp(-dt / tau)
                nd["last_decay_ts"] = now

    def _boost(self, nid: str, label: str, turn: int, now: float) -> None:
        nd = self.nodes.get(nid)
        if nd is None:
            nd = self.nodes[nid] = {"label": label, "activation": 0.0,
                                    "first_seen_turn": turn, "last_boost_turn": turn,
                                    "last_decay_ts": now, "turns": []}
        nd["activation"] = min(1.0, nd["activation"] + float(self.c["b_direct"]))
        nd["last_boost_turn"] = turn
        if not nd["turns"] or nd["turns"][-1] != turn:
            nd["turns"].append(turn)
            nd["turns"] = nd["turns"][-int(self.c["keep_turns"]):]

    def _link(self, ids: list[str], turn: int) -> None:
        uniq = sorted(set(ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                key = (uniq[i], uniq[j])
                ed = self.edges.get(key)
                if ed is None:
                    ed = self.edges[key] = {"weight": 0.0, "turns": [], "last_seen_turn": turn}
                ed["weight"] += 1.0
                ed["last_seen_turn"] = turn
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
        self.frame = [nid for nid, _ in sorted(
            self.nodes.items(), key=lambda kv: (-kv[1]["activation"], kv[0]))][: int(self.c["k_frame"])]

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
        """A few cheap counters (the full instrument is phase 1c)."""
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "frame": list(self.frame), "turns": self._turn}


def _neg_id(s: str) -> tuple:
    """Sort key that orders ids descending, for deterministic tie-breaking on eviction."""
    return tuple(-ord(c) for c in s)


def _neg_key(k: tuple) -> tuple:
    return _neg_id(k[0]) + _neg_id(k[1])
