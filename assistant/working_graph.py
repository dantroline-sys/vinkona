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
  * Grounded (G-6): every node and edge records the turn indices that support it, and every
    node keeps a few short verbatim snippets of where its phrase actually occurred — so a
    carried (dormant) node hands back tangible content on wake, not just a bare label.
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
perhaps still actually basically simply really pretty probably truly literally honestly obviously
certainly definitely essentially generally usually normally maybe sure okay gonna wanna gotta
nothing something anything everything someone anyone everyone somebody anybody everybody nobody none
""".split())

# ── Common content words: high-frequency, domain-neutral nouns/verbs/adjectives that pass the
# stopword filter but carry no memory value on their OWN (world/new/end/laugh).  A BARE such word
# is dropped as a node (see drop_common_singletons); it may still appear INSIDE a multi-word phrase
# ("new world", "mobile device"), where the phrase is the cue.  Tunable — extend for your domain. ──
_COMMON: frozenset = frozenset("""
world worlds people person man men woman women child children life lives home hour hours guy guys
day days week weeks month months year years time times moment moments name names number numbers
fact facts point points part parts place places side sides reason reasons idea ideas rest area
question questions story stories end ends line lines others everybody
make makes made making know knows knew known think thinks thought take takes took taken taking
see sees saw seen seeing come comes came coming want wants wanted look looks looked looking
use uses used using find finds found finding give gives gave given tell tells told call calls
called try tries tried ask asks asked need needs needed feel feels felt become becomes became
leave leaves left put puts mean means meant keep keeps kept begin begins seem seems seemed
help helps helped talk talks talked turn turns turned start starts started show shows showed
hear hears heard play plays played run runs move moves moved live lives lived believe believes
bring brings brought happen happens happened lose loses lost pay meet meets continue learn learns
lead leads follow follows stop stops speak speaks read reads spend grow open opens walk win wins
remember consider appear appears die dies died dying send sends stay stays fall falls reach
remain laugh laughs laughed survive survives survived cost costs chatter
good new first last long great little own other others old right big small large next early
young important public bad same best low late real full hard easy whole free strong true main
simple clear recent likely nice better actual usual
""".split())

_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\sA-Za-z0-9]")   # a word OR a single punctuation mark
_SENT = re.compile(r"[^.!?\n]+[.!?]*")                            # a rough clause/sentence run
_WS = re.compile(r"\s+")

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
    "drop_common_singletons": True,  # a bare high-frequency word (world/new/end) is filler, not a
                                #   cue — drop it as a node; it survives inside a multi-word phrase
    "link_top": 6,              # …and lay co-occurrence edges among the best this many
    "brief_max_chars": 1200,    # hard cap on the rendered briefing
    "brief_threads": 5,         # hottest edges shown as "threads"
    "ground_after_turns": 0,    # 0 = always inject a frame node's source clause (the memory);
                                #   >0 = only once its text has scrolled this many turns off-screen
    "brief_ground_max": 6,      # …at most this many grounded (clause-bearing) lines per briefing
    "brief_ground_chars": 120,  # …each clause truncated to this many chars in the briefing
    "keep_turns": 8,            # bound the per-node / per-edge supporting-turn lists
    # Grounding snippets: with each phrase we keep the clause it came from, so a node is a
    # recall *handle* (content behind it), not just a label.  Surfaced in the briefing ONLY
    # for a carried node that just woke from dormancy — where the LM has no rolling context
    # for it — so within a session the briefing stays lean (the content is still in context).
    "keep_grounds": 3,          # max verbatim snippets kept per node (bounded)
    "ground_max_chars": 160,    # cap each snippet
    "brief_recall": 3,          # max "Earlier (may be stale)" recall lines in a briefing
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


def _sentences(text: str) -> list[str]:
    """Split into rough clause/sentence runs (broken on . ! ? and newlines).  Deterministic,
    whitespace-normalised — used only to attach a phrase back to the clause it appeared in."""
    return [_WS.sub(" ", s).strip() for s in _SENT.findall(text or "") if s.strip()]


def _ground_for(phrase: str, sents_low: list[str], sents: list[str], max_chars: int) -> str:
    """The first clause containing this (normalised) phrase, verbatim and length-capped.
    Empty when no clause matches (e.g. odd whitespace) — grounding is best-effort, never fatal."""
    for i, sl in enumerate(sents_low):
        if phrase in sl:
            return sents[i][:max_chars]
    return ""


# ── The graph ────────────────────────────────────────────────────────────────────────
class WorkingGraph:
    """One conversation's volatile phrase graph.  Owned by a session, discarded with it."""

    def __init__(self, cfg: tp.Optional[dict] = None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.nodes: dict[str, dict] = {}       # id -> {label, activation, first_seen_turn,
                                               #        last_boost_turn, last_decay_ts, turns:[..],
                                               #        grounds:[verbatim clause,..], primed}
        self.edges: dict[tuple, dict] = {}     # (a,b) sorted -> {weight, turns:[..], last_seen_turn}
        self._turn = 0
        self.frame: list[str] = []
        # dormancy-wake bookkeeping (per event): which carried nodes woke this turn, and the
        # content they carried BEFORE this turn re-boosted them (that older snippet is what the
        # LM has no context for, so that is what the briefing recalls).
        self._woke: list[str] = []
        self._woke_grounds: dict[str, list] = {}
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
        self._woke = []
        self._woke_grounds = {}
        self._decay(now)
        phrases = keyphrases(text, min_word_len=self.c["min_word_len"],
                             max_phrase_words=self.c["max_phrase_words"], top_k=self.c["top_k"])
        sents = _sentences(text)
        sents_low = [s.lower() for s in sents]
        gmax = int(self.c["ground_max_chars"])
        drop_common = bool(self.c.get("drop_common_singletons", True))
        mentioned: list[str] = []
        for ph, _score in phrases:
            if drop_common and " " not in ph and ph in _COMMON:
                continue                              # a bare common word is filler, not a memory cue
            nid = "p:" + ph
            self._boost(nid, ph, turn, now, _ground_for(ph, sents_low, sents, gmax))
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

    def _boost(self, nid: str, label: str, turn: int, now: float, ground: str = "") -> None:
        nd = self.nodes.get(nid)
        if nd is None:
            nd = self.nodes[nid] = {"label": label, "activation": 0.0,
                                    "first_seen_turn": turn, "last_boost_turn": turn,
                                    "last_decay_ts": now, "turns": [], "grounds": [],
                                    "primed": False}
        elif nd.get("primed"):
            nd["primed"] = False                          # re-mention lifts a carried node out of dormancy
            self._woke.append(nid)
            self._woke_grounds[nid] = list(nd.get("grounds") or [])   # content carried in from before
        nd["activation"] = min(1.0, nd["activation"] + float(self.c["b_direct"]))
        nd["last_boost_turn"] = turn
        if not nd["turns"] or nd["turns"][-1] != turn:
            nd["turns"].append(turn)
            nd["turns"] = nd["turns"][-int(self.c["keep_turns"]):]
        if ground:                                        # keep the last N distinct clauses, bounded
            gs = nd.setdefault("grounds", [])
            if ground not in gs:
                gs.append(ground)
                del gs[: -int(self.c["keep_grounds"])]

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
        # Grounded frame lines: hand the fast LM the actual CLAUSE behind each held phrase, not
        # just the label — that verbatim content IS the memory (without it the block is a word
        # list).  ONE pass over every frame node, whether it's a live phrase or a carried one that
        # just woke from a past session (those get an "(earlier, may be stale)" flag — the LM has
        # no rolling context for them at all).  Deduped: a clause shared by several phrases from one
        # sentence is shown once (the hottest node owns it, the rest stay bare).  Bounded (count +
        # chars).  No turn-gate: a frozen config must never be able to silence the content.
        lines = [_HEADER]
        gmax = int(self.c.get("brief_ground_max", 6))
        gchars = int(self.c.get("brief_ground_chars", 120))
        seen_clauses: set = set()
        grounded = 0
        for nid in self.frame:
            nd = self.nodes[nid]
            carried = self._woke_grounds.get(nid)              # cross-session clauses (no context at all)
            source = carried if carried else (nd.get("grounds") or [])
            clause = None
            if grounded < gmax:
                for g in reversed(source):                     # most recent distinct clause
                    key = " ".join(g.lower().split())
                    if key and key not in seen_clauses:
                        clause = g[:gchars]; seen_clauses.add(key); break
            if clause:
                mark = " (earlier, may be stale)" if carried else ""
                lines.append(f'- {nd["label"]} — "{clause}"{mark}'); grounded += 1
            else:
                lines.append(f"- {nd['label']}")
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
            "woke": len(self._woke),                             # carried associations reactivated this event
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

    def snapshot(self, max_nodes: int = 40, max_edges: int = 60, max_dormant: int = 18) -> dict:
        """A compact, JSON-able view for the config-screen inspector: the hottest ACTIVE nodes
        plus the hottest DORMANT (carried) ones — so the persistent layer is always visible,
        not crowded out once a new conversation heats up — and the edges among them.  Bounded
        for legibility.  Read-only observability (WM-2: the graph is never rebuilt from this)."""
        active = sorted(((n, d) for n, d in self.nodes.items() if not d.get("primed")),
                        key=lambda kv: (-kv[1]["activation"], kv[0]))[:max_nodes]
        dormant = sorted(((n, d) for n, d in self.nodes.items() if d.get("primed")),
                         key=lambda kv: (-kv[1]["activation"], kv[0]))[:max_dormant]
        top = active + dormant
        ids = {nid for nid, _ in top}
        fr = set(self.frame)
        nodes = [{"id": nid, "label": nd["label"], "activation": round(nd["activation"], 4),
                  "frame": nid in fr, "primed": bool(nd.get("primed")),
                  "grounds": list(nd.get("grounds") or [])} for nid, nd in top]
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
                            "last_ts": nd["last_decay_ts"], "turns": list(nd["turns"]),
                            "grounds": list(nd.get("grounds") or [])}
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
        gmax = int(self.c["ground_max_chars"])
        gkeep = int(self.c["keep_grounds"])
        drop_common = bool(self.c.get("drop_common_singletons", True))
        for nid, nd in sorted((data.get("nodes") or {}).items()):    # fixed order (determinism)
            label = str(nd.get("label", nid))
            # Don't carry legacy filler across sessions: a bare word the CURRENT extractor would
            # never make a node (a stopword, or a common word under the gate) is dropped on load,
            # so an old word-cloud snapshot self-heals instead of being faithfully restored.
            if " " not in label and (label in _STOP or (drop_common and label in _COMMON)):
                continue
            last = float(nd.get("last_ts", now))
            act = float(nd.get("activation", 0.0)) * math.exp(-max(0.0, now - last) / slow) * carry
            if act < floor:
                continue
            turns = [int(t) for t in (nd.get("turns") or [])][-keep:] or [0]   # keep grounding non-empty
            grounds = [str(g)[:gmax] for g in (nd.get("grounds") or [])][-gkeep:]  # carried content
            self.nodes[nid] = {"label": nd.get("label", nid), "activation": act,
                               "first_seen_turn": 0, "last_boost_turn": 0, "last_decay_ts": now,
                               "turns": turns, "grounds": grounds, "primed": True}
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
