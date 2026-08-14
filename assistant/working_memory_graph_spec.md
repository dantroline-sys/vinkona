# VIN-WM-02 — Conversational Working-Memory Graph & Instrument

**Status (post-split, updated 2026-08-14):** PARTIALLY EXECUTED / PARTIALLY SUPERSEDED — read with the notes below.

- **SHIPPED (phase 1a):** `working_graph.py` — the deterministic no-LM half exactly as
  scoped: volatile per-conversation RAKE phrase graph → decaying frame → fenced
  "working notes" briefing; nodes + untyped co-occurrence edges only; deterministic,
  bounded, grounded; off by default (`memory.working_graph`). The instrument + drift
  guard (§6 harvest) landed there too (`metrics()`, trace kind=working_graph).
  The P2 first-person asides lane shipped as chat_logs `role='aside'`.
- **SUPERSEDED (2026-08-09 architecture split):** the slow LM-extraction lane and
  typed relations in THIS volatile graph. Typed knowledge now lives in the DURABLE
  store's `mind_graph.py` (big-LM distillation at dreaming time) — the volatile lane
  stays untyped by design. The staging-copy apply/abort protocol and §9 idle harvest
  fell away with that lane.
- Kept as the reference design: invariants G-1..G-8 remain binding on the shipped
  code, and the slow-lane sections document the road not taken if in-conversation
  typed extraction is ever wanted.

**Original status:** Draft for review (Dan to mark up)
**Owner repository:** `vinkona` (PolyForm)
**Realizes:** VIN-WM-01 phase 1 (the between-turn lane), in the fuller shape Dan specified — live extraction of a *here-and-now* graph, not just an activation overlay.
**Reuses:** VIN-WM-01 phase 0 (`WorkingActivation`, the decaying per-conversation activation already in `recall()`); the deterministic span matcher (`_AhoCorasick.search`); the `lm_lease` yield primitive; the gardening/idle cycle.
**Harvests (proportionately):** VIN-MEM-01 §6 (instrument metrics) and §7 (mood-lock). Deliberately does **not** adopt VIN-MEM-01's Vinur-overlay graph, cognitive-event transaction formalism, frozen hand-off artifact, or G1–G14.
**Non-goal:** a durable knowledge graph. This graph is thrown away at conversation end; anything worth keeping leaves via the *existing* memory-store path, not by persisting the graph.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are per RFC 2119.

---

## 1. Purpose

Keep Vinkona on-thread across a very long session **without a very long context**. The graph is a *compression* strategy, not a retrieval one: instead of re-feeding hundreds of turns (expensive, attention smears) or re-summarising prose each turn (lossy — a photocopy of a photocopy by turn 200), she maintains one small graph that is **updated in place** — add nodes/edges, merge duplicates, decay the stale — and renders it to a compact **briefing** handed to the next turn.

Two things make this distinct from the memory store that exists today (§2):

- **Lifetime.** Session-scoped; discarded at the end. The store is durable and cross-session.
- **Job.** "What are *we* talking about right now / what did we just work out." The store answers "what do I know about X" over 13k+ memories. Different question, different structure.

## 2. Grounding in what exists

An increment over live code, not a greenfield:

| Existing | Role here |
|---|---|
| `WorkingActivation` + `recall(wm=…)` (VIN-WM-01 P0) | The activation field and decay clock. **Generalised**: a base node *is* an activated memory-id. With no extraction, the graph is exactly today's P0 — this spec adds delta nodes and edges on top of the same dynamics. |
| `_AhoCorasick.search()` | Deterministic mention detection for the fast lane. No new matcher. |
| durable `memories` store + `recall()` | Base nodes reference it read-mostly; the briefing shows *activated durable memories* alongside conversation deltas. |
| `lm_lease` | The slow lane leases the model and yields to the reply path. |
| gardening / idle cycle | Where the optional harvest (§9) runs — beside consolidation, not a new subsystem. |
| chat_logs (incl. `role='aside'`, P2) | The authoritative record and the grounding source. Asides colour node `stance`. |

## 3. Cardinal invariants

- **G-1 (Accelerator, not truth).** The briefing is an aid. Every turn MUST be correct with the graph empty, stale, or wrong. The record + store are authoritative; no turn may block on between-turn work.
- **G-2 (Volatile).** The graph has no durable form. Process loss is acceptable and MUST NOT be mitigated by persistence *of the graph* (durable value leaves via §9 harvest instead).
- **G-3 (Non-blocking & preemptible).** Neither lane may add latency to a reply. User input MUST abort in-flight work within the abort budget and discard the staging copy; the live graph is untouched (partial apply is a failure).
- **G-4 (Minimal-mode honest).** The fast lane and the briefing render MUST invoke no model, so they work with VRAM vacated. The slow lane (extraction) is LM work and simply does not run in minimal mode — the graph coasts on decay.
- **G-5 (Fenced & fallible).** The briefing is presented as *her running notes, possibly wrong* — never asserted as fact, never shown to the user verbatim, and no path from it reaches canon (self-model core or asserted user-fact).
- **G-6 (Grounded & checkable).** Every node and edge carries the turn ids that support it. An edge with no support MUST be refused. Every briefing line MUST be traceable to a turn — this is what makes drift auditable rather than silent.
- **G-7 (Bounded).** Nodes, edges, per-extraction output, and the briefing are all hard-capped. The graph MUST NOT grow unbounded under any transcript.
- **G-8 (Deterministic dynamics).** Decay, boost, merge-fold, eviction, frame, and metrics MUST be deterministic — fixed iteration order (ascending id), no randomness, `math.exp`. The **only** stochastic step is extraction (§6), whose *output* is untrusted and whose *fold* back in (§5.3) is deterministic and bounded.

## 4. Data model (the working graph)

### 4.1 Node

```
{ id, origin ∈ {base, delta}, activation: float in [0, A_MAX],
  first_seen_turn: int, last_boost_turn: int, last_decay_ts: float(epoch),
  label: str, source_turns: [int...], stance: [aside_id...] }
```

- `origin=base` → `id` is a durable memory-store id; `label` copied from it (advisory). This is the P0 case.
- `origin=delta` → `id = "delta:<uuid4>"`; a concept the *conversation* introduced with no store counterpart. `uuid4` is minted at node creation (turn processing), never inside a fast-lane event.
- Merging a delta into a base node is FORBIDDEN. If a delta is later found to correspond to a store memory, add an edge `corresponds_to` and keep both.

### 4.2 Edge

```
{ src, dst, relation: str, origin ∈ {base, delta}, weight: float in [0,1],
  support_turns: [int...], last_seen_turn: int }
```

`support_turns` is the grounding (G-6): the turns whose text produced this relation. An edge reaching `support_turns == []` MUST NOT exist.

### 4.3 Frame

The top-`K_FRAME` nodes by activation (tie-break ascending id) = "what is currently in front of her." The frame is the spine of the briefing (§7) and the input to `frame_churn`/`frame_drift` (§8).

## 5. The two lanes

### 5.1 Scheduling & preemption (both lanes)

- Wake only after `T_DEBOUNCE_MS` of quiet; never start during reply generation.
- Work on a **staging copy**; commit by a single reference swap.
- On user input or the lane's time budget, abort: discard the staging copy within `T_ABORT_MAX_MS`; the live graph is byte-identical to its pre-event snapshot. Abort is a discard, not an unwind.

### 5.2 Fast lane — deterministic, no model, every qualifying gap

In order, on the staging copy: **decay** all nodes (`activation *= exp(-dt/TAU_S)`); **boost** (`+B_DIRECT`) nodes whose entities `_AhoCorasick.search` matched in the last turn(s), admitting a base node for a matched-but-absent memory; **strengthen** deterministic co-occurrence edges (relation `co_occurs`) between nodes co-mentioned this turn, appending the turn to `support_turns`; **evict** below `A_MIN` / above caps (delta nodes pinned by an open thread exempt); **recompute** the frame; **update** the instrument (§8). Target well under `T_ABORT_MAX_MS`. Runs in minimal mode.

### 5.3 Slow lane — LM extraction, opportunistic, longer pauses only

This is Dan's "spare cycles during typing" step. Leases a small/fast model (yields to the reply path via `lm_lease`), reads the last `W_EXTRACT_TOK` tokens of the record, and asks for nodes + typed relations (§6). Then **merge-fold** (deterministic):

- Align each proposed node to an existing node — by span match to a resident label, or to a durable memory via the store — before creating anything. A `delta` node is created **only** for a genuinely new concept.
- Add/strengthen proposed edges with their `support_turns`; do not duplicate an existing edge, raise its `weight`.
- Newly folded structure enters at `activation = B_DIRECT` and thereafter decays like everything else — so a concept the conversation drops **fades out on its own**.

The fold is **idempotent** (re-running the same extraction changes nothing new), **best-effort** (MUST NOT be relied on — G-1), and does not run in minimal mode (G-4). Extraction output is untrusted text: sanitise/fence it and cap it (§6) before folding.

**Accumulate, don't replace.** The window is only the *read* aperture; extraction merges into the *standing* graph. Re-extracting fresh each window (and discarding the rest) is FORBIDDEN — it would lose exactly the long-quiet threads this exists to keep.

## 6. Extraction contract

The model is asked, over the recent window, for:

```json
{ "nodes": [{ "label": "str", "turns": [int] }],
  "edges": [{ "src": "label", "dst": "label", "relation": "str", "turns": [int] }] }
```

- `relation` is a short lowercase verb-phrase (`causes`, `blocks`, `is_part_of`, `contradicts`, `decided`, …). Whether to constrain to a controlled vocabulary is an open question (§11); start freeform, watch what recurs.
- Output is capped: ≤ `N_EXTRACT_NODES`, ≤ `N_EXTRACT_EDGES` per pass; anything over is truncated (and the truncation logged, never silently dropped).
- `turns` not present in the read window are discarded (a node/edge must ground to what was actually seen — G-6).
- The prompt is small-model-friendly and states the graph is *working notes*, not a knowledge base — it should prefer *this conversation's* structure over restating world facts.

## 7. The briefing (the payoff)

Each turn, render the graph to a compact block for the reply prompt:

- Header fences it: *"Working notes — my running read of this conversation so far; may be wrong, for continuity only."* (G-5)
- Body: the frame (top-`K_FRAME` nodes) grouped, each with its hottest relations as short lines (`X → blocks → Y`). Base-node lines are activated durable memories; delta-node lines are conversation structure.
- Hard char cap `BRIEF_MAX_CHARS`; bounded and ordered by activation, so it never bloats and always shows the hottest first.
- **Degrades gracefully:** with no extraction yet (cold, or minimal mode), the briefing is just the activated durable memories — i.e. exactly P0. This is the continuity between what's built and what this adds.

The briefing is distinct from the ambient block (§ orientation) and from raw history. It is never spoken and never shown to the user.

## 8. Instrument (concurrent — the drift smoke-detector)

Not a nicety: a graph an LM keeps writing to *will* accumulate errors, and injected every turn those errors compound — the exact "memory trap" this is meant to prevent. The instrument is how that becomes visible. Deterministic functions of the staging copy, emitted as JSONL to the agent state dir, non-blocking:

`working_set_size`, `delta_ratio`, `activation_entropy` (nats), `frame_churn` (`1 − jaccard(frame_now, frame_prev)`), `frame_drift` (`vs. seed`), `open_threads`, `edges_ungrounded` (MUST be 0 — a live assertion of G-6), `extraction_lag` (turns since last successful fold), `briefing_chars`, `completion_rate` (committed / (committed+aborted)). Every record carries the full constant set in force (they are first-guess — §10).

**Drift / mood-lock guard.** If `activation_entropy < E_LOCK` for `K_LOCK` consecutive commits, or `frame_churn == 0` for `K_LOCK` commits, raise a `WorkingGraphStalled` fault: record it and surface it to the operator. It MUST NOT silently mutate activation. A frame that has stopped moving is a fault to flag, not conviction.

## 9. Lifecycle

```
SEED → LIVE → HARVEST(optional) → DISCARD
```

- **SEED.** Warm, not blank (G-1 still holds if seeding fails): high-salience `self_memories()` + top personal memories for the opening topic + the tail of the record, as base nodes at `B_DIRECT`. Bounded to ≤ `N_MAX/4`.
- **LIVE.** Turns + the two lanes.
- **HARVEST (optional).** On close/idle-timeout, inside the existing gardening cycle: a genuinely durable learning may be promoted to the **memory store** (the normal path, canon-fenced); an edge may be promoted only if it recurred across turns. **Nothing durable is kept *as graph* — it becomes an ordinary memory.**
- **DISCARD.** Drop the graph.

**Cross-session carryover — OPEN, not decided (Dan, this turn).** Default is full discard. Whether a thin seed of the *previous* conversation's frame should carry into the next session is deferred (§11-1); the working assumption is that "remembering a bit of last time" is better served by the durable store surfacing it through normal recall than by persisting any part of this graph — which would violate G-2 and risk exactly the stale-context problem we're avoiding. Do not build carryover under this contract.

## 10. Constants (first-guess; every one configurable and logged)

Reuse VIN-WM-01 §8 (`A_MAX=1.0`, `A_MIN=0.02`, `B_DIRECT=0.4`, `TAU_S≈900`, `N_MAX=1024`, `K_FRAME≈12`, `T_DEBOUNCE_MS≈250`, `T_ABORT_MAX_MS≈50`) plus:
`W_EXTRACT_TOK≈600` (read window), `T_EXTRACT_MAX_MS≈4000` (slow-lane budget), `T_EXTRACT_MIN_GAP_S` (don't extract more often than this), `N_EXTRACT_NODES≈16`, `N_EXTRACT_EDGES≈24`, `BRIEF_MAX_CHARS≈700`, `E_LOCK≈1.8`, `K_LOCK≈4`, `EDGE_MIN_WEIGHT` (below which an edge is pruned).

## 11. Phases (ship and measure in order — cheap deterministic payoff before LM risk)

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **P1a** | Fast lane + briefing over **span-matcher deltas only** (no LM): decay/boost/co-occurrence edges/evict/frame, rendered to the fenced briefing. Generalise `WorkingActivation` to carry the node/edge fields. | On held-out ≥50-turn transcripts, the briefing keeps an early thread retrievable at turn N that baseline has lost — **without** raising repetition. If it doesn't move, stop; the LM part is moot. |
| **P1b** | Slow lane: LM extraction + deterministic merge-fold. | Preemptible + idempotent (G-3); minimal-mode coasts (G-4); fold never regresses P1a's briefing; `edges_ungrounded == 0`. |
| **P1c** | Instrument + `WorkingGraphStalled` guard, concurrent. | Trips on a synthetic ossifying transcript within `K_LOCK`; no false trip across ≥200 normal events. |
| **P1d** *(optional)* | Harvest into the store inside gardening. | A durable learning from a session appears via normal recall next session; no graph persisted. |

## 12. Acceptance gates

- **G-ACCEL.** Graph forced empty ⇒ recall and briefing are identical to P0 baseline over a ≥30-turn conversation.
- **G-ABORT.** Abort injected at random points leaves the live graph byte-identical; p99 abort-to-discard ≤ `T_ABORT_MAX_MS`.
- **G-MINIMAL.** Fast lane + briefing complete with LM/embedder unavailable.
- **G-FENCE.** No path from the briefing to canon; the briefing is never emitted as user-facing text; header present.
- **G-GROUND.** No edge exists with empty `support_turns`; every briefing line maps to ≥1 turn id.
- **G-VOLATILE.** Process kill mid-conversation ⇒ empty graph on restart; only store writes + the metrics log persist.
- **G-BOUNDED.** Under an adversarial 10k-turn transcript, node/edge/briefing caps hold; over-cap extraction truncates-and-logs, never grows.
- **G-DRIFT.** The instrument raises `WorkingGraphStalled` on an entropy-collapsing transcript and stays quiet on a healthy one.

## 13. Open questions (do not implement)

1. **Cross-session carryover** (Dan). Default discard vs. a thin frame-seed forward. Leaning discard + let the durable store carry it; revisit once P1c metrics show whether sessions actually reference the previous one.
2. **Extraction model & scheduling.** Small dedicated model vs. borrowing the fast LM under lease vs. the big LM on longer pauses — trades latency risk against extraction quality. Decide from `extraction_lag` under real load.
3. **Relation vocabulary.** Freeform verb-phrases vs. a controlled set. Start freeform; let recurrence in the logs argue for a schema.
4. **Briefing vs. P0 overlap.** A hot durable memory can appear both as a base node and via ordinary recall; dedup policy TBD from P1a data.
5. **Forward-compat with self-authored tooling.** Build the graph as a clean standalone module with its own small store interface (not buried in `recall()`), so the later "she grows it / builds her own tools over it" spec has a first-class object to stand on. Design toward that boundary; build none of it here.
