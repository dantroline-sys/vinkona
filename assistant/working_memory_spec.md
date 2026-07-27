# VIN-WM-01 — Scratch Working Memory & First-Person Asides

**Status:** Draft for implementation
**Owner repository:** `vinkona` (PolyForm)
**Touches:** `assistant/memory.py` (recall hot path, stores), `assistant/cascade.py` (turn loop, idle scheduling), the reply prompt
**Relationship to VIN-MEM-01:** supersedes its *working-memory* intent. VIN-MEM-01 overlaid Vinur's world graph and deferred every user-visible payoff; this spec overlays Vinkona's own `memories` store, changes answers from phase 0, and folds "episodic store / consolidation" into the **existing** gardening + `self_memories()` paths rather than a new subsystem.
**Non-goal:** a persistent knowledge graph. The only new persistent structure is the `asides` table. Edges are conversation-local and volatile until phase 5 earns otherwise.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are per RFC 2119.

---

## 1. Purpose

Give Vinkona **presence of mind** — the right small set of memories and threads active at each moment, carried across turns instead of re-derived cold — and a **checkable inner life** grounded in what she actually noticed at the time, not asserted after the fact.

Two mechanisms, one volatile structure:

- **Scratch graph** — a per-conversation activation overlay on the existing `memories` store. Volatile; discarded at conversation end after a consolidation harvest. Gives presence of mind.
- **Asides** — first-person, private, in-the-moment annotations recorded to a durable channel. Give the episodic record its "what it was like", and are why the scratch graph need not persist.

## 2. Grounding in what exists

This spec is an increment over live code, not a greenfield:

| Existing | Role here |
|---|---|
| `MemoryStore.recall()` ([memory.py:913](memory.py#L913)) | The fusion hot path. Phase 0 adds one term to its score at [memory.py:959](memory.py#L959). |
| `_AhoCorasick.search()` ([memory.py:688](memory.py#L688)) | Deterministic entity mention detection for the fast lane. No new matcher. |
| `_neighbours()` ([memory.py:992](memory.py#L992)) | Today's single-hop associative recall; phase 5 generalises it over earned edges. |
| `log_turn` / `logs_window` | The authoritative conversation record. Source of truth; the scratch graph is only an accelerator over it. |
| `self_memories()` ([memory.py:979](memory.py#L979)) | The continuous-self surface. Consolidated asides land here, canon-fenced. |
| `RhythmStore`, `PeopleStore` | Reflective-aside signals (recurrence, identity). |
| gardening / synthesis cycle | The consolidation harvest (§6) runs inside it, not beside it. |

## 3. Invariants

- **WM-1 (Accelerator, not truth).** Recall MUST return correct results with the scratch graph empty or stale. The record + `memories` store are authoritative. No turn may depend on between-turn work having completed.
- **WM-2 (Volatile).** The scratch graph has no durable form. Process loss is acceptable and MUST NOT be mitigated by persistence.
- **WM-3 (Non-blocking).** Between-turn work MUST NOT delay the next reply. User input preempts it within the abort budget (§5).
- **WM-4 (Minimal-mode safe).** The fast lane and reflective asides MUST invoke no model, so they operate when the box has vacated VRAM.
- **WM-5 (Asides are observed).** No aside may be written to canon (self-model core or asserted user-fact). Asides carry low, capped influence and decay.
- **WM-6 (Fresh, not amnesiac).** A new conversation starts with a warm seed from persistent truth (§6), never with inherited volatile state from the previous session.

## 4. Scratch graph

### 4.1 Node

A scratch node references a `memories` row (or a conversation-local delta) and carries volatile state:

```
{ mem_id | "delta:<uuid>", activation: float, last_boost_turn: int,
  last_decay_ts: float (epoch), stance: [aside_id...] }
```

Activation is a float in `[0, A_MAX]`. `stance` links asides that colour this node this conversation.

### 4.2 Dynamics (deterministic)

- **Decay** — per event, before boosts: `activation *= exp(-dt / TAU_S)`, `dt` in seconds. Store `last_decay_ts` as a float epoch (no ISO round-trip).
- **Boost** — a node is boosted `+B_DIRECT` when its memory is surfaced by recall OR its entities are matched by `_AhoCorasick.search()` in the last turn(s).
- **Spread** (phase 5) — one hop over conversation-earned edges, damped by `DAMP`, using the existing `neighbours`-style adjacency with a fan-out cap.
- **Eviction** — drop below `A_MIN`, cap at `N_MAX`, tie-break deterministically. Delta nodes referenced by an open thread (§7) are exempt.

All iteration MUST be in a fixed order (ascending id); no reliance on dict/set ordering. No randomness.

### 4.3 Feed into recall

Phase 0, at [memory.py:959](memory.py#L959), add `+ w_activation * activation[mid]` to the score. `activation` persists across the conversation and decays between turns.

**Activation vs cooldown MUST be balanced.** Activation says "hot, surface again"; the existing cooldown says "just used, suppress". `w_activation` and the cooldown override MUST be tuned together so the thread stays warm without defeating anti-repetition. This balance is the core tuning task of phase 0 and MUST be measured (§8), not guessed.

## 5. Between-turn lane

### 5.1 Scheduling

- Wake after `T_DEBOUNCE_MS` of quiet (so it never runs mid-utterance); do not start during reply generation.
- Operate on a **staging copy**; commit by a single reference swap.
- On user input or `T_EVENT_MAX_MS`, abort: drop the staging copy within `T_ABORT_MAX_MS`. The live scratch graph MUST be untouched (partial apply is a failure — WM-1/WM-3).

Because the fast lane is deterministic and bounded, abort is a discard, not an unwind.

### 5.2 Fast lane (every qualifying gap)

Deterministic, no model: decay, boost from the last turn's mentions, lay/strengthen conversation-local co-occurrence edges between co-mentioned nodes, evict. Emit reflective asides (§7). Target well under `T_ABORT_MAX_MS` so it fits micro-gaps.

### 5.3 Slow lane (longer pauses only)

Opportunistic LM work (new-memory capture from the last few turns — the existing capture path). Commits idempotently; usually will not finish in a fast exchange, and MUST NOT be relied on (WM-1). Unavailable in minimal mode by design.

## 6. Lifecycle

```
SEED → LIVE → HARVEST → DISCARD
```

- **SEED.** On conversation start, warm the scratch from persistent truth: high-salience `self_memories()` + top personal memories + the tail of the previous record. Fresh, not blank (WM-6).
- **LIVE.** Turns + between-turn lane (§5).
- **HARVEST.** On close or idle timeout, run consolidation inside the existing gardening cycle: promote durable new memories, promote a conversation-earned edge only if it recurred, synthesise a durable aside into a category-`self` memory via the canon-fenced `self_memories()` path. Nothing here writes canon directly.
- **DISCARD.** Drop the scratch graph. The record (with asides) and the updated store carry everything forward.

## 7. Asides

### 7.1 Table (`asides`, in memory.db — the one new persistent structure)

```
id, session_id, turn_ref, source ∈ {reflex, reflective},
kind, text, trust: float, refs: [mem_id...], ts: float,
superseded_by: id | null
```

Distinct from `log_turn` rows so the transcript stays pristine and a guess is never mistaken for what was said.

### 7.2 Reflex asides

Emitted by the reply model on a private channel in the same generation. Kinds: `confidence`, `difficulty`, `user_read`. Rules: only-if-notable, ≤`ASIDE_MAX_PER_TURN` per turn, `trust` low by default (a guess about tone). Never spoken.

### 7.3 Reflective asides

Emitted deterministically by the fast lane on structural events: `recurrence` (activation sustained N turns or across sessions), `contradiction` (turn conflicts with a stored memory), `unresolved` (weak/abstained answer with no later close), `rhythm` (`RhythmStore` hit). No model required. A warm small model MAY narrate to first person; otherwise a template.

### 7.4 Use, correction, safety

- Asides annotate scratch-node `stance` → presence of mind includes felt stance, not just fact.
- A later turn MAY supersede a reflex aside (`superseded_by`) so a misread does not ossify.
- Recall MAY surface a relevant prior aside on topic recurrence ("last time, stumped") for continuity; the reply model may reference it obliquely, never verbatim.
- WM-5 holds absolutely: capped influence, decay, never canon. Operator-visible for debugging; never shown to the user in-conversation.

## 8. Constants (first-guess; every one configurable and logged)

`A_MAX=1.0`, `A_MIN=0.02`, `B_DIRECT=0.4`, `TAU_S≈900`, `DAMP=0.45`, `N_MAX=1024`,
`w_activation` (tune vs cooldown), `T_DEBOUNCE_MS≈250`, `T_EVENT_MAX_MS≈2000`,
`T_ABORT_MAX_MS≈50`, `ASIDE_MAX_PER_TURN=2`, `W_ASIDE_MAX=0.25` (hard influence cap).

## 9. Phases (ship and measure in order)

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **0** | `w_activation` term in `recall()`, persisted+decaying across the conversation; balanced against cooldown. | On held-out multi-turn transcripts, recall surfaces the on-thread memory earlier/more often than baseline **without** raising repetition. If it doesn't move, stop — the rest is moot. |
| **1** | Between-turn fast lane: staging + preempt + deterministic boost/decay/edges. | Abort safety (§10) green; minimal-mode operable; recall unchanged when scratch is cold. |
| **2** | Reflex asides: private reply channel → `asides` table; scratch `stance`. | Volume within cap; zero canon writes; never surfaced verbatim. |
| **3** | Reflective asides + open-thread tracking + supersede path. | Misread-correction demonstrated; reflective asides fire in minimal mode. |
| **4** | Harvest + warm seed inside gardening. | Fresh-not-amnesiac demonstrated across a conversation boundary; consolidated self-aside appears via `self_memories()`. |
| **5** *(deferred)* | Persistent personal graph + multi-hop spread — **only if** phases 1–4 show the same edges re-earned repeatedly. | Evidence of recurring structure. |

## 10. Acceptance gates (proportionate — the risks that actually matter)

- **G-ACCEL.** With the scratch graph forced empty, a ≥30-turn conversation produces identical recall results to baseline. (WM-1)
- **G-ABORT.** Abort injected at random points during the fast lane leaves the live scratch graph byte-identical to its pre-event snapshot; p99 abort-to-discard ≤ `T_ABORT_MAX_MS`. (WM-3)
- **G-MINIMAL.** Fast lane + reflective asides complete with the LM/embedder unavailable (minimal mode). (WM-4)
- **G-CANON.** No path from an aside reaches canon; aside influence on any recall score ≤ `W_ASIDE_MAX`; a superseded aside stops influencing. (WM-5)
- **G-VOLATILE.** Process kill mid-conversation yields an empty scratch graph on restart; only `asides` and normal store writes persist. (WM-2)
- **G-FRESH.** A new conversation does not inherit the previous session's activation vector; it seeds from persistent truth. (WM-6)

## 11. Open questions (do not implement)

1. ~~**Aside emission trigger.**~~ **RESOLVED (2026-07-27):** the reply model is prompted every turn and decides for itself whether anything is worth noting — emit zero or up to `ASIDE_MAX_PER_TURN`. No external gate; the LM's own judgement is the trigger. (Watch the realised rate in phase 2; tighten the prompt, not the mechanism, if it narrates.)
2. **Edge promotion threshold (phase 5).** How many re-earnings across how many sessions justifies a durable edge is unknown until phase 1 logs how often edges recur. That log is the instrument.
3. **Reflex vs. reflective overlap.** A confidence reflex and an `unresolved` reflective aside can describe the same moment; dedup policy deferred to phase 3 data.
