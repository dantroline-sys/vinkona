# Implementation Spec: Operational Self-Improvement Loop (Vinkona skill-LoRA)

**Goal.** Teach the 9B orchestration model (Qwen-9B) to *operate Vinkona* at near-31B quality, at 9B latency and cost, by learning from daily use. The win is **31B-quality orchestration at 9B footprint** — not a smarter model, a model fluent in this system's form.

**Scope — read before anything.** This trains **behaviour/form, NOT knowledge.** It is explicitly *not* a way to put facts into weights. Knowledge stays external in the graph (attributable, editable, supersedable, plastic); only the *skill of operating the system* goes in the adapter. Conflating the two unbuilds the whole architecture (see Appendix).

**Companion:** references the main KB spec — `§13` golden set, `knowledge_gaps` log, `§11` brainstorm faces, `§9` reconciliation banding, the regime firewall (`§8`).

**Hardware.** Training: single RTX 4090, QLoRA (4-bit base). Capture: rides on existing logging. Curation + synthetic generation: idle GPU, shared with the background-reading loop.

---

## 0. Core principles

1. **Cardinal rule — only train on a signal independent of, and better than, the 9B's own judgment.** Train a model on its own unvetted output and it collapses (homogenises, loses the tail, gets confidently worse). Every training example must be *system-accepted*, *31B-corrected*, or *user-corrected* — never self-endorsed.
2. **Only LoRA the genuine LM-skills:** routing, retrieve-vs-answer, reading/synthesising a structured card (not paraphrasing it), regime discipline, the divergent→convergent brainstorm rhythm, extraction JSON, next-best-question phrasing. **Deterministic parts stay in code** (information-gain selection, fit-scoring, RRF) — never train what arithmetic already does correctly.
3. **The ceiling is the 31B's *grounded* competence.** The loop makes the 9B operate the system as well as the 31B does. Only **real** signals (user corrections; objective checks catching teacher errors) push past that ceiling.
4. **Knowledge external & plastic, skill internal & learned — and they improve each other** (the compounding effect, §10).

---

## 1. Prerequisite: freeze the format contract (the actual gate)

A LoRA learns `(input format → output behaviour)`. It bakes in the *exact surface shape* of the context. If the assembly format is still moving, every captured trace silently trains toward a layout you no longer serve — no error, just quiet rot.

- **Freeze the STRUCTURE** (the assembly contract): section order (e.g. `base prompt → date/context → memory stack → retrieved cards → tool results → user turn`), delimiters/section markers, tool-call syntax, JSON schemas, chat template.
- **Content stays variable** — which memories/cards/queries flow through is *supposed* to vary; that's the generalisation surface. Stable envelope, variable contents.
- **Readiness test:** *reprocessing any past trace's inputs under today's assembler yields a byte-identical layout.* If a format change would alter the shape of already-captured traces, you are not ready.
- **Version it** (`format_v1`). Stamp every adapter with the format version it learned. A contract change = new version + re-capture + retrain; the eval-gate (§7) catches a forgotten mismatch.
- Traces captured **before** freeze: keep for analysis, **not** for training.

> Don't chase a *perfect* format — only a *stable, good-enough* one. "I'm no longer changing the skeleton week to week" is the bar.

---

## 2. Signal sources (where "good" comes from)

In ascending value:

1. **System self-judgment (free, objective, continuous).** Normal operation already grades the 9B: did its routing yield a retrieval the user accepted with no re-ask? Did its first-pass extraction get **accepted by reconciliation** (not rejected/rewritten by the 31B)? Did its grammar-constrained JSON **parse first try**? Did its differential's top candidate survive? Agreement with the good outcome → positive example. Divergence + correction → the **correction** is the positive, the 9B's attempt is the negative.
2. **31B-as-teacher, on disagreements only.** Nightly, the 31B redoes that day's orchestration. Where it **agrees** with the 9B → discard (no teaching value). Where it **diverges** → the 31B trace is gold. Targeting only the gap is far more efficient and concentrates training on the 9B's weak spots.
3. **User corrections (rarest, highest).** Edits, re-phrasings, "no, I meant…". The only signal that can teach past the 31B's ceiling → genuine personalisation.

---

## 3. Trace capture (continuous, cheap — rides on logging)

```
trace {
  trace_id, timestamp, format_version,
  input_context: { base, date, memory, cards, tool_results, user },   // the assembled sections
  model_action:  { assistant_turns, tool_calls, emitted_json },        // what the 9B produced
  outcome: {
    retrieval_result, reconciliation_verdict, json_parsed: bool,
    teacher_trace | null,           // filled in §4 only for disagreements
    user_response | null            // accepted | edited:<diff> | re_asked
  },
  labels: { }                        // derived in §4
}
```

---

## 4. Curation (nightly, idle GPU) → training examples

```
for trace in day:
    label(trace)                                   // apply §2 signal rules
    if disagreement_or_failure(trace):
        trace.teacher_trace = run_31B_grounded(trace.input_context)   // §6: retrieve, don't recall
emit:
  SFT_examples      = gold action traces (full role-tagged convo incl tool turns; loss on assistant tokens only)
  preference_pairs  = (rejected = 9B attempt, chosen = correction) for the same context   // DPO
  binary_judgments  = {trace, accepted|rejected}                                          // KTO
filter HARD:
  - keep only vetted (system-accepted / 31B-corrected / user-corrected)
  - objective gates as hard filters: JSON valid · reconciliation-banding match · retrieval relevant · feasibility pass
oversample: hard / rare / disagreement cases (where the learning is)
preserve diversity: do NOT let the set collapse to the easy majority case
append → versioned curated dataset
freeze → a REAL held-out test set, never trained on
```

---

## 5. Training methods (which, and when)

- **SFT — the cold-start.** "Be like these gold traces." Establishes the behaviour; cannot say "*not* that," carries no correction signal. Right for the first adapter.
- **DPO — preference pairs.** Shows the same situation as `rejected` vs `chosen` → learns the *direction of improvement*. **Your system manufactures these for free** (every 9B action the 31B/reconciliation/user corrected). This is the only way to encode *corrections*.
- **KTO — unpaired binary.** Each trace gets accept/reject, no matched pair needed. Best fit for the **noisy system signals** ("this extraction was accepted", "this routing got re-asked").

**Sequence:** SFT cold-start → daily loop shifts to DPO/KTO, because correcting the 9B's specific mistakes (not just showing generic good behaviour) is the entire point.

**Format:** JSONL, exact Qwen chat template, **tool calls + tool results included**, identical to production. The model learns the surface form as much as the substance; a train/serve format mismatch silently degrades everything. (Harvesting real traces gives this consistency for free.)

---

## 6. Synthetic data generation — adversarial, seed-driven (the accelerant)

Self-instruct with a teacher–student split. **Legitimate iff generation is grounded *and* filtered; degenerate ("circle-jerk") if not.** The goal is not volume — it is to **manufacture the 9B's failures on demand**, then harvest the corrections.

### 6.1 The two roles
- **31B-as-user → breadth.** Invents diverse scenarios (every topic, the four `§11` faces, multi-hop, weird phrasings). Variety in *prompts* is the goal and does not degenerate.
- **31B-as-supervisor → MUST RETRIEVE, NOT RECALL.** Every supervisory turn runs the **real** retrieval pipeline and reasons over **real** returned structure with provenance. Correctness comes from the graph, not the teacher's memory — **the external anchor** that separates gold from circle-jerk. Never free-associate from weights.

### 6.2 Seed → perturb → run → filter (the harvest loop)
```
seeds = real conversations, WEIGHTED toward ones that already exposed 9B weakness
        (but NOT only already-failed ones — see note)
while kept_failures < TARGET_N:
    seed    = sample(seeds)
    axis    = sample(PERTURBATION_AXES)            # bend the seed toward a known weak spot
    variant = 31B_user.perturb(seed, axis)
    attempt = 9B.run(variant)                      # provoke the failure
    verdict = judge(attempt)                       # grounded 31B + objective checks (§6.3)
    if verdict == PASS:  discard                    # no training signal
    else:                keep( variant,             # the kept-failure
                               rejected = attempt,
                               chosen   = 31B_grounded_trace )   # the correction

PERTURBATION_AXES = [ ambiguous / under-specified intent · second hop (why → how) ·
  regime-boundary cross (a fact + a fiction/social aside, tests the firewall) ·
  distractor context / length · format-stress (nested JSON, multiple tool calls) ·
  mid-conversation correction ("no, I meant…") · edge-of-scope (graph half-covers it) ]
```
**Generation manufactures failures; curation keeps only the kept-failures.** A *passing* variant is one the 9B already handles — discard it (keeping it is the volume-without-signal trap). The `(rejected = 9B attempt, chosen = grounded 31B)` pairs are the gold (DPO/KTO, §5). Perturb along **difficulty / failure-mode** axes, not just topic — the 9B's deficiencies are structural (chaining, ambiguity, regime boundaries, malformed JSON under load), not topical, so ten subject-swapped paraphrases of a case it already handles give ten more passes, not signal.

> **Seed breadth matters — do not seed only from already-failed interactions.** "Keep only failures" is the rule for the *output* filter, not the *input* seed. Perturb *passing* real conversations too: bending a currently-passing case is how you surface **latent** failures before a user hits them, and early on real failures are too scarce to seed from alone. Seed from real traffic, weighted to known-weak cases; let perturbation find the rest.

### 6.3 Anchors against degeneration (all required)
- **Objective filters:** JSON parses · reconciliation-banding match · retrieval relevance · feasibility pass.
- **Real signal in the blend:** a minority of genuine real-usage traces/corrections tethers the distribution to reality.
- **Real eval only (§7/§9):** the held-out set is never synthetic — a synthetic eval closes the loop and blinds you.
- **Gap-steering:** point the synthetic user at the `knowledge_gaps` log, single-source edges, unresolved `disagrees_with`, queried-but-bare imported edges. **Targeting, not volume, is the superpower.**
- **Coverage > count:** 300 diverse kept-failures beat 1,000 near-duplicates. Watch failure-mode spread; narrow seeds → narrow variants → a model over-tuned to two failure modes.

### 6.4 Data isolation (do not pollute the live system)
Synthetic traces are **training artifacts — not conversation, not knowledge.** Route them to a dedicated **training store**, tagged `synthetic=1`. They **must never** enter the chat / episodic history, the knowledge graph, or reconciliation (main spec §9): this is offline construction, not live conversation, and the variants assert nothing true — they only exercise *form*. The seed corpus is real user content, so keep generation on local/trusted models, scrub anything you wouldn't persist, and treat seeds with the same provenance discipline as the most sensitive data in the system.

> Degenerate case avoided by all of the above: an ungrounded supervisor self-talking inherits the teacher's blind spots, cannot exceed it, and homogenises below it.

---

## 7. Train → gate → deploy cadence

```
TRAIN (when enough new vetted examples accrue to plausibly move the eval — not a fixed schedule):
  fresh QLoRA on the WHOLE accumulated curated set      // NOT a serial fine-tune of the previous adapter
  rank 16–32, all-linear target  →  ~100–300 MB fp16    // rank 8 attn-only ~tens of MB; rank 64 ~0.5–1 GB
  a few hours on the 4090
  mix in the rehearsal anchor set (§9)

GATE (non-negotiable):
  candidate vs current vs base, on:
    - §13 golden set
    - a held-out slice of REAL recent traces        // never synthetic
    - behavioural checks: JSON validity, regime/firewall discipline, no contamination
  promote ONLY if it strictly improves and regresses nothing

DEPLOY:
  hot-swap adapter; keep last few for instant rollback
  log which adapter served which interaction        // regressions stay attributable
  stamp adapter with format_version

STOP retraining when the 9B↔31B agreement rate plateaus near the teacher
  (the distillable gain is harvested; more data just re-teaches what it knows)
```

---

## 8. Readiness & sizing (when to start, how much)

- **The count that matters is vetted, diverse *correction* examples — not raw interactions.** Most daily traffic is the 9B agreeing with itself and teaches nothing. Count the corrections.
- **Ranges (good examples):** SFT cold-start takes with a few hundred; **~500–1,000** synthetic gold gives a solid first adapter. DPO/KTO: **a few hundred to ~1,000** vetted pairs meaningfully move behaviour. Anchor: *hundreds to low thousands* — tens is too noisy, tens of thousands is unnecessary and wastes months.
- **Trigger the first real run when all three hold:** (a) the SFT cold-start set exists, (b) a few hundred *vetted, diverse, real* correction examples have accrued, (c) a held-out eval shows a **measurable gap** between 9B and 31B.
- **Watch:** 9B↔31B agreement rate rising over time → plateau = stop.
- It's a **quality-and-coverage threshold with a measurable-gap gate, not a volume threshold.**

**Two clocks.** The synthetic adversarial harvest (§6) is *fast* — ~1,000 kept-failure pairs in a few days on idle GPU is realistic — and it's the right way to build the **cold-start** adapter. But it is **ceiling-bounded** at the 31B's grounded competence. Real-signal growth (genuine failures and user corrections from live use) accrues *slower*, at the speed of actual use, and it's the only clock that pushes **past** the teacher. Expect a fast first adapter from synthetic harvest, then slower, unbounded refinement from real use. And: generating 1,000 pairs trains *a* LoRA — the **real held-out eval-gate (§7), not the pair count, decides whether it ships.**

---

## 9. Guardrails against collapse (all non-negotiable)

- **Never** train on self-endorsed 9B output.
- **Rehearsal:** always mix a frozen anchor set (golden traces + a little general instruction data) so it doesn't forget base competence or over-specialise. **With heavy adversarial harvest (§6) this is critical:** include a baseline of *real, easy, passing* traces, or the model over-fits to hard cases and starts treating simple queries as traps. Discard passes from the *harvest*; retain a sample of real passes for *rehearsal*. Stress-test heavy, train balanced.
- **Fresh adapter on a growing corpus**, never serial fine-tuning of a fine-tune (compounds drift).
- The **eval-gate is the safety valve**; the frozen base is always the fallback.
- **Preserve diversity**; alarm on dataset collapse to the majority case.
- The **eval set must be real, never synthetic** — a synthetic eval closes the loop and blinds you (the model "improves" on the teacher's idea of good while regressing on reality).

---

## 10. What this buys (and its honest limits)

- **A durable, reusable skill** — the house style of operating Vinkona. The plateau is a **floor you keep, not a wall you hit**; the competence doesn't decay.
- **Format-fluency = the "smooth" feel.** Most local-assistant jank is friction at the seams (malformed tool calls, fumbled retrieval, paraphrasing a card, re-asking a known answer) — exactly the *format* failures a trace-trained LoRA sands down. **Felt improvement likely exceeds benchmark improvement**, because benchmarks measure the world-knowledge/reasoning you've externalised.
- **Specialisation, not a general boost.** The adapter is co-adapted to Vinkona's exact format and **won't transfer** — which is ideal for a single-user, single-system assistant. You trade generality you don't need for fluency you do.
- **Compounding (the second-order win that matters most over months):** cleaner 9B extractions → fewer reconciliation rejections → the graph accretes better/faster → retrieval has more to work with → every answer improves, *independent of the model*. Orchestration skill and knowledge base improve each other.
- **Limit:** ceiling = the 31B's grounded competence; only real signals exceed it.

---

## 11. Build order / sequencing

1. **Instrument capture (just logging) from day one.** Cheap; can precede everything.
2. Build + run the system with the format deliberately **fluid**; iterate freely.
3. **Freeze the format contract; version it (`format_v1`).** ← the real gate.
4. Begin capturing **training-grade** traces (post-freeze only).
5. **Cold-start:** SFT LoRA on 31B grounded-synthetic traces over the `§13` golden set.
6. Stand up **nightly curation** (label → 31B-gold-on-disagreements → build SFT/DPO/KTO sets → objective-filter → dedup → freeze real held-out).
7. **Adversarial synthetic loop** (§6): real seeds → perturb toward weak spots → run the 9B → keep only failures → grounded-31B correction; gap-steered, isolated to the training store.
8. **Train/gate/deploy cadence** + rollback infra + adapter-per-interaction logging.
9. Retrain on accrual; **stop at the agreement plateau.**

> Don't build the loop before the system has run and accumulated real divergence data — you'd be guessing at problems instead of harvesting them. Capture early; build the trainer once the data exists.

---

## Appendix: choices deliberately not taken
- **LoRA the knowledge** — wrong tool (low-rank can't hold the facts), defeats externalisation (kills attribution/edit/supersession/plasticity), and recalls *worse* than retrieval while hallucinating *more*. Knowledge stays in the graph.
- **Serial fine-tuning of the previous adapter** — drift compounds; retrain fresh on the accumulated set each time.
- **Training on raw daily volume** — most traffic is the model agreeing with itself; the correction count is the real denominator.
- **Synthetic eval set** — blinds the loop; the held-out eval must be real.
- **Ungrounded teacher self-talk** — circle-jerk; the supervisor must retrieve and reason over real structure.
- **Fixed retrain cadence** — retrain on vetted-example accrual against a measurable-gap gate; stop at plateau.

## Deferred / open
- Whether **user-correction** signal should be up-weighted in the loss relative to 31B-correction (likely yes — it's the only route past the teacher's ceiling).
- **Per-behaviour adapters** vs one combined adapter (start with one; split only if behaviours interfere).
- Exact **rank / target-modules** (start rank 16–32 all-linear; tune by eval, not by feel).
