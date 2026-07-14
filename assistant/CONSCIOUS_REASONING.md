# Conscious Reasoning in Vinkona

Three interconnected mechanisms make a small LM behave as though it genuinely
understands the user and learns from experience. All three are LIVE — this doc
describes what runs, not a plan. (Two earlier scaffolding modules,
`research_reflection.py` and `retrieval_confidence.py`, were removed 2026-07:
their intent is implemented by the corrections reviewer below and by the
knowledge-host's fit-gate respectively.)

## 1. User model (`user_model.py`)

**What it does:** Tracks the user's domain fluency, communication preferences,
and correction history, in memory.db alongside everything else.

**Why it matters:** A 9B model can't hold everything. But it CAN be given
explicit context about THIS user, which lets it make smarter decisions about
how much detail to include (expert vs. novice), what format to use, and how
much to trust its own reasoning in a domain where it's been corrected before.

**Key tables:**
- `user_domain_fluency` — what the user is expert/intermediate/novice in
- `user_corrections` — when the user said "actually..." (drives expertise updates)
- `user_interactions` — whether the user acted on advice
- `user_communication_pattern` — inferred style preferences

**Live paths:**
- **Capture** — idle reflection (`memory.idle_reflect`) reviews windows of past
  conversation and banks any corrections it spots via
  `user.record_correction(source_ref="idle_reflect")`. No extra LM call — it's
  a third output of the same reflection pass.
- **Injection** — the cascade passes `user_profile_hook` →
  `memory.user.get_user_context_for_lm()` into the LM bridge (cached per
  session). The block lands in the big LM's **briefing** and **deliberation**
  prompts only — the reasoning tier calibrates depth and format to the actual
  user; the fast voice prompt stays lean.

## 2. Corrections → research (the idle reviewer)

**What it does:** Turns banked corrections into durable behavioural knowledge.
A new `corrections` idle task (`memory.review_corrections`) reviews fresh
corrections once (watermark `corrections.review_watermark`), asks the big LM to
frame the generalizable patterns as research questions, and queues them under
session `corrections`. The normal research pipeline answers them, the card-hint
pass shapes each finding, and the knowledge host distills **case/procedure cue
cards** — so the next time the *situation* matches, kb_ask can say what to do
differently.

**Privacy:** the raw correction text never leaves memory.db. The framing prompt
(`DEFAULT_CORRECTIONS_PROMPT`) requires fully general, de-personalised
questions, and `safety.query_privacy` + `people.private_names` mechanically
mask anything that slips through before it reaches the queue.

**Config:** `research.idle.corrections_max` (default 2 per cycle),
`research.idle.corrections_prompt`, task gate `corrections` in
`research.idle.tasks`.

**The loop:**
```
User corrects → idle_reflect banks it → review_corrections generalizes it
  → research queue → sources gathered → card_hint shapes the finding
  → host distills a case/procedure card → kb_ask guidance next time
```

## 3. Retrieval confidence (host-side)

Calibrated trust lives where retrieval lives: the knowledge host's **fit-gate**
scores every candidate answer against the asked situation
(`context_features` vs. card discriminators) and **abstains on a clash** — so
"topically near" never silently becomes "wrong answer". Consequential
questions default to `rigor='high'` (source firewall + strength adjudication).
Nothing on the Vinkona side re-scores results; the confidence the host returns
is the confidence the bridge gates on (`knowledge_host.min_confidence`).

## How they work together

1. **User asks** → kb_ask retrieves, fit-gates, abstains-or-answers
2. **Big LM briefing** carries the user profile → direction calibrated to
   THIS user (depth to expertise, format to preference)
3. **User corrects** → next idle reflection banks it
4. **Idle reviewer** generalizes fresh corrections into research questions
5. **Research → cue cards** → the correction never needs making twice
6. Domain fluency shifts with corrections (`domain_expert` promotes,
   `factual_error` in a domain lowers trust in Vinkona's own reasoning there)

## Remaining wiring (small, tracked in USER_MODEL_INTEGRATION.md)

- [ ] `record_interaction()` on follow-up/action (base-rate signal)
- [ ] config_ui: User Profile tab (view/edit domain fluency, prefs)
- [ ] Retrieval ranking: use fluency to deprioritize basic results for experts
