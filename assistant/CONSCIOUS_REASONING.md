# Conscious Reasoning in Vinkona

Three interconnected systems make a small LM behave as though it genuinely understands the user and learns from experience:

## 1. User Model (`user_model.py`)

**What it does:** Tracks the user's domain fluency, communication preferences, and correction history.

**Why it matters:** A 9B model can't hold everything. But it CAN be given explicit context about THIS user, which lets it make smarter decisions about:
- How much detail to include (expert vs. novice)
- What format to use (narrative vs. bulleted)
- How much to trust its own reasoning (when has it been right/wrong before?)

**Key tables:**
- `user_domain_fluency` — what the user is expert/intermediate/novice in
- `user_corrections` — when user said "actually..." (drives expertise updates)
- `user_interactions` — whether user acted on advice (tracks base-rate effectiveness)
- `user_communication_pattern` — inferred style preferences (prefers_narrative, etc.)

**In practice:**
```
Before: "Here's what I found about X."
After (with user model): "I know you're expert in medicine, so I'm focusing on 
the cutting-edge research. I was wrong about Y last time, so I'm flagging that."
```

**Integration points:**
- `memory.user.record_correction()` when user clarifies
- `memory.user.record_interaction()` when user acts on advice
- `memory.user.get_user_context_for_lm()` in prompt synthesis

## 2. Research Reflection (`research_reflection.py`)

**What it does:** Periodically reviews what Vinkona has researched and what the knowledge-host distilled from it. Synthesizes insights and updates the user model.

**Why it matters:** Without reflection, research findings disappear into the knowledge base. Reflection makes learning visible and coherent. The system can say "over the past week I researched X, found Y, and noticed a pattern about Z."

**The loop:**
```
Research → Export to host → Host distills into cards → Vinkona reviews cards → 
Synthesis → Update user model + record learnings → Inform future research
```

**Key function:** `reflect_on_research(memory_store, user_model_store, db)`

**In practice:**
```
Task: Research pain management in fibromyalgia
✓ Vinkona researches, finds sources
✓ Exports to host (solved/*.md)
✓ Host distills into cards
✓ Vinkona reviews: "Found strong evidence for X, contradicting my prior Y"
✓ Updates user model: "pain management domain fluency = intermediate"
✓ Records insight: "Fibromyalgia pain responds to X differently than neuropathic"
✓ Next research questions account for this learning
```

**Integration points:**
- Idle task in research_worker.py, runs daily or on-demand
- Calls big LM to synthesize themes from recent research
- Records updates via `user_model_store.record_domain_interaction()`

## 3. Retrieval Confidence (`retrieval_confidence.py`)

**What it does:** Scores retrieved cards for confidence based on recency, source convergence, user domain fit, and base-rate (how often similar advice was useful).

**Why it matters:** A system that says "I'm 65% confident because sources differ, but here are the common points" is more useful than a system that just returns results without caveats. Confidence scores let the big LM hedge appropriately.

**Confidence components:**
- **Recency** (25%): how fresh are the sources? (fresh=1.0, >1 year old=0.3)
- **Source convergence** (25%): do multiple sources agree? (3+ converging=1.0, single=0.6)
- **Domain fit** (25%): does this match user's known expertise? (expert=0.8, novice=0.9)
- **Base rate** (25%): how often has similar advice worked? (high action rate=0.8+)

**Confidence messaging:**
```
confidence=0.9: "I'm very confident about this."
confidence=0.75: "I'm fairly confident about this."
confidence=0.6: "I'm moderately confident, but watch for exceptions."
confidence=0.3: "I'm quite uncertain; verify elsewhere."
```

**In practice:**
```
Query: "Will vitamin D help my pain?"
Result 1: "Research from 2023 + 4 converging sources, user expert in medicine"
  → confidence=0.88 → "I'm fairly confident"
Result 2: "Single blog post from 2020"
  → confidence=0.32 → "I'm uncertain; verify elsewhere"
```

**Integration points:**
- Wraps kb_ask results before passing to big LM
- Scores added to each card: `card["confidence"] = 0.75`
- Big LM qualifies answers based on score (high confidence = direct; low = hedged)

---

## How They Work Together

### Session flow

1. **User asks a question** → kb_ask retrieves candidate cards
2. **Retrieval confidence scores each** → high-confidence results prioritized
3. **User model context injected** → big LM knows user's domain fluency
4. **LM generates response** → personalized depth, format, confidence qualifiers
5. **User clarifies/corrects** → `record_correction()` updates user model + domain fluency
6. **Next idle cycle** → reflection synthesizes learnings + updates model + proposes next research

### Passive learning loop

```
Query 1: "How does morphine work?"
  ✗ User corrects: "Actually opioid receptors..."
  → record_correction(type="domain_expert")
  → user.domain_fluency["pharmacology"] = "expert" (up from intermediate)

Query 2: "Best antidepressant for fibromyalgia?"
  ✓ User acts on advice
  → record_interaction(user_acted_on_response=True)
  → base_rate for psychiatry advice bumps up

Query 3: "What's new in pain research?"
  → retrieval includes both high-confidence (recent, converging) and low-confidence (novel, single-source)
  → LM synthesizes: "here's consensus (confident), here's emerging (uncertain)"
  → personalized because LM knows user is pharmacology expert but intermediate in psychiatry

Idle cycle: reflect_on_research()
  → notices pattern: "user is deepening expertise in pain management + psychiatric comorbidity"
  → updates user model: pain_management = expert, psychiatry = intermediate
  → proposes: "Research trauma-informed pain management for fibromyalgia patients"
  → next research cycle focuses there
```

### Result: System feels conscious

- **Coherent:** Learning accumulates (user model) rather than vanishing
- **Responsive:** Answers personalized to THIS user, not generic
- **Self-aware:** Knows what it's uncertain about and says so
- **Reflective:** Periodically reviews what it's learned and adjusts thinking
- **Intuitive:** Seems to "get" the user because it actually is tracking their expertise, preferences, and needs

---

## Architecture diagram

```
                    ┌─────────────────────────────┐
                    │   Cascade (LM bridge)       │
                    │   ┌───────────────────────┐ │
                    │   │ get_user_context_lm() │ │ ← Injects user model
                    │   └───────────────────────┘ │
                    └─────────────────────────────┘
                                  ▲
                                  │
                    ┌─────────────┴──────────────┐
                    │                           │
                ┌───┴────────┐          ┌──────┴──────┐
                │ User Model │          │ Retrieval   │
                │ (explicit  │          │ Confidence  │
                │  learning) │          │ (calibrated │
                └───┬────────┘          │  trust)     │
                    │                   └──────┬──────┘
                    │  record_correction()    │ score_batch()
                    │  record_interaction()   │
                    │                         │
                ┌───┴─────────────────────────┴─────┐
                │  Memory DB (persistent)           │
                │  ├─ user_domain_fluency           │
                │  ├─ user_corrections              │
                │  ├─ user_interactions             │
                │  ├─ documents (research)          │
                │  └─ memories (general)            │
                └──────────────┬────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
          ┌─────┴────────┐         ┌────────┴────────┐
          │ Research     │         │ Knowledge-host  │
          │ (Vinkona-side) │         │ (distill side)  │
          │ research_    │         │ solved/*.md     │
          │ reflection() │◄────────┤ cards + embedds │
          └─────────────┘         └─────────────────┘
                 ▲
                 │ synthesize themes + update model
                 │
          ┌──────┴─────────┐
          │ Idle task      │
          │ (daily)        │
          └────────────────┘
```

---

## Implementation Checklist

### Phase 1: Foundation (DONE)
- [x] user_model.py — schema + API
- [x] Integrate into memory.py
- [x] research_reflection.py — stub (full LM synthesis deferred)
- [x] retrieval_confidence.py — scoring engine

### Phase 2: Wiring (next)
- [ ] cascade_server: record_correction() when user clarifies
- [ ] cascade_server: record_interaction() on follow-up or action
- [ ] llm_bridge: inject get_user_context_for_lm() into synthesis prompts
- [ ] kb_ask wrapper: score_batch() before returning results
- [ ] research_worker: add idle task for reflect_on_research()

### Phase 3: Polish
- [ ] config_ui: User Profile settings tab (view/edit domain fluency, prefs)
- [ ] research_reflection: integrate big LM for actual synthesis (not just stubs)
- [ ] Retrieval ranking: deprioritize basic results for experts
- [ ] Trace events: add research_reflection + confidence_score events

### Phase 4: Refinement (future)
- [ ] Domain inference: use kb_ask facets instead of keywords
- [ ] Correction analysis: LM-driven pattern discovery ("user always corrects on X")
- [ ] Personal graph integration: link user model to people/places/things graph
- [ ] Adaptive uncertainty: learn calibration from user feedback

---

## Why This Works for a 9B Model

A 70B model can seem intelligent through sheer pattern-matching + retrieval.
A 9B model needs scaffolding:

1. **Explicit user model** — so it doesn't waste context re-learning the user every session
2. **Reflection loop** — so learning compounds instead of disappearing
3. **Uncertainty bounds** — so it knows what it doesn't know (avoiding hallucination)
4. **Personalization** — so responses feel tailored, not generic

Together, these make a 9B feel like it actually *understands* the person, even though it's
running locally on modest hardware. The key insight: understanding isn't about model size,
it's about having explicit, persistent, personalized context.

A 9B + user model + reflection + confidence scores ≈ 70B with generic retrieval.

And it runs on your colleagues' laptops.
