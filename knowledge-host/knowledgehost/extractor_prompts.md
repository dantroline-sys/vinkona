# Extractor Prompts (companion to the KB spec)

These are the offline LM prompts the distillation worker runs. The hard one is the **causal-edge extractor**, because `mechanism` and `modifiers` are what make diagnostic "why" work — get them vague and every differential collapses into mush.

---

## 0. How to run these (engineering wrapper)

- **Temperature 0.0–0.2.** Extraction is deterministic work, not creative.
- **Constrain the output grammar — do not trust the prompt alone.** Use grammar-constrained decoding against the JSON schema: GBNF (llama.cpp), or `xgrammar`/`outlines` (vLLM). This *guarantees* parseable JSON. Keep a JSON-repair fallback only if you can't constrain.
- **Provenance is attached by the worker, not the model.** The model never knows `doc_id`. It emits a short `evidence` span (verbatim, ≤25 words) for verification; the worker stamps `doc_id/section/chunk_id`.
- **Node linking is downstream.** The model proposes a canonical `label` + `aliases` + `kind` for each concept; the worker's `link_to_node` resolves to an existing node or creates one. Instruct the model to normalise labels (lowercase, singular, no articles) so linking is stable.
- **Empty is valid.** If a chunk has nothing of the target type, return `[]`. Do not invent.
- **One call can yield many items.** Output is always a JSON array.

### The shared feature vocabulary (the thing that makes diagnosis work)
The causal extractor's `discriminators` and the query-side context extractor (§7) **must draw from the same feature vocabulary**, or `score_context_fit` compares apples to oranges. Maintain a `feature_vocabulary` table; inject the current top-N feature names into both prompts so the model reuses them instead of inventing synonyms. Seed cross-domain core:

```
onset (sudden|gradual|delayed_<t>)      laterality (unilateral|bilateral|focal|diffuse)
timing (immediate|delayed|episodic)     quality (burning|sharp|dull|gritty|itchy|aching|...)
severity (mild|moderate|severe)         trigger (what brings it on)
relieved_by / aggravated_by             associated (co-occurring features)
location (anatomical/spatial)           reversibility (transient|persistent)
dose/threshold (exposure needed)        population/context (who/when it applies)
```
Each discriminator is a `{feature, value}` pair so fit-scoring is feature-overlap, not fuzzy text match. The vocabulary is **extensible** — allow new `feature` names, but encourage reuse of existing ones.

---

## 1. Causal-edge extractor (the centrepiece)

**System prompt:**
```
You extract CAUSAL relationships from a text passage into structured JSON for a
knowledge base used to explain and diagnose. You do NOT summarise prose. You output
a JSON array only — no markdown, no commentary. Return [] if the passage states no
causal relationship.

A causal edge is: CAUSE --[mechanism, under conditions]--> EFFECT.

For each causal relationship explicitly stated or directly supported by the passage,
emit one object:

{
  "cause":  {"label": <canonical noun phrase>, "kind": <entity|phenomenon|condition|action|agent>, "aliases": [<surface forms in text>]},
  "effect": {"label": <canonical noun phrase>, "kind": ..., "aliases": [...]},
  "polarity": "causes" | "prevents" | "exacerbates" | "reduces",
  "mechanism": <the INTERMEDIATE chain by which cause produces effect — the actual
                physiological/physical/logical steps, NOT a restatement of "X causes Y".
                If the passage gives no mechanism, write the best-supported chain and
                set mechanism_basis to "inferred"; never leave a bare restatement.>,
  "mechanism_basis": "stated" | "inferred",
  "modifiers": {
     "conditions": [<what must hold for this cause to produce this effect>],
     "discriminators": [{"feature": <from shared vocabulary>, "value": <...>}],
        // discriminators = how THIS cause's presentation differs from other causes of
        // the same effect. This is the field diagnosis depends on. Be specific:
        // onset, laterality, quality, timing, associated features, threshold.
     "threshold": <exposure/dose needed, or null>,
     "context": <population/setting where it applies, or null>
  },
  "strength": <0.0–1.0: how strongly the passage asserts AND how established it is.
               Stated mechanism + strong claim → high; mere association → low>,
  "evidence": <verbatim supporting span, ≤25 words>
}

RULES:
- Mechanism must EXPLAIN, not restate. "wind causes dry eye because wind causes dryness"
  is rejected; "wind accelerates evaporation of the tear film, thinning it until the
  ocular surface is exposed" is correct.
- Discriminators must distinguish this cause from siblings sharing the same effect.
  If the passage contrasts several causes of one effect, the contrasts ARE the
  discriminators — capture them.
- Distinguish causation from correlation. If the passage only reports association,
  set mechanism_basis "inferred" and lower strength.
- Normalise labels: lowercase, singular, no articles. Keep concepts atomic (one
  phenomenon per node), not whole sentences.
- Reuse feature names from the provided vocabulary wherever they fit.
```

**Few-shot example A — input passage:**
```
Exposure to wind, especially in low-humidity conditions, accelerates evaporation of
the tear film, producing a burning, bilateral discomfort typical of evaporative dry
eye. Wind-blown sand, by contrast, abrades the corneal epithelium, giving a sudden,
sharp, usually unilateral foreign-body sensation. Ultraviolet light reflected off
sand and water can cause photokeratitis, whose symptoms are characteristically
delayed by 6 to 12 hours and present as severe, bilateral pain with photophobia.
```

**Expected output A:**
```json
[
  {
    "cause": {"label": "wind exposure", "kind": "phenomenon", "aliases": ["wind", "windy conditions"]},
    "effect": {"label": "evaporative dry eye", "kind": "condition", "aliases": ["dry eye"]},
    "polarity": "causes",
    "mechanism": "wind accelerates evaporation of the tear film; with insufficient tear replacement the film thins and the ocular surface is exposed, irritating surface nerves",
    "mechanism_basis": "stated",
    "modifiers": {
      "conditions": ["low ambient humidity", "prolonged exposure"],
      "discriminators": [
        {"feature": "onset", "value": "gradual"},
        {"feature": "laterality", "value": "bilateral"},
        {"feature": "quality", "value": "burning"}
      ],
      "threshold": null,
      "context": "outdoor exposure"
    },
    "strength": 0.9,
    "evidence": "wind ... accelerates evaporation of the tear film, producing a burning, bilateral discomfort"
  },
  {
    "cause": {"label": "wind-blown sand", "kind": "agent", "aliases": ["sand", "airborne particulate"]},
    "effect": {"label": "corneal abrasion", "kind": "condition", "aliases": ["foreign-body sensation"]},
    "polarity": "causes",
    "mechanism": "solid particulate strikes and mechanically abrades the corneal epithelium, exposing subepithelial nerve endings",
    "mechanism_basis": "stated",
    "modifiers": {
      "conditions": ["airborne particulate present", "no eye protection"],
      "discriminators": [
        {"feature": "onset", "value": "sudden"},
        {"feature": "laterality", "value": "unilateral"},
        {"feature": "quality", "value": "sharp/gritty foreign-body"}
      ],
      "threshold": null,
      "context": "windy beach/desert"
    },
    "strength": 0.9,
    "evidence": "wind-blown sand ... abrades the corneal epithelium, giving a sudden, sharp, usually unilateral foreign-body sensation"
  },
  {
    "cause": {"label": "reflected ultraviolet radiation", "kind": "phenomenon", "aliases": ["UV", "glare off sand and water"]},
    "effect": {"label": "photokeratitis", "kind": "condition", "aliases": []},
    "polarity": "causes",
    "mechanism": "UV photochemically injures the corneal epithelium; the inflammatory/pain response develops only after a latent period as damaged cells signal",
    "mechanism_basis": "inferred",
    "modifiers": {
      "conditions": ["reflective surfaces (sand, water)", "sun exposure without UV protection"],
      "discriminators": [
        {"feature": "onset", "value": "delayed_6-12h"},
        {"feature": "laterality", "value": "bilateral"},
        {"feature": "severity", "value": "severe"},
        {"feature": "associated", "value": "photophobia"}
      ],
      "threshold": "prolonged UV exposure",
      "context": "bright outdoor/reflective environments"
    },
    "strength": 0.85,
    "evidence": "ultraviolet light reflected off sand and water can cause photokeratitis ... delayed by 6 to 12 hours"
  }
]
```

**Few-shot example B — different domain, shows `prevents` polarity + functional link:**
```
Forcing a saw through the cut makes the blade wander off the line: lateral pressure
deflects the thin blade. Letting the saw's own weight do the work keeps the kerf
straight.
```
```json
[
  {
    "cause": {"label": "lateral pressure on saw blade", "kind": "action", "aliases": ["forcing the saw"]},
    "effect": {"label": "blade wander", "kind": "phenomenon", "aliases": ["cut deviating off the line"]},
    "polarity": "causes",
    "mechanism": "lateral force deflects the thin, low-stiffness blade sideways, so the kerf follows the deflection rather than the marked line",
    "mechanism_basis": "stated",
    "modifiers": {
      "conditions": ["thin/flexible blade", "operator applying sideways force"],
      "discriminators": [{"feature": "trigger", "value": "pushing/forcing"}],
      "threshold": null,
      "context": "hand sawing"
    },
    "strength": 0.85,
    "evidence": "Forcing a saw through the cut makes the blade wander ... lateral pressure deflects the thin blade"
  },
  {
    "cause": {"label": "letting saw weight do the work", "kind": "action", "aliases": ["light touch sawing"]},
    "effect": {"label": "blade wander", "kind": "phenomenon", "aliases": []},
    "polarity": "prevents",
    "mechanism": "removing lateral force lets the blade track its own kerf, so no sideways deflection accumulates",
    "mechanism_basis": "inferred",
    "modifiers": {
      "conditions": ["minimal downward/sideways force"],
      "discriminators": [{"feature": "trigger", "value": "light, unforced strokes"}],
      "threshold": null,
      "context": "hand sawing"
    },
    "strength": 0.8,
    "evidence": "Letting the saw's own weight do the work keeps the kerf straight"
  }
]
```

---

## 2. Procedure-card extractor

**System prompt:**
```
You convert a passage describing HOW to do something into a structured procedure card
as JSON. Output a single JSON object, or [] if the passage describes no procedure.
No prose, no markdown. Strip all authorial voice; steps must be imperative.

{
  "node": {"label": <canonical procedure name, lowercase>, "kind": "procedure", "aliases": [...]},
  "domain": <field>,
  "goal": <what completing it achieves>,
  "preconditions": [<state/conditions required before starting>],
  "tools": [...], "materials": [...],
  "steps": [{"n": 1, "action": <imperative instruction>, "rationale": <why this step / what it prevents>}],
  "tips": [...], "mistakes": [<common errors + their consequence>], "safety": [...],
  "relations": [{"type": "part_of"|"alternative_to"|"requires"|"prerequisite_of", "label": <other procedure/concept>}],
  "evidence": <verbatim span, ≤25 words>
}

RULES:
- Every step imperative ("Clamp the workpiece"), each with a one-line rationale.
- Capture parameters inline in the action ("set the fence to 5–7 TPI"), so "how much"
  questions resolve from step text.
- Merge multi-source variants is NOT your job; emit one card for this passage. The
  worker handles cross-source merge.
```

---

## 3. Typed-edge extractor (non-causal relations: who / where / taxonomy / function)

**System prompt:**
```
You extract NON-CAUSAL relationships between concepts into JSON edges. Output a JSON
array; [] if none. No prose. (Causal relationships are handled separately — ignore them.)

Edge families and types:
  taxonomic:  is_a, instance_of, subtype_of
  meronymic:  part_of, has_part, component_of
  spatial:    located_in, adjacent_to, contains
  epistemic:  described_by, authored_by, taught_by, cites      (the semantic "who")
  temporal:   precedes, follows, concurrent_with               (non-procedural "when")
  functional: treats, used_for, requires, produces             (links conditions ↔ procedures)

{
  "src": {"label": ..., "kind": ..., "aliases": [...]},
  "dst": {"label": ..., "kind": ..., "aliases": [...]},
  "family": <one of above>,
  "type": <specific type>,
  "modifiers": {"conditions": [...], "context": <or null>},
  "strength": <0.0–1.0>,
  "evidence": <verbatim span, ≤25 words>
}

RULES:
- Normalise labels (lowercase, singular, no articles).
- Direction matters: "the cornea is part of the eye" → src=cornea, type=part_of, dst=eye.
- functional edges are how a "why" (condition) connects to a "how" (procedure):
  "irrigation treats chemical eye exposure" → treats(irrigation → chemical eye exposure).
```

---

## 4. Content-router labeller (bootstrap only)

Used to label a sample so you can train the cheap multi-label classifier — not run online.
```
Label this passage with every category that applies. Output JSON only:
{"labels": [<"procedural"|"causal"|"relational"|"declarative">], "rationale": <≤15 words>}

procedural  = tells you HOW to do something (steps/method)
causal      = states WHY: cause→effect, mechanism, condition→result
relational  = non-causal relations between concepts (is-a, part-of, located-in, authored-by)
declarative = defines/describes WHAT something is, no procedure/cause/relation of note

A passage may carry several labels. "declarative" alone → no distillation needed.
```

---

## 5. Retrieval-surface generator (questions + propositions)

Run per card / node / edge after distillation. Builds the text users' queries land on.
```
Given this knowledge item, generate the questions it answers and the atomic facts it
asserts, so casual user queries can retrieve it. Output JSON only:
{
  "questions": [<5–10 ways a user might ask this, varied phrasing & specificity>],
  "propositions": [<3–8 atomic, self-contained factual statements>]
}

For CAUSAL items, include diagnostic phrasings:
  "why would <cause> cause <effect>?", "what makes <effect> happen after <cause>?",
  "<effect> after <context>, why?", "could <cause> be why I have <effect>?"
For PROCEDURE items, include "how do I <goal>?", "what's the right way to <action>?",
  and parameter questions ("what <tool/setting> for <task>?").
```

---

## 6. (Reference) Worker integration

```
edges = grammar_constrained_generate(causal_prompt, chunk.text, schema=CAUSAL_SCHEMA, temp=0.1)
for e in edges:
    e.cause = link_to_node(e.cause); e.effect = link_to_node(e.effect)   # resolve/create
    register_features(e.modifiers.discriminators)                         # grow vocabulary
    e.provenance = {doc_id, section, chunk_id}                            # worker stamps
    h = hash(canonical(e))
    if exists(h): continue
    m = hnsw_edges.search(e.embedding, k=1)
    if m and sim(m,e) > MERGE_THRESH: merge_edge(m, e)   # append provenance, reconcile modifiers
    else: insert_edge(e)
```

---

## 7. Query-side: intent + context extractor (mirrors the causal vocabulary)

This pairs with `diagnostic_why`. It MUST use the same feature vocabulary as §1's discriminators.
```
Classify the user's question and, if it is a "why/what-caused/what-could-this-be"
question, extract the effect and the context features. Output JSON only:
{
  "intent": "what"|"how"|"why_mechanistic"|"why_diagnostic"|"what_if"|"who"|"where"|"taxonomy"|"which"|"what_else",
  "effect":  {"label": ..., "aliases": [...]}  | null,   // for diagnostic/mechanistic
  "cause":   {"label": ..., "aliases": [...]}  | null,   // for mechanistic/what_if
  "context_features": [{"feature": <shared vocabulary>, "value": ...}]   // for diagnostic
}

Example — "why are my eyes sore after a windy day at the beach?"
{
  "intent": "why_diagnostic",
  "effect": {"label": "ocular surface irritation", "aliases": ["sore eyes"]},
  "cause": null,
  "context_features": [
    {"feature": "trigger", "value": "wind"},
    {"feature": "trigger", "value": "sand"},
    {"feature": "context", "value": "beach"},
    {"feature": "context", "value": "sun/UV exposure"}
  ]
}
```
`diagnostic_why` then pulls every incoming causal edge to `effect`, and `score_context_fit`
compares these `context_features` against each edge's `discriminators` — same vocabulary,
so the comparison is a clean feature overlap. The ranked result is your differential.

---

## 8. Narrative / interpretive extractor (fiction-regime)

The fiction-regime counterpart to §1. The content router sends literary/fictional chunks here. Its job is **not** to assert facts but to *sort* a passage into regime-tagged items that fan out into the substrates: diegetic facts (scope=work), implied psychological causality, reusable social conventions (regime=conventional), and character beliefs (firewalled, scope=character). Same engineering wrapper as the others (temp 0–0.2, grammar-constrained JSON, evidence spans ≤25 words). The eight analysis questions map one-to-one onto the output arrays.

**System prompt:**
```
You analyse a passage of fiction/narrative prose and output STRUCTURED JSON ONLY — no
prose, no markdown. You do not assert anything about the real world. Every item you emit
is tagged with a REGIME and a SCOPE so that downstream reasoning never confuses a
character's belief with a fact, or one story's events with general truth.

REGIMES: fictional (true within this work) | conventional (a reusable social/behavioural
pattern) | interpretive (a belief/attitude/judgement held by someone).
SCOPES:  "work" (this story) | "character:<name>" | "general".

Reconstruct implied/off-page content where the text licenses it, but set basis="inferred".

Output this object (omit empty arrays):
{
  "entities":   [{"label","kind","aliases":[],"role"}],                 // Q5: who/what appears
  "relations":  [{"src","type","dst","scope":"work"}],                  // Q5: is_a/part_of/son_of/attends/...
  "diegetic_causal": [                                                   // Q2: stated or implied cause→effect IN the story
     {"cause","effect","mechanism","basis":"stated|inferred","scope":"work","evidence"}],
  "character_states": [                                                  // Q4: feelings/motives/self-image revealed
     {"holder","state","trigger","reveals","evidence"}],
  "beliefs": [                                                           // Q1+Q6: attitudes/judgements — FIREWALLED
     {"holder","belief","regime":"interpretive","scope":"character:<holder>",
      "narrative_stance":"endorsed|undercut|neutral|channelled","evidence"}],
  "conventions": [                                                       // Q3: social codes assumed operative
     {"pattern","domain","regime":"conventional","evidence"}],
  "general_patterns": [                                                  // Q7: the reusable payload
     {"instance","generalisation","regime":"conventional","evidence"}],
  "setting": {"inference","evidence","confidence":0.0_to_1.0}           // Q8: inferred context + how sure
}

RULES:
- NEVER emit a character's belief, judgement, or perception as an entity/relation/fact.
  "Hindu gods squabble" is a belief held by a character → beliefs[], scope=character,
  never a relation or world-fact. This rule is the firewall; treat it as inviolable.
- conventions[] and general_patterns[] are the only items meant to generalise beyond the
  work; phrase them so they're reusable (not tied to these characters' names).
- diegetic_causal mechanism must explain, not restate; mark reconstructed off-page causes
  basis="inferred".
- narrative_stance: is the belief endorsed by the narration, undercut/ironised, or merely
  channelled through a POV? (Convert-psychology channelled ≠ authorial endorsement.)
```

**Few-shot — input passage:**
```
She blushed more deeply at the reference to Father Olivier. She did not know how to reply.
It was true that she liked the young French priest, with his funny accent and blue eyes, but
she had been going to the Catholic chapel for months before he arrived. She loved its cheerful
hymns, and its simple creed of one god - so different from those miserable Hindu gods who
squabbled with each other like her sisters at home. But, added to that, the vulgarity of Ravi's
remark bewildered her because his family were known for their breeding. People always said that
Ravi would be a man of honour, like his father.
```

**Expected output:**
```json
{
  "entities": [
    {"label": "the protagonist", "kind": "character", "aliases": ["she"], "role": "Hindu-background woman drawn to the Catholic chapel"},
    {"label": "Father Olivier", "kind": "character", "aliases": ["the young French priest"], "role": "French Catholic priest"},
    {"label": "Ravi", "kind": "character", "aliases": [], "role": "man from a high-status family"},
    {"label": "the Catholic chapel", "kind": "place", "aliases": [], "role": "place of worship she attends"}
  ],
  "relations": [
    {"src": "the protagonist", "type": "attends", "dst": "the Catholic chapel", "scope": "work"},
    {"src": "Ravi", "type": "son_of", "dst": "Ravi's father", "scope": "work"},
    {"src": "Ravi", "type": "member_of", "dst": "a family known for breeding", "scope": "work"}
  ],
  "diegetic_causal": [
    {"cause": "Ravi's remark referencing Father Olivier", "effect": "the protagonist blushes deeply",
     "mechanism": "the insinuation lands because she does like the priest; partial truth + exposure produces embarrassment",
     "basis": "stated", "scope": "work", "evidence": "She blushed more deeply at the reference to Father Olivier"},
    {"cause": "vulgarity from a man whose family is known for breeding", "effect": "the protagonist's bewilderment",
     "mechanism": "expectation violation: crude speech contradicts the high conduct his lineage led her to expect",
     "basis": "stated", "scope": "work", "evidence": "the vulgarity of Ravi's remark bewildered her because his family were known for their breeding"},
    {"cause": "Ravi insinuated her chapel attendance is really about the priest", "effect": "her blush and defensive recall that she attended before he arrived",
     "mechanism": "reconstructed off-page remark that the passage responds to",
     "basis": "inferred", "scope": "work", "evidence": "she had been going to the Catholic chapel for months before he arrived"}
  ],
  "character_states": [
    {"holder": "the protagonist", "state": "embarrassment", "trigger": "the reference to Father Olivier",
     "reveals": "she does like him and is self-conscious that it shows", "evidence": "She blushed more deeply ... She did not know how to reply"},
    {"holder": "the protagonist", "state": "defensiveness about her faith's sincerity", "trigger": "the implied insinuation",
     "reveals": "she pre-empts the accusation by stressing prior, independent devotion", "evidence": "going to the Catholic chapel for months before he arrived"}
  ],
  "beliefs": [
    {"holder": "the protagonist", "belief": "Hindu gods are petty and quarrelsome", "regime": "interpretive",
     "scope": "character:the protagonist", "narrative_stance": "channelled",
     "evidence": "those miserable Hindu gods who squabbled with each other like her sisters"},
    {"holder": "the protagonist", "belief": "a simple monotheistic creed is preferable", "regime": "interpretive",
     "scope": "character:the protagonist", "narrative_stance": "channelled",
     "evidence": "its simple creed of one god"}
  ],
  "conventions": [
    {"pattern": "a deep blush signals concealed attraction or embarrassment", "domain": "social/emotional", "regime": "conventional",
     "evidence": "She blushed more deeply at the reference to Father Olivier"},
    {"pattern": "vulgar speech is read as a breach of good breeding", "domain": "social/honour", "regime": "conventional",
     "evidence": "the vulgarity of Ravi's remark bewildered her because his family were known for their breeding"}
  ],
  "general_patterns": [
    {"instance": "Ravi expected to be honourable because his father is", "generalisation": "honour cultures presume character is inherited from the father", "regime": "conventional",
     "evidence": "Ravi would be a man of honour, like his father"},
    {"instance": "she stresses she attended the chapel before the priest arrived", "generalisation": "people defend against an accusation of mixed motives by citing a prior, independent motivation", "regime": "conventional",
     "evidence": "going to the Catholic chapel for months before he arrived"}
  ],
  "setting": {
    "inference": "a French-Catholic mission context within Hindu India (e.g. French colonial enclave such as Pondicherry)",
    "evidence": "a French priest, a Hindu protagonist, the names Ravi, an honour/breeding discourse",
    "confidence": 0.55
  }
}
```

**Routing of these outputs (where reconciliation sends them):**
- `conventions` + `general_patterns` → **conventional** regime; corroborate across many works (lots of stories showing "a blush conceals feeling" strengthen the pattern).
- `entities`/`relations`/`diegetic_causal` → **fictional** regime, scope=work; never corroborate or contradict anything outside this story; queryable as "in ⟨work⟩, …".
- `beliefs` → **interpretive**, scope=character; firewalled — never enter empirical corroboration; surfaced as "the character believes …", with `narrative_stance` qualifying it.
- `setting` → fictional, scope=work, carried with its confidence.

**Query-side pairing:** add `world`/`work` scoping to the §7 intent extractor for fictional queries, so "why did she blush?" retrieves only within the work, while "what does a blush signify?" retrieves the corroborated conventional pattern across works.
