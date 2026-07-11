# VINKONA-DMAP-01 — Drug-Source → Conflict / Mechanism / Indication Import Mapping

**Status:** Draft for implementation · **Doc version:** 1.0 · **Date:** 2026-07-08
**Component:** `vinkona-dmap import` — offline, write-time. Consumes RxClass/MED-RT/FDASPL class data and the ONC High-Priority DDI list; emits nodes and edges consumed by VINKONA-CONF-01 (conflict checker) and the retrieval graph.
**Depends on:** VINKONA-CONF-01 (edge/mechanism schema, with the §4 scope amendment below), VINKONA-LEX-01 (for resolving source disease/drug strings to VINKONA nodes).

RFC 2119 keywords are normative.

---

## 1. Purpose and the central distinction

RxClass exposes several relationship families. They do **not** all mean the same thing, and the single most important rule of this import is to route each family to the right place:

- **Class / grouper relations** (ATC, MeSH-PA, VA class, FDASPL `has_EPC`) → the **is_a-style hierarchy**. These are the scaffolding the CONF-01 ancestor-walk climbs; they are **not** conflicts.
- **Mechanism relations** (MED-RT/FDASPL `has_MoA`, `has_PE`, `has_PK`) → **mechanism nodes + `acts_via` edges**. These populate the "why" substrate CONF-01 routes conflicts through; they **never fire** anything themselves.
- **Contraindication relation** (MED-RT `CI_with`) → the **one** genuine `conflict_edge` family this source yields (drug × *condition*).
- **Indication relations** (MED-RT `may_treat`, `may_prevent`) → **indication edges** on the action's *upstream* side; **not** conflicts.
- **Drug × drug interactions** are **not in RxClass** (they left with the discontinued NLM interaction API). They come from the **ONC High-Priority DDI list** (§5) and map to a *separate* `conflict_edge` shape.

Getting this split right is the whole point: if the importer treats `has_MoA` as a conflict, the checker fires on mechanism membership; if it treats `CI_with` as a class, contraindications never fire. Keep them in their lanes.

## 2. Master mapping table

| Source relation (RxClass rela / source) | Semantic | VINKONA target | Edge type / `relation_type` | `fire_when` shape | Fires? |
|---|---|---|---|---|---|
| ATC (ATCPROD), MeSH-PA, VA class, FDASPL `has_EPC` | class membership | class node + membership edge | `member_of` (hierarchy) | — | no (scaffold) |
| MED-RT / FDASPL `has_MoA` | mechanism of action | mechanism node + link | `acts_via` | — | no (substrate) |
| MED-RT / FDASPL `has_PE` | physiologic effect | mechanism node + link | `acts_via` (effect role) | — | no (substrate) |
| MED-RT / FDASPL `has_PK` | pharmacokinetics | pathway node + link | `has_pk` | — | no (future PK-interaction substrate) |
| **MED-RT `CI_with`** | **contraindicated with condition** | **`conflict_edge`** | **`contraindicated`** | **`presence(condition:X)`** | **yes (once ratified)** |
| MED-RT `may_treat` | indication | indication edge (upstream of action) | `indicated_for` | — | no (indication) |
| MED-RT `may_prevent` | prophylaxis | indication edge | `prevents` | — | no (indication) |
| **ONC-High DDI pair** (not RxClass) | **drug–drug interaction** | **`conflict_edge` ×2 (symmetric)** | **`contraindicated`** or **`antagonizes`** | **`presence(substance:B)`** | **yes (once ratified)** |

Everything in the "no" rows is imported so the graph is *richer* and so conflicts have a hierarchy to inherit along and a mechanism to cite — but only the two bold rows produce firing safety edges, and both enter as `status: proposed` (§3).

## 3. Import rules (normative)

**R1 — Everything auto-imported that could fire is `status: proposed`.** Per CONF-01 §5.1, a proposed edge surfaces as `flag_for_human`/`unratified_rule` and can never `fire` or clear. Bulk import therefore yields thousands of *reviewable* contraindication/interaction edges, none of which fire until a human (optionally model-assisted) ratifies them. This is the safety invariant of the whole system applied to import: **no machine-minted safety edge is ever authoritative on arrival.**

**R2 — Subject is the substance, not the product or the action.** A `CI_with` or DDI edge is a property of the *ingredient/substance*, so its `conflict_edge.subject` MUST be the VINKONA **substance node** (e.g. `substance:adrenaline`), never a specific trade product or a single administration action. One edge per substance-contraindication then covers every product and route via the §4 scope extension — no duplication across brands.

**R3 — Provenance and authority are mandatory.** Every imported edge sets `source_ref` to `<source>/<rela>/<version>` (e.g. `MED-RT/CI_with/2026AA`, `ONC-HPDDI/2024`) and `authority = 'pub'` (RxNorm/RxClass/MED-RT/ONC are US-public and broadly usable). `rationale` is templated from the source (e.g. "MED-RT: adrenaline contraindicated with narrow-angle glaucoma").

**R4 — Severity is provisional until ratification.** MED-RT `CI_with` carries no severity → default `caution`. ONC-High is a high-priority list by definition → default `severe`. Because proposed edges do not fire, this default is only a triage hint; the **ratifying reviewer assigns the authoritative severity**, which is what the checker will use once `status` becomes `ratified`.

**R5 — Condition/drug resolution failures go to the research queue, not to /dev/null.** A `CI_with` names a MED-RT disease; it MUST be resolved (via LEX-01 + a crosswalk) to an VINKONA condition node (ICD-11 / SNOMED / Wikidata, per your authority scheme). If the drug or the condition cannot be resolved, the importer MUST NOT drop the row — it emits an VINKONA-CONF-02 research item (`unresolved-mapping`) carrying the source triple and the failed side, so coverage gaps are visible rather than silent. (Same never-event logic as the checker: a dropped contraindication is invisible harm.)

**R6 — Deterministic dedup by authority prefix.** If a substance/condition already exists from a higher-authority source (`pub:` SNOMED/ICD-11/Wikidata), the import attaches to the existing node rather than minting a duplicate; the RxNorm/MED-RT identifier is retained as an alias/xref on that node. Same-source re-import is idempotent (keyed by `(subject, relation_type, fire_when-signature, source_ref)`).

## 4. Node/edge model additions and the CONF-01 scope amendment

The recast introduces (or makes explicit) these node/edge types alongside CONF-01's `conflict_edge`/`mechanism`/`conflict_override`:

```
node kinds:   substance | product | action | class | mechanism | condition | pathway
edge kinds:   member_of(child → class)            -- ATC/MeSH-PA/VA/EPC hierarchy
              acts_via(substance → mechanism, role:'moa'|'pe')   -- mechanism routing substrate
              has_pk(substance → pathway)          -- future PK-interaction substrate
              administers(action → substance)      -- links an administration action to its ingredient(s)
              indicated_for(action → condition)    -- may_treat (upstream indication side)
              prevents(action → condition)         -- may_prevent
```

**Mechanism nodes are a typed subset of class nodes.** A `has_MoA`/`has_PE` target is both a grouper (drugs sharing it) and a mechanism descriptor. Represent it once as a node carrying mechanism metadata (`label`, `explanation`, `conditionality_class`) so it can be **cited as `mechanism_id`** by a `conflict_edge` *and* act as a class for scope. Import sets `conditionality_class` from a small MoA/PE→class lookup where known (e.g. receptor-competition MoAs → `acute_competition`; enzyme-inhibition → `steady_state`), else `none`.

**CONF-01 §7.1 K1 scope amendment (required for this import to work).** The checker's candidate-gather MUST extend the subject scope from is_a-only to include the substance(s) an action administers, and their class ancestors:

```
scope(action A) = {A}
               ∪ isa_ancestors(A)
               ∪ administered_substances(A)                       -- REQUIRED
               ∪ ⋃_{s ∈ administered_substances(A)} isa_ancestors(s)   -- REQUIRED
               ∪ ⋃_{s} member_of-groupers(s) of type {ATC, EPC}   -- RECOMMENDED
               ∪ ⋃_{s} member_of-groupers(s) of type {MoA, PE}    -- OPTIONAL (see caution)
```

Rationale: a contraindication asserted on `substance:adrenaline` must reach `act:administer_adrenaline` (which is *not* an is_a descendant of the substance — it's linked by `administers`). ATC/EPC groupers are safe to include (a class-level contraindication is usually valid for members). **MoA/PE grouper inclusion is optional and off by default**: sharing a mechanism is a weaker basis for inheriting a contraindication and risks false positives (alarm fatigue) — enable only for specific curated mechanism classes, never blanket.

## 5. Drug × drug interactions (ONC High-Priority list)

RxClass does not supply these. The ONC-High list gives ingredient×ingredient pairs. Map each pair (A, B) to a **symmetric pair** of edges so it fires regardless of which drug the card names:

```json
[
  {"edge_id":"onc:<A>:<B>","subject":"substance:A","relation_type":"contraindicated",
   "severity":"severe","status":"proposed","mechanism_id":null,
   "fire_when":{"op":"presence","pred":"substance:B"},
   "authority":"pub","source_ref":"ONC-HPDDI/2024",
   "rationale":"ONC high-priority interaction: A + B"},
  {"edge_id":"onc:<B>:<A>","subject":"substance:B","relation_type":"contraindicated",
   "severity":"severe","status":"proposed","mechanism_id":null,
   "fire_when":{"op":"presence","pred":"substance:A"},
   "authority":"pub","source_ref":"ONC-HPDDI/2024",
   "rationale":"ONC high-priority interaction: A + B"}
]
```

Use `relation_type: antagonizes` instead of `contraindicated` where the interaction is efficacy-opposition rather than a hazard (author/reviewer decides at ratification). `presence(substance:B)` works because CONF-01's active set already includes every substance administered by the card's actions (via `administers`), so a co-prescribed interacting drug is "present."

## 6. The payoff: mechanism substrate enables *derived* interaction proposals

Importing `has_MoA`/`has_PE` for **every** substance is what later unlocks CONF-01 Appendix B (conditionality/interaction derived from mechanism) — without ever auto-firing. Because the mechanism graph is now present, your deliberative model can, offline, **propose** drug×drug conflicts by mechanism opposition and hand them to ratification:

Worked logic — adrenaline × propranolol:
- `substance:adrenaline —acts_via(moa)→ mech:alpha_agonism` and `—acts_via(moa)→ mech:beta_agonism`
- `substance:propranolol —acts_via(moa)→ mech:beta_antagonism`
- opposition detector: `beta_antagonism` cancels `beta_agonism`, leaving `alpha_agonism` unopposed → **propose** `conflict_edge(subject:substance:adrenaline, contraindicated, fire_when:presence(substance:propranolol), mechanism_id:mech:unopposed_alpha, status:proposed)`.

This proposal is exactly a CONF-02 emission and enters at `status: proposed` — surfaced for human ratification, never fired on the model's say-so. The mechanism import pays for itself twice: explanation now, and interaction discovery later.

## 7. Worked example — `substance:adrenaline` import

Given RxClass data for adrenaline (ATC `C01CA24`; MED-RT `has_MoA` Adrenergic alpha/beta-Agonists; `has_PE` Vasoconstriction / Increased Heart Rate; `CI_with` Narrow-Angle Glaucoma; `may_treat` Anaphylaxis), the importer emits:

```json
{
  "nodes": [
    {"id":"substance:adrenaline","kind":"substance","xref":["RXCUI:3992","ATC:C01CA24"]},
    {"id":"class:atc.C01CA","kind":"class","label":"Adrenergic and dopaminergic agents"},
    {"id":"mech:alpha_agonism","kind":"mechanism","label":"Adrenergic alpha-agonism","conditionality_class":"acute_competition"},
    {"id":"mech:beta_agonism","kind":"mechanism","label":"Adrenergic beta-agonism","conditionality_class":"acute_competition"},
    {"id":"condition:narrow_angle_glaucoma","kind":"condition"},
    {"id":"condition:anaphylaxis","kind":"condition"}
  ],
  "edges": [
    {"kind":"member_of","from":"substance:adrenaline","to":"class:atc.C01CA","source_ref":"ATC/2026"},
    {"kind":"acts_via","from":"substance:adrenaline","to":"mech:alpha_agonism","role":"moa","source_ref":"MED-RT/has_MoA/2026AA"},
    {"kind":"acts_via","from":"substance:adrenaline","to":"mech:beta_agonism","role":"moa","source_ref":"MED-RT/has_MoA/2026AA"},
    {"kind":"conflict_edge","edge_id":"cirt:adrenaline:narrow_angle_glaucoma","subject":"substance:adrenaline",
     "relation_type":"contraindicated","severity":"caution","status":"proposed","mechanism_id":null,
     "fire_when":{"op":"presence","pred":"condition:narrow_angle_glaucoma"},
     "authority":"pub","source_ref":"MED-RT/CI_with/2026AA",
     "rationale":"MED-RT: adrenaline contraindicated with narrow-angle glaucoma"},
    {"kind":"indicated_for","from":"act:administer_adrenaline","to":"condition:anaphylaxis","source_ref":"MED-RT/may_treat/2026AA"}
  ]
}
```

Note what did and did not become a conflict: the single `CI_with` became one **proposed** `conflict_edge`; the two MoAs became **`acts_via`** substrate (no firing); the ATC code became **hierarchy**; `may_treat` became an **upstream indication** edge. The action `act:administer_adrenaline —administers→ substance:adrenaline` link (authored elsewhere) is what carries the glaucoma contraindication to the action at check time via the §4 scope extension.

## 8. Non-goals / deferred
- **No drug×drug data from RxClass** — that gap is filled by ONC-High (§5) and, later, mechanism-derived proposals (§6). Commercial interaction sets remain a separate, licensed upgrade path.
- **PK-interaction firing** (CYP inhibitor × substrate) — `has_pk`/pathway nodes are imported as substrate only; deriving PK conflicts is deferred to the same mechanism-proposal pipeline (§6), never auto-fired.
- **Severity/mechanism enrichment of ONC-High edges** — done at ratification, not import.
- **ATC redistribution** — this import consumes ATC *via RxClass mappings under NLM/UMLS terms* and stores ATC codes as node xrefs/identifiers; it MUST NOT bulk-redistribute the WHOCC ATC index as a standalone artifact (commercial redistribution of the index is restricted).
