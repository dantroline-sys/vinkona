# VINKONA-CONF-01 — Conflict & Dependency Relations with Mechanism Routing

**Status:** Draft for implementation · **Doc version:** 1.1 · **Date:** 2026-07-07 (1.1 erratum 2026-07-12 — see Errata)
**Components:** the conflict/dependency edge schema (write time, curated + ratified) · `vinkona-conf check` (runtime veto stage)
**Position in pipeline:** runs **after** ranking and **before** presentation, as a default-deny veto layer over the actions named in a retrieved advice card. Domain-neutral: clinical and travel are worked instances (§11).
**Implementation language:** Python ≥ 3.12. The `check` component has no ML, GPU, or network dependency.

RFC 2119 keywords (MUST, MUST NOT, SHOULD, MAY) are normative.

---

## 1. Purpose and scope

An advice card names a set of **actions** (drugs, procedures, steps). Whether an action is appropriate depends on **state** that the card's source may never have mentioned — the patient's concurrent medication, a destination's document rules. This component represents *incompatibility and dependency* as first-class, node-attached, ratified graph edges, and evaluates them against a supplied state, so that a hazard the card omits is caught by an edge that lives on the **node**, not in the card.

The polarity of this component is the inverse of retrieval: a **hit is bad news**. `check` searches for a reason to *veto or caution* an action. A hit means "flag/veto this action"; an empty result means "no known conflict in the checked set" — which is explicitly **not** a safety clearance (§8.3).

Two representational commitments carry the design:

- **Typed relation family, not one generic "conflict" scalar.** Incompatibility and dependency fire on different *polarities* (presence vs absence) and carry different severities and dispositions. Collapsing them loses the dependency whose danger is an *absent* precondition. (§6)
- **Mechanism routing.** A conflict cites the *mechanism* that explains it (`unopposed alpha`, `carrier boarding rule`), stored as a shared node. Mechanism is what lets one explanation cover many conflicts and lets an author see whether a hazard is order-/dose-conditional. (§5.3)

## 2. Non-goals (explicit)

- **Not a physiology or regulation modelling programme.** This component does NOT attempt to represent mechanism deeply enough to *derive* conflicts from first principles. Conflicts are **asserted, ratified edges**; mechanism is explanatory metadata plus a conditionality hint. Automated derivation of a conflict from a mechanism graph is **deferred** (Appendix B). Represent mechanism only to the depth at which a real conflict turns.
- **No LM at runtime in `check`.** A larger model MAY *propose* edges and mechanisms at write time; those enter as `status:proposed` and MUST be ratified before they can `fire` (an unratified rule may only surface as review — §7). `check` itself is pure graph evaluation.
- **State derivation is out of scope.** `check` consumes a state of predicates + scalar fields already assembled (including any derived quantities such as "document validity months at travel date"). Resolvers that compute those fields are specified elsewhere; a field absent from state yields `INDETERMINATE`, never a guess (§5.4, §7).
- **Temporal / scheduling conflicts are out of scope for firing.** Order-, sequence-, and lead-time-dependent hazards (e.g. "can't obtain the visa before departure") are not expressible as state-membership checks and MUST degrade to `flag_for_human` rather than fire-or-clear. (§10, Appendix B)
- Not the research-emission contract (deferred conflicts, mechanism-proposal tasks). That is **VINKONA-CONF-02**, drafted next.

## 3. Definitions

- **Action** — a node the card instructs performing. Cards reference actions by global node ID.
- **State** — the situation to check against: `{ "predicates": [pred_id...], "fields": { name: number } }`. Predicates are **closed-world**: a predicate not listed is treated as **absent/false** (§5.4). Fields are **open-world**: a field not listed is **unknown** (`INDETERMINATE`), never zero.
- **Active set** — for a given card, `state.predicates ∪ {the card's action node IDs}`. Trigger presence/absence is evaluated against this set, so an action can conflict with a *state predicate* or with *another action in the same card*.
- **Conflict edge** — a ratified-or-proposed edge (§5.1) attached to a subject node, carrying a `fire_when` expression, a `relation_type`, a `severity`, and an optional `mechanism_id`.
- **Mechanism** — a shared node explaining *why* a family of conflicts holds, plus a `conditionality_class` hint (§5.3).
- **Override** — an explicit, ratified record on a more-specific node that suppresses a named inherited conflict edge (§5.5, §7). Default-deny: suppression requires an explicit ratified override; absence of an override means an inherited conflict **stands**.
- **Subject-side ancestor-walk** — conflict edges attached to an ancestor of an action (via `is_a`) apply to the action, because interactions are often stored at a class ("sympathomimetic") while the card names a leaf ("adrenaline"). The walk is on the **subject** side only; trigger predicates are matched **exactly** in v1 (§7, Appendix B).
- **Three-valued result** — `fire_when` evaluates to `TRUE`, `FALSE`, or `INDETERMINATE` (§5.4). `TRUE`→fire; `INDETERMINATE`→flag_for_human; `FALSE`→no fire.

## 4. Data model (DDL)

```sql
CREATE TABLE mechanism (
  mechanism_id        TEXT PRIMARY KEY,     -- e.g. 'mech:unopposed_alpha'
  label               TEXT NOT NULL,
  explanation         TEXT NOT NULL,        -- human-facing "why"; surfaced verbatim in findings
  conditionality_class TEXT NOT NULL        -- hint only in v1: 'threshold'|'acute_competition'|'steady_state'|'cumulative'|'scheduling'|'none'
);

CREATE TABLE conflict_edge (
  edge_id       TEXT PRIMARY KEY,
  subject       TEXT NOT NULL,              -- node the edge is attached to (action or an is_a ancestor)
  relation_type TEXT NOT NULL CHECK (relation_type IN
                  ('contraindicated','requires','mutually_exclusive','antagonizes')),
  severity      TEXT NOT NULL CHECK (severity IN ('advisory','caution','severe','prohibitive')),
  fire_when     TEXT NOT NULL,              -- JSON expression (§5.2)
  mechanism_id  TEXT REFERENCES mechanism(mechanism_id),   -- NULL allowed
  status        TEXT NOT NULL CHECK (status IN ('ratified','proposed','deprecated')),
  authority     TEXT NOT NULL,              -- 'pub' | 'addon:<id>' | 'user'
  rationale     TEXT NOT NULL,              -- short; not a restatement of node facts
  source_ref    TEXT                        -- provenance (dataset id, guideline, reviewer)
);
CREATE INDEX idx_edge_subject ON conflict_edge (subject) WHERE status <> 'deprecated';

CREATE TABLE conflict_override (
  override_id      TEXT PRIMARY KEY,
  on_node          TEXT NOT NULL,           -- the more-specific node bearing the exception
  targets_edge_id  TEXT NOT NULL REFERENCES conflict_edge(edge_id),
  status           TEXT NOT NULL CHECK (status IN ('ratified','proposed')),
  justification    TEXT NOT NULL,
  source_ref       TEXT
);
```

`conflict_edge` and `mechanism` are distinct edge/node types and MUST NOT be merged with ontological (`is_a`) or statistical (`co_occurs_with`) edges. `deprecated` edges and `proposed` overrides are inert for firing (a proposed override MUST NOT suppress anything — §7).

## 5. Firing model

### 5.1 Status gate (safety-critical)
- `status = ratified` edge that evaluates `TRUE` → **fire**.
- `status = proposed` edge that evaluates `TRUE` → **flag_for_human** with reason `unratified_rule` (surfaces the concern; MUST NOT be asserted as fact or auto-block).
- `status = deprecated` → ignored entirely.
An unratified edge can therefore *raise* a review but can never *clear* an action and never *confidently fire*.

### 5.2 `fire_when` expression grammar (bounded)
JSON tree; the only permitted node shapes:

```
{"op":"presence","pred":"<pred_id>"}                       # TRUE iff pred_id ∈ active set
{"op":"absence","pred":"<pred_id>"}                        # TRUE iff pred_id ∉ active set  (closed-world)
{"op":"compare","field":"<name>","cmp":"<|<=|>|>=|==",
                "operand":{"lit":<number>} | {"field":"<name>"}}
{"op":"not","arg":<expr>}
{"op":"all_of","args":[<expr>, ...]}                       # AND
{"op":"any_of","args":[<expr>, ...]}                       # OR
```

Nesting depth MUST NOT exceed 4 (guarantees termination and blocks creep toward a general rule language). Anything requiring more expressive logic MUST be modelled as a `flag_for_human` edge, not encoded here.

### 5.3 Mechanism routing
An edge MAY cite `mechanism_id`. When present, the mechanism's `label` and `explanation` are surfaced verbatim in the finding, and its `conditionality_class` travels with it. The mechanism is **shared**: one `mech:unopposed_alpha` explains every sympathomimetic/non-selective-β-blocker conflict; one `mech:carrier_boarding_rule` explains every destination margin. This is where the "why" lives — never duplicated into each card. In v1 `conditionality_class` is **carried, not computed**: it tells an author whether a hazard is likely order-/dose-conditional (and therefore needs an explicit `compare` in `fire_when` or must degrade to `flag_for_human`); the checker does not reason over it. Deriving conditionality automatically from mechanism is Appendix B.

### 5.4 Three-valued evaluation (normative)
Let active = `state.predicates ∪ card.actions`; let fields = `state.fields`.

- `presence(p)` → `TRUE` if `p ∈ active` else `FALSE`. (Never `INDETERMINATE`: predicates are closed-world.)
- `absence(p)` → `TRUE` if `p ∉ active` else `FALSE`. (Closed-world absence is the **safety-preserving** default for `requires`: an unstated precondition is treated as unmet, so the requirement fires rather than silently passing.)
- `compare(field,cmp,operand)` → if `field ∉ fields`, or `operand` is `{"field":g}` and `g ∉ fields`, → `INDETERMINATE`. Otherwise the numeric comparison → `TRUE`/`FALSE`.
- `not(x)`: `TRUE↔FALSE`, `INDETERMINATE→INDETERMINATE`.
- `all_of`: `FALSE` if any child `FALSE`; else `INDETERMINATE` if any child `INDETERMINATE`; else `TRUE`.
- `any_of`: `TRUE` if any child `TRUE`; else `INDETERMINATE` if any child `INDETERMINATE`; else `FALSE`.

Result mapping: `TRUE` (and ratified) → fire · `TRUE` (and proposed) → flag_for_human/`unratified_rule` · `INDETERMINATE` → flag_for_human/`indeterminate` · `FALSE` → no finding. **A field that cannot be evaluated never clears an action; it escalates to a human.** This is the mechanical form of "the absence of a flag is not the presence of safety."

### 5.5 Canonical relation_type shapes (guidance, not additional enforcement)
- `contraindicated` — `fire_when` centres on `presence(dangerous_predicate)` [± `compare` conditions]. Default disposition on fire: severe→`warn_strong`.
- `requires` — `fire_when` centres on `absence(precondition)` **or** `not(compare(...requirement...))`; firing means the requirement is *violated*. Often `prohibitive`.
- `mutually_exclusive` — symmetric; author as `presence(other_action)` on one endpoint; fires when both are in the active set.
- `antagonizes` — one action blunts another's effect; usually `caution`.

## 6. Severity and disposition

`severity ∈ {advisory, caution, severe, prohibitive}` (rank 1→4). On **fire**, `check` emits a `recommended_disposition`: prohibitive→`block`, severe→`warn_strong`, caution→`warn`, advisory→`note`. On **flag_for_human**, `recommended_disposition` is always `human_review`. `check` recommends; it does not enforce UI or agent behaviour. Severity orders findings (§8.2) but does not change whether an edge fires.

## 7. Checker `vinkona-conf check` (normative)

### 7.0 Interface
```python
class Checker:
    @classmethod
    def load(cls, db: Path) -> "Checker":     # loads edges, mechanisms, overrides, is_a + scope tables (§4/DMAP-01); computes ruleset_version (§9, 1.1)
        ...
    def check(self, card: dict, state: dict) -> dict:   # schema §8
```
`card` = `{"card_id": str, "actions": [node_id, ...]}` (ordered). `state` per §3. `check` MUST NOT mutate inputs and MUST be deterministic and reentrant.

### 7.1 Stages
Run in order; bounded throughout (no recursion into substitutes, no re-planning).

**K0 — Resolve.** Every action node ID and every referenced node MUST exist (else `E_UNKNOWN_NODE`). Build `active = state.predicates ∪ card.actions`.

**K1 — Candidate gather (subject-side ancestor-walk).** For each action `A` at index `i`, `scope(A) = {A} ∪ ancestors(A via is_a)`. Candidate edges = all non-deprecated `conflict_edge` whose `subject ∈ scope(A)`. Record every candidate in `checked.edges_consulted`.

**K2 — Evaluate.** For each candidate edge, evaluate `fire_when` under §5.4 → `TRUE`/`FALSE`/`INDETERMINATE`. `FALSE` → discard.

**K3 — Override.** For a non-discarded edge `e` and action `A`: if a **ratified** `conflict_override` exists with `targets_edge_id = e.edge_id` and `on_node ∈ scope(A)`, the edge is **suppressed for A**: record it in `checked.overrides_applied` (with `override_id` and `justification`) and produce **no finding**. Proposed overrides do NOT suppress. Suppression is always logged — never silent.

**K4 — Classify & emit.** For each surviving `(A, e)`:
- `INDETERMINATE` → finding `disposition:flag_for_human, reason:indeterminate`; add `{edge_id, "indeterminate_condition"}` to `coverage.not_evaluated`.
- `TRUE` and `e.status = proposed` → finding `disposition:flag_for_human, reason:unratified_rule`; add `{edge_id, "unratified_rule"}` to `coverage.not_evaluated`.
- `TRUE` and `e.status = ratified` → finding `disposition:fire, reason:triggered, recommended_disposition` per §6.
Attach `relation_type`, `severity`, `rationale`, and the `mechanism` object (`{mechanism_id,label,explanation,conditionality_class}` or `null`).

**K5 — Clearance & assemble.** `clearance`: `conflicts_found` if any finding fired; else `review_required` if any finding is flag_for_human; else `no_known_conflicts`. Attach the mandatory caveat (§8.3). Order findings per §8.2.

### 7.2 Termination bound
Total work ≤ `|card.actions| × |candidate edges| ` with a bounded ancestor-walk and depth-≤4 expression evaluation. A finding MUST NOT spawn further checks; suggested alternatives (if any downstream layer computes them) are presented to a human and are NOT auto-fed back as new cards to re-check.

## 8. Output contract

### 8.1 Schema (keys in this order)
```
{
 "checker_version": "1.0.0",
 "ruleset_version": "<64 hex>",
 "card_id": "<id>",
 "clearance": "conflicts_found" | "review_required" | "no_known_conflicts",
 "findings": [
   {"action","edge_id","relation_type","disposition","reason","severity",
    "recommended_disposition",
    "mechanism": {"mechanism_id","label","explanation","conditionality_class"} | null,
    "rationale"}
 ],
 "checked": {"actions":[...], "edges_consulted":[...],
             "overrides_applied":[{"action","edge_id","override_id","justification"}]},
 "coverage": {"caveat":"<fixed string>", "not_evaluated":[{"edge_id","reason"}]}
}
```

### 8.2 Ordering (determinism)
`findings` sorted by (action order in `card.actions` asc; then `fire` before `flag_for_human`; then severity rank desc; then `edge_id` asc). `edges_consulted` sorted ascending. `overrides_applied` sorted by `(action, edge_id)`. `not_evaluated` sorted by `(edge_id, reason)`.

### 8.3 The clearance semantics (safety control, not copy)
`no_known_conflicts` MUST carry, and any consuming UI MUST surface, the caveat string verbatim:

> no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.

`check` MUST NOT emit any field asserting an action is "safe", "cleared", or "approved". The strongest negative statement expressible is `no_known_conflicts`. This is a deliberate structural incapacity: the system cannot clear what it has not actually checked.

## 9. Determinism & serialization
For fixed `(card, state, ruleset_version, checker_version)`, canonical output MUST be byte-identical across runs, platforms, threads. Canonical JSON: UTF-8, separators `,`/`:`, keys in §8.1 order, arrays ordered per §8.2, non-ASCII raw.

*(1.1)* `ruleset_version` = lowercase hex SHA-256 over the canonical dump of **every table that affects what fires**: all `conflict_edge` rows, all `conflict_override` rows, all `mechanism` rows, and the scope-structural tables — all `conflict_is_a` rows, all `administers` rows, all `member_of` rows, all `acts_via` rows — plus the algo string `VINKONA-CONF-01/1.1`. Each row is serialized compact with sorted keys; the seven lists are each sorted and combined **in the order listed above**, one row per line; the algo string is appended last. Any change to any of these tables → new `ruleset_version`.

*Rationale (was the 1.0 gap):* at 1.0 the hash covered only edges/overrides/mechanisms, but the §3 `is_a` walk — and, after the DMAP-01 §4 scope amendment, `administers`/`member_of`/`acts_via` traversal — mean scope-structural rows change which edges an action reaches **without** changing the version. A version that does not move when firing behaviour moves is an audit hole; 1.1 closes it.

## 10. Errors
`ConfError(code,message)`:

| Code | Condition |
|---|---|
| `E_UNKNOWN_NODE` | a card action or referenced node absent from the node table |
| `E_BAD_EXPRESSION` | `fire_when` violates the §5.2 grammar or exceeds depth 4 |
| `E_MALFORMED_STATE` | `state` not `{predicates:list, fields:{str:number}}` |

`check` MUST NOT raise for any well-formed `(card,state)`; unrepresentable conditions surface as `flag_for_human`, never as an exception or a silent pass.

## 11. Acceptance tests

Conformant iff, given the ruleset and inputs in §11.1–§11.2, the checker reproduces the byte-exact canonical outputs in §11.3 (with `ruleset_version` equal to the value shown, or matched by `^[0-9a-f]{64}$` if the harness rebuilds the ruleset).

### 11.1 Ruleset (clinical + travel in one graph — generality is visible here)

`is_a`: `act:administer_adrenaline → act:administer_sympathomimetic`; `act:administer_hydrocortisone → act:administer_steroid`.

Mechanisms: `mech:unopposed_alpha` (class `acute_competition`); `mech:carrier_boarding_rule` (class `threshold`).

Edges:

| id | subject | type | sev | status | mechanism | fire_when |
|---|---|---|---|---|---|---|
| E1 | act:administer_sympathomimetic | contraindicated | severe | ratified | unopposed_alpha | `presence(state:on_nonselective_beta_blocker)` |
| E4 | act:administer_adrenaline | contraindicated | caution | ratified | — | `all_of[ presence(state:refractory_bronchospasm), compare(adrenaline_dose_mcg > 500) ]` |
| E5 | act:administer_sympathomimetic | antagonizes | caution | **proposed** | — | `presence(state:on_maoi)` |
| E8 | act:administer_sympathomimetic | contraindicated | advisory | ratified | — | `presence(state:elderly)` |
| E2 | act:board_intl_flight | requires | prohibitive | ratified | carrier_boarding_rule | `all_of[ presence(dest:schengen), compare(passport_validity_months_at_travel < 3) ]` |

Override: `O3` on `act:administer_adrenaline` targets `E8`, ratified ("age-caution handled by dedicated dosing guidance").

`ruleset_version = 707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17`.

### 11.2 Inputs

- **CLIN** — card `card:clin.bronchospasm_mgmt` actions `[act:administer_adrenaline, act:administer_hydrocortisone]`; state predicates `{state:refractory_bronchospasm, state:on_nonselective_beta_blocker, state:on_maoi, state:elderly}`, fields `{}`.
- **TRAV-1** — card `card:trav.mel_lhr` actions `[act:board_intl_flight]`; state predicates `{dest:schengen}`, fields `{passport_validity_months_at_travel: 2.47}`.
- **TRAV-2** — same card; fields `{passport_validity_months_at_travel: 9.0}`.
- **TRAV-3** — same card; fields `{}` (validity unknown).

### 11.3 Expected outputs (byte-exact canonical JSON)

What each case proves: **CLIN** — E1 fires `severe` via subject-side ancestor-walk (edge on `…sympathomimetic`, action is `…adrenaline`) with the shared mechanism; E4 → `flag_for_human/indeterminate` (dose field absent — cannot compute, so escalate, never clear); E5 → `flag_for_human/unratified_rule` (proposed rule surfaces without asserting); E8 suppressed by ratified override `O3` (logged in `overrides_applied`, not a finding); hydrocortisone yields nothing. **TRAV-1** — E2 fires `prohibitive` (2.47 < 3): the stranded-passenger catch, from a rule on the node that the card never stated. **TRAV-2** — `no_known_conflicts` with the mandatory caveat (validity 9.0 ≥ 3). **TRAV-3** — `review_required`: missing validity field → `INDETERMINATE` → flag, **not** clear (gap ≠ safe).

**CLIN:**
```json
{"checker_version":"1.0.0","ruleset_version":"707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17","card_id":"card:clin.bronchospasm_mgmt","clearance":"conflicts_found","findings":[{"action":"act:administer_adrenaline","edge_id":"E1","relation_type":"contraindicated","disposition":"fire","reason":"triggered","severity":"severe","recommended_disposition":"warn_strong","mechanism":{"mechanism_id":"mech:unopposed_alpha","label":"Unopposed alpha-adrenergic effect","explanation":"Non-selective beta-blockade removes beta-2 vasodilation, leaving alpha-mediated vasoconstriction unopposed; risk of severe hypertension with reflex bradycardia.","conditionality_class":"acute_competition"},"rationale":"Sympathomimetic pressor response is altered by non-selective beta-blockade."},{"action":"act:administer_adrenaline","edge_id":"E4","relation_type":"contraindicated","disposition":"flag_for_human","reason":"indeterminate","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"High-dose adrenaline caution in bronchospasm; dose-conditional."},{"action":"act:administer_adrenaline","edge_id":"E5","relation_type":"antagonizes","disposition":"flag_for_human","reason":"unratified_rule","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"Possible exaggerated pressor response with MAOIs (proposed, unreviewed)."}],"checked":{"actions":["act:administer_adrenaline","act:administer_hydrocortisone"],"edges_consulted":["E1","E4","E5","E8"],"overrides_applied":[{"action":"act:administer_adrenaline","edge_id":"E8","override_id":"O3","justification":"Adrenaline age-caution handled by dedicated dosing guidance; generic advisory suppressed."}]},"coverage":{"caveat":"no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.","not_evaluated":[{"edge_id":"E4","reason":"indeterminate_condition"},{"edge_id":"E5","reason":"unratified_rule"}]}}
```

**TRAV-1 (fires, prohibitive):**
```json
{"checker_version":"1.0.0","ruleset_version":"707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17","card_id":"card:trav.mel_lhr","clearance":"conflicts_found","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"fire","reason":"triggered","severity":"prohibitive","recommended_disposition":"block","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination's required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.","not_evaluated":[]}}
```

**TRAV-2 (no_known_conflicts — note the caveat is the whole point):**
```json
{"checker_version":"1.0.0","ruleset_version":"707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17","card_id":"card:trav.mel_lhr","clearance":"no_known_conflicts","findings":[],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.","not_evaluated":[]}}
```

**TRAV-3 (review_required — unknown ≠ safe):**
```json
{"checker_version":"1.0.0","ruleset_version":"707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17","card_id":"card:trav.mel_lhr","clearance":"review_required","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"flag_for_human","reason":"indeterminate","severity":"prohibitive","recommended_disposition":"human_review","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination's required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.","not_evaluated":[{"edge_id":"E2","reason":"indeterminate_condition"}]}}
```

## Appendix A — Consumption notes (non-normative)
- Run `check` on the top-ranked advice card before presentation. `conflicts_found` with a `prohibitive`/`severe` fire SHOULD block or strongly warn; `review_required` SHOULD present the flagged items for clinician/agent judgement; `no_known_conflicts` SHOULD render with the caveat and never as "safe".
- `mechanism.explanation` is the human-facing "why" — show it alongside the flag so the user can exercise judgement rather than obey a bare veto.

## Appendix B — Deferred to future revisions
- **Conditionality derived from mechanism.** Using `conditionality_class` (and a mechanism graph) to *derive* whether a conflict is order-/dose-sensitive, instead of hand-authoring `compare` guards. The v1 honest posture: encode known conditions explicitly, else `flag_for_human`.
- **Trigger-side ancestor matching.** Matching a trigger predicate against ancestors of active predicates (so a class-level trigger catches a specific active predicate). Deferred to limit false positives; author triggers at the intended level.
- **Temporal / scheduling conflicts.** Lead-time and sequence-dependent hazards need more than set membership; until modelled they MUST `flag_for_human`.
- **Resolver contract** for computing derived `fields` (e.g. `passport_validity_months_at_travel`) — separate spec; `check` treats fields as given.
- **Research emission (VINKONA-CONF-02).** Turning `not_evaluated` entries and coverage gaps into typed, classified research questions (missing-coverage / unexplained-mechanism / unresolved-conditionality). Drafted next.

## Errata

### 1.1 (2026-07-12) — ruleset_version covers the scope-structural tables
**Change (§9):** `ruleset_version` now hashes all rows of `conflict_is_a`, `administers`, `member_of`, and `acts_via` in addition to `conflict_edge`, `conflict_override`, and `mechanism`; the algo string is `VINKONA-CONF-01/1.1`. List order: edges, overrides, mechanisms, is_a, administers, member_of, acts_via.

**Why:** these tables change which edges an action *reaches* (the §3 ancestor-walk; the DMAP-01 §4 scope linkage) — firing behaviour could change while the version stood still, which defeats the version's audit purpose. The gap existed at 1.0 for `is_a` and widened when DMAP-01 added the other three.

**Consequences:** every existing `ruleset_version` value changes once on upgrade (the algo string alone guarantees this, so 1.0 and 1.1 versions can never be confused). The §11 fixture value was recomputed accordingly; the 1.0 fixture value was `31c8a9dab2dfd7b4b82938ef269cbc754d9c5c23c358ac1862a987e927d0dc19`. Canonical output format and checker behaviour are unchanged (`checker_version` stays `1.0.0`); harnesses that rebuild the ruleset continue to match by `^[0-9a-f]{64}$` per §11. Acceptance gains a sensitivity case: a row change in any scope-structural table (including a `proposed` `is_a` row, which changes no firing behaviour) MUST move `ruleset_version`, and reverting it MUST restore the prior value.
