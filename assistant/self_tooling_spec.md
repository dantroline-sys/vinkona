# VIN-TOOL-01 — Vinkona Self-Tooling Contract

**Status:** Adopted 2026-08-18, with the Adaptations in §0 (normative for this tree)
**Supersedes:** the free-form codegen approach (toolsmith.py's runtime loop)
**Depends on:** VIN-BIND-01 and VIN-MEM-01 *as re-pointed in §0.5–0.6* — neither exists
here as written; their obligations bind to concrete local mechanisms instead.

---

## 0. Adaptations for this tree (normative)

The contract below is adopted whole; this section records the deviations §2.3 requires
in writing, plus the re-pointing of the two referenced specs.  Where §0 and the body
conflict, §0 wins.

### 0.1 Pydantic — REJECTED (written rejection per §2.3)
The dependency ratchet (`test_dependencies.py`) holds the hard runtime surface to four
packages, and pydantic v2 ships a compiled core (`pydantic-core`) — exactly the
supply-chain and cross-platform surface this tree ratcheted out (it was removed once
already as a dead declaration).  **Replacement:** each block declares a plain
declarative params spec; JSON schemas are *derived* mechanically by
`blocks.params_json_schema()` at call time.  No schema is ever hand-written or stored,
so §11's single-source-of-truth intent holds: there is nothing to drift.

### 0.2 networkx — REJECTED (written rejection per §2.3)
Tool graphs here are ≤ ~15 nodes.  Cycle detection and topological ordering are
Kahn's algorithm (~25 lines in `toolgraph.py`), pinned by tests.  A graph library for
this contradicts the ratchet's whole point.

### 0.3 llama.cpp schema→GBNF — ADOPTED, at request time not build time
The pinned llama-server converts `response_format: json_schema` to a grammar
server-side, per request.  There are therefore **no grammar artifacts in the repo at
all** — §11's "no hand-written grammars" is satisfied by construction.  The CI check
becomes: the emission payload builder MUST attach a schema derived from the live
palette (asserted by test), so out-of-vocabulary emission is impossible at the server.

### 0.4 Haystack — build-thin option chosen (the table's own second alternative)
No vendoring.  The wiring validator's behaviour (typed sockets, pre-run connection
validation) is diffed against Haystack 2.x semantics in `test_toolgraph.py` cases;
the engine's code stays original and small.

### 0.5 VIN-BIND-01 re-pointed
No waypoint registry exists in this tree.  The rollback obligations bind to:
`mutate`/`fs-write` blocks declare their write surface (a path or table inside the
own-tools store); the engine snapshots it before execution and can restore it in one
step; dry-runs redirect all writes to a scratch area inside the store.  The store
boundary is the existing own-tools write-containment root — the one hard rule (read
anywhere, write only in the store) is unchanged and OS-enforced as before.

### 0.6 VIN-MEM-01 re-pointed
"Cognitive events" land as asides (`chat_logs` role `aside`), the surface that
already exists.  Her §6.3 self-appraisal of a dry-run is an aside, not a boolean.

### 0.7 Signing → hash manifest
Single local box; self-built graphs never ship (existing policy).  §7's "signed
repository" is a sha256 manifest over each block's source, verified at load;
mismatch refuses the block.  Real signatures can arrive if blocks ever ship.

### 0.8 Palette v0 is ~12 blocks, not 25
The news/document core (the use-case that actually failed under codegen):
fetch → parse → extract → filter → dedupe → rank → summarise → digest → notify,
plus the table trio and adapters.  Growth is driven by Gap Reports — the mechanism
proving itself.

### 0.9 Toolsmith demoted, not deleted
The §10 developer pathway **is** `toolsmith.py` — its spec→build→bank-failure→
re-analyse loop, moved off the runtime idle loop and behind the developer's hand.
At runtime, T3 files a Gap Report; nothing authors code.  (`own_tools.toolsmith`
stays opt-in-off; since TG5 the idle pass composes graphs by default and the
old codegen loop runs only under `toolsmith.codegen_dev`.)

### 0.10 Capability names mapped to this tree
`net` routes via the amiga_net egress broker (no direct sockets from blocks);
`process` runs in the toolbox podman/bwrap sandbox; `fs-read` honours the existing
read_paths/denylist seam; `sensor`/`biometric` are reserved words with no blocks.
Blocks that call her faculties go through the faculties dispatch with its allow-list
and hard-denies (`revise_self`/`deliberate`/`note_person`) unchanged.

### 0.11 Implementation stages
TG0 this document · TG1 types + block contract + registry + fixtures runner
(`blocks.py`, `palette.py`) · TG2 wiring validator (`toolgraph.py`) · TG3 constrained
emission (T1 configure first) · TG4 executor + self-test protocol (shadow, dry-run
snapshot/rollback, probation) · TG5 Gap Reports replace runtime codegen + panel/desk
Tools surfaces · TG6 palette growth toward the §9 core table.

---

## 1. Purpose and governing principle

Vinkona extends her own capabilities by **composing pre-verified blocks**, not by
writing code.  De-novo code generation is demoted to a rare, human-gated authoring
pathway that exists outside her runtime loop.

> **Prime rule:** At runtime, the only executable code is code that was written,
> tested, versioned, and signed at build time.  Vinkona emits *data* (tool graphs);
> the runtime executes *blocks*.  The two never mix.

Rationale: a ~30B local model configuring a closed vocabulary under a grammar is
near-deterministic; the same model adapting or authoring open-ended Python fails
occasionally and silently.  In a clinical-adjacent product, silent wrongness is the
worst failure class.  Verification, not generation, is the scarce resource — so the
design maximises what can be verified in advance.

## 2. Architecture: two layers

### 2.1 Wiring layer (Vinkona's output)
- Format: **tool-call sequences** (OpenAI-style function calls with JSON-schema
  parameters) — the most heavily post-trained structured format in this model class,
  and simultaneously a closed vocabulary.
- Emission is **grammar-constrained**: the model cannot produce a token outside the
  schema (see §0.3 for how).  Grammars/schemas are machine-generated from block
  declarations — never hand-authored.
- Multi-step tools: a sequence of calls where each call's output binds to a named
  slot (`$step1.out`) referenced by later calls.  This sequence *is* the tool graph.
- No loops, no conditionals beyond declared gate blocks, no imports, no inline
  expressions.  Anything resembling a programming language in the wiring layer is a
  spec violation.
- Context budget per emission: ≤ 8k tokens.  Thinking disabled or budget-capped.

### 2.2 Block layer (developer's output)
- Language: Python, stdlib + the pinned allowlist.  Chosen for developer velocity,
  **not** because the model touches it — the model never sees block source.
- Each block is a pure-ish callable with declared ports, capabilities, fixtures, and
  a version (§3).  Side-effect implementations (net, fs, LM) are injected via a
  context object, so shadow/replay modes swap them wholesale.
- Blocks run in the flow engine's process by default; blocks with `process` or
  untrusted-input capability run in the podman/bwrap-isolated worker.

### 2.3 Foundations — build-vs-import policy
Licensing rule (PolyForm codebase): MIT/BSD/Apache-2.0 MAY be imported or vendored
(preserve LICENSE/NOTICE); GPL/AGPL MUST NOT be imported or ported — design study
only; fair-code (SUL, e.g. n8n) MUST NOT be copied in any form.  The original table's
rows resolve, for this tree, per §0.1–0.4.  Hand-rolling where a table option exists
remains a violation unless rejected in writing — §0 is that writing.

## 3. Block contract

Every block MUST declare:

| Field | Requirement |
|---|---|
| `name`, `version` | Semver.  Saved graphs pin block versions (§7). |
| `summary` | One sentence, written **for retrieval** — what the model matches on. |
| `ports_in` / `ports_out` | Semantic types from the closed set (§4).  Never bare `dict`/`list`/`str`. |
| `params` | Declarative spec (schema derived per §0.1), with defaults.  Everything user-specific is a param, never a code change. |
| `capabilities` | Subset of: `net`, `fs-read`, `fs-write`, `mutate`, `process`, `sensor`, `biometric`.  Empty = pure transform. |
| `fixtures` | ≥ 2 canned input→output pairs, incl. one edge case.  Run by CI and the self-test protocol (§6). |
| `failure_modes` | Enumerated; missing-data policy is fail / degrade — never invent. |
| `rollback` | Required iff `mutate` or `fs-write`: snapshot per §0.5, no exceptions. |
| `docstring` | Human-readable; **no Python signatures ever enter model context**. |

**Static guarantees this buys:** a graph's capability set = the union of its blocks',
computable before execution; a graph with no `mutate`/`fs-write` block provably needs
no rollback artifact; type-checking every edge rejects invalid pipelines before
anything runs.

## 4. Semantic type set (closed)

`FeedRef` · `Document` · `Passage` · `Record` · `Table` · `Ranked[T]` · `Event` ·
`FileRef` · `ImageRef` · `AudioRef` · `Query` · `Digest`

- Blocks declare e.g. `List[Document] → Ranked[Document]`, never `list → list`.
- Extending the set is a developer decision, never a runtime one.
- Type mismatches are bridged by explicit **adapter blocks** (`table_to_records`),
  keeping coercion visible in the graph rather than hidden in loosened ports.

## 5. Capability tiers (runtime pathway selection)

| Tier | What Vinkona does | Gate |
|---|---|---|
| **T1 Configure** | Set params on an existing block/graph (new feed URL, topic filter, ranking weights).  Expected to cover most needs. | Self-test only (§6).  Auto-deploy permitted. |
| **T2 Compose** | Wire existing blocks into a new graph. | Static type + capability check → self-test → user notified in plain language.  Approval required if capabilities include `net`+`fs-write`, `mutate`, or `biometric`. |
| **T3 Author** | A genuinely new block is needed.  **She does not write it.**  She files a Gap Report (§8). | Developer authors, tests, signs via §10. |

## 6. Self-test protocol ("testing herself")

Before any new or modified graph deploys:

1. **Static pass** — schema validity, type-check all edges, capability computation,
   version resolution.
2. **Fixture pass** — the graph runs against its blocks' fixtures composed
   end-to-end, in shadow mode: injected replay context, no real side effects.
3. **Live dry-run** — one execution against real inputs with all `mutate`/`fs-write`
   redirected to the scratch area (§0.5 snapshot).  She inspects the output herself —
   does it type-check, is it non-empty when inputs were non-empty, does it match the
   stated intent? — and her appraisal is logged as an aside (§0.6), not a boolean.
4. **Probation** — the first N real runs (default 5) keep shadow artifacts and are
   flagged in provenance.  A regression during probation auto-disables the graph and
   files a Gap Report.
5. **Deploy** — the graph registers with pinned block versions.

**Anti-invisibility requirements** (the user never reads code, so the system
substitutes for review): every surfaced result carries provenance one tap away
(graph, blocks, sources, which filter fired); "nothing matched" renders distinctly
from "no new data"; periodic **exclusion sampling** surfaces a small sample of items
a filter *rejected*, so a filter eating the wrong things is catchable in seconds.

## 7. Versioning and change control

- Saved graphs pin exact block versions; a block upgrade never silently changes a
  deployed tool.  Upgrading a graph re-triggers the full §6 protocol.
- The block manifest is hash-verified (§0.7); the runtime refuses mismatched blocks.
- Self-built graphs are per-installation and never ship.

## 8. Gap Reports (limitation discovery)

When she wants a capability the palette lacks, or attributes a self-test failure to
a missing/deficient block, she files a structured report rather than attempting
authorship:

```json
{
  "kind": "gap | defect | enhancement",
  "goal": "plain-language statement of what the user needed",
  "attempted_graph": "...",
  "missing": "block or type or capability she could not find",
  "closest_existing": ["blocks considered and why rejected"],
  "evidence": "fixture output / failure trace if applicable",
  "urgency": "user-blocking | quality | nice-to-have"
}
```

Reports queue locally (the existing ideas store, schema-extended) and surface to the
developer.  Her limitations become the project's backlog — the deployment is a
requirements-discovery instrument instead of a codegen risk.

## 9. Palette enumeration

Organised by **verb × type**, domain-neutral (no "oncology filter" — domain knowledge
lives in Vinur, expressed as params and queries).  Target well under 50 blocks; §0.8
stages the budget, starting at ~12.  The original core/extended/perception tables are
retained as the roadmap; `biometric` remains a distinct capability, never bundled into
`sensor`, permanently excluded from auto-deploy, opt-in per deployment, and out of any
clinically-pitched build.

## 10. Escape hatch: authored blocks (developer pathway)

For T3, the developer (optionally assisted by a large model, offline) authors new
blocks: contract first (name, ports, params, capabilities, fixtures — before any
implementation); implementation against the contract; fixtures pass in the sandbox;
evidence-driven repair loop (traceback + failing fixture only), max 3 attempts before
the human takes over; sign, version, release.  This loop is `toolsmith.py` (§0.9).
Vinkona's runtime never invokes it.

## 11. Acceptance criteria

- [ ] Emission cannot produce out-of-vocabulary calls (schema enforced at the server, §0.3).
- [ ] The type checker rejects a deliberately mis-wired graph before execution.
- [ ] A capability summary renders in plain language for a sample T2 graph.
- [ ] A `mutate` block in dry-run leaves the real store untouched and produces a valid snapshot.
- [ ] Exclusion sampling surfaces rejected items on schedule.
- [ ] A forced fixture regression during probation auto-disables the graph and files a well-formed Gap Report.
- [ ] The core palette passes full fixture CI in under one minute.
- [ ] No hand-written JSON schemas or grammar files exist in the repository; schemas are derived from block declarations at call time, verified by a CI check.
- [ ] Dependency audit passes: the ratchet's hard surface is unchanged; no GPL/AGPL/SUL code in the tree.
