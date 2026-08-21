# VIN-INIT-01 — Conversational Initiative Contract

**Status:** Adopted 2026-08-18, with the §0 Adaptations (normative for this tree)
**Purpose:** cure greeting-time mirroring ("what's new?" → reflected back) by giving
her something *of her own* to raise — one pre-selected, grounded item, verbalised
naturally or dropped freely.

> **Prime rule:** Initiative is *selected deterministically and verbalised by the
> model.*  The LLM never chooses what to raise; it is handed at most **one**
> pre-selected item with context.  The queue produces recitation; nothing produces
> mirroring; one item produces a friend.

---

## 0. Adaptations for this tree (normative)

0.1 **Mid-conversation initiative (§6.3) = the existing spontaneity segue lane,
unchanged.**  Dan's prior ruling stands: no lull detector.  VIN-INIT's new
machinery is the conversation-OPEN lane + the persistent queue; spontaneity stays
the mid-stream consumer (optionally drawing from this queue at IN5).

0.2 **Relational fit** comes from the mind graph (her own people/things/places
store) + the user model — term overlap between an item's grounding text and graph
entity labels / user interests, computed by FEEDERS at write time and stored on
the item.  A thin graph degrades fit to 0 (backlog/self_state still work); never
node-distance theatre, never a guess.

0.3 **Invitation detection** ("what's new?", "what shall we talk about?") is a
deterministic phrase list, not a classifier.  The frequency gate's *p* applies
only to uninvited openers.

0.4 **Tact labels** (severity / sensitivity / corroboration) are authored at FEED
time (news feeder: corroboration by ≥2 distinct sources in the archive is a
query; sensitivity by the big LM during dreaming) and gated at SELECTION
deterministically, **fail-closed**: an unlabeled or uncorroborated news item can
never open a conversation.  `grief_adjacent` = respond-never-initiate, exactly as
§6.2 writes it.

0.5 **VIN-MEM-01 re-pointed** as established: cognitive events land as asides;
open-loop extraction (§4.1) and reflection questions (§4.5) are additional
prompts in the existing dreaming/idle-reflect passes, not a new subsystem.

0.6 **Placement (§9):** a table in memory.db — per-profile automatically,
clearable from the panel, provenance inspectable ("why did you ask me that?" is a
lookup, answered from `grounding`).

0.7 **Outcome capture** reuses spontaneity's watermark/outcome-judge pattern —
one mechanism, two lanes.

0.8 **Empty queue (§10 first criterion) is prompt work too:** the greeting
instruction tells her to answer "what's new?" from the ambient block and her real
self-state when nothing is queued — never to volley the question back.

0.9 **Stages:** IN0 this doc · IN1 deterministic core (queue/scorer/governors,
`initiative.py`, offline-tested incl. the 20-greeting simulation) · IN2 LM-free
feeders (self_state from trace events; backlog from research plans) · IN3
selector + session-open injection + invitation bypass + outcome write-back +
panel surface · IN4 dreaming feeders (open loops w/ strict windows, reflection
questions, news×graph w/ corroboration + tact labels) · IN5 bounded per-user
channel weights + channel-starvation gap reports + spontaneity unification.

---

## 3. Initiative item schema

channel `open_loop | backlog | news | self_state | reflection` · `pointer`
(one-sentence topic marker) · `relational_context` (why this matters to this
user / to her) · `grounding` (**mandatory** — node ids / source refs; ungrounded
initiative is inadmissible: initiative must never be confabulated) · `created` /
`expires` · optional `time_window` (earliest/latest natural moment) · stored
`fit` (§0.2) · `sensitivity none | personal | grief_adjacent` · `corroborated`
(news) · `respond_only` · `raised_count` / `deflections` / `last_outcome
unraised | engaged | deflected`.

## 4. Feeder channels (priority = how human each feels)

4.1 **Open loops** — unresolved threads with a resolution point the user hasn't
reported (dreaded event, pending decision, mentioned plan).  Carries a
`time_window`; asking about Thursday's list three weeks later reads as broken —
expiry is strict (window close + 7 days).  Extracted in dreaming.
4.2 **Research-question backlog** — open `research` plan questions become her
genuine curiosity ("I never did settle X").  Long expiry, low urgency, excellent
filler.
4.3 **News × user-graph** — from the news archive, scored against the mind
graph; fast expiry (48h); full tact filter (§6.2).  The sharpest edge here.
4.4 **Self-state** — completed ingests, deployed graphs, gap reports, digests:
"I built myself a news tool overnight."  Cheap, honest, observed aliveness.  72h.
4.5 **Reflection questions** — "what higher-level questions am I holding?" in
idle reflection; stored as pointers.

## 5. Scoring (deterministic; weights in config, not code)

`salience = w_t·timeliness + w_r·fit − w_n·novelty_penalty − w_s·sensitivity_penalty − w_f·fatigue`

Timeliness: high inside `time_window`, decays past it; no window → mild
freshness decay.  Novelty: penalise channel repetition (two news openers in a
row) and topic overlap with recent working-memory markers.  Fatigue:
`raised_count` discounts steeply, `deflected` harder; **two deflections retire
the item**.  Queue hard-capped (default 12), lowest-salience pruned on insert;
expired items removed at every dreaming pass.

## 6. Governors

6.1 **Frequency:** eligible at conversation open with probability *p* (default
0.6) OR whenever the opener is an explicit invitation (bypasses *p*).  Never
twice in one conversation unless the user engaged the first and asks for more.
6.2 **Tact (news + third parties):** severity×proximity gating (high-severity
local events admissible as *concern for the user* only, no event detail);
multiple-source rule (≥2 independent sources or not speakable — manufactured
alarm from a thin report is the worst failure here); grief-adjacent exclusion
(deaths/diagnoses/disasters touching named people in the graph are NEVER opener
material — stored respond-only so she answers knowledgeably if the user raises
them); surveillance-tone check (shared knowledge, never inference about the
user's movements).
6.3 Mid-conversation: §0.1 — the spontaneity lane, as built.

## 7. Verbalisation contract (the model's side)

One structured pointer injected beside the greeting context: the pointer, why it
matters, and the instruction — *raise it naturally if the opening gives room;
weave it as your own thought, not a report; if the user arrives with their own
agenda, drop it without mention.*  Exactly one item, never the queue, never
channel metadata.  A drop returns the item undiscounted (dropping for fit is
correct behaviour).  Post-turn, the outcome judge maps what happened to
`last_outcome`.

## 8. Feedback

`engaged` nudges the channel's per-user weight up, `deflected` down — bounded so
no channel can die or dominate (IN5).  A channel producing nothing admissible
for 14 days files a Gap Report instead of starving silently.

## 9. Privacy

Queue lives in memory.db (per-profile), inspectable and clearable from the
panel; "why did you ask me that?" answerable truthfully from `grounding`; no
item may be built from data classes denied to conversational use.

## 10. Acceptance criteria

- [ ] Empty queue + "what's new?" → graceful honest answer, not mirroring, not fabrication (§0.8 prompt + injection tests).
- [ ] A planted open-loop item is raised within its window, never before, never after expiry.
- [ ] A single-source high-severity news item is NOT speakable; corroborated, it becomes speakable as concern without detail.
- [ ] A grief_adjacent item is never initiated but is available when the user raises it.
- [ ] Two deflections retire an item; never twice in one conversation (unless engaged + invited).
- [ ] "Why did you ask me that?" yields grounding truthfully.
- [ ] Across 20 simulated greetings, initiative rate ≈ *p*; explicit invitations always bypass.
