# Retrieval eval harness

The regression gate for the knowledge-host read path (retrieval contract §8). **No
threshold tuning or "improvement" to retrieval is trusted without a run of this.**

## Run it

```bash
python -m knowledgehost eval                 # score the current read path (baseline)
python -m knowledgehost eval --trace         # + per-query top card ids (use this to grade)
python -m knowledgehost eval --retriever current_path --gold eval/gold.jsonl
```

Prints the report and writes a JSON artifact to `eval-runs/<run_id>.json` (or
`eval_out_dir`). Every retriever is scored through the same `retrieval.retrieve` seam,
so when we add card-BM25 / a cross-encoder / a trained query→card linker they register
as new retrievers and compare against `current_path` on identical gold.

## The headline metrics (what actually gates a change)

| metric | meaning | accept |
|---|---|---|
| **false-confidence** | Tier-1 answers whose top card is graded 0–1 | **≤ 2%** |
| **out-of-graph correctness** | unanswerable queries returned Tier 3/4 | **≥ 90%** |
| abstention overreach | answerable queries wrongly abstained | report, ≤ 25% target |
| recall@5 / recall@20 / nDCG@10 | ranking on answerable queries | track |

A run "FAILs" if false-confidence > 2% or out-of-graph < 90%, *regardless of nDCG*.

## Grading the gold set (the maintainer, ~2h, one time)

`gold.jsonl` ships with the **out-of-graph** rows already usable (they need no card grades
— their gold is "the system abstains"). The **in-graph** rows are templates with empty
`judgments`; fill them:

1. `python -m knowledgehost eval --trace` — see the top card id each query returns.
2. For each in-graph query, add the relevant card ids to `judgments` with a grade:
   `3` directly answers · `2` partially · `1` related context · `0` irrelevant.
3. Aim for ≥60 total (100+ ideal), ~15% out-of-graph, spread across the categories.

Candidate queries may be LM-drafted, but **grades are reviewed by a human once** — do not
self-certify (contract §8.1).

## Fixture schema

```json
{"id":"fact01","query":"…","category":"lookup_fact",
 "judgments":{"<card_id>":3,"<card_id>":1},
 "context_features":{"trigger":"…"},        // optional, drives the fit-gate
 "caller":"reflection","mode":null,"rigor":null,"note":"…"}
```

`category ∈ lookup_fact | lookup_definition | relation | change | multi_hop | out_of_graph`.
