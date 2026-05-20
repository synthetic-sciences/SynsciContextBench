# datasets/curated/

Hand-built test cases owned by this repo. These are edited by humans, ship
with the repo, and back the phases that are most specific to real agent
context use.

| File | Phase | Cases | Notes |
|------|-------|------:|-------|
| `retrieval_ground_truth.json` | 1 (retrieval) | 100 | File+keyword ground truth across 15 popular repos |
| `multihop_test_cases.json` | 2 | 100 | Cross-file hops |
| `code_qa_test_cases.json` | 3 | 100 | Function/symbol QA |
| `adversarial_test_cases.json` | 4 | 100 | Same name / wrong context decoys |
| `hallucination_test_cases.json` | 5 | 100 | Anti-fabrication checks |
| `swe_agent_test_cases.json` | 9 | 25 | Real-shape SWE tasks, stratified by knowledge tier |
| `atlas_test_cases.json` | 10 | 20 | Tool contracts / graph memory / paper QA / synthesis |
| `session_replay_cases.json` | 11 | 10 | Real production-session losses with labeled causes |

## Schema reference

Every curated dataset starts with metadata fields:

```jsonc
{
  "_description": "what this dataset measures",
  "_methodology": "how cases were authored, how scoring works",
  "_categories": { "...": "..." },  // optional, when cases are bucketed
  "test_cases": [
    { ... per-case fields ... }
  ]
}
```

Per-case fields vary by phase but always include `id`, `query` (or `question`),
and at least one ground-truth signal (`expected_files`, `expected_evidence`,
`expected_anchors`, `acceptance_criteria`, etc.).

## atlas_test_cases.json

Categories: `tool_contract`, `graph_memory`, `artifact`, `paper_qa`,
`multi_turn`, `prior_decision`, `avoid_repeat`, `synthesis`.

Each case has:
- `question` — what an agent would ask
- `expected_evidence` — keyword set that should appear in retrieved content
- `expected_anchors` — paths / IDs the right surface would name
- `judge_rubric` — optional LLM rubric for the Atlas judge
- `negative_signals` — phrases that indicate fabrication

## session_replay_cases.json

Each case names the engine that originally **won** and the engine that
originally **lost**, along with the labeled failure cause and the
minimum-relevance threshold the replay must clear to count as a "resolved"
regression.

## adversarial_test_cases.json (and other Phase 4 cases)

These are decoys: the query targets a specific symbol/file, but multiple
look-alikes exist. Discrimination score rewards engines that surface the
correct one over its near-misses.
