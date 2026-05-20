# Phases

Eleven phases, each measuring one axis of context-engine quality. The runner
executes them in order; you can isolate any one with `--<phase>-only` or
suppress it with `--skip-<phase>`.

## Phase 1 — Retrieval Quality

| | |
|--|--|
| **Code** | `benchmarks/runner.py:run_retrieval_benchmark` |
| **Dataset** | `benchmarks/datasets/curated/retrieval_ground_truth.json` |
| **Cases** | 100 queries across 15 popular Python repos |
| **Metrics** | MRR, Precision@K, Recall@K, NDCG@K (K ∈ {1, 3, 5, 10}) |

Ground truth is file-path + keyword-set per query. Recall@K de-duplicates
results by `id` so an engine returning multiple chunks per file can no
longer exceed 1.0.

## Phase 2 — Multi-Hop Retrieval

| | |
|--|--|
| **Code** | `benchmarks/phases/multihop.py` |
| **Dataset** | `benchmarks/datasets/curated/multihop_test_cases.json` |
| **Cases** | 100 queries requiring context from 2+ files or repos |
| **Metrics** | hop coverage, average hop MRR, hop recall@K |

## Phase 3 — Code QA

| | |
|--|--|
| **Code** | `benchmarks/phases/code_qa.py` |
| **Dataset** | `benchmarks/datasets/curated/code_qa_test_cases.json` |
| **Cases** | 100 questions about definitions, call sites, imports, inheritance, return types |
| **Metrics** | accuracy (LLM judge or structural keyword match) |

## Phase 4 — Adversarial Near-Miss

| | |
|--|--|
| **Code** | `benchmarks/phases/adversarial.py` |
| **Dataset** | `benchmarks/datasets/curated/adversarial_test_cases.json` |
| **Cases** | 100 queries with decoys (same name / wrong context, test vs prod, version-X vs version-Y) |
| **Metrics** | discrimination score — does the engine prefer the right one over its near-miss? |

## Phase 5 — Hallucination Rate

| | |
|--|--|
| **Code** | `benchmarks/phases/hallucination.py` |
| **Dataset** | `benchmarks/datasets/curated/hallucination_test_cases.json` |
| **Cases** | 100 prompts where the LLM is invited to make up an API |
| **Metrics** | overall rate, true-hallucination rate, abstention rate, context-miss rate, per-error-type breakdown |

Lower is better. The "true" rate subtracts cases where the engine simply
abstained from inventing.

## Phase 6 — Validated Datasets

| | |
|--|--|
| **Code** | `benchmarks/phases/validated_eval.py` |
| **Dataset** | `benchmarks/datasets/validated/*` (CodeSearchNet, CoSQA, AdvTest, CodeFeedback-ST, StackOverflow-QA, APPS) |
| **Cases** | ~100/dataset by default; configurable |
| **Metrics** | MRR + the rest of the Phase 1 metric set |

Match modes: `content` (token similarity), `file` (file path), `hybrid`
(either), `llm` (Claude/Gemini judge). The LLM judge respects
`--judge-top-k` (default 10); previously hard-coded to 3, which silently
forced rank-4+ results to be irrelevant.

## Phase 7 — LLM-as-Judge (3D, fairness pass)

| | |
|--|--|
| **Code** | `benchmarks/judges/llm_judge.py` |
| **Dataset** | reuses validated-dataset chunks |
| **Cases** | ~100/dataset |
| **Metrics** | relevance + completeness + faithfulness, judged blind |

## Phase 8 — Enhanced Judge (position-debiased, 4D + RAGAS)

| | |
|--|--|
| **Code** | `benchmarks/judges/enhanced_judge.py` |
| **Dataset** | reuses validated-dataset chunks |
| **Cases** | up to 200/dataset, 2× passes per case with shuffled chunk order |
| **Metrics** | relevance + completeness + specificity + faithfulness, plus RAGAS-style context precision/recall, plus inter-pass position consistency |

## Phase 9 — SWE-Agent

| | |
|--|--|
| **Code** | `benchmarks/phases/swe_agent.py` (+ `swe_real_patch.py`) |
| **Dataset** | `benchmarks/datasets/curated/swe_agent_test_cases.json` |
| **Cases** | 25, stratified by knowledge tier (A: well-known, B: niche, C: version-specific) |
| **Metrics** | 4D judge composite (correctness, completeness, code-quality, no-hallucination), structural pass rate, context utilization |

Each case runs once **without context** (baseline) and once with each engine,
so the report quantifies the engine's value-add directly.

### Phase 9b — Real-patch (opt-in)

`--real-patch` enables `swe_real_patch.run_real_patch`. The harness clones
the case's repo, applies the generated patch (or overwrites the target file
for standalone solutions), and runs the case's test command in a sandbox.
Test commands are screened by a safety filter; cases without `repo_url` and
`test_command` are skipped.

## Phase 10 — Atlas Workflow

| | |
|--|--|
| **Code** | `benchmarks/phases/atlas.py` |
| **Dataset** | `benchmarks/datasets/curated/atlas_test_cases.json` |
| **Cases** | 20 across 8 categories |
| **Metrics** | anchor hit, evidence recall, optional LLM judge, hallucination signals, composite |

### Categories

| Category | Tests |
|----------|-------|
| `tool_contract` | MCP tool schemas, required parameters |
| `graph_memory` | Recall prior nodes, hypotheses, outcomes |
| `artifact` | Locate the table/plot/log/diff that supports a claim |
| `paper_qa` | Answer with paper citations |
| `multi_turn` | Continue work from an existing branch |
| `prior_decision` | Find the rationale behind a choice |
| `avoid_repeat` | Surface prior failed experiments |
| `synthesis` | Combine paper + graph + code context |

Composite without judge: `0.5 * anchor_hit + 0.5 * evidence_recall`.
With judge: `0.4 * anchor_hit + 0.4 * evidence_recall + 0.2 * judge_score`.

## Phase 11 — Real-Session Replay

| | |
|--|--|
| **Code** | `benchmarks/phases/session_replay.py` |
| **Dataset** | `benchmarks/datasets/curated/session_replay_cases.json` |
| **Cases** | 10 — production-session moments where one engine visibly beat another |
| **Metrics** | win rate, regression resolution rate, labeled vs re-classified cause |

Each case names the engine that originally won and the engine that
originally lost. The replay re-classifies failures under the live
`scoring.failure_taxonomy` so the report can show "10 cases labeled
`bad_retrieval`, now 6 still `bad_retrieval` / 2 `bad_ranking` / 2 resolved."

## Post-phase aggregation

After all phases finish the runner emits two cross-cutting reports:

- **Per-category leaderboards** (`scoring/leaderboards.py`):
  `code_retrieval`, `docs_lookup`, `paper_qa`, `atlas_graph`,
  `tool_contract`, `swe_patch`, `context_utilization`,
  `hallucination_inverted`. Each is sorted independently.
- **Failure taxonomy** (`scoring/failure_taxonomy.py`): per-engine bucket
  counts across all phases plus a representative slice of failure
  examples.

And, when two or more engines are configured, **pairwise statistical
significance** (`scoring/statistical_analysis.py`): paired t-test,
Wilcoxon signed-rank, bootstrap CI, Cohen's d, Cliff's delta with Holm
correction.
