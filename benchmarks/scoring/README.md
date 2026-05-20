# scoring/

Deterministic scoring, metrics, leaderboards, and analysis.

```
scoring/
├── metrics.py                core IR metrics (MRR, NDCG, P@K, R@K, MAP, ...)
├── semantic_metrics.py       CodeBLEU + AST similarity for code-shaped results
├── context_grounding.py      citation / utilization / hallucination-reduction
├── leaderboards.py           per-category leaderboards (replaces single-winner)
├── failure_taxonomy.py       classify failures into actionable buckets
└── statistical_analysis.py   paired tests, bootstrap CIs, effect sizes
```

## Why no judge in here?

`scoring/` is for deterministic, reproducible numbers. The LLM-driven judges
live in [`judges/`](../judges/README.md) so reviewers can reason about each
half separately. Phases blend the two in their report dataclasses.

## Per-module guide

### `metrics.py`

Implements MRR, NDCG@K, Precision@K, Recall@K, Average Precision, MAP,
Success@K, R-Precision. References: Husain et al. 2019, Thakur et al. 2021.

Two diagnosis-driven fixes live here:
- `recall_at_k` now de-duplicates by `result.id` and clamps at 1.0 — the
  old implementation could produce values > 1 when multiple chunks shared
  one ground-truth source file.
- `_match_relevance` (in `runner.py`, but mirrored here in spirit) no
  longer awards relevance on a single keyword match; single tokens like
  `BaseModel` would otherwise mark almost any chunk from that library as
  relevant.

### `context_grounding.py`

Four signals the diagnosis flagged as missing:
- `citation_share`     — fraction of retrieved chunks the answer references
- `utilization`        — fraction of identifiable facts in chunks that
                         appear in the answer
- `answer_change`      — how different an answer is with vs. without context
- `hallucination_reduction` — negative-signal hits removed by adding context

### `leaderboards.py`

The diagnosis was explicit: single-line "Engine X wins" hides reality.
Eight leaderboards are emitted side-by-side so an engine winning code
retrieval but losing the diff-aware phase-context is visibly shown to lose:

`code_retrieval`, `docs_lookup`, `paper_qa`, `diff_aware_graph`, `tool_contract`,
`swe_patch`, `context_utilization`, `hallucination_inverted`.

### `failure_taxonomy.py`

Every failure is bucketed into one of:

- `missing_index_coverage` — engine returned no results
- `bad_retrieval`          — results returned, none relevant
- `bad_ranking`            — relevant present but outside the reporting window
- `bad_packaging`          — right symbol, wrong neighborhood
- `tool_ergonomics`        — adapter error / required knob missing
- `benchmark_blind_spot`   — correct surface, rubric still scored low

### `statistical_analysis.py`

Paired t-test, Wilcoxon signed-rank, bootstrap CIs (10K resamples by
default), Cohen's d, Cliff's delta, Bonferroni / Holm correction.
