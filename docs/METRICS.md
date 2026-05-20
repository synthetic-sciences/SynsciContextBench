# Metrics

Reference for every score the harness reports. Higher is better unless
explicitly noted.

## IR metrics — `benchmarks/scoring/metrics.py`

These follow the BEIR / Husain conventions.

### MRR

```
MRR(query) = 1 / rank_of_first_relevant_result
```

`0` if no result is relevant. Reported as the mean over queries.

### Precision@K

```
Precision@K = #{relevant in top K} / K
```

Divides by K (not the number of returned results) so engines returning
fewer than K items are not artificially inflated.

### Recall@K

```
Recall@K = unique_relevant_in_top_K / total_relevant
```

Two fixes from the diagnosis:

1. We de-duplicate by `RetrievalResult.id` before counting, so multiple
   chunks from one source file count once.
2. We clamp at 1.0 as a defense-in-depth check.

Without these, an engine returning five chunks per relevant file produced
`Recall@10 ≈ 5 × total_relevant / total_relevant ≈ 5`, which is
mathematically impossible.

### NDCG@K

```
DCG@K  = Σ_{i=1..K}  grade_i / log2(i + 1)
IDCG@K = Σ_{i=1..K}  max_grade / log2(i + 1)   (over min(total_relevant, K) docs)
NDCG@K = DCG@K / IDCG@K
```

Grades are 0 (irrelevant), 1 (file or 2+ keyword match), 2 (file + keyword).
Ideal DCG uses the corpus-level relevant count, not just what the engine
returned — per TREC/BEIR convention.

### Average Precision, MAP, Success@K, R-Precision

Standard. Implemented in `scoring/metrics.py:average_precision`,
`success_at_k`, `r_precision`. MAP is the mean of per-query AP.

## Relevance grade — `benchmarks/runner.py:_match_relevance`

Stricter than the previous OR-of-two-booleans implementation:

- File match alone        → grade 1
- File match + keyword    → grade 2
- 2+ distinct keyword hits → grade 1
- Otherwise               → grade 0

A single keyword match no longer flips relevance, because tokens like
`BaseModel` or `Lock` would otherwise mark any chunk from the target
library as relevant and inflate recall.

## Context-grounding signals — `benchmarks/scoring/context_grounding.py`

### `citation_count`, `cited_chunks`, `citation_share`

Detects `[1]`, `chunk N`, `source N`, `according to file.py`, and similar
forms. `cited_chunks / total_chunks` is the share.

### `utilization`

```
utilization = facts_used / facts_available
```

A "fact" is an identifiable symbol the chunks introduce (function name,
class name, import target, decorator, parameter name). The Phase 9
`swe_agent` aggregator already uses this; the helper is shared so any
phase can call it.

### `answer_change`

`1 - SequenceMatcher(no_context_answer, with_context_answer).ratio()`,
clamped to [0,1]. Zero means context did nothing.

### `hallucination_reduction`

`negative_signals_before - negative_signals_after`. Positive means
context prevented some fabrications.

## Atlas composite — `benchmarks/phases/atlas.py`

For one case:

- `anchor_hit`         ∈ {0, 1} — 1 if any expected anchor appears in
  retrieved chunk path or content.
- `evidence_recall`    ∈ [0, 1] — fraction of expected evidence terms
  surfaced anywhere across chunks.
- `judge_score`        ∈ [0, 1] — optional LLM rubric. 0 if no judge
  creds.
- `hallucination_signals` ∈ ℕ — count of negative-signal phrases matched.

Composite:

```
with judge   :  0.4 * anchor + 0.4 * evidence_recall + 0.2 * judge
without judge:  0.5 * anchor + 0.5 * evidence_recall
```

Without a judge the judge's weight is folded into the structural terms so
the composite magnitudes are still comparable.

## SWE-Agent composite — `benchmarks/phases/swe_agent.py`

Per case: `(correctness + completeness + code_quality + no_hallucination) / 4`,
each scored 0-5 by the judge. Reported on a 0-1 scale by dividing by 5.
Position debiasing runs each case twice with the chunk order swapped and
averages.

## Real-patch — `benchmarks/phases/swe_real_patch.py`

When `--real-patch` is on and a case ships `repo_url` + `test_command`,
the harness clones, applies, runs the test command in a sandbox, and
reports:

- `success`           — exit code 0
- `test_pass_rate`    — parsed pytest `N passed, M failed`
- `runtime_ms`        — full wall-clock
- `skipped_reason`    — populated on safety-filter rejections, missing
                        fields, clone failures, etc.

## Session-replay — `benchmarks/phases/session_replay.py`

Per case:

- `score = 0.5 * anchor_hit + 0.5 * evidence_recall`
- `passed_threshold = score >= case.minimum_relevance`
- `regression_resolved = is_loser AND passed_threshold`
- `re_classified_cause` — bucket from the live failure taxonomy when
  `passed_threshold` is false

Per engine:

- `win_rate` — share of cases that cleared their threshold
- `historical_losses` — count of cases where this engine was the labeled loser
- `losses_resolved` / `losses_still_failing` — split of those losses now

## Latency — `benchmarks/infra/latency.py`

`LatencyMeter` reports five numbers per call: `total_ms`, `request_ms`,
`retry_ms`, `sleep_ms`, `other_ms`. The number adapters return must be
`total_ms` (full user-visible wall-clock) so rate-limit sleeps and retry
backoffs are accounted for. The Nia and Context7 adapters now do this by
construction.

## Statistical significance — `benchmarks/scoring/statistical_analysis.py`

For each metric where ≥ 2 engines have per-query scores:

- Paired t-test
- Wilcoxon signed-rank
- Bootstrap 95 % CI (10K resamples by default)
- Cohen's d
- Cliff's delta
- Holm correction over the family of pairwise comparisons
