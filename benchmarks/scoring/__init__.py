"""Scoring, metrics, and analysis helpers.

- ``metrics``               Core IR metrics: MRR, NDCG, Precision@K, Recall@K,
                            MAP, Success@K, R-Precision (BEIR / Husain et al.).
- ``semantic_metrics``      CodeBLEU + AST similarity, used as auxiliary
                            signals for code-shaped results.
- ``context_grounding``     Citation detection, fact utilization,
                            answer-change-with-context, and
                            hallucination-reduction signals.
- ``leaderboards``          Per-category leaderboards. Replaces single-winner
                            reporting so an engine that wins code retrieval
                            but loses the diff-aware phase context is visibly shown to lose.
- ``failure_taxonomy``      Classifies every failure into one of six buckets
                            (missing_index_coverage, bad_retrieval,
                            bad_ranking, bad_packaging, tool_ergonomics,
                            benchmark_blind_spot).
- ``statistical_analysis``  Paired t-test, Wilcoxon, bootstrap CIs, Cohen's d,
                            Cliff's delta, Bonferroni / Holm correction.
"""

from .context_grounding import GroundingMetrics, grounding_metrics
from .failure_taxonomy import build_failure_taxonomy, print_failure_taxonomy
from .leaderboards import build_leaderboards, print_leaderboards
from .metrics import (
    AggregateMetrics,
    HallucinationResult,
    QueryEvaluation,
    RetrievalResult,
    aggregate,
    aggregate_hallucinations,
    evaluate_query,
)

__all__ = [
    "AggregateMetrics",
    "GroundingMetrics",
    "HallucinationResult",
    "QueryEvaluation",
    "RetrievalResult",
    "aggregate",
    "aggregate_hallucinations",
    "build_failure_taxonomy",
    "build_leaderboards",
    "evaluate_query",
    "grounding_metrics",
    "print_failure_taxonomy",
    "print_leaderboards",
]
