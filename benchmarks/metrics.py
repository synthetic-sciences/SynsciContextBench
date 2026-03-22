"""Information Retrieval metrics for benchmark evaluation.

Implements NDCG, MRR, Precision@K, Recall@K, MAP, Success@K, R-Precision,
and hallucination rate.

References:
    Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot
        Evaluation of Information Retrieval Models. NeurIPS Datasets Track.
    Husain et al. (2019). CodeSearchNet Challenge. arXiv:1909.09436.
    Urbano et al. (2019). On the Measurement of Test Collection Reliability. SIGIR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieved item with its relevance."""

    id: str
    score: float
    content: str
    is_relevant: bool = False  # set by ground-truth matching
    relevance_grade: int = 0  # 0=irrelevant, 1=partial, 2=exact


@dataclass
class QueryEvaluation:
    """Evaluation result for a single query."""

    query: str
    engine: str
    results: list[RetrievalResult]
    latency_ms: float = 0.0

    # Computed metrics (populated by evaluate())
    precision_at: dict[int, float] = field(default_factory=dict)
    recall_at: dict[int, float] = field(default_factory=dict)
    ndcg_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


@dataclass
class HallucinationResult:
    """Result from a single hallucination test case."""

    query: str
    engine: str
    generated_code: str
    errors: list[dict]  # [{type: "invented_method", detail: "..."}]
    hallucination_rate: float = 0.0


# ---------------------------------------------------------------------------
# Core IR metrics
# ---------------------------------------------------------------------------


def precision_at_k(results: list[RetrievalResult], k: int) -> float:
    """Fraction of top-k results that are relevant.

    Divides by k (not the number of returned results) so that engines
    returning fewer results are not artificially inflated.
    """
    if k == 0:
        return 0.0
    top_k = results[:k]
    return sum(1 for r in top_k if r.is_relevant) / k


def recall_at_k(results: list[RetrievalResult], k: int, total_relevant: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if total_relevant == 0:
        return 0.0
    top_k = results[:k]
    found = sum(1 for r in top_k if r.is_relevant)
    return found / total_relevant


def mrr(results: list[RetrievalResult]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result."""
    for i, r in enumerate(results):
        if r.is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(results: list[RetrievalResult], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    total = 0.0
    for i, r in enumerate(results[:k]):
        total += r.relevance_grade / math.log2(i + 2)  # i+2 because log2(1)=0
    return total


def ndcg_at_k(
    results: list[RetrievalResult],
    k: int,
    total_relevant: int,
    max_grade: int = 2,
) -> float:
    """Normalized DCG — compares actual ranking to the corpus-level ideal.

    The ideal DCG is built from ALL known relevant documents (not just the
    ones the engine returned).  This follows TREC / BEIR convention and
    prevents inflated scores when an engine returns few results.
    """
    actual_dcg = dcg_at_k(results, k)

    # Build corpus-level ideal: total_relevant docs at max_grade, ranked
    # perfectly, truncated to k positions.
    ideal_count = min(total_relevant, k)
    ideal_dcg = sum(
        max_grade / math.log2(i + 2) for i in range(ideal_count)
    )

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(results: list[RetrievalResult], total_relevant: int) -> float:
    """Average Precision (AP) for a single query.

    AP = (1/R) * sum_{k=1}^{N} (Precision@k * rel_k)

    This is the BEIR standard metric alongside NDCG@10 (Thakur et al. 2021).
    More sensitive to recall than NDCG because it considers all relevant docs.
    """
    if total_relevant == 0:
        return 0.0
    score = 0.0
    relevant_so_far = 0
    for i, r in enumerate(results):
        if r.is_relevant:
            relevant_so_far += 1
            score += relevant_so_far / (i + 1)
    return score / total_relevant


def success_at_k(results: list[RetrievalResult], k: int) -> float:
    """Success@K: 1 if any relevant result in top-K, else 0.

    Binary metric used alongside MRR in CodeSearchNet evaluations.
    """
    return 1.0 if any(r.is_relevant for r in results[:k]) else 0.0


def r_precision(results: list[RetrievalResult], total_relevant: int) -> float:
    """R-Precision: Precision at rank R where R = total relevant docs.

    Standard TREC metric. If there are R relevant documents, measures
    the fraction of top-R results that are relevant.
    """
    if total_relevant == 0:
        return 0.0
    top_r = results[:total_relevant]
    return sum(1 for r in top_r if r.is_relevant) / total_relevant


def evaluate_query(
    query_eval: QueryEvaluation,
    k_values: list[int],
    total_relevant: int,
) -> QueryEvaluation:
    """Compute all metrics for a single query evaluation."""
    for k in k_values:
        query_eval.precision_at[k] = precision_at_k(query_eval.results, k)
        query_eval.recall_at[k] = recall_at_k(query_eval.results, k, total_relevant)
        query_eval.ndcg_at[k] = ndcg_at_k(query_eval.results, k, total_relevant)
    query_eval.mrr = mrr(query_eval.results)
    return query_eval


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries for one engine."""

    engine: str
    num_queries: int = 0
    avg_precision_at: dict[int, float] = field(default_factory=dict)
    avg_recall_at: dict[int, float] = field(default_factory=dict)
    avg_ndcg_at: dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # BEIR-standard metrics (Thakur et al. 2021)
    map_score: float = 0.0  # Mean Average Precision
    avg_success_at: dict[int, float] = field(default_factory=dict)  # Success@K
    avg_r_precision: float = 0.0  # R-Precision

    # Hallucination (optional)
    hallucination_rate: float | None = None
    error_breakdown: dict[str, int] = field(default_factory=dict)


def aggregate(
    evaluations: list[QueryEvaluation],
    k_values: list[int],
    per_query_total_relevant: list[int] | None = None,
) -> AggregateMetrics:
    """Aggregate metrics across multiple query evaluations.

    Args:
        evaluations: Per-query evaluations
        k_values: Which K values were evaluated
        per_query_total_relevant: Total relevant docs per query (for MAP/R-Prec).
            If provided, computes BEIR-standard MAP, Success@K, R-Precision.
    """
    if not evaluations:
        return AggregateMetrics(engine="unknown")

    engine = evaluations[0].engine
    n = len(evaluations)
    agg = AggregateMetrics(engine=engine, num_queries=n)

    for k in k_values:
        agg.avg_precision_at[k] = sum(e.precision_at.get(k, 0) for e in evaluations) / n
        agg.avg_recall_at[k] = sum(e.recall_at.get(k, 0) for e in evaluations) / n
        agg.avg_ndcg_at[k] = sum(e.ndcg_at.get(k, 0) for e in evaluations) / n

    agg.avg_mrr = sum(e.mrr for e in evaluations) / n

    latencies = sorted(e.latency_ms for e in evaluations)
    agg.avg_latency_ms = sum(latencies) / n
    agg.p95_latency_ms = latencies[int(n * 0.95)] if n > 1 else latencies[0]

    # BEIR-standard extended metrics (Thakur et al. 2021)
    # Success@K (always computable)
    for k in k_values:
        agg.avg_success_at[k] = sum(success_at_k(e.results, k) for e in evaluations) / n

    if per_query_total_relevant and len(per_query_total_relevant) == n:
        # MAP: Mean Average Precision
        ap_scores = [
            average_precision(e.results, tr) for e, tr in zip(evaluations, per_query_total_relevant)
        ]
        agg.map_score = sum(ap_scores) / n

        # R-Precision
        agg.avg_r_precision = (
            sum(r_precision(e.results, tr) for e, tr in zip(evaluations, per_query_total_relevant))
            / n
        )

    return agg


def aggregate_hallucinations(
    results: list[HallucinationResult],
) -> tuple[float, dict[str, int]]:
    """Aggregate hallucination results.

    Returns:
        (overall_hallucination_rate, error_type_counts)
    """
    if not results:
        return 0.0, {}

    total_errors = 0
    total_checks = len(results)
    breakdown: dict[str, int] = {}

    for r in results:
        if r.errors:
            total_errors += 1
        for err in r.errors:
            err_type = err.get("type", "unknown")
            breakdown[err_type] = breakdown.get(err_type, 0) + 1

    rate = total_errors / total_checks
    return rate, breakdown
