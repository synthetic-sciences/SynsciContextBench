"""Unit tests for the IR metrics — the math the headline numbers depend on."""
from __future__ import annotations

import math

from benchmarks.scoring.metrics import (
    AggregateMetrics,
    QueryEvaluation,
    RetrievalResult,
    aggregate,
    average_precision,
    dcg_at_k,
    evaluate_query,
    mrr,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    success_at_k,
)


def _r(idx: str, relevant: bool, grade: int = 0) -> RetrievalResult:
    return RetrievalResult(id=idx, score=1.0, content="", is_relevant=relevant, relevance_grade=grade)


def test_precision_at_k_divides_by_k() -> None:
    results = [_r("a", True), _r("b", False), _r("c", True)]
    assert precision_at_k(results, 2) == 0.5  # 1 of top-2
    assert precision_at_k(results, 4) == 0.5  # 2 relevant / k=4
    assert precision_at_k(results, 0) == 0.0


def test_recall_dedups_by_id() -> None:
    # Two relevant hits but same id -> counts once.
    results = [_r("dup", True), _r("dup", True), _r("x", True)]
    assert recall_at_k(results, 10, total_relevant=2) == 1.0
    assert recall_at_k([], 10, total_relevant=0) == 0.0


def test_recall_capped_at_one() -> None:
    results = [_r("a", True), _r("b", True), _r("c", True)]
    assert recall_at_k(results, 10, total_relevant=2) == 1.0


def test_mrr() -> None:
    assert mrr([_r("a", False), _r("b", True)]) == 0.5
    assert mrr([_r("a", True)]) == 1.0
    assert mrr([_r("a", False)]) == 0.0


def test_dcg_and_ndcg() -> None:
    # First-position graded hit -> perfect nDCG against ideal of one rel doc.
    results = [_r("a", True, grade=2)]
    assert dcg_at_k(results, 1) == 2.0 / math.log2(2)
    assert ndcg_at_k(results, 1, total_relevant=1, max_grade=2) == 1.0
    # Same hit at rank 2 scores lower.
    results2 = [_r("x", False, 0), _r("a", True, 2)]
    assert ndcg_at_k(results2, 2, total_relevant=1, max_grade=2) < 1.0


def test_average_precision() -> None:
    # rel at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    results = [_r("a", True), _r("b", False), _r("c", True)]
    expected = (1.0 + (2 / 3)) / 2
    assert abs(average_precision(results, total_relevant=2) - expected) < 1e-9


def test_success_and_r_precision() -> None:
    results = [_r("a", False), _r("b", True)]
    assert success_at_k(results, 2) == 1.0
    assert success_at_k(results, 1) == 0.0
    # R-precision at R=2: top-2 has 1 relevant -> 0.5
    assert r_precision(results, total_relevant=2) == 0.5


def test_evaluate_and_aggregate() -> None:
    qe1 = QueryEvaluation(
        query="q1", engine="mock",
        results=[_r("a", True, 2), _r("b", False, 0)],
    )
    qe2 = QueryEvaluation(
        query="q2", engine="mock",
        results=[_r("c", False, 0), _r("d", True, 2)],
    )
    for qe, tr in ((qe1, 1), (qe2, 1)):
        evaluate_query(qe, [1, 3], tr)
    agg = aggregate([qe1, qe2], [1, 3], per_query_total_relevant=[1, 1])
    assert isinstance(agg, AggregateMetrics)
    assert agg.num_queries == 2
    # qe1 MRR=1, qe2 MRR=0.5 -> avg 0.75
    assert abs(agg.avg_mrr - 0.75) < 1e-9
    assert agg.map_score > 0
    assert 1 in agg.avg_success_at


def test_empty_aggregate_is_safe() -> None:
    agg = aggregate([], [1, 3])
    assert agg.num_queries == 0
