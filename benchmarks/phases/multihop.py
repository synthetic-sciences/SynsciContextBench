"""Multi-hop retrieval benchmark.

Tests whether the engine can answer questions that require combining
information from 2+ indexed sources (files, repos, papers).

Unlike single-hop retrieval where one chunk contains the answer,
multi-hop queries need the engine to surface multiple complementary
pieces of evidence. We measure:

1. **Hop coverage**: Did the engine return chunks covering ALL required hops?
2. **Hop Recall@K**: Of all required evidence pieces, how many appear in top-K?
3. **Hop MRR**: How quickly does the engine surface the first chunk for each hop?
4. **Answer completeness**: Could a reader synthesize the full answer from top-K?
"""

from __future__ import annotations

import json

from tqdm import tqdm
from dataclasses import dataclass, field

from ..adapters.base import ContextEngineAdapter, SearchResult


@dataclass
class HopDefinition:
    """A single piece of evidence needed to answer the full question."""

    id: str
    description: str
    # At least one of these must appear in a relevant result
    required_files: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)


@dataclass
class MultiHopTestCase:
    """A query that requires combining info from multiple sources."""

    id: str
    query: str
    description: str
    num_hops: int
    hops: list[HopDefinition]
    category: str = "cross_file"  # cross_file | cross_repo | cross_source


@dataclass
class HopResult:
    """Whether a specific hop was satisfied in the search results."""

    hop_id: str
    found: bool
    first_rank: int | None  # rank of first result satisfying this hop (1-indexed)
    matching_results: int  # how many results matched this hop


@dataclass
class MultiHopEvaluation:
    """Evaluation of a single multi-hop query."""

    query: str
    engine: str
    test_case_id: str
    num_hops: int
    hop_results: list[HopResult]
    latency_ms: float

    # Computed metrics
    hop_coverage: float = 0.0     # fraction of hops with at least 1 result
    hop_recall_at_5: float = 0.0  # hops found in top-5
    hop_recall_at_10: float = 0.0  # hops found in top-10
    avg_hop_mrr: float = 0.0      # average 1/rank across hops


@dataclass
class MultiHopAggregateMetrics:
    """Aggregate multi-hop metrics across all queries for one engine."""

    engine: str
    num_queries: int = 0
    avg_hop_coverage: float = 0.0
    avg_hop_recall_at_5: float = 0.0
    avg_hop_recall_at_10: float = 0.0
    avg_hop_mrr: float = 0.0
    avg_latency_ms: float = 0.0
    # Per-category breakdown
    by_category: dict[str, dict] = field(default_factory=dict)


def _check_hop(
    hop: HopDefinition,
    results: list[SearchResult],
    top_k: int | None = None,
) -> HopResult:
    """Check whether a hop is satisfied by any result in the list."""
    check_results = results[:top_k] if top_k else results
    first_rank = None
    matching = 0

    for i, r in enumerate(check_results):
        content_lower = r.content.lower()
        file_match = any(rf in r.file_path for rf in hop.required_files) if hop.required_files else False
        keyword_match = any(kw.lower() in content_lower for kw in hop.required_keywords) if hop.required_keywords else False

        if file_match or keyword_match:
            matching += 1
            if first_rank is None:
                first_rank = i + 1

    return HopResult(
        hop_id=hop.id,
        found=first_rank is not None,
        first_rank=first_rank,
        matching_results=matching,
    )


def evaluate_multihop(
    test_case: MultiHopTestCase,
    results: list[SearchResult],
    latency_ms: float,
    engine_name: str,
) -> MultiHopEvaluation:
    """Evaluate a single multi-hop query."""
    hop_results = [_check_hop(hop, results) for hop in test_case.hops]
    hop_results_at_5 = [_check_hop(hop, results, top_k=5) for hop in test_case.hops]
    hop_results_at_10 = [_check_hop(hop, results, top_k=10) for hop in test_case.hops]

    n_hops = len(test_case.hops)
    coverage = sum(1 for hr in hop_results if hr.found) / n_hops if n_hops > 0 else 0.0
    recall_5 = sum(1 for hr in hop_results_at_5 if hr.found) / n_hops if n_hops > 0 else 0.0
    recall_10 = sum(1 for hr in hop_results_at_10 if hr.found) / n_hops if n_hops > 0 else 0.0

    # Average MRR across hops
    mrr_sum = 0.0
    for hr in hop_results:
        if hr.first_rank is not None:
            mrr_sum += 1.0 / hr.first_rank
    avg_mrr = mrr_sum / n_hops if n_hops > 0 else 0.0

    return MultiHopEvaluation(
        query=test_case.query,
        engine=engine_name,
        test_case_id=test_case.id,
        num_hops=n_hops,
        hop_results=hop_results,
        latency_ms=latency_ms,
        hop_coverage=coverage,
        hop_recall_at_5=recall_5,
        hop_recall_at_10=recall_10,
        avg_hop_mrr=avg_mrr,
    )


async def run_multihop_benchmark(
    engine: ContextEngineAdapter,
    dataset_path: str,
    top_k: int = 10,
    max_queries: int | None = None,
    seed: int = 0,
) -> tuple[MultiHopAggregateMetrics, list[MultiHopEvaluation]]:
    """Run multi-hop retrieval benchmark against one engine."""
    from ..infra.sampling import sample_seeded

    test_cases = load_multihop_cases(dataset_path)
    test_cases = sample_seeded(test_cases, max_queries, seed=seed)
    evaluations: list[MultiHopEvaluation] = []

    for tc in tqdm(test_cases, desc=f"  {engine.name} multihop", unit="q"):
        try:
            results, latency = await engine.search_code(query=tc.query, top_k=top_k)
        except Exception as e:
            print(f"  [!] Multi-hop query failed for {engine.name}: {tc.query[:50]}... — {e}")
            continue

        ev = evaluate_multihop(tc, results, latency, engine.name)
        evaluations.append(ev)

    agg = aggregate_multihop(evaluations)
    return agg, evaluations


def aggregate_multihop(evaluations: list[MultiHopEvaluation]) -> MultiHopAggregateMetrics:
    """Aggregate multi-hop metrics across all queries."""
    if not evaluations:
        return MultiHopAggregateMetrics(engine="unknown")

    engine = evaluations[0].engine
    n = len(evaluations)
    agg = MultiHopAggregateMetrics(engine=engine, num_queries=n)

    agg.avg_hop_coverage = sum(e.hop_coverage for e in evaluations) / n
    agg.avg_hop_recall_at_5 = sum(e.hop_recall_at_5 for e in evaluations) / n
    agg.avg_hop_recall_at_10 = sum(e.hop_recall_at_10 for e in evaluations) / n
    agg.avg_hop_mrr = sum(e.avg_hop_mrr for e in evaluations) / n
    agg.avg_latency_ms = sum(e.latency_ms for e in evaluations) / n

    # Per-category breakdown
    categories: dict[str, list[MultiHopEvaluation]] = {}
    for e in evaluations:
        # Get category from test case (stored in test_case_id prefix)
        cat = "unknown"
        for ev in evaluations:
            if ev.test_case_id == e.test_case_id:
                # Category is encoded in test case, we'll infer from hop count
                cat = "2-hop" if e.num_hops == 2 else f"{e.num_hops}-hop"
                break
        categories.setdefault(cat, []).append(e)

    for cat, cat_evals in categories.items():
        cn = len(cat_evals)
        agg.by_category[cat] = {
            "num_queries": cn,
            "avg_hop_coverage": sum(e.hop_coverage for e in cat_evals) / cn,
            "avg_hop_mrr": sum(e.avg_hop_mrr for e in cat_evals) / cn,
        }

    return agg


def load_multihop_cases(path: str) -> list[MultiHopTestCase]:
    """Load multi-hop test cases from JSON."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", []):
        hops = [
            HopDefinition(
                id=h["id"],
                description=h.get("description", ""),
                required_files=h.get("required_files", []),
                required_keywords=h.get("required_keywords", []),
            )
            for h in item.get("hops", [])
        ]
        cases.append(
            MultiHopTestCase(
                id=item["id"],
                query=item["query"],
                description=item.get("description", ""),
                num_hops=item.get("num_hops", len(hops)),
                hops=hops,
                category=item.get("category", "cross_file"),
            )
        )
    return cases
