"""Run-to-run consistency and query paraphrase stability analysis.

Measures:
1. Rank consistency: Run same queries twice, compute Kendall's tau on result rankings
2. Paraphrase stability: Same intent different wording, measure rank overlap (RBO)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..adapters.base import ContextEngineAdapter
from .logging_config import get_logger

logger = get_logger("consistency")


@dataclass
class ConsistencyResult:
    """Results from run-to-run consistency analysis."""

    engine: str = ""
    num_queries: int = 0
    avg_kendall_tau: float = 0.0
    avg_jaccard_at_5: float = 0.0
    avg_jaccard_at_10: float = 0.0
    avg_rbo: float = 0.0
    per_query: list[dict] = field(default_factory=list)


@dataclass
class ParaphraseResult:
    """Results from paraphrase stability analysis."""

    engine: str = ""
    num_pairs: int = 0
    avg_jaccard_at_5: float = 0.0
    avg_jaccard_at_10: float = 0.0
    avg_rbo: float = 0.0
    avg_rank_shift: float = 0.0
    per_pair: list[dict] = field(default_factory=list)


def _kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Compute Kendall's tau between two rankings (by item ID)."""
    # Build position maps
    all_items = list(set(ranking_a) | set(ranking_b))
    if len(all_items) < 2:
        return 1.0

    pos_a = {item: i for i, item in enumerate(ranking_a)}
    pos_b = {item: i for i, item in enumerate(ranking_b)}

    # Assign max rank for items not in a ranking
    max_rank = len(all_items)
    for item in all_items:
        pos_a.setdefault(item, max_rank)
        pos_b.setdefault(item, max_rank)

    concordant = 0
    discordant = 0
    for i in range(len(all_items)):
        for j in range(i + 1, len(all_items)):
            a_diff = pos_a[all_items[i]] - pos_a[all_items[j]]
            b_diff = pos_b[all_items[i]] - pos_b[all_items[j]]
            if a_diff * b_diff > 0:
                concordant += 1
            elif a_diff * b_diff < 0:
                discordant += 1

    n = concordant + discordant
    if n == 0:
        return 1.0
    return (concordant - discordant) / n


def _jaccard_at_k(results_a: list[str], results_b: list[str], k: int) -> float:
    """Jaccard similarity of top-k result sets."""
    set_a = set(results_a[:k])
    set_b = set(results_b[:k])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _rbo(ranking_a: list[str], ranking_b: list[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap — a top-weighted rank similarity measure.

    p controls the top-heaviness (0.9 = focus on top results).
    Returns a value in [0, 1].
    """
    depth = min(len(ranking_a), len(ranking_b))
    if depth == 0:
        return 0.0

    overlap = 0.0
    rbo_sum = 0.0
    for d in range(1, depth + 1):
        set_a = set(ranking_a[:d])
        set_b = set(ranking_b[:d])
        overlap = len(set_a & set_b) / d
        rbo_sum += p ** (d - 1) * overlap

    return (1 - p) * rbo_sum


async def run_consistency_check(
    engine: ContextEngineAdapter,
    queries: list[dict],
    top_k: int = 10,
) -> ConsistencyResult:
    """Run each query twice and measure ranking consistency.

    Args:
        engine: The engine to test
        queries: List of query dicts with at least 'id' and 'query' keys
        top_k: Number of results to retrieve per query
    """
    result = ConsistencyResult(engine=engine.name, num_queries=len(queries))
    taus = []
    j5s = []
    j10s = []
    rbos = []

    for q in queries:
        query_text = q["query"]
        query_id = q.get("id", "")

        try:
            results_a, _ = await engine.search_code(query=query_text, top_k=top_k)
            results_b, _ = await engine.search_code(query=query_text, top_k=top_k)
        except Exception as e:
            logger.warning("Consistency check failed for %s: %s", query_id, e)
            continue

        ids_a = [r.id for r in results_a]
        ids_b = [r.id for r in results_b]

        tau = _kendall_tau(ids_a, ids_b)
        j5 = _jaccard_at_k(ids_a, ids_b, 5)
        j10 = _jaccard_at_k(ids_a, ids_b, 10)
        rbo_score = _rbo(ids_a, ids_b)

        taus.append(tau)
        j5s.append(j5)
        j10s.append(j10)
        rbos.append(rbo_score)

        result.per_query.append({
            "query_id": query_id,
            "kendall_tau": tau,
            "jaccard@5": j5,
            "jaccard@10": j10,
            "rbo": rbo_score,
        })

    result.avg_kendall_tau = sum(taus) / len(taus) if taus else 0.0
    result.avg_jaccard_at_5 = sum(j5s) / len(j5s) if j5s else 0.0
    result.avg_jaccard_at_10 = sum(j10s) / len(j10s) if j10s else 0.0
    result.avg_rbo = sum(rbos) / len(rbos) if rbos else 0.0

    logger.info(
        "Consistency: %s — tau=%.3f J@5=%.3f J@10=%.3f RBO=%.3f (%d queries)",
        engine.name, result.avg_kendall_tau, result.avg_jaccard_at_5,
        result.avg_jaccard_at_10, result.avg_rbo, len(taus),
        extra={"engine": engine.name},
    )
    return result


async def run_paraphrase_stability(
    engine: ContextEngineAdapter,
    paraphrase_pairs: list[dict],
    top_k: int = 10,
) -> ParaphraseResult:
    """Measure ranking stability across paraphrased queries.

    Args:
        engine: The engine to test
        paraphrase_pairs: List of dicts with 'original', 'paraphrase', and 'id' keys
        top_k: Number of results to retrieve per query
    """
    result = ParaphraseResult(engine=engine.name, num_pairs=len(paraphrase_pairs))
    j5s = []
    j10s = []
    rbos = []
    shifts = []

    for pair in paraphrase_pairs:
        pair_id = pair.get("id", "")
        original = pair["original"]
        paraphrase = pair["paraphrase"]

        try:
            results_orig, _ = await engine.search_code(query=original, top_k=top_k)
            results_para, _ = await engine.search_code(query=paraphrase, top_k=top_k)
        except Exception as e:
            logger.warning("Paraphrase check failed for %s: %s", pair_id, e)
            continue

        ids_orig = [r.id for r in results_orig]
        ids_para = [r.id for r in results_para]

        j5 = _jaccard_at_k(ids_orig, ids_para, 5)
        j10 = _jaccard_at_k(ids_orig, ids_para, 10)
        rbo_score = _rbo(ids_orig, ids_para)

        # Average rank shift for items in both results
        shared = set(ids_orig) & set(ids_para)
        if shared:
            pos_orig = {id_: i for i, id_ in enumerate(ids_orig)}
            pos_para = {id_: i for i, id_ in enumerate(ids_para)}
            shift = sum(abs(pos_orig[id_] - pos_para[id_]) for id_ in shared) / len(shared)
        else:
            shift = float(top_k)

        j5s.append(j5)
        j10s.append(j10)
        rbos.append(rbo_score)
        shifts.append(shift)

        result.per_pair.append({
            "pair_id": pair_id,
            "original": original[:100],
            "paraphrase": paraphrase[:100],
            "jaccard@5": j5,
            "jaccard@10": j10,
            "rbo": rbo_score,
            "avg_rank_shift": shift,
        })

    result.avg_jaccard_at_5 = sum(j5s) / len(j5s) if j5s else 0.0
    result.avg_jaccard_at_10 = sum(j10s) / len(j10s) if j10s else 0.0
    result.avg_rbo = sum(rbos) / len(rbos) if rbos else 0.0
    result.avg_rank_shift = sum(shifts) / len(shifts) if shifts else 0.0

    logger.info(
        "Paraphrase stability: %s — J@5=%.3f J@10=%.3f RBO=%.3f shift=%.1f (%d pairs)",
        engine.name, result.avg_jaccard_at_5, result.avg_jaccard_at_10,
        result.avg_rbo, result.avg_rank_shift, len(j5s),
        extra={"engine": engine.name},
    )
    return result
