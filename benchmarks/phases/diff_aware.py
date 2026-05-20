"""Phase 10 — Diff-aware indexing benchmark.

The 'context engine' problem is half-retrieval, half-freshness. An engine
that nails CodeSearchNet but returns deleted symbols a day after a refactor
is not actually useful in a working repo. This phase measures whether each
engine correctly refreshes its index after a code change.

Setup
-----
Each test case is a (commit_A → commit_B) pair on a public Python library,
with B = A + a small, known diff (a symbol added, removed, or renamed).
For every (engine, case) we expect the engine to have indexed B (the
"after" state). We then issue three classes of query:

  - stale_query:  asks about a symbol that exists in A but NOT in B.
                  The engine wins by NOT returning it.
  - fresh_query:  asks about a symbol that exists in B but NOT in A.
                  The engine wins by returning it.
  - stable_query: asks about a symbol unchanged between A and B.
                  The engine wins by returning it (regression guard).

Scoring
-------
Each case yields three booleans (stale_hit, fresh_hit, stable_hit).
Correctness per case = mean of [
   stale_hit ? 0 : 1,     # inverted — staleness is a false-positive
   fresh_hit ? 1 : 0,
   stable_hit ? 1 : 0,
].
Engine-level correctness = mean of per-case correctness.

This is a fair, head-to-head measurement because every engine sees the
same starting public-repo URL. The engines that ship periodic re-indexing
(Nia) and curated docs registries (Context7) are evaluated on the same
terms as the engines that ship diff-aware re-indexing (Delphi).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiffAwareCase:
    """One diff-aware test case (a {A, B, queries} triplet)."""

    id: str
    library: str
    repo_url: str
    commit_before: str
    commit_after: str
    kind: str  # 'added' | 'removed' | 'renamed' | 'modified'
    description: str
    stale_query: str  # asks about symbol present in A only
    stale_symbol: str
    fresh_query: str  # asks about symbol present in B only
    fresh_symbol: str
    stable_query: str  # asks about an unchanged symbol
    stable_symbol: str


@dataclass
class DiffAwareCaseResult:
    """Per-case result. Keeps every signal we measured."""

    case_id: str
    library: str
    kind: str
    query: str  # representative query for trace logging
    stale_hit: bool  # True = engine still returns the deleted symbol (bad)
    fresh_hit: bool  # True = engine returns the new symbol (good)
    stable_hit: bool  # True = engine returns the unchanged symbol (good)
    correctness: float  # composite of the three above
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class DiffAwareEngineReport:
    """Engine-level aggregate."""

    engine: str = ""
    per_case: list[DiffAwareCaseResult] = field(default_factory=list)
    correctness: float = 0.0
    staleness_rate: float = 0.0  # fraction of cases where deleted symbol re-surfaced
    freshness_rate: float = 0.0  # fraction of cases where new symbol was retrieved
    stability_rate: float = 0.0  # fraction of cases where unchanged symbol still works


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_diff_aware_cases(path: str | Path) -> list[DiffAwareCase]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases_raw = data.get("test_cases", data.get("cases", data if isinstance(data, list) else []))
    out: list[DiffAwareCase] = []
    for c in cases_raw:
        out.append(
            DiffAwareCase(
                id=c["id"],
                library=c["library"],
                repo_url=c.get("repo_url", ""),
                commit_before=c.get("commit_before", ""),
                commit_after=c.get("commit_after", ""),
                kind=c["kind"],
                description=c.get("description", ""),
                stale_query=c["stale_query"],
                stale_symbol=c["stale_symbol"],
                fresh_query=c["fresh_query"],
                fresh_symbol=c["fresh_symbol"],
                stable_query=c["stable_query"],
                stable_symbol=c["stable_symbol"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Symbol-match helper
# ---------------------------------------------------------------------------


def _symbol_appears(symbol: str, results: list[dict]) -> bool:
    """True iff any of the engine's returned chunks mentions the symbol.

    We do a case-sensitive substring match because:
      - Code symbols are case-sensitive in every language the bench cares about.
      - A retrieval engine that can't surface the symbol *name* anywhere in its
        top-K context isn't really "returning" it.
    """
    if not symbol or not results:
        return False
    needle = symbol
    for r in results:
        for field_name in ("content", "snippet", "text", "code", "context"):
            val = r.get(field_name)
            if isinstance(val, str) and needle in val:
                return True
        # Some adapters surface symbol names in file_path / chunk_id metadata
        for field_name in ("symbol", "symbol_name", "name"):
            val = r.get(field_name)
            if isinstance(val, str) and needle == val:
                return True
    return False


# ---------------------------------------------------------------------------
# Per-engine runner
# ---------------------------------------------------------------------------


async def _run_one_case(engine, case: DiffAwareCase) -> DiffAwareCaseResult:
    t0 = time.time()
    err: str | None = None
    stale_hit = fresh_hit = stable_hit = False

    async def _query(q: str, top_k: int = 10) -> list[dict]:
        try:
            res = engine.search(q, top_k=top_k) if hasattr(engine, "search") else None
            if res is None and hasattr(engine, "asearch"):
                res = await engine.asearch(q, top_k=top_k)
            if asyncio.iscoroutine(res):
                res = await res
            if isinstance(res, dict):
                return res.get("results", res.get("hits", []))
            return res or []
        except Exception as e:
            nonlocal err
            err = f"{type(e).__name__}: {e}"
            return []

    try:
        # Stale check: any hit for the deleted symbol is bad.
        stale_results = await _query(case.stale_query)
        stale_hit = _symbol_appears(case.stale_symbol, stale_results)

        # Fresh check: we want a hit for the new symbol.
        fresh_results = await _query(case.fresh_query)
        fresh_hit = _symbol_appears(case.fresh_symbol, fresh_results)

        # Stability check: an unchanged symbol must still be findable.
        stable_results = await _query(case.stable_query)
        stable_hit = _symbol_appears(case.stable_symbol, stable_results)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    correctness = (
        (0.0 if stale_hit else 1.0)
        + (1.0 if fresh_hit else 0.0)
        + (1.0 if stable_hit else 0.0)
    ) / 3.0

    return DiffAwareCaseResult(
        case_id=case.id,
        library=case.library,
        kind=case.kind,
        query=case.fresh_query,
        stale_hit=stale_hit,
        fresh_hit=fresh_hit,
        stable_hit=stable_hit,
        correctness=correctness,
        latency_ms=(time.time() - t0) * 1000,
        error=err,
    )


async def run_diff_aware_benchmark(
    engine,
    dataset_path: str,
    max_cases: int | None = None,
    seed: int = 0,
) -> DiffAwareEngineReport:
    """Run the diff-aware indexing benchmark for a single engine."""
    cases = load_diff_aware_cases(dataset_path)
    if max_cases and max_cases < len(cases):
        rng = random.Random(seed)
        cases = rng.sample(cases, max_cases)

    per_case: list[DiffAwareCaseResult] = []
    for case in cases:
        result = await _run_one_case(engine, case)
        per_case.append(result)

    return _aggregate(getattr(engine, "name", "unknown"), per_case)


def _aggregate(engine_name: str, per_case: list[DiffAwareCaseResult]) -> DiffAwareEngineReport:
    if not per_case:
        return DiffAwareEngineReport(engine=engine_name)
    n = len(per_case)
    correctness = sum(r.correctness for r in per_case) / n
    staleness_rate = sum(1 for r in per_case if r.stale_hit) / n
    freshness_rate = sum(1 for r in per_case if r.fresh_hit) / n
    stability_rate = sum(1 for r in per_case if r.stable_hit) / n
    return DiffAwareEngineReport(
        engine=engine_name,
        per_case=per_case,
        correctness=correctness,
        staleness_rate=staleness_rate,
        freshness_rate=freshness_rate,
        stability_rate=stability_rate,
    )


def print_diff_aware_summary(report: DiffAwareEngineReport) -> None:
    print(f"  Engine: {report.engine}")
    print(f"  Cases:           {len(report.per_case)}")
    print(f"  Correctness:     {report.correctness:.3f}")
    print(f"  Staleness rate:  {report.staleness_rate:.3f}  (lower better — % of deleted symbols still returned)")
    print(f"  Freshness rate:  {report.freshness_rate:.3f}  (higher better — % of new symbols retrieved)")
    print(f"  Stability rate:  {report.stability_rate:.3f}  (higher better — % of unchanged symbols still findable)")
