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

import json
import logging
import random
import re
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
    # False when the engine cannot pin a commit (no diff-aware re-index). Such
    # cases are reported separately and excluded from the headline freshness
    # score instead of silently flooring it.
    supported: bool = True
    reindexed: bool = False  # True if the engine actually re-indexed A->B


@dataclass
class DiffAwareEngineReport:
    """Engine-level aggregate."""

    engine: str = ""
    per_case: list[DiffAwareCaseResult] = field(default_factory=list)
    correctness: float = 0.0
    staleness_rate: float = 0.0  # fraction of cases where deleted symbol re-surfaced
    freshness_rate: float = 0.0  # fraction of cases where new symbol was retrieved
    stability_rate: float = 0.0  # fraction of cases where unchanged symbol still works
    supported_cases: int = 0  # cases where the engine actually re-indexed A->B
    unsupported_cases: int = 0  # cases scored single-pass (no commit pinning)
    supports_commit_pinning: bool = False


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


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _result_fields(r) -> tuple[str, str]:
    """Extract (text, symbol_metadata) from a SearchResult OR a raw dict.

    The adapters return ``SearchResult`` dataclasses, but legacy callers and
    some engines hand back dicts — accept both so the matcher never silently
    sees nothing.
    """
    if isinstance(r, dict):
        text_parts = [
            str(r.get(k, "") or "")
            for k in ("content", "snippet", "text", "code", "context", "file_path")
        ]
        sym = ""
        for k in ("symbol", "symbol_name", "name"):
            if r.get(k):
                sym = str(r[k])
                break
        meta = r.get("metadata") or {}
        if not sym and isinstance(meta, dict):
            sym = str(meta.get("symbol", "") or "")
        return "\n".join(text_parts), sym
    # SearchResult dataclass (or any object with attributes)
    text = "\n".join(
        str(getattr(r, k, "") or "") for k in ("content", "file_path")
    )
    meta = getattr(r, "metadata", {}) or {}
    sym = str(meta.get("symbol", "")) if isinstance(meta, dict) else ""
    return text, sym


def _symbol_appears(symbol: str, results: list) -> bool:
    """True iff the engine's returned chunks surface the symbol.

    Matches the symbol as a whole identifier token (word boundary) in any
    returned chunk's content/path, or as an exact symbol-metadata value. Also
    matches the final dotted component (``Class.method`` -> ``method``), which
    the previous naive substring match missed for qualified names — the strict
    matcher was a documented cause of the phase flooring at 1/3.
    """
    if not symbol or not results:
        return False
    last = symbol.split(".")[-1].split("::")[-1]
    needles = {symbol, last}
    for r in results:
        text, sym = _result_fields(r)
        if sym and sym in needles:
            return True
        tokens = set(_IDENT_RE.findall(text))
        if needles & tokens:
            return True
    return False


def compute_correctness(stale_hit: bool, fresh_hit: bool, stable_hit: bool) -> float:
    """Composite per-case correctness.

    Staleness is a false positive (the deleted symbol must NOT resurface), so it
    is inverted; freshness and stability are true positives. Pure function so it
    is unit-testable without an engine.
    """
    return (
        (0.0 if stale_hit else 1.0)
        + (1.0 if fresh_hit else 0.0)
        + (1.0 if stable_hit else 0.0)
    ) / 3.0


# ---------------------------------------------------------------------------
# Per-engine runner
# ---------------------------------------------------------------------------


async def _query(engine, q: str, top_k: int = 10) -> tuple[list, str | None]:
    """Query an engine via the adapter contract (``search_code`` -> results).

    The previous implementation called ``engine.search`` / ``engine.asearch``,
    which no adapter implements — so every query returned ``[]`` and every
    engine floored at correctness 1/3. The adapter contract is
    ``search_code(query, top_k) -> (list[SearchResult], latency)``.
    """
    try:
        results, _latency = await engine.search_code(query=q, top_k=top_k)
        return results or [], None
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"


async def _reindex_at(engine, repo_url: str, ref: str) -> str | None:
    try:
        await engine.index_repository_at_commit(repo_url, ref)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


async def _run_one_case(engine, case: DiffAwareCase, top_k: int = 10) -> DiffAwareCaseResult:
    t0 = time.time()
    err: str | None = None
    stale_hit = fresh_hit = stable_hit = False

    supports = bool(getattr(engine, "supports_commit_pinning", False))
    can_pin = supports and bool(case.repo_url and case.commit_before and case.commit_after)
    reindexed = False

    if can_pin:
        # Freshness can only be measured by actually moving the index A -> B.
        err = await _reindex_at(engine, case.repo_url, case.commit_before)
        if err is None:
            stale_at_a, e1 = await _query(engine, case.stale_query, top_k)
            err = e1
            # The deleted symbol exists at A; it should disappear after B.
            err2 = await _reindex_at(engine, case.repo_url, case.commit_after)
            err = err or err2
            reindexed = err2 is None
            stale_after_b, e2 = await _query(engine, case.stale_query, top_k)
            fresh_after_b, e3 = await _query(engine, case.fresh_query, top_k)
            stable_after_b, e4 = await _query(engine, case.stable_query, top_k)
            err = err or e2 or e3 or e4
            stale_hit = _symbol_appears(case.stale_symbol, stale_after_b)
            fresh_hit = _symbol_appears(case.fresh_symbol, fresh_after_b)
            stable_hit = _symbol_appears(case.stable_symbol, stable_after_b)
    else:
        # Engine can't pin a commit: run a single pass against whatever is
        # indexed. Recorded as unsupported so it is excluded from the headline
        # freshness score rather than masquerading as a real measurement.
        stale_results, e1 = await _query(engine, case.stale_query, top_k)
        fresh_results, e2 = await _query(engine, case.fresh_query, top_k)
        stable_results, e3 = await _query(engine, case.stable_query, top_k)
        err = e1 or e2 or e3
        stale_hit = _symbol_appears(case.stale_symbol, stale_results)
        fresh_hit = _symbol_appears(case.fresh_symbol, fresh_results)
        stable_hit = _symbol_appears(case.stable_symbol, stable_results)

    correctness = compute_correctness(stale_hit, fresh_hit, stable_hit)

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
        supported=can_pin,
        reindexed=reindexed,
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
    supported = [r for r in per_case if r.supported]
    unsupported = [r for r in per_case if not r.supported]
    # Headline freshness metrics are computed over cases the engine could
    # actually demonstrate freshness on (i.e. it re-indexed A->B). Falling back
    # to all cases only when no case supported pinning keeps single-pass engines
    # visible while not crediting them with a freshness measurement.
    scored = supported or per_case
    n = len(scored)
    return DiffAwareEngineReport(
        engine=engine_name,
        per_case=per_case,
        correctness=sum(r.correctness for r in scored) / n,
        staleness_rate=sum(1 for r in scored if r.stale_hit) / n,
        freshness_rate=sum(1 for r in scored if r.fresh_hit) / n,
        stability_rate=sum(1 for r in scored if r.stable_hit) / n,
        supported_cases=len(supported),
        unsupported_cases=len(unsupported),
        supports_commit_pinning=bool(supported),
    )


def print_diff_aware_summary(report: DiffAwareEngineReport) -> None:
    print(f"  Engine: {report.engine}")
    print(f"  Cases:           {len(report.per_case)}")
    print(
        f"  Re-index (A->B): {'yes' if report.supports_commit_pinning else 'no — single-pass'} "
        f"(supported={report.supported_cases}, unsupported={report.unsupported_cases})"
    )
    print(f"  Correctness:     {report.correctness:.3f}")
    print(f"  Staleness rate:  {report.staleness_rate:.3f}  (lower better — % of deleted symbols still returned)")
    print(f"  Freshness rate:  {report.freshness_rate:.3f}  (higher better — % of new symbols retrieved)")
    print(f"  Stability rate:  {report.stability_rate:.3f}  (higher better — % of unchanged symbols still findable)")
