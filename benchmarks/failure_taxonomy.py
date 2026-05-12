"""Failure-cause classification for benchmark results.

The diagnosis asked for a per-failure taxonomy so that "engine X lost" turns
into "engine X lost because of bucket Y", which is actionable. The buckets
come straight from the diagnosis:

- ``missing_index_coverage`` — engine returned zero results, or all results
  came from a tiny, unrelated portion of the corpus. Symptom: the engine
  probably never indexed the right source.
- ``bad_retrieval`` — engine returned results but none touched the expected
  files / keywords / anchors. Symptom: candidate generation failed.
- ``bad_ranking`` — engine returned at least one relevant chunk, but it was
  ranked outside the reporting window. Symptom: the candidate set was OK,
  but the scorer demoted it.
- ``bad_packaging`` — engine returned chunks that contain the right symbol
  but lack neighboring context, imports, or test code (the agent could not
  act on them). Detected via short chunks or low context-utilization scores.
- ``tool_ergonomics`` — engine returned an error (auth, timeout, rate
  limit, server error), or required an extra knob to retrieve correctly.
- ``benchmark_blind_spot`` — engine produced the right surface (paper /
  graph / tool contract) but the test case scoring missed it. Used when
  hallucination signals are zero but the case still scored low.

Each engine gets a per-bucket count over all evaluated cases plus a small
list of representative failure examples so the report can quote them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureExample:
    case_id: str
    query: str
    benchmark: str
    cause: str
    detail: str = ""


@dataclass
class EngineFailureReport:
    engine: str
    total_failures: int = 0
    buckets: dict[str, int] = field(default_factory=dict)
    examples: list[FailureExample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


_BUCKETS = (
    "missing_index_coverage",
    "bad_retrieval",
    "bad_ranking",
    "bad_packaging",
    "tool_ergonomics",
    "benchmark_blind_spot",
)


def _classify_retrieval_failure(qr: dict) -> tuple[str, str]:
    """Bucket a single retrieval failure record.

    Expects a dict with at least ``num_results``, ``mrr``, ``precision_at``.
    """
    num_results = int(qr.get("num_results", 0))
    mrr = float(qr.get("mrr", 0.0))
    p_at_1 = float((qr.get("precision_at") or {}).get(1, 0.0))
    p_at_10 = float((qr.get("precision_at") or {}).get(10, 0.0))

    if num_results == 0:
        return "missing_index_coverage", "engine returned zero results"
    if mrr == 0.0 and p_at_10 == 0.0:
        return "bad_retrieval", "no top-10 result was relevant"
    if mrr > 0.0 and mrr < 0.2 and p_at_10 > 0:
        return "bad_ranking", "relevant result present but ranked outside top-5"
    if p_at_1 == 0.0 and p_at_10 < 0.2:
        return "bad_ranking", "low precision concentrated in lower ranks"
    return "benchmark_blind_spot", "structural metric flagged failure but graded counts disagree"


def _classify_thesis_failure(case: dict) -> tuple[str, str]:
    """Bucket a single Thesis-case result."""
    n_chunks = int(case.get("num_chunks", 0))
    anchor = int(case.get("anchor_hit", 0))
    recall = float(case.get("evidence_recall", 0.0))
    halls = int(case.get("hallucination_signals", 0))
    error = case.get("error") or ""

    if error:
        return "tool_ergonomics", error[:200]
    if n_chunks == 0:
        return "missing_index_coverage", "Thesis case returned no chunks"
    if anchor == 0 and recall == 0.0:
        return "bad_retrieval", "no anchor + no evidence keywords surfaced"
    if anchor == 0 and recall > 0.0 and recall < 0.5:
        return "bad_ranking", "some evidence present but no anchor hit"
    if halls > 0:
        return "bad_packaging", "negative signals matched in surfaced content"
    if anchor == 1 and recall > 0.4:
        return "benchmark_blind_spot", "retrieved right surface but rubric still low"
    return "bad_packaging", "evidence present but composite below threshold"


def _classify_swe_failure(case: dict) -> tuple[str, str]:
    """Bucket a single SWE-Agent case result."""
    judge = float(case.get("judge_composite", 0.0))
    crit = float(case.get("criteria_pass_rate", 0.0))
    util = float(case.get("context_utilization_score", 0.0))
    halls = int(case.get("hallucination_count", 0))

    if judge == 0.0 and crit == 0.0:
        return "bad_retrieval", "no useful context produced"
    if util < 0.1 and judge < 0.6:
        return "bad_packaging", "context returned but solution did not cite it"
    if halls > 1:
        return "bad_packaging", "hallucinated symbols despite context"
    if judge >= 0.6 and crit < 0.5:
        return "benchmark_blind_spot", "judge approved but structural checks failed"
    return "bad_ranking", "context present but did not improve solution enough"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _bump(rep: EngineFailureReport, bucket: str, example: FailureExample) -> None:
    rep.total_failures += 1
    rep.buckets[bucket] = rep.buckets.get(bucket, 0) + 1
    if len(rep.examples) < 12:  # keep representative slice
        rep.examples.append(example)


def build_failure_taxonomy(report: dict) -> dict[str, dict]:
    """Walk a `BenchmarkReport` dict and return per-engine failure taxonomy."""
    engines = list(report.get("engines") or [])
    per_engine: dict[str, EngineFailureReport] = {
        e: EngineFailureReport(engine=e, buckets={b: 0 for b in _BUCKETS})
        for e in engines
    }

    # --- Retrieval phase
    for eng, qrs in (report.get("query_results") or {}).items():
        if eng not in per_engine:
            per_engine[eng] = EngineFailureReport(engine=eng, buckets={b: 0 for b in _BUCKETS})
        for qr in qrs or []:
            if float(qr.get("mrr", 0.0)) >= 0.5:
                continue  # passing
            bucket, detail = _classify_retrieval_failure(qr)
            _bump(per_engine[eng], bucket, FailureExample(
                case_id=str(qr.get("query_id") or qr.get("query", "")[:40]),
                query=str(qr.get("query", ""))[:200],
                benchmark="retrieval",
                cause=bucket,
                detail=detail,
            ))

    # --- Thesis phase
    for eng, th_data in (report.get("thesis") or {}).items():
        if not isinstance(th_data, dict) or eng not in per_engine:
            continue
        for case in th_data.get("per_case") or []:
            if float(case.get("composite", 0.0)) >= 0.5:
                continue
            bucket, detail = _classify_thesis_failure(case)
            _bump(per_engine[eng], bucket, FailureExample(
                case_id=str(case.get("case_id", "")),
                query=str(case.get("question", ""))[:200],
                benchmark=f"thesis/{case.get('category', '')}",
                cause=bucket,
                detail=detail,
            ))

    # --- SWE-Agent phase
    swe_per_case = (report.get("swe_agent") or {}).get("_per_case") or []
    for case in swe_per_case:
        eng = case.get("engine") or ""
        if eng not in per_engine or eng in ("no_context", "baseline"):
            continue
        if float(case.get("judge_composite", 0.0)) >= 0.7:
            continue
        bucket, detail = _classify_swe_failure(case)
        _bump(per_engine[eng], bucket, FailureExample(
            case_id=str(case.get("test_case_id", "")),
            query=str(case.get("title", case.get("query", "")))[:200],
            benchmark="swe_agent",
            cause=bucket,
            detail=detail,
        ))

    # Drop empty engines (e.g., baseline-only entries) and convert to dicts.
    out: dict[str, dict] = {}
    for eng, rep in per_engine.items():
        if rep.total_failures == 0 and all(v == 0 for v in rep.buckets.values()):
            continue
        out[eng] = {
            "engine": eng,
            "total_failures": rep.total_failures,
            "buckets": rep.buckets,
            "examples": [
                {
                    "case_id": ex.case_id,
                    "query": ex.query,
                    "benchmark": ex.benchmark,
                    "cause": ex.cause,
                    "detail": ex.detail,
                }
                for ex in rep.examples
            ],
        }
    return out


def print_failure_taxonomy(tax: dict[str, dict]) -> None:
    if not tax:
        return
    print("\n=== Failure Taxonomy ===")
    print("  Buckets: missing_index_coverage, bad_retrieval, bad_ranking,")
    print("           bad_packaging, tool_ergonomics, benchmark_blind_spot")
    for eng, data in tax.items():
        print(f"\n  [{eng}]  total_failures={data['total_failures']}")
        for bucket, count in (data.get("buckets") or {}).items():
            if count:
                print(f"    {bucket:<24} {count}")
        if data.get("examples"):
            print("    sample failures:")
            for ex in data["examples"][:3]:
                print(f"      - {ex['benchmark']} {ex['case_id']}: {ex['cause']} — {ex['detail'][:100]}")
