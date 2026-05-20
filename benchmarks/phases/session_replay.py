"""Real-session replay benchmark.

Cases come from production the diff-aware phase sessions where one engine visibly beat
another (most of them: Nia beats Delphi on the diff-aware phase-flavored questions). The
replay runs every engine on the same query and reports:

- ``win_rate``   — share of cases where the engine cleared the
  ``minimum_relevance`` threshold (i.e., would have surfaced the right
  evidence in the real session).
- ``regression`` — for cases the engine originally lost, did it still
  fail in the replay? If yes, the gap is real; if no, the underlying
  engine has improved since the case was recorded.
- ``failure_cause`` — preserved label from the original incident, then
  re-classified by the live taxonomy classifier so we can see whether the
  cause changed (e.g., "still missing_index_coverage" vs "now bad_ranking").
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..adapters.base import ContextEngineAdapter, SearchResult
from ..scoring.failure_taxonomy import _classify_diff_aware_failure
from ..infra.logging_config import get_logger
from ..infra.sampling import sample_seeded

logger = get_logger("session_replay")


@dataclass
class ReplayCase:
    id: str
    category: str
    query: str
    loser_engine: str
    winner_engine: str
    labeled_cause: str
    labeled_cause_notes: str
    expected_evidence: list[str]
    expected_anchors: list[str]
    minimum_relevance: float = 0.5


@dataclass
class ReplayResult:
    case_id: str
    category: str
    engine: str
    query: str
    score: float = 0.0
    anchor_hit: int = 0
    evidence_recall: float = 0.0
    passed_threshold: bool = False
    labeled_cause: str = ""
    re_classified_cause: str = ""
    is_loser: bool = False
    is_winner: bool = False
    regression_resolved: bool = False
    latency_ms: float = 0.0


@dataclass
class ReplayEngineReport:
    engine: str
    num_cases: int = 0
    win_rate: float = 0.0
    avg_score: float = 0.0
    historical_losses: int = 0
    losses_resolved: int = 0
    losses_still_failing: int = 0
    by_category: dict[str, float] = field(default_factory=dict)
    by_labeled_cause: dict[str, dict] = field(default_factory=dict)
    per_case: list[ReplayResult] = field(default_factory=list)


def load_replay_cases(path: str | Path) -> list[ReplayCase]:
    with open(path) as f:
        data = json.load(f)
    out: list[ReplayCase] = []
    for raw in data.get("cases", []):
        out.append(ReplayCase(
            id=raw["id"],
            category=raw.get("category", "unknown"),
            query=raw["query"],
            loser_engine=raw.get("loser_engine", ""),
            winner_engine=raw.get("winner_engine", ""),
            labeled_cause=raw.get("labeled_cause", ""),
            labeled_cause_notes=raw.get("labeled_cause_notes", ""),
            expected_evidence=list(raw.get("expected_evidence", [])),
            expected_anchors=list(raw.get("expected_anchors", [])),
            minimum_relevance=float(raw.get("minimum_relevance", 0.5)),
        ))
    return out


def _anchor_hit(chunks: list[SearchResult], anchors: list[str]) -> int:
    if not anchors:
        return 0
    for c in chunks:
        text = ((c.file_path or "") + "\n" + (c.content or "")).lower()
        if any(a.lower() in text for a in anchors):
            return 1
    return 0


def _evidence_recall(chunks: list[SearchResult], evidence: list[str]) -> float:
    if not evidence:
        return 0.0
    haystack = "\n".join((c.content or "") + " " + (c.file_path or "") for c in chunks).lower()
    hits = sum(1 for e in evidence if e.lower() in haystack)
    return hits / len(evidence)


def _score(anchor: int, recall: float) -> float:
    return 0.5 * anchor + 0.5 * recall


async def _replay_for_engine(
    engine: ContextEngineAdapter,
    cases: list[ReplayCase],
    top_k: int,
) -> ReplayEngineReport:
    per_case: list[ReplayResult] = []

    for case in tqdm(cases, desc=f"  {engine.name} session-replay", unit="case"):
        t0 = time.perf_counter()
        try:
            chunks, _ = await engine.search_code(query=case.query, top_k=top_k)
        except Exception as e:
            logger.warning("Replay search failed [%s/%s]: %s", engine.name, case.id, e)
            chunks = []

        anchor = _anchor_hit(chunks, case.expected_anchors)
        recall = _evidence_recall(chunks, case.expected_evidence)
        score = _score(anchor, recall)
        passed = score >= case.minimum_relevance

        is_loser = (case.loser_engine == engine.name)
        is_winner = (case.winner_engine == engine.name)
        regression_resolved = is_loser and passed

        re_class = ""
        if not passed:
            # Re-classify with the live taxonomy
            re_class, _ = _classify_diff_aware_failure({
                "num_chunks": len(chunks),
                "anchor_hit": anchor,
                "evidence_recall": recall,
                "hallucination_signals": 0,
                "error": "",
            })

        per_case.append(ReplayResult(
            case_id=case.id,
            category=case.category,
            engine=engine.name,
            query=case.query,
            score=score,
            anchor_hit=anchor,
            evidence_recall=recall,
            passed_threshold=passed,
            labeled_cause=case.labeled_cause,
            re_classified_cause=re_class,
            is_loser=is_loser,
            is_winner=is_winner,
            regression_resolved=regression_resolved,
            latency_ms=(time.perf_counter() - t0) * 1000,
        ))

    return _aggregate(engine.name, per_case)


def _aggregate(engine_name: str, per_case: list[ReplayResult]) -> ReplayEngineReport:
    rep = ReplayEngineReport(engine=engine_name, num_cases=len(per_case))
    if not per_case:
        return rep

    rep.win_rate = sum(1 for r in per_case if r.passed_threshold) / len(per_case)
    rep.avg_score = sum(r.score for r in per_case) / len(per_case)

    losers = [r for r in per_case if r.is_loser]
    rep.historical_losses = len(losers)
    rep.losses_resolved = sum(1 for r in losers if r.regression_resolved)
    rep.losses_still_failing = rep.historical_losses - rep.losses_resolved

    # Category breakdown
    by_cat: dict[str, list[ReplayResult]] = {}
    for r in per_case:
        by_cat.setdefault(r.category, []).append(r)
    rep.by_category = {
        cat: round(sum(rr.score for rr in rs) / len(rs), 4)
        for cat, rs in by_cat.items()
    }

    # Cause breakdown — group failures by their labeled cause and the
    # re-classified cause so we can show "10 cases labeled bad_retrieval, now
    # 6 still bad_retrieval / 2 bad_ranking / 2 passed".
    by_cause: dict[str, dict] = {}
    for r in per_case:
        if not r.labeled_cause:
            continue
        slot = by_cause.setdefault(r.labeled_cause, {"total": 0, "resolved": 0, "by_now": {}})
        slot["total"] += 1
        if r.passed_threshold:
            slot["resolved"] += 1
        else:
            cur = r.re_classified_cause or "uncategorized"
            slot["by_now"][cur] = slot["by_now"].get(cur, 0) + 1
    rep.by_labeled_cause = by_cause
    rep.per_case = per_case
    return rep


async def run_session_replay_benchmark(
    engines: list[ContextEngineAdapter],
    dataset_path: str | Path,
    top_k: int = 10,
    max_cases: int | None = None,
    seed: int = 0,
) -> dict[str, ReplayEngineReport]:
    cases = load_replay_cases(dataset_path)
    cases = sample_seeded(cases, max_cases, seed=seed)
    out: dict[str, ReplayEngineReport] = {}
    for engine in engines:
        out[engine.name] = await _replay_for_engine(engine, cases, top_k)
    return out


def print_session_replay_summary(reports: dict[str, ReplayEngineReport]) -> None:
    if not reports:
        return
    print("\n=== Session Replay Benchmark ===")
    print("  (cases where one engine visibly beat another in a real the diff-aware phase session)")
    for engine, rep in reports.items():
        print(f"\n  [{engine}]  n={rep.num_cases}")
        print(f"    win_rate={rep.win_rate:.3f}  avg_score={rep.avg_score:.3f}")
        print(
            f"    historical_losses={rep.historical_losses} "
            f"resolved={rep.losses_resolved} still_failing={rep.losses_still_failing}"
        )
        if rep.by_labeled_cause:
            print("    per labeled cause:")
            for cause, slot in rep.by_labeled_cause.items():
                print(
                    f"      - {cause:<24} total={slot['total']} "
                    f"resolved={slot['resolved']}  "
                    f"now={dict(slot['by_now'])}"
                )
