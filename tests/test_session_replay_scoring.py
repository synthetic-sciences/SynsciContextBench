"""Tests for the session-replay phase scoring functions."""
from __future__ import annotations

from benchmarks.adapters.base import SearchResult
from benchmarks.phases.session_replay import (
    ReplayResult,
    _aggregate,
    _anchor_hit,
    _evidence_recall,
    _score,
)


def _sr(content: str = "", file_path: str = "") -> SearchResult:
    return SearchResult(id="x", content=content, score=1.0, file_path=file_path)


def test_anchor_hit() -> None:
    chunks = [_sr(content="def authenticate_user(): ...")]
    assert _anchor_hit(chunks, ["authenticate_user"]) == 1
    assert _anchor_hit(chunks, ["nonexistent"]) == 0
    assert _anchor_hit(chunks, []) == 0
    # anchors can match on file path
    assert _anchor_hit([_sr(file_path="auth/login.py")], ["login.py"]) == 1


def test_evidence_recall() -> None:
    chunks = [_sr(content="alpha beta"), _sr(content="gamma")]
    assert _evidence_recall(chunks, ["alpha", "gamma"]) == 1.0
    assert _evidence_recall(chunks, ["alpha", "missing"]) == 0.5
    assert _evidence_recall(chunks, []) == 0.0


def test_score_blend() -> None:
    assert _score(1, 1.0) == 1.0
    assert _score(0, 0.0) == 0.0
    assert _score(1, 0.0) == 0.5
    assert _score(0, 1.0) == 0.5


def test_aggregate_win_rate_and_regression() -> None:
    cases = [
        ReplayResult(
            case_id="a", category="bad_retrieval", engine="delphi", query="q",
            score=0.8, passed_threshold=True, labeled_cause="bad_retrieval",
            is_loser=True, regression_resolved=True,
        ),
        ReplayResult(
            case_id="b", category="bad_ranking", engine="delphi", query="q",
            score=0.2, passed_threshold=False, labeled_cause="bad_ranking",
            is_loser=True, regression_resolved=False, re_classified_cause="bad_ranking",
        ),
    ]
    rep = _aggregate("delphi", cases)
    assert rep.num_cases == 2
    assert rep.win_rate == 0.5
    assert rep.historical_losses == 2
    assert rep.losses_resolved == 1
    assert rep.losses_still_failing == 1
    assert "bad_retrieval" in rep.by_labeled_cause


def test_aggregate_empty() -> None:
    rep = _aggregate("eng", [])
    assert rep.num_cases == 0
    assert rep.win_rate == 0.0
