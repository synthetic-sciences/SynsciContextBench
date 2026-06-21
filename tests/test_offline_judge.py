"""Tests for the deterministic offline judge."""
from __future__ import annotations

from benchmarks.judges.offline_judge import offline_judge_match, offline_relevance_grade


def test_full_match_high_grade() -> None:
    grade = offline_relevance_grade(
        query="validate session token",
        ground_truth="def validate_token(session): ...",
        result="def validate_token(session): return verify(token)",
    )
    assert grade >= 2


def test_irrelevant_low_grade() -> None:
    grade = offline_relevance_grade(
        query="validate session token",
        ground_truth="def validate_token(): ...",
        result="completely unrelated text about rendering html templates",
    )
    assert grade <= 1


def test_match_returns_relevance_flag() -> None:
    is_rel, grade = offline_judge_match(
        "parse json config", "def parse_json_config(): ...", "def parse_json_config(path): ..."
    )
    assert is_rel is (grade >= 2)


def test_empty_inputs() -> None:
    assert offline_relevance_grade("", "", "") == 0
    assert offline_relevance_grade("q", "gt", "") == 0
