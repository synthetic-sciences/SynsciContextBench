"""Tests for the diff-aware (freshness) phase — the previously-floored phase.

These prove two things the README flagged as broken:
  1. The matcher surfaces symbols (incl. qualified names) from SearchResult
     objects, not just dicts.
  2. The re-index protocol distinguishes a *fresh* engine (re-indexes A->B,
     scores 1.0) from a *frozen/stale* engine (ignores B, floors at 1/3) —
     which the old code could never do because it called a non-existent
     ``engine.search`` and never re-indexed.
"""
from __future__ import annotations

from benchmarks.adapters.base import SearchResult
from benchmarks.adapters.mock import MockAdapter
from benchmarks.phases.diff_aware import (
    DiffAwareCase,
    _aggregate,
    _run_one_case,
    _symbol_appears,
    compute_correctness,
)


def test_symbol_appears_searchresult_and_dict() -> None:
    sr = SearchResult(id="1", content="def validate_token(t): ...", score=1.0)
    assert _symbol_appears("validate_token", [sr]) is True
    assert _symbol_appears("missing_symbol", [sr]) is False
    # dict form still works
    assert _symbol_appears("foo", [{"content": "def foo(): pass"}]) is True


def test_symbol_appears_qualified_name() -> None:
    sr = SearchResult(id="1", content="class S:\n    def start(self): ...", score=1.0)
    # Qualified name should match on its final component.
    assert _symbol_appears("Server.start", [sr]) is True


def test_symbol_appears_word_boundary() -> None:
    # 'get' must not match inside 'forget' (whole-identifier match).
    sr = SearchResult(id="1", content="x = forgetful", score=1.0)
    assert _symbol_appears("get", [sr]) is False


def test_symbol_appears_metadata() -> None:
    sr = SearchResult(id="1", content="opaque", score=1.0, metadata={"symbol": "thing"})
    assert _symbol_appears("thing", [sr]) is True


def test_compute_correctness() -> None:
    assert compute_correctness(False, True, True) == 1.0  # perfect
    assert abs(compute_correctness(True, False, True) - (1 / 3)) < 1e-9  # frozen floor
    assert compute_correctness(True, False, False) == 0.0


def _case() -> DiffAwareCase:
    return DiffAwareCase(
        id="c1", library="lib", repo_url="https://github.com/x/y",
        commit_before="A", commit_after="B", kind="modified", description="",
        stale_query="old_func", stale_symbol="old_func",
        fresh_query="new_func", fresh_symbol="new_func",
        stable_query="keep_func", stable_symbol="keep_func",
    )


def _corpus() -> list[dict]:
    return [
        {"id": "d1", "content": "def old_func(): pass", "symbol": "old_func", "commits": ["A"]},
        {"id": "d2", "content": "def new_func(): pass", "symbol": "new_func", "commits": ["B"]},
        {"id": "d3", "content": "def keep_func(): pass", "symbol": "keep_func", "commits": ["A", "B"]},
    ]


async def test_fresh_engine_scores_perfect() -> None:
    engine = MockAdapter(corpus=_corpus(), name="fresh", perfect=True)
    result = await _run_one_case(engine, _case())
    assert result.supported is True
    assert result.reindexed is True
    assert result.stale_hit is False  # deleted symbol gone after B
    assert result.fresh_hit is True  # new symbol present after B
    assert result.stable_hit is True
    assert result.correctness == 1.0


async def test_frozen_engine_floors() -> None:
    # perfect=False => index frozen at first ref (A); never reflects B.
    engine = MockAdapter(corpus=_corpus(), name="frozen", perfect=False)
    result = await _run_one_case(engine, _case())
    assert result.supported is True  # it *claims* pinning, just behaves stale
    assert result.stale_hit is True  # deleted symbol still surfaces (bad)
    assert result.fresh_hit is False  # new symbol never appears
    assert abs(result.correctness - (1 / 3)) < 1e-9


async def test_unsupported_engine_marked_not_supported() -> None:
    class _NoPin(MockAdapter):
        supports_commit_pinning = False  # type: ignore[assignment]

    engine = _NoPin(corpus=_corpus(), name="nopin")
    result = await _run_one_case(engine, _case())
    assert result.supported is False


def test_aggregate_scores_over_supported_cases() -> None:
    from benchmarks.phases.diff_aware import DiffAwareCaseResult

    perfect = DiffAwareCaseResult(
        case_id="c1", library="l", kind="k", query="q",
        stale_hit=False, fresh_hit=True, stable_hit=True, correctness=1.0, supported=True,
    )
    floored = DiffAwareCaseResult(
        case_id="c2", library="l", kind="k", query="q",
        stale_hit=True, fresh_hit=False, stable_hit=True, correctness=1 / 3, supported=False,
    )
    rep = _aggregate("eng", [perfect, floored])
    # Headline correctness computed over the supported case only -> 1.0
    assert rep.correctness == 1.0
    assert rep.supported_cases == 1
    assert rep.unsupported_cases == 1
