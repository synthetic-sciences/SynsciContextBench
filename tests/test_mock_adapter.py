"""Tests for the deterministic offline MockAdapter."""
from __future__ import annotations

from benchmarks.adapters.mock import MockAdapter


def _corpus() -> list[dict]:
    return [
        {"id": "auth", "content": "def validate_token(t): return verify(t)", "language": "python"},
        {"id": "db", "content": "def get_connection(dsn): return Pool(dsn)", "language": "python"},
        {"id": "cache", "content": "class Cache: def evict(self): pass", "language": "python"},
    ]


async def test_ranks_relevant_first() -> None:
    eng = MockAdapter(corpus=_corpus())
    results, latency = await eng.search_code("validate token", top_k=3)
    assert results[0].id == "auth"
    assert latency >= 0


async def test_deterministic() -> None:
    eng1 = MockAdapter(corpus=_corpus())
    eng2 = MockAdapter(corpus=_corpus())
    r1, _ = await eng1.search_code("connection pool", top_k=3)
    r2, _ = await eng2.search_code("connection pool", top_k=3)
    assert [r.id for r in r1] == [r.id for r in r2]


async def test_language_filter() -> None:
    corpus = _corpus() + [{"id": "js", "content": "function validateToken() {}", "language": "javascript"}]
    eng = MockAdapter(corpus=corpus)
    results, _ = await eng.search_code("validate token", top_k=5, language="javascript")
    assert all(r.language == "javascript" for r in results)


async def test_commit_pinning_fresh_vs_frozen() -> None:
    corpus = [
        {"id": "old", "content": "def old_func(): pass", "commits": ["A"]},
        {"id": "new", "content": "def new_func(): pass", "commits": ["B"]},
    ]
    fresh = MockAdapter(corpus=corpus, perfect=True)
    await fresh.index_repository_at_commit("repo", "A")
    await fresh.index_repository_at_commit("repo", "B")
    res, _ = await fresh.search_code("new_func", top_k=5)
    assert any(r.id == "new" for r in res)  # fresh engine sees B

    frozen = MockAdapter(corpus=corpus, perfect=False)
    await frozen.index_repository_at_commit("repo", "A")
    await frozen.index_repository_at_commit("repo", "B")
    res2, _ = await frozen.search_code("new_func", top_k=5)
    assert not any(r.id == "new" for r in res2)  # frozen never reflects B


def test_supports_commit_pinning_flag() -> None:
    assert MockAdapter().supports_commit_pinning is True
