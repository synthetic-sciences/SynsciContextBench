"""Deterministic in-memory adapter for offline runs and tests.

The rest of the harness needs a live engine on a socket plus paid LLM-judge
calls to run at all. That makes the benchmark impossible to unit-test or to
exercise in CI, and it makes every scorer change a leap of faith. ``MockAdapter``
fixes that: a fully deterministic, dependency-free engine backed by a small
in-memory corpus, with lexical-overlap ranking.

It also models **commit-pinned re-indexing** (``index_repository_at_commit``),
which is what the diff-aware phase needs to actually measure freshness: index at
commit A, query, re-index at commit B, query. A real engine demonstrates
freshness by changing what it returns after the re-index; a frozen index does
not. The mock lets us prove the *scorer* is correct independent of any live
engine.
"""
from __future__ import annotations

import re

from .base import ContextEngineAdapter, IndexResult, SearchResult

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        out.add(tok)
        # identifier subwords so "get_user" matches "user"
        for part in re.split(r"[_\W]+", tok):
            for sub in re.findall(r"[a-z]+|[0-9]+", part):
                out.add(sub)
    return out


class MockAdapter(ContextEngineAdapter):
    """A deterministic lexical search engine over an in-memory corpus.

    Each corpus doc is a dict with at least ``id`` and ``content``; optional
    ``file_path``, ``language``, ``symbol``, ``commits`` (the set of commit refs
    in which the doc exists — used by the diff-aware phase). When ``commits`` is
    omitted the doc exists at every commit.
    """

    def __init__(
        self,
        corpus: list[dict] | None = None,
        name: str = "mock",
        perfect: bool = False,
    ) -> None:
        self.name = name
        self._corpus = list(corpus or [])
        self._active_commit: str | None = None
        # When True, the engine is omniscient about commits (always fresh).
        # When False, it behaves like a *frozen* index pinned to the first
        # commit it ever saw — useful for modeling a stale engine in tests.
        self._perfect = perfect
        self._frozen_commit: str | None = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        q = _tokens(query)
        scored: list[tuple[float, dict]] = []
        for doc in self._visible_docs():
            if language and doc.get("language") and doc["language"] != language:
                continue
            overlap = len(q & _tokens(doc.get("content", "")))
            if overlap:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("id"))))
        results: list[SearchResult] = []
        for score, doc in scored[:top_k]:
            results.append(
                SearchResult(
                    id=str(doc.get("id", "")),
                    content=doc.get("content", ""),
                    score=float(score) / max(len(q), 1),
                    file_path=doc.get("file_path", ""),
                    language=doc.get("language", ""),
                    repo_name=doc.get("repo_name", "mock-repo"),
                    metadata={"symbol": doc.get("symbol", "")},
                )
            )
        return results, 1.0

    async def search_papers(
        self, query: str, top_k: int = 10
    ) -> tuple[list[SearchResult], float]:
        return await self.search_code(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Indexing (incl. commit-pinned re-index for diff-aware)
    # ------------------------------------------------------------------
    async def index_repository(self, repo_url: str) -> IndexResult:
        return IndexResult(success=True, resource_id=repo_url, duration_ms=1.0)

    async def index_repository_at_commit(self, repo_url: str, ref: str) -> IndexResult:
        """Re-index the repo pinned to a specific commit ref.

        A fresh engine tracks the new ref; a frozen engine ignores everything
        after the first ref it saw (modeling a stale index).
        """
        if self._frozen_commit is None:
            self._frozen_commit = ref
        self._active_commit = ref if self._perfect else self._frozen_commit
        return IndexResult(success=True, resource_id=f"{repo_url}@{ref}", duration_ms=1.0)

    async def index_paper(self, arxiv_id: str) -> IndexResult:
        return IndexResult(success=True, resource_id=arxiv_id, duration_ms=1.0)

    async def list_repositories(self) -> list[dict]:
        return [{"repo_name": "mock-repo", "repo_id": "mock"}]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _visible_docs(self) -> list[dict]:
        if self._active_commit is None:
            return self._corpus
        visible = []
        for doc in self._corpus:
            commits = doc.get("commits")
            if commits is None or self._active_commit in commits:
                visible.append(doc)
        return visible

    @property
    def supports_commit_pinning(self) -> bool:
        return True
