"""Base adapter interface for context engines.

Every engine (synsc-context, Nia, future engines) implements this ABC
so the benchmark runner can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Normalized search result returned by any adapter."""

    id: str
    content: str
    score: float
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    language: str = ""
    repo_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    success: bool
    resource_id: str = ""
    duration_ms: float = 0.0
    error: str = ""


class ContextEngineAdapter(ABC):
    """Abstract base class for context engine adapters."""

    name: str = "base"

    @abstractmethod
    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Search code and return (results, latency_ms)."""
        ...

    @abstractmethod
    async def search_papers(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """Search indexed papers and return (results, latency_ms)."""
        ...

    @abstractmethod
    async def index_repository(self, repo_url: str) -> IndexResult:
        """Index a GitHub repository."""
        ...

    @abstractmethod
    async def index_paper(self, arxiv_id: str) -> IndexResult:
        """Index an arXiv paper."""
        ...

    @abstractmethod
    async def list_repositories(self) -> list[dict]:
        """List all indexed repositories."""
        ...

    async def cleanup(self) -> None:
        """Optional cleanup (close HTTP clients, etc.)."""
        pass
