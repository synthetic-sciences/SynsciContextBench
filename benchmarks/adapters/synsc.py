"""Adapter for synsc-context (our engine)."""

from __future__ import annotations

import time

import httpx

from ..logging_config import get_logger
from .base import ContextEngineAdapter, IndexResult, SearchResult

logger = get_logger("adapter.synsc")


class SynscAdapter(ContextEngineAdapter):
    """Adapter for synsc-context HTTP API.

    Accepts an optional ``quality_mode`` (default ``agent``). Servers that
    do not yet implement the agent-quality endpoints just ignore the field.
    The benchmark is set up so users can compare ``quality_mode=default`` vs
    ``quality_mode=agent`` head-to-head.
    """

    name = "synsc-context"

    def __init__(self, api_url: str, api_key: str, quality_mode: str = "agent"):
        self.api_url = api_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers=self.headers,
            timeout=120.0,
        )
        self._quality_mode = quality_mode

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        payload: dict = {
            "query": query,
            "top_k": top_k,
            "quality_mode": self._quality_mode,
        }
        if repo_ids:
            payload["repo_ids"] = repo_ids
        if language:
            payload["language"] = language

        start = time.perf_counter()
        # Prefer the new context-pack endpoint when the server exposes it;
        # fall back to /v1/search/code on 404. We do not retry here, so a
        # missing endpoint just costs one extra request.
        endpoint = "/v1/search/context_pack" if self._quality_mode == "agent" else "/v1/search/code"
        resp = await self._client.post(endpoint, json=payload)
        if resp.status_code == 404 and endpoint != "/v1/search/code":
            resp = await self._client.post("/v1/search/code", json=payload)
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    id=r.get("chunk_id", ""),
                    content=r.get("content", ""),
                    score=r.get("relevance_score", 0.0),
                    file_path=r.get("file_path", ""),
                    start_line=r.get("start_line", 0),
                    end_line=r.get("end_line", 0),
                    language=r.get("language", ""),
                    repo_name=r.get("repo_name", ""),
                )
            )

        server_latency = data.get("search_time_ms", latency)
        logger.debug(
            "search_code: %d results in %.0fms (server: %.0fms) query=%s",
            len(results), latency, server_latency, query[:60],
            extra={"engine": self.name, "latency_ms": latency, "server_latency_ms": server_latency, "num_results": len(results)},
        )
        return results, latency

    async def search_papers(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        payload = {"query": query, "top_k": top_k}

        start = time.perf_counter()
        resp = await self._client.post("/v1/search/papers", json=payload)
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    id=r.get("chunk_id", r.get("id", "")),
                    content=r.get("content", ""),
                    score=r.get("relevance_score", r.get("similarity", 0.0)),
                    file_path=r.get("paper_title", ""),
                    metadata={
                        "section": r.get("section_title", ""),
                        "page": r.get("page_number"),
                    },
                )
            )

        return results, latency

    async def index_repository(self, repo_url: str) -> IndexResult:
        start = time.perf_counter()
        resp = await self._client.post(
            "/v1/repositories/index",
            json={"url": repo_url},
            timeout=600.0,
        )
        duration = (time.perf_counter() - start) * 1000

        if resp.status_code >= 400:
            return IndexResult(
                success=False, duration_ms=duration, error=resp.text
            )

        data = resp.json()
        return IndexResult(
            success=data.get("success", True),
            resource_id=data.get("repo_id", ""),
            duration_ms=duration,
        )

    async def index_paper(self, arxiv_id: str) -> IndexResult:
        start = time.perf_counter()
        resp = await self._client.post(
            "/v1/papers/index",
            json={"arxiv_id": arxiv_id},
            timeout=300.0,
        )
        duration = (time.perf_counter() - start) * 1000

        if resp.status_code >= 400:
            return IndexResult(
                success=False, duration_ms=duration, error=resp.text
            )

        data = resp.json()
        return IndexResult(
            success=True,
            resource_id=data.get("paper_id", ""),
            duration_ms=duration,
        )

    async def list_repositories(self) -> list[dict]:
        resp = await self._client.get("/v1/repositories")
        resp.raise_for_status()
        data = resp.json()
        return data.get("repositories", [])

    async def cleanup(self) -> None:
        await self._client.aclose()
