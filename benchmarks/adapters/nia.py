"""Adapter for Nia (trynia.ai) context engine.

Based on Nia's public API docs: https://docs.trynia.ai/api-guide
Nia exposes both REST and MCP endpoints. We use REST for benchmarking
since it's easier to measure latency precisely.

NOTE: Nia's exact API schema may change — update this adapter if their
endpoints or response formats differ from what's documented here.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from .base import ContextEngineAdapter, IndexResult, SearchResult


class NiaAdapter(ContextEngineAdapter):
    """Adapter for Nia (trynia.ai) REST API."""

    name = "nia"

    def __init__(self, api_url: str, api_key: str, request_delay: float = 3.0):
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers=self.headers,
            timeout=120.0,
        )
        self._request_delay = request_delay  # seconds between requests to avoid 429

    # ------------------------------------------------------------------
    # NOTE: Nia's REST API endpoints are based on their public docs.
    # If their API differs, update the paths and payload shapes here.
    # You can also swap this out for MCP tool calls if preferred.
    # ------------------------------------------------------------------

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        # Nia v2 universal-search endpoint
        payload: dict = {"query": query}
        if repo_ids:
            payload["repositories"] = repo_ids  # expects ["owner/repo"] strings
        if language:
            payload["language"] = language

        # Throttle to stay under Nia's rate limit
        if self._request_delay > 0:
            await asyncio.sleep(self._request_delay)

        # Retry with backoff on 429
        max_retries = 5
        for attempt in range(max_retries):
            start = time.perf_counter()
            resp = await self._client.post("/v2/universal-search", json=payload)
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [nia] 429 rate limited, waiting {wait}s (attempt {attempt+1})...")
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            break

        data = resp.json()

        results = []
        for r in data.get("results", data.get("data", [])):
            # Nia nests file_path inside a "source" object
            source = r.get("source", {})
            file_path = (
                r.get("file_path")
                or r.get("path")
                or source.get("file_path")
                or source.get("path", "")
            )
            repo_name = (
                r.get("repository")
                or r.get("repo_name")
                or source.get("repository", "")
            )
            results.append(
                SearchResult(
                    id=r.get("id", r.get("chunk_id", "")),
                    content=r.get("content", r.get("text", "")),
                    score=r.get("score", r.get("relevance", 0.0)),
                    file_path=file_path,
                    start_line=r.get("start_line", 0),
                    end_line=r.get("end_line", 0),
                    language=r.get("language", ""),
                    repo_name=repo_name,
                )
            )

        return results[:top_k], latency

    async def search_papers(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        payload = {"query": query}

        start = time.perf_counter()
        resp = await self._client.post("/v2/universal-search", json=payload)
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", data.get("data", [])):
            results.append(
                SearchResult(
                    id=r.get("id", ""),
                    content=r.get("content", r.get("text", "")),
                    score=r.get("score", r.get("relevance", 0.0)),
                    file_path=r.get("title", ""),
                    metadata={
                        "section": r.get("section", ""),
                        "page": r.get("page"),
                    },
                )
            )

        return results, latency

    async def index_repository(self, repo_url: str) -> IndexResult:
        start = time.perf_counter()
        resp = await self._client.post(
            "/v2/repositories",
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
            success=True,
            resource_id=data.get("id", data.get("repository_id", "")),
            duration_ms=duration,
        )

    async def index_paper(self, arxiv_id: str) -> IndexResult:
        start = time.perf_counter()
        resp = await self._client.post(
            "/v2/research-papers",
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
            resource_id=data.get("id", data.get("paper_id", "")),
            duration_ms=duration,
        )

    async def list_repositories(self) -> list[dict]:
        resp = await self._client.get("/v2/repositories")
        resp.raise_for_status()
        data = resp.json()
        return data.get("repositories", data.get("data", []))

    async def cleanup(self) -> None:
        await self._client.aclose()
