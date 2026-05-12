"""Adapter for Context7 (context7.com) context engine.

Context7 provides pre-indexed library documentation. It exposes:
  - HTTP API: GET /api/v2/context?libraryId=...&query=...&type=json
  - MCP server: npx -y @upstash/context7-mcp (tools: resolve-library-id, query-docs)

We use the HTTP API for benchmarking (consistent with synsc/nia adapters,
easier latency measurement). MCP is used only for library ID resolution
when the HTTP API needs an exact /owner/repo ID.

Unlike synsc-context and Nia, Context7 does NOT support custom repo indexing.
It has pre-indexed docs for thousands of popular libraries.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx

from ..infra.logging_config import get_logger
from .base import ContextEngineAdapter, IndexResult, SearchResult

logger = get_logger("adapter.context7")

# Map common benchmark repo names to Context7 library IDs (/owner/repo format)
_REPO_TO_CONTEXT7_ID: dict[str, str] = {
    "fastapi": "/fastapi/fastapi",
    "pydantic": "/pydantic/pydantic",
    "httpx": "/encode/httpx",
    "flask": "/pallets/flask",
    "django": "/django/django",
    "requests": "/psf/requests",
    "sqlalchemy": "/sqlalchemy/sqlalchemy",
    "numpy": "/numpy/numpy",
    "pandas": "/pandas-dev/pandas",
    "pytorch": "/pytorch/pytorch",
    "tensorflow": "/tensorflow/tensorflow",
    "react": "/facebook/react",
    "next.js": "/vercel/next.js",
    "nextjs": "/vercel/next.js",
    "express": "/expressjs/express",
    "langchain": "/langchain-ai/langchain",
    "llamaindex": "/run-llama/llama_index",
    "starlette": "/encode/starlette",
    "uvicorn": "/encode/uvicorn",
}


def _extract_context7_id(repo_url: str) -> str:
    """Extract a Context7 library ID from a GitHub URL.

    e.g. "https://github.com/tiangolo/fastapi" -> "/tiangolo/fastapi"
    Then check known mappings for the canonical Context7 ID.
    """
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("/")
    repo_name = parts[-1].lower() if parts else repo_url.lower()

    # Check known mappings first
    for key, ctx7_id in _REPO_TO_CONTEXT7_ID.items():
        if key in repo_name:
            return ctx7_id

    # Fallback: use /owner/repo from the URL
    if len(parts) >= 2:
        return f"/{parts[-2]}/{parts[-1]}"
    return f"/{repo_name}/{repo_name}"


def _infer_library_from_query(query: str) -> str | None:
    """Try to infer a Context7 library ID from a natural language query."""
    query_lower = query.lower()
    for key, ctx7_id in _REPO_TO_CONTEXT7_ID.items():
        if key in query_lower:
            return ctx7_id
    return None


def _extract_library_candidates(query: str) -> list[str]:
    """Extract potential library/framework names from a natural language query.

    Prioritizes single technical words that are likely library names,
    then falls back to hyphenated combinations (e.g., scikit-learn).
    """
    # Common non-library words to skip
    stop_words = {
        "how", "to", "in", "the", "a", "an", "is", "it", "do", "does",
        "can", "what", "why", "when", "where", "which", "with", "from",
        "for", "on", "of", "by", "at", "as", "or", "and", "not", "no",
        "get", "set", "use", "using", "create", "make", "find", "check",
        "if", "else", "return", "function", "method", "class", "file",
        "string", "list", "dict", "array", "object", "int", "float",
        "bool", "type", "value", "key", "name", "data", "code", "error",
        "read", "write", "parse", "convert", "sort", "filter", "map",
        "loop", "iterate", "print", "output", "input", "variable",
        "import", "module", "package", "install", "run", "test",
        "python", "javascript", "java", "ruby", "go", "php", "rust",
        "typescript", "html", "css", "sql", "bash", "shell",
        "empty", "line", "into", "result", "query", "call", "number",
        "readonly", "declaring", "token", "multiple", "char", "byte",
        "index", "count", "text", "null", "none", "true", "false",
        "path", "directory", "folder", "table", "column", "row",
    }

    candidates: list[str] = []

    # 1. Words with dots/hyphens first (e.g., next.js, scikit-learn)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9._-]+", query)
    for word in words:
        lower = word.lower()
        if ("." in word or "-" in word) and lower not in stop_words:
            candidates.append(lower)

    # 2. CamelCase / PascalCase words (e.g., FastAPI, NumPy, TensorFlow)
    for word in words:
        lower = word.lower()
        if lower in stop_words or len(lower) < 3:
            continue
        if any(c.isupper() for c in word[1:]):
            candidates.append(lower)

    # 3. Single words that aren't stop words (most likely library names)
    for word in words:
        lower = word.lower()
        if lower not in stop_words and lower not in candidates and len(lower) >= 4:
            candidates.append(lower)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:5]  # Limit to top 5 candidates


def _parse_txt_to_results(raw_text: str, library_id: str) -> list[SearchResult]:
    """Parse Context7's TXT response into SearchResult objects.

    Context7 TXT format uses '---' separators between sections,
    each with a heading, source URL, description, and code block.
    """
    results: list[SearchResult] = []

    # Split on the separator lines
    sections = re.split(r"\n-{20,}\n", raw_text)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 20:
            continue

        # Extract heading (### line)
        heading = ""
        heading_match = re.match(r"^#{1,3}\s+(.+)", section)
        if heading_match:
            heading = heading_match.group(1).strip()

        # Extract source URL
        source = ""
        source_match = re.search(r"Source:\s*(https?://\S+)", section)
        if source_match:
            source = source_match.group(1)

        # Extract file path from source URL (e.g. github blob path)
        file_path = heading or source or f"{library_id}/docs"

        # Extract language from code fence
        lang = ""
        code_match = re.search(r"```(\w+)", section)
        if code_match:
            lang = code_match.group(1)

        # Extract code content if present, otherwise use full section
        code_block = re.search(r"```\w*\n(.*?)```", section, re.DOTALL)
        content = code_block.group(1).strip() if code_block else section

        if len(content) < 10:
            continue

        results.append(
            SearchResult(
                id=f"ctx7_{i}",
                content=content,
                score=1.0 - (i * 0.05),  # Descending relevance by position
                file_path=file_path,
                language=lang or "python",
                repo_name=library_id,
                metadata={"source": "context7", "source_url": source},
            )
        )

    return results


def _parse_json_to_results(data: list[dict], library_id: str) -> list[SearchResult]:
    """Parse Context7's JSON response into SearchResult objects."""
    results: list[SearchResult] = []

    for i, item in enumerate(data):
        content = item.get("content", item.get("code", ""))
        if not content or len(content) < 10:
            continue

        results.append(
            SearchResult(
                id=f"ctx7_{i}",
                content=content,
                score=item.get("score", 1.0 - (i * 0.05)),
                file_path=item.get("title", item.get("source", f"{library_id}/docs")),
                language=item.get("language", "python"),
                repo_name=library_id,
                metadata={
                    "source": "context7",
                    "source_url": item.get("source", ""),
                },
            )
        )

    return results


class Context7Adapter(ContextEngineAdapter):
    """Adapter for Context7 HTTP API.

    Uses the public REST API at context7.com/api/v2/context.
    Falls back to MCP stdio for library ID resolution if needed.
    """

    name = "context7"

    def __init__(
        self,
        api_url: str = "https://context7.com",
        api_key: str = "",
        npx_command: str = "npx",
        request_delay: float = 0.5,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._npx_command = npx_command
        self._request_delay = request_delay
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=120.0,
            headers=headers,
        )
        # MCP subprocess for library ID resolution (lazy-started)
        self._mcp_process: asyncio.subprocess.Process | None = None
        self._mcp_request_id = 0
        self._mcp_initialized = False
        self._mcp_lock = asyncio.Lock()
        # Cache resolved library IDs
        self._library_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # MCP stdio helpers (used for library ID resolution as fallback)
    # ------------------------------------------------------------------

    async def _ensure_mcp_started(self) -> None:
        """Start the MCP server subprocess if not already running."""
        if self._mcp_process is not None and self._mcp_process.returncode is None:
            return

        env = os.environ.copy()
        env.setdefault("DEFAULT_MINIMUM_TOKENS", "10000")

        self._mcp_process = await asyncio.create_subprocess_exec(
            self._npx_command, "-y", "@upstash/context7-mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._mcp_initialized = False
        self._mcp_request_id = 0

        # MCP handshake
        await self._mcp_send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "synsc-benchmark", "version": "1.0.0"},
        })
        # Send initialized notification
        assert self._mcp_process.stdin
        msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self._mcp_process.stdin.write(msg.encode())
        await self._mcp_process.stdin.drain()
        await asyncio.sleep(0.1)
        self._mcp_initialized = True

    async def _mcp_send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC 2.0 request to the MCP subprocess."""
        assert self._mcp_process and self._mcp_process.stdin and self._mcp_process.stdout

        self._mcp_request_id += 1
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": self._mcp_request_id,
            "method": method,
            "params": params,
        }) + "\n"
        self._mcp_process.stdin.write(msg.encode())
        await self._mcp_process.stdin.drain()

        line = await asyncio.wait_for(
            self._mcp_process.stdout.readline(), timeout=60.0
        )
        if not line:
            raise RuntimeError("Context7 MCP server closed stdout unexpectedly")

        resp = json.loads(line.decode())
        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(f"Context7 MCP error: {err.get('message', err)}")
        return resp.get("result", {})

    async def _request_with_retry(
        self, method: str, url: str, **kwargs: object
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff on 429 and 5xx."""
        max_retries = 5
        for attempt in range(max_retries):
            resp = await self._client.request(method, url, **kwargs)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"    [context7] Rate limited (429), retrying in {retry_after}s...")
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code == 301:
                try:
                    redirect_data = resp.json()
                    redirect_url = redirect_data.get("redirectUrl", "")
                    if redirect_url:
                        print(f"    [context7] Library redirected to {redirect_url}")
                        params = dict(kwargs.get("params", {}))  # type: ignore[arg-type]
                        params["libraryId"] = redirect_url
                        kwargs = {**kwargs, "params": params}
                        continue
                except Exception:
                    pass

            if resp.status_code == 202:
                print("    [context7] Library not finalized (202), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

            return resp

        return resp  # Return last response even if retries exhausted

    async def _query_http(
        self, library_id: str, query: str, response_type: str = "txt"
    ) -> str | list[dict]:
        """Query Context7's HTTP API."""
        resp = await self._request_with_retry(
            "GET",
            f"{self._api_url}/api/v2/context",
            params={
                "libraryId": library_id,
                "query": query,
                "type": response_type,
            },
        )
        resp.raise_for_status()

        if response_type == "json":
            return resp.json()
        return resp.text

    async def _search_library_http(self, library_name: str) -> str:
        """Resolve a library name to a Context7 library ID via HTTP API.

        Uses GET /api/v2/libs/search?libraryName=<name>&query=<name>
        """
        try:
            resp = await self._request_with_retry(
                "GET",
                f"{self._api_url}/api/v2/libs/search",
                params={
                    "libraryName": library_name,
                    "query": library_name,
                },
            )
            if resp.status_code != 200:
                return ""

            data = resp.json()
            # API returns {"results": [...]} or a plain list
            libraries = data.get("results", data) if isinstance(data, dict) else data
            if libraries and isinstance(libraries, list) and len(libraries) > 0:
                best = libraries[0]
                library_id = best.get("id", "")
                if library_id:
                    return library_id
        except Exception as e:
            print(f"    [context7] HTTP library search failed for '{library_name}': {e}")

        return ""

    async def _resolve_library_mcp(self, library_name: str) -> str:
        """Resolve a library name to a Context7 library ID via MCP."""
        async with self._mcp_lock:
            await self._ensure_mcp_started()

            result = await self._mcp_send_request("tools/call", {
                "name": "resolve-library-id",
                "arguments": {"query": library_name, "libraryName": library_name},
            })

        # Parse the text content for a library ID
        texts = []
        for part in result.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        text = "\n".join(texts)

        # Look for /owner/repo patterns
        for line in text.strip().splitlines():
            id_match = re.search(r"(/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", line)
            if id_match:
                return id_match.group(1)

        return ""

    async def _resolve_library(self, library_name: str) -> str:
        """Resolve a library name to Context7 library ID.

        Priority: cache -> static mapping -> HTTP search API -> MCP -> guess.
        """
        cache_key = library_name.lower()
        if cache_key in self._library_cache:
            return self._library_cache[cache_key]

        # Check static mapping
        if cache_key in _REPO_TO_CONTEXT7_ID:
            self._library_cache[cache_key] = _REPO_TO_CONTEXT7_ID[cache_key]
            return self._library_cache[cache_key]

        # Primary: HTTP search API
        library_id = await self._search_library_http(library_name)
        if library_id:
            self._library_cache[cache_key] = library_id
            return library_id

        # Fallback: MCP resolution
        try:
            library_id = await self._resolve_library_mcp(library_name)
            if library_id:
                self._library_cache[cache_key] = library_id
                return library_id
        except Exception as e:
            print(f"    [context7] MCP resolve failed for '{library_name}': {e}")

        # Last resort: guess /name/name
        fallback = f"/{library_name}/{library_name}"
        self._library_cache[cache_key] = fallback
        return fallback

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Search Context7's library docs via HTTP API."""
        # Determine which libraries to search
        library_ids: list[str] = []
        if repo_ids:
            for rid in repo_ids:
                library_ids.append(_extract_context7_id(rid))
        else:
            inferred = _infer_library_from_query(query)
            if inferred:
                library_ids.append(inferred)

        # If static mapping failed, try HTTP library search API
        if not library_ids:
            candidates = _extract_library_candidates(query)
            for candidate in candidates:
                try:
                    resolved = await self._search_library_http(candidate)
                    if resolved:
                        library_ids.append(resolved)
                        # Cache for future queries
                        self._library_cache[candidate.lower()] = resolved
                        break  # Use first successful match
                except Exception:
                    continue

        # If no library could be resolved at all, skip search
        if not library_ids:
            return [], 0.0

        all_results: list[SearchResult] = []
        total_latency_ms = 0.0
        # End-to-end "user time": includes rate-limit sleeps and retries the
        # caller cannot avoid. The benchmark reports both numbers so we can
        # show the gap.
        user_start = time.perf_counter()

        for library_id in library_ids:
            try:
                # Rate-limiting delay — counted as part of user-visible
                # latency. Skipping it here understates Context7's real
                # request cost (the diagnosis flagged this exact gap).
                if self._request_delay > 0:
                    await asyncio.sleep(self._request_delay)

                start = time.perf_counter()
                raw = await self._query_http(library_id, query, response_type="json")
                request_latency = (time.perf_counter() - start) * 1000
                total_latency_ms += request_latency

                if isinstance(raw, list):
                    results = _parse_json_to_results(raw, library_id)
                else:
                    # Fallback if server returns text despite json request
                    results = _parse_txt_to_results(str(raw), library_id)
                all_results.extend(results)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    try:
                        resolved = await self._resolve_library(
                            library_id.strip("/").split("/")[-1]
                        )
                        if resolved and resolved != library_id:
                            start = time.perf_counter()
                            raw = await self._query_http(resolved, query, response_type="json")
                            request_latency = (time.perf_counter() - start) * 1000
                            total_latency_ms += request_latency
                            if isinstance(raw, list):
                                results = _parse_json_to_results(raw, resolved)
                            else:
                                results = _parse_txt_to_results(str(raw), resolved)
                            all_results.extend(results)
                    except Exception as inner_e:
                        print(f"    [context7] Fallback failed for {library_id}: {inner_e}")
                else:
                    print(f"    [context7] HTTP {e.response.status_code} for {library_id}")
            except Exception as e:
                print(f"    [context7] Error searching {library_id}: {e}")

        # Full user-visible latency: includes rate-limit sleeps + retries.
        user_latency_ms = (time.perf_counter() - user_start) * 1000
        # Return the user-visible number so the benchmark reports the cost
        # the agent actually paid; the request-only number is still
        # available in `total_latency_ms` for debugging adapters.
        return all_results[:top_k], user_latency_ms

    async def search_papers(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """Context7 does not index papers — returns empty results."""
        return [], 0.0

    async def index_repository(self, repo_url: str) -> IndexResult:
        """Context7 uses pre-indexed libraries — no custom indexing.

        We verify the library exists by making a test query.
        """
        start = time.perf_counter()
        library_id = _extract_context7_id(repo_url)

        try:
            raw = await self._query_http(library_id, "overview", response_type="txt")
            duration = (time.perf_counter() - start) * 1000
            return IndexResult(
                success=True,
                resource_id=library_id,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return IndexResult(
                success=False,
                duration_ms=duration,
                error=f"Library {library_id} not found in Context7: {e}",
            )

    async def index_paper(self, arxiv_id: str) -> IndexResult:
        """Context7 does not support paper indexing."""
        return IndexResult(
            success=False,
            duration_ms=0,
            error="Context7 does not support paper indexing",
        )

    async def list_repositories(self) -> list[dict]:
        """Context7 doesn't expose a list endpoint — return cached libraries."""
        return [
            {"name": lib, "context7_id": cid}
            for lib, cid in self._library_cache.items()
        ]

    async def cleanup(self) -> None:
        """Close HTTP client and terminate MCP subprocess."""
        await self._client.aclose()

        if self._mcp_process and self._mcp_process.returncode is None:
            self._mcp_process.terminate()
            try:
                await asyncio.wait_for(self._mcp_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._mcp_process.kill()
        self._mcp_process = None
        self._mcp_initialized = False
        self._library_cache.clear()
