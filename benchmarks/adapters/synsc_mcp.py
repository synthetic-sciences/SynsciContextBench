"""Delphi (synsc-context) adapter that talks to the MCP proxy.

The HTTP adapter in ``synsc.py`` is the easiest path for benchmarking, but the
diagnosis points out that real agent usage is through the MCP proxy, which
historically only exposed ``index_repository(url, branch)`` and did not let
the agent ask for ``quality_mode=agent`` / ``build_context_pack``. Delphi's
own changes add those controls — and this adapter exercises them so the
benchmark reflects the realistic agent path.

Connection model:
    - Spawn ``synsc-context-proxy`` (npx + node) on stdin/stdout.
    - Send JSON-RPC 2.0 requests for ``tools/call``.
    - Each call accepts the new ``quality_mode`` argument; if the
      installed Delphi build does not yet expose it, the proxy ignores
      the field and we fall back to the default endpoint.

Latency accounting: like the patched HTTP adapters, this returns the
user-visible wall-clock latency (including subprocess setup the first time
and any reads waiting on the server).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time

from ..logging_config import get_logger
from .base import ContextEngineAdapter, IndexResult, SearchResult

logger = get_logger("adapter.synsc-mcp")


class SynscMCPAdapter(ContextEngineAdapter):
    """Delphi via its MCP proxy."""

    name = "synsc-context-mcp"

    def __init__(
        self,
        proxy_command: str = "synsc-context-proxy",
        api_url: str = "http://localhost:8742",
        api_key: str = "",
        quality_mode: str = "agent",
    ):
        self._proxy_command = proxy_command
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._quality_mode = quality_mode
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        # Cache the set of tool names the proxy advertises so we know
        # whether `build_context_pack` is available without retrying.
        self._tool_names: set[str] = set()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        # Resolve binary (allow override via PATH or absolute path)
        cmd = shutil.which(self._proxy_command) or self._proxy_command
        env = os.environ.copy()
        env.setdefault("SYNSC_API_URL", self._api_url)
        if self._api_key:
            env.setdefault("SYNSC_API_KEY", self._api_key)

        self._process = await asyncio.create_subprocess_exec(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._initialized = False
        self._request_id = 0
        await self._handshake()

    async def _handshake(self) -> None:
        assert self._process and self._process.stdin and self._process.stdout
        await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "synsc-context-bench", "version": "1.0.0"},
        })
        # initialized notification
        msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self._process.stdin.write(msg.encode())
        await self._process.stdin.drain()
        await asyncio.sleep(0.05)
        # List tools to know what the proxy supports
        try:
            tools = await self._rpc("tools/list", {})
            self._tool_names = {t.get("name", "") for t in tools.get("tools", [])}
        except Exception:
            self._tool_names = set()
        self._initialized = True

    async def _rpc(self, method: str, params: dict) -> dict:
        async with self._lock:
            await self._ensure_started_inner()
            assert self._process and self._process.stdin and self._process.stdout
            self._request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            msg = json.dumps(payload) + "\n"
            self._process.stdin.write(msg.encode())
            await self._process.stdin.drain()

            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=120.0)
            if not line:
                raise RuntimeError("synsc-context-proxy closed stdout")
            resp = json.loads(line.decode())
            if "error" in resp:
                err = resp["error"]
                raise RuntimeError(f"synsc-context-proxy error: {err.get('message', err)}")
            return resp.get("result", {})

    async def _ensure_started_inner(self) -> None:
        """Internal version (already holding the lock)."""
        if self._process is not None and self._process.returncode is None and self._initialized:
            return
        if self._initialized:
            return
        # Already-locked variant of _ensure_started:
        if self._process is None or self._process.returncode is not None:
            cmd = shutil.which(self._proxy_command) or self._proxy_command
            env = os.environ.copy()
            env.setdefault("SYNSC_API_URL", self._api_url)
            if self._api_key:
                env.setdefault("SYNSC_API_KEY", self._api_key)
            self._process = await asyncio.create_subprocess_exec(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._request_id = 0
        # Handshake without re-acquiring the lock
        if not self._initialized:
            self._request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "synsc-context-bench", "version": "1.0.0"},
                },
            }
            assert self._process and self._process.stdin and self._process.stdout
            self._process.stdin.write((json.dumps(payload) + "\n").encode())
            await self._process.stdin.drain()
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=120.0)
            if not line:
                raise RuntimeError("synsc-context-proxy closed stdout during init")
            self._process.stdin.write(
                (json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode()
            )
            await self._process.stdin.drain()
            await asyncio.sleep(0.05)
            # tools/list
            self._request_id += 1
            self._process.stdin.write(
                (json.dumps({
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "tools/list",
                    "params": {},
                }) + "\n").encode()
            )
            await self._process.stdin.drain()
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
            try:
                resp = json.loads(line.decode())
                tools = resp.get("result", {}).get("tools", [])
                self._tool_names = {t.get("name", "") for t in tools}
            except Exception:
                self._tool_names = set()
            self._initialized = True

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_text_results(text: str) -> list[SearchResult]:
        """Best-effort parse for MCP tools that return plain text blocks."""
        out: list[SearchResult] = []
        # Split by '----' or blank-line patterns; each block becomes one result.
        blocks = [b.strip() for b in text.split("\n----\n") if b.strip()]
        if not blocks:
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        for i, blk in enumerate(blocks):
            path = ""
            for line in blk.splitlines():
                if line.startswith("file: ") or line.startswith("path: "):
                    path = line.split(":", 1)[1].strip()
                    break
            out.append(SearchResult(
                id=f"synsc-mcp-{i}",
                content=blk[:4000],
                score=1.0 - (i * 0.02),
                file_path=path,
                repo_name="",
            ))
        return out

    @staticmethod
    def _parse_tool_response(resp: dict) -> list[SearchResult]:
        """MCP tools return {content: [{type, text}, ...]}."""
        for part in resp.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text", "")
                # Try JSON first
                try:
                    data = json.loads(txt)
                    if isinstance(data, dict) and "results" in data:
                        rows = data["results"]
                    elif isinstance(data, list):
                        rows = data
                    else:
                        rows = []
                    out: list[SearchResult] = []
                    for i, r in enumerate(rows):
                        out.append(SearchResult(
                            id=r.get("chunk_id", r.get("id", f"synsc-mcp-{i}")),
                            content=r.get("content", r.get("text", "")),
                            score=r.get("relevance_score", r.get("score", 1.0 - i * 0.02)),
                            file_path=r.get("file_path", r.get("path", "")),
                            start_line=r.get("start_line", 0),
                            end_line=r.get("end_line", 0),
                            language=r.get("language", ""),
                            repo_name=r.get("repo_name", ""),
                            metadata={"source": "synsc-mcp"},
                        ))
                    if out:
                        return out
                except Exception:
                    pass
                return SynscMCPAdapter._parse_text_results(txt)
        return []

    # ------------------------------------------------------------------
    # ContextEngineAdapter interface
    # ------------------------------------------------------------------

    async def search_code(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], float]:
        user_start = time.perf_counter()
        # Prefer `build_context_pack` when the proxy exposes it (Delphi's new
        # agent-quality endpoint). Falls back to search_code otherwise.
        tool = "build_context_pack" if "build_context_pack" in self._tool_names else "search_code"
        args: dict = {
            "query": query,
            "top_k": top_k,
            "quality_mode": self._quality_mode,
        }
        if repo_ids:
            args["repo_ids"] = repo_ids
        if language:
            args["language"] = language
        try:
            resp = await self._rpc("tools/call", {"name": tool, "arguments": args})
        except Exception as e:
            logger.warning("MCP search_code failed (%s): %s", tool, e)
            return [], (time.perf_counter() - user_start) * 1000
        results = self._parse_tool_response(resp)
        return results[:top_k], (time.perf_counter() - user_start) * 1000

    async def search_papers(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        user_start = time.perf_counter()
        tool = "search_papers" if "search_papers" in self._tool_names else "search_code"
        args = {"query": query, "top_k": top_k, "quality_mode": self._quality_mode}
        try:
            resp = await self._rpc("tools/call", {"name": tool, "arguments": args})
        except Exception as e:
            logger.warning("MCP search_papers failed: %s", e)
            return [], (time.perf_counter() - user_start) * 1000
        results = self._parse_tool_response(resp)
        return results[:top_k], (time.perf_counter() - user_start) * 1000

    async def index_repository(self, repo_url: str) -> IndexResult:
        start = time.perf_counter()
        args = {"url": repo_url, "quality_mode": self._quality_mode}
        try:
            resp = await self._rpc("tools/call", {"name": "index_repository", "arguments": args})
        except Exception as e:
            return IndexResult(success=False, duration_ms=(time.perf_counter() - start) * 1000, error=str(e))
        # Extract success/repo_id heuristically
        text_blocks = [p.get("text", "") for p in resp.get("content", []) if isinstance(p, dict)]
        joined = "\n".join(text_blocks)
        success = "error" not in joined.lower()
        return IndexResult(
            success=success,
            duration_ms=(time.perf_counter() - start) * 1000,
            resource_id="",
            error="" if success else joined[:500],
        )

    async def index_paper(self, arxiv_id: str) -> IndexResult:
        start = time.perf_counter()
        try:
            resp = await self._rpc("tools/call", {
                "name": "index_paper",
                "arguments": {"arxiv_id": arxiv_id, "quality_mode": self._quality_mode},
            })
        except Exception as e:
            return IndexResult(success=False, duration_ms=(time.perf_counter() - start) * 1000, error=str(e))
        return IndexResult(success=True, duration_ms=(time.perf_counter() - start) * 1000)

    async def list_repositories(self) -> list[dict]:
        try:
            resp = await self._rpc("tools/call", {"name": "list_repositories", "arguments": {}})
        except Exception:
            return []
        text_blocks = [p.get("text", "") for p in resp.get("content", []) if isinstance(p, dict)]
        out: list[dict] = []
        for blk in text_blocks:
            try:
                data = json.loads(blk)
                if isinstance(data, dict) and "repositories" in data:
                    out.extend(data["repositories"])
                elif isinstance(data, list):
                    out.extend(data)
            except Exception:
                continue
        return out

    async def cleanup(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
        self._process = None
        self._initialized = False
        self._tool_names.clear()
