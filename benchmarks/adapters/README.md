# adapters/

Engine adapters. Each adapter wraps a single context engine and exposes the
common `ContextEngineAdapter` interface defined in `base.py` so the runner
can treat engines uniformly.

```
adapters/
├── base.py         abstract base + SearchResult / IndexResult dataclasses
├── synsc.py        Delphi HTTP API (also speaks the new context_pack endpoint)
├── synsc_mcp.py    Delphi via the MCP proxy — agent-realistic path
├── nia.py          Nia REST (full-latency accounting incl. rate-limit sleeps)
└── context7.py     Context7 HTTP + MCP-only library resolution fallback
```

## The contract

```python
class ContextEngineAdapter:
    name: str
    async def search_code(query, top_k, repo_ids=None, language=None) -> (list[SearchResult], latency_ms)
    async def search_papers(query, top_k) -> (list[SearchResult], latency_ms)
    async def index_repository(repo_url) -> IndexResult
    async def index_paper(arxiv_id) -> IndexResult
    async def list_repositories() -> list[dict]
    async def cleanup() -> None
```

Two non-obvious rules every adapter must follow:

1. **Latency is user-visible**, not request-only. The number an adapter
   returns must include any rate-limit sleeps and retry backoffs. The
   diagnosis flagged the previous behavior (excluding sleeps) as a
   fairness gap; both `nia.py` and `context7.py` have been updated.

2. **`SearchResult.id` should be stable across chunks**. The `metrics`
   layer de-duplicates by id when computing `Recall@K`, so returning the
   same chunk under multiple ids inflates recall.

## Adding a new engine

1. Subclass `ContextEngineAdapter` in a new file, set `name`.
2. Implement all five methods. Return `[]` and `IndexResult(success=False,...)`
   for unsupported operations.
3. Export from `adapters/__init__.py`.
4. Add a CLI option in `benchmarks/__main__.py` (`--engines my-engine`).
5. Add the engine to the engine-resolution block in `__main__.py:main()`.
6. Document any API keys it needs in `benchmarks/.env.local.example`.

## Quality mode (Delphi only)

The Delphi adapters accept `quality_mode={"agent","default"}`. When set to
`agent`, the HTTP adapter prefers `/v1/search/context_pack` and the MCP
adapter prefers the `build_context_pack` tool. Both gracefully fall back to
`search_code` / `/v1/search/code` on 404. This was added so the benchmark
exercises the new agent-quality endpoints introduced in Delphi alongside
the diagnosis.
