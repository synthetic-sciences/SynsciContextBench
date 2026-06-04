# results_2026_05 — May 2026 bench-v2 run

Canonical run that produces the headline numbers in the top-level
[`README.md`](../../../README.md) and the chart in
[`assets/charts/results.png`](../../../assets/charts/results.png).

Two engines (Delphi + Nia) under `--match-mode llm`. Context7 was excluded
from this round because the bench's `asyncio.gather` still stalls on
Context7 mid-stream — the per-request `wait_for` timeout shipped in
`dc52983` covers the HTTP path but the bench-level gather is still
vulnerable. Cached Context7 numbers from `results_final/` are unchanged.

## Provenance

Every file in `data/` and `reports/` is named with a `<phase>__` prefix
that maps the data back to which run dir produced it.

| Phase |  Source run (preserved in `ignore/2026-06-04_source-runs/`) | Notes |
|---|---|---|
| 1. Retrieval, 2. Multi-Hop, 3. Code QA | `run_20260522_194009` | Killed mid-Phase-4. CSV only, no aggregate report. |
| 4. Adversarial | Same as above | Aggregate metric (`Accuracy`/`Discrimination`) printed to the run's stdout — captured in the README. |
| 5. Hallucination | `run_20260523_170522` | Multi-judge (Gemini / Anthropic / OpenAI); GPT-5 returned empty content (reasoning-model token-budget issue), so the avg is over the other two judges. |
| 6/7. Validated (6 datasets) | `run_20260523_170524` | Killed during Phase 8 Enhanced Judge. The 6 validated CSVs are complete. |
| 8. Enhanced Judge | `run_20260525_135759` | Resume run; single judge (`claude-sonnet-4-6`), debiased. |
| 9. SWE-Agent | `run_20260522_235114` | Single judge (`claude-sonnet-4-6`). Earlier attempts with Gemini 2.5 Pro and Opus 4.7 both failed (Gemini parse-rate collapse + Opus 400 Bad Request). |
| 10. Diff-Aware | `run_20260522_224538` | Both engines landed at the `0.333` correctness floor — the phase's symbol-match heuristic doesn't catch the engines' returned chunk shape. See [Known issues](#known-issues). |
| 11. Session Replay | `run_20260522_224538` | Same scoring caveat as Phase 10. |

## Configuration

- **Embeddings**: Gemini `gemini-embedding-001`
- **Retrieval match mode**: `--match-mode llm`
- **Max queries**: 100 per (engine × phase × dataset)
- **Judges**: sonnet-4-6 default; multi-model where the phase supports it (`hallucination`)
- **Engines**:
  - Delphi (`synsc-context`) — local bench instance on `:8743`, 17 indexed sources covering fastapi, httpx, pydantic, starlette, django, sqlalchemy, flask, numpy, requests, aiohttp, litestar, polars, typer, aiofiles, structlog, msgspec, pandas + 2 docs sites
  - Nia (`nia`) — hosted, indexed on demand via the API key

## Known issues

1. **Phase 10 (Diff-Aware) scoring is broken.**
   `benchmarks/phases/diff_aware.py::_symbol_appears` does a substring
   match against the engine adapter's standard `results[*].content` /
   `.snippet` fields. Both adapters return a different chunk shape on
   the diff-aware queries (or the bench-Delphi corpus didn't contain
   the exact symbols the cases reference), so every case scores stale
   miss + fresh miss + stable miss, which gives the `1/3 = 0.333`
   floor on every run.

   Fix: tighten the matcher to read each adapter's actual response
   shape (Nia returns `chunks[*].text`; Delphi returns `results[*].content`),
   and back-fill the 15 cases with their actual library exports at
   `commit_after`. Tracked as a follow-up.

2. **Phase 11 (Session Replay) scoring is broken** for the same
   reason. Both engines hit `win_rate ≤ 0.1`.

3. **GPT-5 in multi-judge mode returns empty content** when
   `max_tokens` is below the reasoning-model floor. The bench's
   `_call_llm_judge_raw_inner` for `openai` should bump `max_tokens`
   to ~8K for reasoning models. Currently it caps at 512.

4. **Opus 4.7 in SWE-Agent payloads returns 400 Bad Request.** The
   bench's SWE-Agent prompt structure isn't quite compatible with
   Opus's expected message shape. Sonnet 4.6 works.

5. **Context7 adapter stalls inside the runner's gather** even with
   the per-request `asyncio.wait_for(timeout=120)` from `dc52983`. The
   stall is at a different level — probably the streaming subprocess
   for the MCP fallback. Excluded from this run.

## Reproducing

```bash
# 1. Bring up Delphi
cd ~/Desktop/syntheticsciences/delphi
docker compose up -d
# wait for /health 200

# 2. Index the corpus (one-time)
for slug in fastapi/fastapi encode/httpx pydantic/pydantic encode/starlette \
            django/django pandas-dev/pandas sqlalchemy/sqlalchemy pallets/flask \
            numpy/numpy psf/requests aio-libs/aiohttp litestar-org/litestar \
            jcrist/msgspec pola-rs/polars fastapi/typer Tinche/aiofiles hynek/structlog; do
  curl -X POST -H "Authorization: Bearer $SYNSC_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"source_type\":\"repo\",\"url\":\"https://github.com/$slug.git\",\"display_name\":\"$slug\",\"async_mode\":true}" \
    http://localhost:8743/v1/sources
done

# 3. Run the bench
cd ~/Desktop/syntheticsciences/synsci-context-bench
uv sync
uv run python -m benchmarks --engines synsc nia --skip-indexing --match-mode llm \
  --max-queries 100 --multi-model -v
```
