# Changelog

This repo follows the spirit of [Keep a Changelog](https://keepachangelog.com).
Versions track the benchmark methodology, not just code edits.

## Unreleased

### Added — package reorganization

- `benchmarks/` is now subdivided into `phases/`, `judges/`, `scoring/`,
  `infra/`, `utils/`. Each subpackage has its own README documenting its
  contract.
- `benchmarks/datasets/` split into `curated/` (in-repo cases) and
  `validated/` (downloaded standard datasets).
- New top-level docs: `ARCHITECTURE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `docs/PHASES.md`, `docs/METRICS.md`, plus a README in every subfolder.
- `BenchmarkConfig` exposes `curated_dir` and `validated_dir` properties
  so phases reference paths through config, not raw strings.

## 1.1.0 — the diff-aware phase & session replay, fairness fixes

### Added

- **Phase 10 — Diff-Aware Indexing.** 20 hand-curated cases across 8
  categories (`tool_contract`, `graph_memory`, `artifact`, `paper_qa`,
  `multi_turn`, `prior_decision`, `avoid_repeat`, `synthesis`). Per-category
  composite blends anchor-hit + evidence recall + optional LLM rubric.
- **Phase 11 — Real-Session Replay.** 10 production-session losses with
  labeled causes; the replay re-classifies each failure under the live
  taxonomy so the report can show "still missing_index_coverage" vs
  "now bad_ranking" vs "resolved".
- **Per-category leaderboards** (`scoring/leaderboards.py`). Replaces the
  single-winner report with `code_retrieval`, `docs_lookup`, `paper_qa`,
  `the phase-10 benchmark_graph`, `tool_contract`, `swe_patch`, `context_utilization`,
  `hallucination_inverted`.
- **Failure taxonomy classifier** (`scoring/failure_taxonomy.py`). Buckets:
  `missing_index_coverage`, `bad_retrieval`, `bad_ranking`, `bad_packaging`,
  `tool_ergonomics`, `benchmark_blind_spot`.
- **Delphi MCP adapter** (`adapters/synsc_mcp.py`). Uses
  `build_context_pack` when the proxy advertises it, falls back to
  `search_code` otherwise. Both adapters take `quality_mode` (default
  `agent`).
- **`scoring/context_grounding.py`** — citation share, fact utilization,
  answer-change, hallucination-reduction.
- **`infra/latency.py`** — `LatencyMeter` with request / retry / sleep
  buckets.
- **`phases/swe_real_patch.py`** — opt-in real-patch SWE evaluation
  (`--real-patch`) that clones, applies, and runs tests.
- New CLI flags: `--diff-aware-only`, `--session-replay-only`,
  `--skip-diff-aware`, `--skip-session-replay`, `--engines synsc-mcp`,
  `--synsc-quality-mode`, `--judge-top-k`, `--seed`, `--num-seeds`,
  `--real-patch`.

### Fixed

- `Recall@K` was producing values > 1.0 when the engine returned multiple
  chunks for the same ground-truth source file. `scoring/metrics.py` now
  de-duplicates by `result.id` and clamps at 1.0.
- `_match_relevance` no longer awards relevance on a single keyword match;
  single tokens like `BaseModel` would otherwise mark almost any chunk
  from that library as relevant.
- Validated LLM judge previously hard-coded `top_k=3`, silently forcing
  rank-4+ irrelevant and capping `Recall@10`. Now respects
  `--judge-top-k` (default 10).
- All `queries[:max_queries]` truncations replaced with
  `infra.sampling.sample_seeded` / `stratified_sample`. The same `--seed`
  produces the same draw across phases.
- SWE-Agent test cases are stratified by knowledge tier on sub-sampling
  so the A/B/C mix is preserved.
- Context7 and Nia adapters now record full user-visible latency,
  including rate-limit sleeps and 429 retries. The previous behavior
  hid Context7's per-request delay and Nia's 429 backoffs.
- `adapters/synsc.py` now sends `quality_mode=agent` and prefers
  `/v1/search/context_pack`, falling back to `/v1/search/code` on 404.

## 1.0.0 — Initial public release

9-phase benchmark across Delphi / Context7 / Nia: retrieval, multi-hop,
code-QA, adversarial near-miss, hallucination, CodeSearchNet / CoSQA /
AdvTest, 3D LLM judge, position-debiased enhanced judge, SWE-Agent.
