<div align="center">

# SynSci Context Bench

Benchmark harness for comparing code context engines head-to-head.

Tests [Delphi](https://github.com/synthetic-sciences/synsc-delphi), [Context7](https://context7.com), and [Nia](https://trynia.ai) across **11 benchmark phases** — code retrieval, multi-hop, adversarial, hallucination, validated datasets, LLM-as-judge, position-debiased enhanced judge, SWE-Agent code generation, **diff-aware indexing** (does the engine correctly refresh its index after a real commit?), and **real-session replay** of cases where one engine visibly beat another.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Suites](https://img.shields.io/badge/phases-11-58a6ff?style=for-the-badge)]()
[![Engines](https://img.shields.io/badge/engines-3-f78166?style=for-the-badge)]()
[![Judge](https://img.shields.io/badge/judge-Claude_Sonnet_4.6-blueviolet?style=for-the-badge&logo=anthropic&logoColor=white)]()

<br/>

<img src="assets/charts/results.png" alt="Benchmark Results" width="950"/>

</div>

---

## Results

May 2026 run, 100 queries per engine per phase, all phases scored under `--match-mode llm`. Two engines: Delphi vs Nia. Judge: Claude Sonnet 4.6 (single-judge phases) + Gemini 2.5 Pro / Claude Opus 4.7 / GPT-5 ensemble (multi-judge phases). Raw CSVs + per-phase reports live in [`benchmarks/results/results_2026_05/`](benchmarks/results/results_2026_05/) with a manifest mapping each file to its source run. Full per-phase methodology in [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

### Core retrieval phases

| Phase | Metric | Delphi | Nia | Delphi delta |
|-------|--------|:---:|:---:|:---:|
| Retrieval | MRR | **0.676** | 0.505 | **+34 %** |
| Multi-Hop | hop coverage | 0.774 | **0.785** | −1.4 % |
| Code QA | accuracy (LLM-judged) | **0.465** | 0.208 | **+124 %** |
| Adversarial | discrimination | **0.427** | 0.185 | **+131 %** |
| Hallucination | avoidance (1 − rate, multi-judge avg) | 60.3 % | **65.7 %** | −5 pp |

### Enhanced LLM judge (position-debiased, 4D scoring, 100 queries per dataset)

Each query scored twice with swapped chunk ordering to eliminate positional bias. Sonnet 4.6 judges Relevance, Completeness, Specificity, and Faithfulness (0–3 each).

| Dataset | Engine | Total (0-12) | Wins |
|---------|--------|:-:|:-:|
| CodeSearchNet | **Delphi** | **0.517** | **33** |
| CodeSearchNet | Nia | 0.340 | 10 |
| CoSQA | **Delphi** | **0.922** | **47** |
| CoSQA | Nia | 0.745 | 27 |
| AdvTest | **Delphi** | **0.472** | **30** |
| AdvTest | Nia | 0.350 | 17 |
| **All-up** | **Delphi** | | **110 wins** |
| **All-up** | Nia | | 54 wins |

Delphi is higher than Nia on *every* dimension across *every* dataset.

### SWE-Agent Benchmark (25 tasks · sonnet-4-6 judge)

Does feeding context engine results to an LLM actually produce better code?

| Metric | Baseline (no context) | Delphi | Nia |
|--------|:---:|:---:|:---:|
| Judge composite | 0.681 | **0.784** | 0.801 |
| Criteria pass | 89 % | 90.7 % | **94 %** |
| Context utilization | — | 17.7 % | **29.5 %** |
| Parse rate | 60 % | **80 %** | 72 % |

Delphi wins parse rate by 11 pp (LLM produces more machine-readable outputs from Delphi context). Nia wins composite by 2 pp.

### Validated datasets (LLM-judged, 6 public datasets)

| Dataset | Delphi MRR | Nia MRR |
|---------|:---:|:---:|
| CoSQA | **0.352** | 0.179 |
| CodeFeedback-ST | 0.017 | **0.047** |
| AdvTest | **0.045** | 0.010 |
| StackOverflow-QA | 0.000 | **0.015** |
| CodeSearchNet | **0.007** | 0.000 |
| APPS | 0.000 | 0.000 |

LLM-judge mode is strict by design — both engines score low in absolute terms on the heavily-obfuscated datasets (AdvTest, StackOverflow-QA). On CoSQA (real natural-language queries) Delphi wins by **+96 %**.

### Phase 10 (diff-aware) and Phase 11 (session replay) — *scoring issue, not reported*

Both phases ran on both engines, but the symbol-match heuristic in `benchmarks/phases/diff_aware.py::_symbol_appears` is too strict for the chunk shapes the live adapters return — both engines landed at the `1/3 = 0.333` correctness floor on every diff-aware case and at `win_rate ≤ 0.1` on session replay. The phases ship with the bench but their numbers are not in the headline tables until the scorer is tightened. Details and a fix sketch in [`benchmarks/results/results_2026_05/README.md#known-issues`](benchmarks/results/results_2026_05/README.md#known-issues).

---

## Reproducing

```bash
# 1. Bring up Delphi (or any MCP-context engine on :8743)
cd ~/Desktop/syntheticsciences/delphi
docker compose up -d

# 2. Index a code corpus into Delphi (one-time, ~10 min for the May 2026 set)
#    Run dirs in results_2026_05/ used: fastapi, httpx, pydantic, starlette,
#    django, pandas, sqlalchemy, flask, numpy, requests, aiohttp, litestar,
#    msgspec, polars, typer, aiofiles, structlog
#    (See benchmarks/results/results_2026_05/README.md for the exact curl loop.)

# 3. Configure keys
cd ~/Desktop/syntheticsciences/synsci-context-bench
cp benchmarks/.env.local.example benchmarks/.env.local
$EDITOR benchmarks/.env.local     # set SYNSC_API_KEY, NIA_API_KEY, BENCH_*

# 4. Run the bench. ~2-3 hrs end to end.
uv sync
uv run python -m benchmarks --engines synsc nia --skip-indexing \
    --match-mode llm --max-queries 100 --multi-model -v
```

---

## Testing & offline reproducibility

The harness can run **without API keys, network, or Docker** via a deterministic
`MockAdapter` and offline judge, so the scorers are unit-testable and CI can
verify the pipeline end-to-end.

```bash
uv sync --extra dev
uv run pytest          # 33 tests: metrics, diff-aware, session-replay, mock, offline judge
```

The test suite pins the behavior of every scorer (catching the class of bug that
floored Phase 10) and includes a validated-eval pipeline check proving MRR is
~1.0 when the relevant document is actually retrieved — i.e. near-zero validated
MRRs are a corpus-coverage issue, not a metric bug.

See [`docs/THREATS_TO_VALIDITY.md`](docs/THREATS_TO_VALIDITY.md) for the full
validity treatment (contamination/leakage, vendor bias, judge reliability,
statistical-conclusion validity) written to publication standard.

---

## Fairness

This benchmark addresses common fairness concerns in engine comparison:

- **LLM-as-Judge** — Claude Sonnet evaluates result quality regardless of format (code vs docs), replacing biased file-path matching
- **Wall-clock latency** — all engines measured identically with `time.perf_counter()`. Delphi averages ~2.2-2.5s/query on production deployment.
- **Position debiasing** — enhanced judge shuffles chunk order to eliminate ~10% positional bias
- **Equal adapter treatment** — no artificial handicaps, equalized timeouts (120s), no fallback libraries
- **Consistent sample size** — 100 queries per engine per phase

---

## What's Being Tested

| # | Phase | Tests | Queries |
|:-:|-------|-------|:------:|
| 1 | **Retrieval Quality** | P@K, Recall@K, NDCG@K, MRR against known ground truth | 100 |
| 2 | **Multi-Hop Retrieval** | Queries needing context from 2+ files/repos | 100 |
| 3 | **Code QA** | Definitions, call sites, imports, inheritance, return types | 100 |
| 4 | **Adversarial Near-Miss** | Decoys: same name/wrong context, test vs prod, version confusion | 100 |
| 5 | **Hallucination Rate** | Does engine context prevent LLMs from making stuff up? | 100 |
| 6 | **CodeSearchNet** | Function-level code search (Husain et al. 2019) | 100 |
| 7 | **CoSQA** | Real web search queries (Huang et al. 2021) | 100 |
| 8 | **Enhanced Judge** | Position-debiased 4D + faithfulness + RAGAS metrics | 200 |
| 9 | **SWE-Agent** | Code generation with/without context, no-context baseline | 25 |
| 10 | **Diff-Aware Indexing** | Does the engine correctly refresh its index after a real commit? Stale / fresh / stability checks. | 15 |
| 11 | **Session Replay** | Real moments from production research sessions where one engine beat another | 10 |
| — | *AdvTest (supplementary)* | Adversarial/obfuscated code queries | 100 |

The diagnosis that drove the Phase 10/11 additions: pure code-retrieval benchmarks understate context-engine value-add. Real work needs the index to be *fresh* (not just accurate) and to handle the messy, mid-session moments where the engine that wins yesterday loses today. Phases 10 and 11 measure those.

### Phase 10: Diff-Aware Indexing

15 cases, each a real `(commit_A, commit_B)` pair on a public Python library (fastapi, httpx, pydantic, sqlalchemy, polars, litestar, msgspec, requests). Every case ships three queries:

| Query class | What it asks | Engine wins by |
|-------------|--------------|----------------|
| `stale_query` | A symbol that existed in A but was deleted in B | **Not** returning the deleted symbol |
| `fresh_query` | A symbol added in B (didn't exist in A) | Returning the new symbol |
| `stable_query` | A symbol unchanged across A and B | Returning the unchanged symbol (regression guard) |

Composite correctness = mean of `[1−stale_hit, fresh_hit, stable_hit]`. Three diagnostic rates are also reported: staleness (lower better), freshness, stability.

### Phase 11: Session Replay

Real moments from production research sessions, each labeled with the original failure cause (`missing_index_coverage`, `bad_retrieval`, `bad_ranking`, `bad_packaging`, `tool_ergonomics`, `benchmark_blind_spot`). The replay then re-classifies the failure under the live taxonomy so the report can show "10 cases labeled `bad_retrieval`, now 6 still failing / 2 reclassified as `bad_ranking` / 2 resolved." See `benchmarks/scoring/failure_taxonomy.py`.

---

## Quick Start

```bash
uv sync
cp benchmarks/.env.local.example benchmarks/.env.local
# fill in your API keys
```

```bash
# run everything
uv run python -m benchmarks

# run specific phases
uv run python -m benchmarks --skip-indexing --max-queries 100 --match-mode llm -v

# single engine
uv run python -m benchmarks --engines synsc --skip-indexing --max-queries 50

# validated datasets only
uv run python -m benchmarks --validated-only --match-mode llm --dataset codesearchnet cosqa advtest

# enhanced judge only
uv run python -m benchmarks --enhanced-judge-only
```

### Environment Variables

| Variable | What it does |
|----------|-------------|
| `SYNSC_API_URL` | Delphi server URL (default `http://localhost:8742`) |
| `SYNSC_API_KEY` | Delphi API key |
| `NIA_API_KEY` | Nia API key |
| `CONTEXT7_ENABLED` | Set `true` for Context7 |
| `BENCH_LLM_PROVIDER` | `anthropic`, `gemini`, or `openai` |
| `BENCH_LLM_MODEL` | Model ID for judge benchmarks |
| `BENCH_LLM_API_KEY` | API key for the judge LLM |

<details>
<summary>All CLI flags</summary>

| Flag | Effect |
|------|--------|
| `--engines synsc nia context7` | Pick engines |
| `--engines synsc-mcp` | Use Delphi via the MCP proxy (agent-realistic) |
| `--synsc-quality-mode {agent,default}` | Pass-through to the Delphi adapter |
| `--skip-indexing` | Skip repo indexing |
| `--skip-retrieval` | Skip retrieval phase |
| `--skip-multihop` | Skip multi-hop phase |
| `--skip-code-qa` | Skip code QA phase |
| `--skip-adversarial` | Skip adversarial phase |
| `--skip-hallucination` | Skip hallucination phase |
| `--skip-validated` | Skip validated dataset phases |
| `--skip-diff-aware` | Skip diff-aware indexing (Phase 10) |
| `--skip-session-replay` | Skip Session Replay (Phase 11) |
| `--diff-aware-only` | Run only diff-aware indexing |
| `--session-replay-only` | Run only session replay |
| `--validated-only` | Run only validated datasets |
| `--enhanced-judge-only` | Run only enhanced judge |
| `--match-mode llm` | Use LLM judge for validated scoring |
| `--judge-top-k N` | Judge top-N (was hard-coded to 3; default now 10) |
| `--seed N` | RNG seed for query sampling |
| `--num-seeds N` | Run N seeds and aggregate (3-5 for CIs) |
| `--no-debiasing` | Disable position debiasing (2x faster) |
| `--no-significance` | Skip statistical analysis |
| `--dataset cosqa codesearchnet advtest` | Pick specific datasets |
| `--swe-agent-only` | Run only SWE-Agent benchmark |
| `--skip-swe-agent` | Skip SWE-Agent benchmark |
| `--real-patch` | Enable real-patch SWE evaluation (clone + run tests) |
| `--with-agent-queries` | Also run AI-generated queries (default: gold only) |
| `--engines none` | Baseline-only mode (with `--swe-agent-only`) |
| `--max-queries N` | Limit query count per phase |
| `-v` | Verbose logging |

</details>

---

## Statistical Rigor

Every pairwise comparison includes paired t-tests, Wilcoxon signed-rank, bootstrap CIs (10K resamples), Cohen's d, Cliff's delta, and Bonferroni correction. The enhanced judge tracks its own reliability with Cohen's kappa and Position Consistency metrics.

---

## Engine Adapters

| Engine | Adapter | Notes |
|--------|---------|-------|
| **Delphi** | `benchmarks/adapters/synsc.py` | HTTP API, needs indexed repos |
| **Nia** | `benchmarks/adapters/nia.py` | REST API, global knowledge search |
| **Context7** | `benchmarks/adapters/context7.py` | HTTP API, pre-crawled docs |

Add a new engine by implementing `ContextEngineAdapter` from `benchmarks/adapters/base.py`.

---

## Project Structure

```
synsci-context-bench/
├── README.md                  this file
├── ARCHITECTURE.md            high-level design + diagnosis traceability
├── CHANGELOG.md               release notes (1.0 → 1.1 → reorg)
├── CONTRIBUTING.md            how to add phases / engines / metrics
├── pyproject.toml
│
├── benchmarks/
│   ├── README.md              package overview
│   ├── __main__.py            cli entry point
│   ├── runner.py              phase orchestrator
│   ├── config.py              env + path config (curated_dir, validated_dir, seeds)
│   │
│   ├── adapters/              one file per engine
│   │   ├── synsc.py           Delphi HTTP (quality_mode aware)
│   │   ├── synsc_mcp.py       Delphi MCP-proxy (build_context_pack)
│   │   ├── nia.py             Nia REST (full-latency accounted)
│   │   ├── context7.py        Context7 HTTP (full-latency accounted)
│   │   └── base.py            ContextEngineAdapter interface
│   │
│   ├── phases/                one module per benchmark phase
│   │   ├── multihop.py        Phase 2
│   │   ├── code_qa.py         Phase 3
│   │   ├── adversarial.py     Phase 4
│   │   ├── hallucination.py   Phase 5
│   │   ├── validated_eval.py  Phase 6 — CodeSearchNet / CoSQA / AdvTest
│   │   ├── swe_agent.py       Phase 9 — code generation benchmark
│   │   ├── swe_real_patch.py  Phase 9b — real-patch eval (opt-in)
│   │   ├── diff_aware.py    Phase 10 — diff-aware indexing
│   │   └── session_replay.py  Phase 11 — production session replay
│   │
│   ├── judges/                LLM-as-judge implementations
│   │   ├── llm_judge.py       3D blind scoring
│   │   └── enhanced_judge.py  4D position-debiased + RAGAS
│   │
│   ├── scoring/               deterministic scoring + analysis
│   │   ├── metrics.py         MRR, NDCG, P@K, R@K (dedup-fixed), MAP
│   │   ├── semantic_metrics.py     CodeBLEU + AST similarity
│   │   ├── context_grounding.py    citation, utilization, hallucination-reduction
│   │   ├── leaderboards.py    per-category leaderboards
│   │   ├── failure_taxonomy.py     classify failures into actionable buckets
│   │   └── statistical_analysis.py paired tests, bootstrap, effect sizes
│   │
│   ├── infra/                 operational glue
│   │   ├── logging_config.py  structured logging + per-query traces
│   │   ├── sampling.py        seeded + stratified sampling
│   │   ├── latency.py         end-to-end latency meter
│   │   └── consistency.py     repeat-run consistency checks
│   │
│   ├── utils/                 standalone helpers
│   │   ├── dataset_loader.py  downloads CodeSearchNet / CoSQA / ...
│   │   └── create_benchmark_repo.py  fixture builder
│   │
│   ├── datasets/
│   │   ├── curated/           hand-built cases owned by this repo
│   │   │   ├── retrieval_ground_truth.json
│   │   │   ├── multihop_test_cases.json
│   │   │   ├── code_qa_test_cases.json
│   │   │   ├── adversarial_test_cases.json
│   │   │   ├── hallucination_test_cases.json
│   │   │   ├── swe_agent_test_cases.json
│   │   │   ├── diff_aware_test_cases.json
│   │   │   └── session_replay_cases.json
│   │   └── validated/         downloaded standard datasets
│   │       ├── codesearchnet_benchmark.json
│   │       ├── cosqa_benchmark.json
│   │       ├── advtest_benchmark.json
│   │       └── ...
│   │
│   └── results/               run_<ts>/ directories — traces, manifests, CSVs
│
├── docs/
│   ├── README.md              docs index
│   ├── PHASES.md              per-phase deep dive
│   ├── METRICS.md             per-metric reference
│   └── BENCHMARK_REPORT.md    last full-run report
│
├── scripts/
│   └── generate_charts.py     regenerate assets/charts/results.png
│
└── assets/
    └── charts/
```

Every subdirectory has its own `README.md` that describes what's in it
and the local conventions.

---

## Regenerate the Chart

```bash
python scripts/generate_charts.py
```

---

## References

Husain et al. (2019) CodeSearchNet Challenge. arXiv:1909.09436 ·
Huang et al. (2021) CoSQA. ACL 2021 ·
Zheng et al. (2023) Judging LLM-as-a-Judge ·
Shi et al. (2025) Judging the Judges ·
Es et al. (2024) RAGAS ·
Ren et al. (2020) CodeBLEU ·
Thakur et al. (2021) BEIR. NeurIPS

---

<div align="center">

Built by the [Synthetic Sciences](https://github.com/synthetic-sciences) team

Questions? **hello@syntheticsciences.ai**

</div>
