<div align="center">

# SynSci Context Bench

Benchmark harness for comparing code context engines head-to-head.

Tests [Delphie](https://github.com/synthetic-sciences/synsc-context), [Context7](https://context7.com), and [Nia](https://trynia.ai) across 8 benchmark suites, ~2,000 queries, using automated IR metrics and a position-debiased LLM judge.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Suites](https://img.shields.io/badge/suites-8-58a6ff?style=for-the-badge)]()
[![Engines](https://img.shields.io/badge/engines-3-f78166?style=for-the-badge)]()
[![Judge](https://img.shields.io/badge/judge-Claude_Sonnet_4.6-blueviolet?style=for-the-badge&logo=anthropic&logoColor=white)]()

<br/>

<img src="assets/charts/results.png" alt="Benchmark Results" width="950"/>

</div>

---

## Results

All numbers pulled from `benchmarks/results/results.json`. Full methodology in [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

### Custom Benchmarks (hand-crafted queries, 3 engines)

| Benchmark | Metric | Delphie | Context7 | Nia |
|-----------|--------|:---:|:---:|:---:|
| Retrieval | MRR | **0.817** | 0.350 | 0.345 |
| Multi-Hop | Coverage | **0.967** | 0.850 | 0.783 |
| Code QA | Accuracy | **0.867** | 0.200 | 0.154 |
| Adversarial | Accuracy | **0.800** | 0.000 | 0.000 |
| Hallucination | Rate (lower is better) | **40%** | **40%** | 55.6% |

### Industry-Standard Datasets (CoSQA + CodeSearchNet, 450 queries each)

| Dataset | Metric | Delphie | Context7 | Nia |
|---------|--------|:---:|:---:|:---:|
| CodeSearchNet | MRR | **0.940** | 0.000 | 0.053 |
| CodeSearchNet | NDCG@10 | **0.941** | 0.000 | 0.053 |
| CoSQA | MRR | **0.636** | 0.002 | 0.003 |
| CoSQA | NDCG@10 | **0.642** | 0.002 | 0.004 |

### Enhanced LLM Judge (position-debiased, 4D scoring, ~500 queries per dataset)

Each query is scored twice with swapped chunk ordering and averaged. This eliminates the ~10% positional bias documented in LLM evaluations (Zheng et al. 2023). Scored on Relevance, Completeness, Specificity, and Faithfulness (0-3 each).

| Dataset | Engine | Relevance | Completeness | Specificity | Faithfulness | Total | Wins |
|---------|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| CodeSearchNet | **Delphie** | **1.98** | **1.88** | **1.36** | **2.04** | **1.82** | **436** |
| CodeSearchNet | Context7 | 0.63 | 0.30 | 0.40 | 1.30 | 0.66 | 36 |
| CoSQA | Delphie | 1.16 | 0.55 | 0.68 | 1.60 | 1.00 | 109 |
| CoSQA | **Context7** | **1.76** | **1.35** | **1.41** | **2.13** | **1.66** | **341** |

Delphie wins **88%** of CodeSearchNet queries (code-to-code retrieval). Context7 wins **68%** of CoSQA queries (documentation-style "how do I..." questions). See the note on ecological validity below.

### A note on CoSQA vs CodeSearchNet

CoSQA queries look like "how to parse json in python" or "python read csv file". An LLM already knows those answers and would never call a context engine for them. CodeSearchNet queries ("find the function that does X") match how agents actually use context engines in practice. CoSQA scores should be weighted lower when evaluating engines for real agent workflows.

---

## What's Being Tested

| # | Suite | Tests | Size |
|:-:|-------|-------|:----:|
| 1 | **Retrieval Quality** | P@K, Recall@K, NDCG@K, MRR against known ground truth | 10q |
| 2 | **Multi-Hop Retrieval** | Queries needing context from 2+ files/repos | 10q |
| 3 | **Code QA** | Definitions, call sites, imports, inheritance, return types | 15q |
| 4 | **Adversarial Near-Miss** | Decoys: same name/wrong context, test vs prod, version confusion | 10q |
| 5 | **Hallucination Rate** | Does engine context prevent LLMs from making stuff up? | 10q |
| 6 | **Validated Datasets** | CoSQA + CodeSearchNet from HuggingFace | ~900q |
| 7 | **LLM-as-Judge** | Blind 3D scoring (relevance, completeness, specificity) | ~1,000q |
| 8 | **Enhanced Judge** | Position-debiased 4D + faithfulness + RAGAS metrics | ~1,000q |

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

# run specific suites
uv run python -m benchmarks --judge-only --engines synsc context7
uv run python -m benchmarks --retrieval-only --skip-indexing
uv run python -m benchmarks --enhanced-judge-only

# quick iteration
uv run python -m benchmarks --judge-only --engines synsc --max-queries 50

# download datasets from HuggingFace
uv run python -m benchmarks --download-datasets
```

### Environment Variables

| Variable | What it does |
|----------|-------------|
| `SYNSC_API_URL` | Delphie server URL (default `http://localhost:8742`) |
| `SYNSC_API_KEY` | Delphie API key |
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
| `--skip-indexing` | Skip repo indexing |
| `--no-debiasing` | Disable position debiasing (2x faster) |
| `--no-significance` | Skip statistical analysis |
| `--bootstrap-n N` | Bootstrap resamples (default 10,000) |
| `--dataset cosqa\|codesearchnet` | Run one dataset |
| `--multi-model` | Hallucination across model tiers |
| `--max-queries N` | Limit query count |

</details>

---

## Statistical Rigor

Every pairwise comparison includes paired t-tests, Wilcoxon signed-rank, bootstrap CIs (10K resamples), Cohen's d, Cliff's delta, and Bonferroni correction. The enhanced judge tracks its own reliability with Cohen's kappa and Position Consistency metrics.

---

## Engine Adapters

| Engine | Adapter | Notes |
|--------|---------|-------|
| **Delphie** | `benchmarks/adapters/synsc.py` | HTTP API, needs indexed repos |
| **Nia** | `benchmarks/adapters/nia.py` | REST API, global knowledge search |
| **Context7** | `benchmarks/adapters/context7.py` | HTTP API, pre-crawled docs |

Add a new engine by implementing `ContextEngineAdapter` from `benchmarks/adapters/base.py`.

---

## Project Structure

```
benchmarks/
  __main__.py             cli entry point
  runner.py               orchestrates suites
  config.py               env config
  metrics.py              NDCG, MRR, P@K, R@K, MAP
  semantic_metrics.py     CodeBLEU, AST similarity
  statistical_analysis.py paired tests, bootstrap, effect sizes
  llm_judge.py            3D blind scoring
  enhanced_judge.py       4D debiased + RAGAS
  validated_eval.py       CoSQA / CodeSearchNet
  hallucination.py        hallucination rate
  multihop.py             multi-hop retrieval
  code_qa.py              code QA
  adversarial.py          adversarial near-miss
  adapters/               engine adapters
  datasets/               ground truth + downloads
  results/                output JSON
docs/
  BENCHMARK_REPORT.md     full analysis
  RESULTS.md              tabulated results
  WHITEPAPER.md           technical whitepaper
scripts/
  generate_charts.py      regenerate the chart
```

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

Questions? **team@syntheticsciences.ai**

</div>
