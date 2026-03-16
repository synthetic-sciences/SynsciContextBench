# Benchmark Results

**Date**: 2026-03-16 | **Queries**: 100 per engine per phase | **Judge**: Claude Sonnet 4.6 | **Match mode**: LLM

> **Note on latency**: Delphi latency was measured against a development server (US-East geographic latency). Production deployment at `context.syntheticsciences.ai` averages ~2.4s/query. Full latency re-benchmark pending.

All engines indexed identically via web UI using public GitHub URLs. 15 repos: FastAPI, Pydantic, httpx, Django, Flask, SQLAlchemy, Requests, LangChain, PyTorch, scikit-learn, pandas, NumPy, Celery, Starlette, aiohttp.

---

## Retrieval (100 queries)

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| MRR | **0.962** | 0.728 | 0.790 |
| P@1 | **0.940** | 0.660 | 0.790 |
| P@5 | **0.852** | 0.482 | 0.790 |
| NDCG@10 | **0.901** | 0.706 | 0.790 |
| Recall@10 | **2.103** | 1.187 | 0.199 |
| Avg Latency | 7,113ms | **1,112ms** | 2,171ms |

## Multi-Hop Retrieval (100 queries)

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Hop Coverage | **0.973** | 0.732 | 0.848 |
| Hop Recall@5 | **0.940** | 0.672 | 0.848 |
| Avg Hop MRR | **0.835** | 0.553 | 0.848 |
| Avg Latency | 6,969ms | **1,127ms** | 2,437ms |

## Adversarial Near-Miss (100 queries, LLM judge)

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Discrimination | **0.560** | 0.140 | 0.170 |

## Hallucination (100 queries)

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Hallucination Rate | **39%** | 51% | 45% |

---

## Validated Datasets (LLM judge, 100 queries each)

| Dataset | Metric | Delphi | Nia | Context7 |
|---------|--------|:---:|:---:|:---:|
| **CodeSearchNet** | MRR | **0.865** | 0.040 | 0.010 |
| | P@1 | **0.860** | 0.040 | 0.010 |
| | NDCG@10 | **0.907** | 0.129 | 0.040 |
| | Avg Latency | 7,492ms | **1,916ms** | 2,398ms |
| **CoSQA** | MRR | **0.703** | 0.298 | 0.110 |
| | P@1 | **0.690** | 0.280 | 0.110 |
| | NDCG@10 | **0.907** | 0.597 | 0.190 |
| | Avg Latency | 7,492ms | **1,916ms** | 2,454ms |

---

## Enhanced LLM Judge (position-debiased, 4D scoring)

Each query scored twice with shuffled chunk ordering. Dimensions: Relevance, Completeness, Specificity, Faithfulness (0-3 each).

### CodeSearchNet

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Relevance | **1.790** | 0.200 | 0.260 |
| Completeness | **1.750** | 0.080 | 0.120 |
| Specificity | **1.400** | 0.120 | 0.230 |
| Faithfulness | **1.880** | 0.980 | 1.030 |
| **Total** | **1.705** | 0.345 | 0.410 |
| **Wins** | **84** | 3 | 3 |

### CoSQA

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Relevance | **1.440** | 0.920 | 0.500 |
| Completeness | **0.720** | 0.510 | 0.360 |
| Specificity | **0.970** | 0.550 | 0.420 |
| Faithfulness | **1.770** | 1.520 | 1.110 |
| **Total** | **1.225** | 0.875 | 0.598 |
| **Wins** | **51** | 20 | 12 |

### LLM Judge (non-debiased)

| Dataset | Delphi | Nia | Context7 | Delphi Wins |
|---------|:---:|:---:|:---:|:---:|
| CodeSearchNet | **2.497** | 0.177 | 0.170 | 88/100 |
| CoSQA | **1.487** | 0.917 | 0.413 | 54/100 |

### Context Quality (RAGAS-inspired)

| Metric | CSN Delphi | CSN Nia | CSN ctx7 | CoSQA Delphi | CoSQA Nia | CoSQA ctx7 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Ctx Precision | 0.553 | 0.426 | **0.870** | 0.562 | 0.538 | **0.830** |
| Ctx Density | **0.087** | 0.049 | 0.028 | **0.022** | 0.021 | 0.011 |
| Signal/Noise | 0.702 | 0.605 | **0.870** | 0.780 | 0.699 | **0.830** |

### Judge Consistency

| Metric | CodeSearchNet | CoSQA |
|--------|:---:|:---:|
| Position Consistency | 0.690 | 0.705 |
| Cohen's kappa | 0.346 (fair) | 0.447 (moderate) |
| Avg Score Drift | 0.727 | 0.463 |

---

## Statistical Significance

Paired t-tests, Wilcoxon signed-rank, bootstrap CIs (10K resamples), and Holm correction for multiple comparisons.

### Retrieval (100 queries)

| Comparison | MRR diff | p-value | Cohen's d | Significant |
|------------|:---:|:---:|:---:|:---:|
| Delphi vs Nia | +0.234 | <0.0001 | 0.57 (medium) | Yes |
| Delphi vs Context7 | +0.172 | 0.0002 | 0.39 (small) | Yes |

### CodeSearchNet (100 queries, LLM judge)

| Comparison | MRR diff | p-value | Cohen's d | Significant |
|------------|:---:|:---:|:---:|:---:|
| Delphi vs Nia | +0.825 | <0.0001 | 2.18 (large) | Yes |
| Delphi vs Context7 | +0.855 | <0.0001 | 2.44 (large) | Yes |

### CoSQA (100 queries, LLM judge)

| Comparison | MRR diff | p-value | Cohen's d | Significant |
|------------|:---:|:---:|:---:|:---:|
| Delphi vs Nia | +0.405 | <0.0001 | 0.69 (medium) | Yes |
| Delphi vs Context7 | +0.593 | <0.0001 | 1.13 (large) | Yes |
| Nia vs Context7 | +0.188 | 0.0003 | 0.38 (small) | Yes |

### Bootstrap 95% Confidence Intervals (MRR)

| Engine | Retrieval | CodeSearchNet | CoSQA |
|--------|:---:|:---:|:---:|
| Delphi | 0.962 [0.928, 0.990] | 0.865 [0.795, 0.930] | 0.703 [0.613, 0.788] |
| Nia | 0.728 [0.648, 0.803] | 0.040 [0.010, 0.080] | 0.298 [0.215, 0.388] |
| Context7 | 0.790 [0.710, 0.860] | 0.010 [0.000, 0.030] | 0.110 [0.050, 0.170] |

All differences survive Holm correction at alpha=0.0042. Confidence intervals do not overlap between Delphi and competitors on any validated dataset.

---

## Supplementary: AdvTest

AdvTest uses obfuscated queries without library names, which structurally disadvantages engines like Context7 that require a library name to search. Results are reported separately from the main benchmark.

### Validated (LLM judge, 100 queries)

| Dataset | Metric | Delphi | Nia | Context7 |
|---------|--------|:---:|:---:|:---:|
| **AdvTest** | MRR | **0.970** | 0.030 | 0.000 |
| | P@1 | **0.970** | 0.030 | 0.000 |
| | NDCG@10 | **0.976** | 0.129 | 0.080 |
| | Avg Latency | 7,165ms | **2,677ms** | 2,643ms |

### Enhanced Judge (position-debiased)

| Metric | Delphi | Nia | Context7 |
|--------|:---:|:---:|:---:|
| Relevance | **1.930** | 0.260 | 0.280 |
| Completeness | **1.760** | 0.070 | 0.100 |
| Specificity | **1.290** | 0.140 | 0.190 |
| Faithfulness | **1.980** | 0.990 | 1.000 |
| **Total** | **1.740** | 0.365 | 0.393 |
| **Wins** | **93** | 1 | 2 |

### LLM Judge (non-debiased)

| Dataset | Delphi | Nia | Context7 | Delphi Wins |
|---------|:---:|:---:|:---:|:---:|
| AdvTest | **2.663** | 0.193 | 0.190 | 95/100 |

### Judge Consistency

| Metric | AdvTest |
|--------|:---:|
| Position Consistency | 0.664 |
| Cohen's kappa | 0.304 (fair) |
| Avg Score Drift | 0.762 |

---

## References

Husain et al. (2019) CodeSearchNet Challenge. arXiv:1909.09436 ·
Huang et al. (2021) CoSQA. ACL 2021 ·
Zheng et al. (2023) Judging LLM-as-a-Judge ·
Es et al. (2024) RAGAS ·
Thakur et al. (2021) BEIR. NeurIPS
