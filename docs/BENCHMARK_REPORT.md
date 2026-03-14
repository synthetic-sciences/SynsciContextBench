# Benchmark Report: synsc-context vs Nia vs Context7

**Version**: 4.0
**Date**: 2026-03-14
**Authors**: Synthetic Sciences Engineering Team
**Test Corpora**: FastAPI, Pydantic, httpx (custom); CoSQA (industry-standard); CodeSearchNet (industry-standard)
**Judge LLM**: Claude Sonnet 4.6 (Anthropic) — with position debiasing (Zheng et al. 2023)
**Benchmark Harness**: `synsc-context/benchmarks/` (8 test suites + enhanced judge + statistical analysis)
**Engines Tested**: synsc-context, Nia (trynia.ai), Context7 (context7.com)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What Changed in v4.0](#what-changed-in-v40)
3. [Benchmark Overview](#benchmark-overview)
4. [Part I — Custom Benchmarks (Hand-Crafted Datasets)](#part-i--custom-benchmarks)
5. [Part II — Industry-Standard Benchmarks (Validated Datasets)](#part-ii--industry-standard-benchmarks)
6. [Part III — LLM-as-Judge (Fair Cross-Engine Evaluation)](#part-iii--llm-as-judge)
7. [Part IV — Enhanced Judge (Position-Debiased, 4D Scoring)](#part-iv--enhanced-judge)
8. [Context Quality Metrics](#context-quality-metrics)
9. [Judge Consistency Analysis](#judge-consistency-analysis)
10. [Context Enrichment Analysis](#context-enrichment-analysis)
11. [Latency Comparison](#latency-comparison)
12. [Methodology & Fairness](#methodology--fairness)
13. [Ecological Validity](#ecological-validity)
14. [Strengths & Weaknesses](#strengths--weaknesses)
15. [Limitations & Overfitting Risk](#limitations--overfitting-risk)
16. [Conclusions & Next Steps](#conclusions--next-steps)

---

## Executive Summary

synsc-context was evaluated against Nia and Context7 across 8 benchmark suites plus a position-debiased enhanced LLM judge with statistical consistency analysis. The evaluation evolved through multiple rounds to ensure fairness, culminating in a blind, position-debiased LLM judge with 4-dimensional scoring and RAGAS-inspired context quality metrics.

In v4.0, we added a **position-debiased enhanced judge** (Zheng et al. 2023) that evaluates each query twice in both chunk orderings and averages scores, a 4th scoring dimension (**faithfulness**), and context quality metrics (precision, density, signal-to-noise, chunk diversity). We also added judge self-consistency metrics (Cohen's κ, Position Consistency) to measure how reliable the judge itself is.

### Key Results at a Glance

| Benchmark Type | Dataset | synsc-context | Nia | Context7 | Winner |
|---|---|:---:|:---:|:---:|:---:|
| Retrieval (MRR) | Custom (10q) | **0.950** | 0.175 | — | synsc |
| Multi-Hop (Coverage) | Custom (10q) | **0.967** | 0.867 | — | synsc |
| Code QA (Accuracy) | Custom (15q) | **1.000** | 0.333 | — | synsc |
| Adversarial (Accuracy) | Custom (10q) | **0.800** | 0.000 | — | synsc |
| Hallucination Rate | Custom (10q) | **0%** | N/A | — | synsc |
| Validated Retrieval (MRR) | CoSQA (500q) | **0.629** | 0.004 | — | synsc |
| Validated Retrieval (MRR) | CodeSearchNet (497q) | **0.940** | — | — | synsc |
| LLM Judge (Total 0-3) | CodeSearchNet (497q) | **2.685** | — | 0.420 | synsc (6.4x) |
| LLM Judge (Total 0-3) | CoSQA (500q) | 1.181 | — | **1.498** | Context7 (1.27x) |
| **Enhanced Judge (4D, debiased)** | **CodeSearchNet (497q)** | **1.815** | — | 0.655 | **synsc (2.8x)** |
| **Enhanced Judge (4D, debiased)** | **CoSQA (500q)** | 0.998 | — | **1.664** | **Context7 (1.67x)** |

synsc-context dominates code-to-code retrieval (CodeSearchNet: 2.8x on debiased 4D scoring, 436 wins vs 36). Context7 leads on natural-language "how do I..." queries (CoSQA: 1.67x, 341 wins vs 109) — though these queries have low ecological validity for agent workflows (see [Ecological Validity](#ecological-validity)).

---

## What Changed in v4.0

| Change | Impact |
|--------|--------|
| **Position-debiased enhanced judge** | Evaluates each query twice (original + reversed chunk order), averages scores — eliminates ~10% positional bias (Zheng et al. 2023) |
| **4-dimensional scoring** | Added **faithfulness** (0-3) alongside relevance, completeness, specificity |
| **Judge self-consistency metrics** | Cohen's κ, Position Consistency (PC), avg score drift — measures judge reliability |
| **RAGAS-inspired context quality metrics** | Context precision, density, signal-to-noise, chunk diversity — no LLM needed |
| **Concurrent dataset evaluation** | CodeSearchNet + CoSQA run in parallel via asyncio.gather |
| **Ecological validity analysis** | Discussion of benchmark-vs-real-world relevance |

### Prior Changes (v3.0)

| Change | Impact |
|--------|--------|
| Added Context7 as third engine | Three-way comparison |
| Post-retrieval context enrichment | Function signatures + docstrings prepended to results |
| Chunk relationships (adjacent + same_class edges) | Built at index time for future graph traversal |
| Full-scale LLM judge | 497-500 queries per dataset (up from 50) |
| Auth + response caching | Reduced repeated Supabase round-trips |
| Batch embedding inserts | 100-per-batch instead of 1-at-a-time |

### Context Enrichment (v3.0)

Search results now include structural context prepended to each chunk:

```
# function: async def app(request: Request) -> Response:
# Docstring: Takes a function or coroutine func(request) -> response,
# and returns an ASGI application.
# preceding context:
#     if not errors:
#         response_field = dependant.response_field
#         ...

<actual code chunk>
```

This helps LLM consumers understand the code's purpose without reading the entire file.

### Enrichment Impact on LLM Judge Scores

| Dataset | v2.0 (no enrichment) | v3.0 (with enrichment) | Delta |
|---------|:---:|:---:|:---:|
| **CodeSearchNet Relevance** | 2.830 | **2.861** | +0.031 |
| **CodeSearchNet Completeness** | 2.790 | **2.813** | +0.023 |
| **CodeSearchNet Specificity** | 2.300 | **2.382** | +0.082 |
| **CodeSearchNet Total** | 2.640 | **2.685** | **+0.045** |
| **CoSQA Relevance** | 1.490 | **1.522** | +0.032 |
| **CoSQA Completeness** | 0.970 | **1.032** | +0.062 |
| **CoSQA Specificity** | 1.040 | 0.988 | -0.052 |
| **CoSQA Total** | 1.167 | **1.181** | **+0.014** |

Specificity saw the biggest improvement on CodeSearchNet (+0.082), confirming that function signatures help the judge recognize that retrieved chunks are specific to the query.

---

## Benchmark Overview

The evaluation was conducted in three phases, each designed to address fairness concerns from the prior phase:

| Phase | Approach | Fairness Level | Why |
|-------|----------|:-:|-----|
| **Phase 1**: Custom benchmarks | Hand-crafted queries against FastAPI/Pydantic/httpx | Medium | Custom datasets may favor synsc-context's design |
| **Phase 2**: Validated benchmarks | Industry-standard CoSQA + CodeSearchNet | High | Standard datasets, but content-matching penalizes text transformation |
| **Phase 3**: LLM-as-Judge | Claude Sonnet 4.6 blindly scores retrieved context | **Highest** | Fully engine-agnostic — no content or file matching bias |

---

## Part I — Custom Benchmarks

**Test corpus**: FastAPI, Pydantic, httpx (indexed on both synsc-context and Nia)
**Date**: 2026-03-09
**Note**: Context7 was not evaluated on custom benchmarks (added later).

### 1. Retrieval Quality (Precision@K / NDCG / MRR)

*10 natural language queries — measuring how well the engine surfaces the most relevant code.*

| Metric | synsc-context | Nia | Delta |
|--------|:---:|:---:|:---:|
| **MRR** | **0.950** | 0.175 | +443% |
| **P@1** | **0.900** | 0.100 | +800% |
| **P@3** | **0.833** | 0.067 | +1143% |
| **P@5** | **0.720** | 0.160 | +350% |
| **P@10** | **0.560** | 0.210 | +167% |
| **NDCG@10** | **0.890** | 0.262 | +240% |
| Avg latency | **3,713ms** | 10,017ms | 2.7x faster |

**Analysis**: synsc-context returns the correct result in the top position 90% of the time vs 10% for Nia. Nia's universal-search endpoint mixes documentation, issues, and code, diluting code-specific results.

---

### 2. Multi-Hop Retrieval

*10 queries requiring context from multiple files/repos — e.g., "How does FastAPI validate request bodies using Pydantic models?"*

| Metric | synsc-context | Nia | Delta |
|--------|:---:|:---:|:---:|
| **Hop Coverage** | **0.967** | 0.867 | +12% |
| **Hop Recall@5** | **0.867** | 0.733 | +18% |
| **Hop MRR** | **0.913** | 0.616 | +48% |
| Avg latency | **3,575ms** | 7,025ms | 2.0x faster |

**Analysis**: Both engines handle multi-hop queries reasonably well. synsc-context's 0.967 coverage means it surfaces code from nearly all required libraries in a single query.

---

### 3. Code-Specific QA

*15 queries testing code understanding: definitions, call sites, imports, inheritance, argument usage, return types.*

| Metric | synsc-context | Nia | Delta |
|--------|:---:|:---:|:---:|
| **Accuracy** | **1.000** | 0.333 | +200% |
| **MRR** | **1.000** | 0.333 | +200% |
| **Symbol accuracy** | **1.000** | 0.600 | +67% |
| **Chunk coherence** | **1.000** | 0.333 | +200% |
| Avg latency | **3,549ms** | 7,137ms | 2.0x faster |

**Breakdown by QA type:**

| QA Type | synsc-context | Nia |
|---------|:---:|:---:|
| definition | **1.000** | 0.000 |
| call_site | **1.000** | 0.400 |
| import | **1.000** | 0.000 |
| inheritance | **1.000** | 1.000 |
| argument_usage | **1.000** | 1.000 |
| return_type | **1.000** | 0.000 |

**Analysis**: synsc-context achieves perfect scores. Nia struggles with definitions, imports, and return types — fundamental code navigation tasks. synsc-context's chunk-level indexing with AST-extracted symbols provides a clear advantage.

---

### 4. Adversarial Near-Miss

*10 pairs of semantically similar but functionally different code — e.g., `json.dumps()` vs `json.loads()`, test code vs production code.*

| Metric | synsc-context | Nia | Delta |
|--------|:---:|:---:|:---:|
| **Accuracy** | **0.800** | 0.000 | — |
| **Discrimination** | **0.550** | 0.000 | — |
| **Decoy confusion** | 30.0% | 0.0% | — |
| Avg latency | **3,851ms** | 7,233ms | 1.9x faster |

**Analysis**: Nia scored 0.000 across all adversarial categories — it failed to return the correct target for any pair. synsc-context scores 0.800 accuracy but only 0.550 discrimination, indicating the right code is found but decoys sometimes rank higher. This is synsc-context's weakest area and the biggest improvement opportunity (reranking, symbol boosting).

---

### 5. Hallucination Rate

*10 code generation tasks — feed engine context to Gemini 2.5 Flash, validate generated code against known API surfaces.*

| Metric | synsc-context | Nia |
|--------|:---:|:---:|
| **True hallucination rate** | **0%** | N/A |
| Correct abstentions | 30% | — |
| Context miss rate | 0% | — |
| Overall failure rate (legacy) | 30% | 30% |

**Breakdown**: For 7 queries where the engine had relevant context, the LLM generated correct code in all cases — **0% true hallucination rate**. The 30% overall rate comes from 3 queries about un-indexed libraries (SQLAlchemy, PyTorch, Tailwind CSS), where the LLM correctly abstained. Nia's 30% was due to rate limiting (2 search failures + 1 generation failure).

---

## Part II — Industry-Standard Benchmarks

**Methodology**: Both engines indexed the same GitHub repositories containing benchmark corpus code. Retrieval is measured by content similarity matching (Jaccard + SequenceMatcher, threshold 0.5) and file-path matching.

### CoSQA (Code Search and Question Answering)

**Source**: Huang et al., 2021 — 500 real web search queries for Python code, human-annotated relevance labels.
**Corpus**: [KB-syntheticsciences/benchmark-corpus-cosqa](https://github.com/KB-syntheticsciences/benchmark-corpus-cosqa)
**Queries**: 500 (synsc-context), 497 (Nia — 3 timeouts)

| Metric | synsc-context | Nia | Delta |
|--------|:---:|:---:|:---:|
| **MRR** | **0.629** | 0.004 | +157x |
| **P@1** | **0.612** | 0.004 | +153x |
| **Recall@10** | **0.656** | 0.004 | +164x |
| **NDCG@10** | **0.634** | 0.004 | +159x |
| Avg latency | 4,036ms | 7,686ms | 1.9x faster |
| P95 latency | 5,152ms | 14,041ms | 2.7x faster |

**Analysis**: synsc-context achieves strong retrieval on this keyword-style query dataset. Nia's near-zero scores stem from its universal-search returning results from all indexed sources (documentation, other repos) instead of the target benchmark repository — the `repositories` filter parameter appears to have no effect.

### CodeSearchNet

**Source**: Husain et al., 2019 — the de facto standard for code search evaluation.
**Corpus**: [KB-syntheticsciences/benchmark-corpus-codesearchnet](https://github.com/KB-syntheticsciences/benchmark-corpus-codesearchnet) (Python subset)
**Queries**: 497 (docstring → code matching)

| Metric | synsc-context |
|--------|:---:|
| **MRR** | **0.940** |
| **P@1** | **0.938** |
| **Recall@10** | **0.948** |
| **NDCG@10** | **0.942** |
| Avg latency | 4,205ms |
| P95 latency | 5,592ms |

**Analysis**: synsc-context achieves exceptional performance on CodeSearchNet — 94% of queries return the correct function as the top result. This benchmark uses natural language docstrings as queries, which align well with synsc-context's context-enriched embeddings.

*Nia was not evaluated on CodeSearchNet due to the repository scoping issue identified in CoSQA.*

---

## Part III — LLM-as-Judge

**The fairest benchmark.** Previous benchmarks had a structural bias: content-matching penalizes engines that transform text during indexing, and file-path matching requires proper repository scoping. The LLM-as-Judge approach eliminates both biases.

### Methodology

1. Send each query to each engine
2. Collect raw retrieved context (no filtering or transformation)
3. Feed (query, context) to **Claude Sonnet 4.6** as a blind judge
4. Judge scores three dimensions (0-3 each):
   - **Relevance**: Does the context contain code related to the query?
   - **Completeness**: Could you fully answer the query from this context?
   - **Specificity**: Is the context targeted (not generic boilerplate)?
5. Compare scores across engines; track wins/ties

### Results — CodeSearchNet (497 queries)

| Metric | synsc-context | Context7 |
|--------|:---:|:---:|
| **Avg Relevance (0-3)** | **2.861** | 0.612 |
| **Avg Completeness (0-3)** | **2.813** | 0.318 |
| **Avg Specificity (0-3)** | **2.382** | 0.330 |
| **Avg Total (0-3)** | **2.685** | 0.420 |
| Avg latency | 5,365ms | 2,988ms |
| Queries | 497 | 497 |

**Analysis**: synsc-context scores **6.4x higher** than Context7 on CodeSearchNet. This dataset uses docstrings as queries to find the corresponding function — synsc-context's chunk-level indexing with AST-extracted symbols is purpose-built for this task. Context7 returns documentation pages which rarely contain the specific function implementation the query asks for.

### Results — CoSQA (500 queries)

| Metric | synsc-context | Context7 |
|--------|:---:|:---:|
| **Avg Relevance (0-3)** | 1.522 | **1.762** |
| **Avg Completeness (0-3)** | 1.032 | **1.402** |
| **Avg Specificity (0-3)** | 0.988 | **1.330** |
| **Avg Total (0-3)** | 1.181 | **1.498** |
| Avg latency | 5,014ms | 2,946ms |
| Queries | 500 | 500 |

**Analysis**: Context7 edges ahead on CoSQA by **1.27x**. CoSQA queries are natural-language web search style ("how to sort a list in python", "check if file exists") — these are better served by documentation/tutorial content than raw code snippets. This is Context7's strength: it indexes documentation sites and returns human-readable explanations.

### Results — CoSQA v2.0 (50 queries, synsc vs Nia head-to-head)

*From v2.0 report — kept for reference.*

| Metric | synsc-context | Nia |
|--------|:---:|:---:|
| **Avg Total (0-3)** | **1.067** | 0.267 |
| **Head-to-Head Wins** | **39 (78%)** | 3 (6%) |
| Ties | 8 (16%) | 8 (16%) |

### Three-Engine Summary (Basic Judge — 3D, no debiasing)

| Dataset | synsc-context | Nia | Context7 | Best Engine |
|---------|:---:|:---:|:---:|:---:|
| **CodeSearchNet** | **2.685** | — | 0.420 | synsc (6.4x) |
| **CoSQA** | 1.181 | 0.267 | **1.498** | Context7 (1.27x vs synsc) |

**Key Insight**: Each engine has a clear strength:
- **synsc-context** dominates **code retrieval** (finding specific functions, classes, implementations)
- **Context7** wins on **documentation queries** (how-to, conceptual questions)
- **Nia** struggles with scoped retrieval in all cases

---

## Part IV — Enhanced Judge

**The most rigorous benchmark.** The enhanced judge improves on the basic LLM judge with position debiasing, a 4th scoring dimension (faithfulness), and self-consistency metrics.

### Methodology

1. Send each query to each engine, collect top-5 results
2. **Pass 1**: Feed (query, context) to Claude Sonnet 4.6 — score 4 dimensions (0-3 each)
3. **Pass 2 (debiasing)**: Reverse the order of context chunks, re-evaluate with the same judge
4. **Average** Pass 1 and Pass 2 scores for the final debiased score
5. Compute **context quality metrics** (no LLM needed — pure token analysis)
6. Track **judge self-consistency** across both passes

**Scoring dimensions (0-3 each):**

| Dimension | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| **Relevance** | No snippet relates | Tangentially related | Partially relevant | Directly addresses query |
| **Completeness** | Cannot answer | Partial with gaps | Mostly complete | Fully answerable |
| **Specificity** | Generic boilerplate | Relevant + noise | Mostly targeted | Precisely targeted |
| **Faithfulness** | Misleading/wrong info | Some inaccuracies | Mostly accurate | Fully accurate, no hallucination |

### Results — CodeSearchNet (497 queries, debiased)

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| **Avg Relevance (0-3)** | **1.980** | 0.626 | 3.2x |
| **Avg Completeness (0-3)** | **1.877** | 0.296 | 6.3x |
| **Avg Specificity (0-3)** | **1.360** | 0.402 | 3.4x |
| **Avg Faithfulness (0-3)** | **2.044** | 1.296 | 1.6x |
| **Avg Total (0-3)** | **1.815** | 0.655 | **2.8x** |
| Wins | **436** | 36 | 12:1 |
| Ties | 25 | 25 | — |

### Results — CoSQA (500 queries, debiased)

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| **Avg Relevance (0-3)** | 1.164 | **1.762** | 0.66x |
| **Avg Completeness (0-3)** | 0.552 | **1.352** | 0.41x |
| **Avg Specificity (0-3)** | 0.680 | **1.412** | 0.48x |
| **Avg Faithfulness (0-3)** | 1.596 | **2.130** | 0.75x |
| **Avg Total (0-3)** | 0.998 | **1.664** | **0.60x** |
| Wins | 109 | **341** | 1:3.1 |
| Ties | 50 | 50 | — |

### Debiased vs Non-Debiased Comparison

Position debiasing consistently lowers scores (averaging across orderings corrects for positional inflation):

| Dataset | Non-Debiased Total | Debiased Total | Drift |
|---------|:---:|:---:|:---:|
| CodeSearchNet (synsc) | 2.785 | 1.815 | -0.970 |
| CodeSearchNet (ctx7) | 0.760 | 0.655 | -0.105 |
| CoSQA (synsc) | 1.265 | 0.998 | -0.267 |
| CoSQA (ctx7) | 1.680 | 1.664 | -0.016 |

synsc-context's scores are more position-sensitive (larger drift) because code chunks have stronger ordering effects — the first chunk often contains the function signature, making it more valuable in position 1.

---

## Context Quality Metrics

RAGAS-inspired metrics computed without LLM calls — pure token-level analysis of retrieved context.

| Metric | Definition |
|--------|------------|
| **Context Precision** | Position-weighted relevance: chunks containing query terms score higher when ranked earlier |
| **Context Density** | Fraction of retrieved tokens that contain query-relevant terms |
| **Signal-to-Noise** | Ratio of useful content (code, identifiers) vs noise (whitespace, boilerplate) |
| **Chunk Diversity** | 1 − avg pairwise Jaccard similarity (higher = more diverse results) |

### CodeSearchNet

| Metric | synsc-context | Context7 | Winner |
|--------|:---:|:---:|:---:|
| Context Precision | **0.481** | 0.191 | synsc (2.5x) |
| Context Density | **0.101** | 0.058 | synsc (1.7x) |
| Signal-to-Noise | **0.595** | 0.389 | synsc (1.5x) |
| Chunk Diversity | **0.945** | 0.925 | synsc |

### CoSQA

| Metric | synsc-context | Context7 | Winner |
|--------|:---:|:---:|:---:|
| Context Precision | **0.491** | 0.235 | synsc (2.1x) |
| Context Density | 0.030 | **0.034** | Context7 |
| Signal-to-Noise | **0.643** | 0.517 | synsc (1.2x) |
| Chunk Diversity | 0.927 | **0.938** | Context7 |

**Analysis**: synsc-context consistently returns higher-precision, higher-signal context. Even on CoSQA where Context7 wins on judge scores, synsc has 2.1x better context precision — the issue isn't retrieval precision, it's that code chunks don't answer "how do I..." questions as well as documentation does.

---

## Judge Consistency Analysis

Position debiasing reveals how sensitive the LLM judge is to chunk ordering.

| Metric | CodeSearchNet | CoSQA | Interpretation |
|--------|:---:|:---:|:---|
| **Position Consistency** | 0.553 | 0.785 | Fraction of queries where the winner doesn't change when order is swapped |
| **Cohen's κ** | 0.290 (fair) | 0.537 (moderate) | Agreement between pass 1 and pass 2 beyond chance |
| **Avg Score Drift** | 1.089 | 0.382 | Mean absolute difference between pass 1 and pass 2 scores |
| **N Evaluated** | 994 | 1,000 | Total (query, engine) pairs evaluated across both passes |

**Interpretation**:
- CodeSearchNet has **lower consistency** (κ=0.290, drift=1.089). Code chunks are highly order-sensitive — the function signature in chunk 1 is disproportionately valuable, so swapping it to the end significantly impacts scores.
- CoSQA is **more stable** (κ=0.537, drift=0.382). Documentation-style content is less order-dependent since each chunk tends to be more self-contained.
- These consistency metrics are themselves a finding: **LLM judges of code retrieval should always use position debiasing**, as the ordering effect is substantial.

**Reference**: Shi et al. (2025) report Position Consistency of 0.82±0.14 for Claude-3.5-Sonnet on MT-Bench (general text). Our code-retrieval PC of 0.553 is lower, confirming that code chunks exhibit stronger positional effects than natural language.

---

## Context Enrichment Analysis

synsc-context implements two layers of context enrichment:

### 1. Pre-embedding enrichment (at index time)

Structural metadata prepended to chunks before embedding (inspired by [supermemoryai/code-chunk](https://github.com/supermemoryai/code-chunk)):

```
# File: src/auth/middleware.py
# Scope: AuthMiddleware > validate_token
# Defines: validate_token, decode_jwt
# Uses: import jwt, import datetime
# After: refresh_token, revoke_token
# Before: AuthMiddleware.__init__
```

### 2. Post-retrieval enrichment (v3.0 — at search time)

After retrieval, each result is enriched with:
- **Enclosing function/class signature** from the `symbols` table
- **Docstring** (first 3 lines, truncated at 200 chars)
- **Preceding context** (last 5 lines of the previous chunk)

This runs on at most `top_k` results (~10) using two batched SQL queries (~10-30ms on prod).

### Impact on Retrieval Quality

| Dataset | v2.0 (pre-embedding only) | v3.0 (+ post-retrieval) | Delta |
|---------|:---:|:---:|:---:|
| **CodeSearchNet Total** | 2.640 | **2.685** | +0.045 |
| **CoSQA Total** | 1.167 | **1.181** | +0.014 |

**Analysis**: Post-retrieval enrichment provides consistent improvement across both datasets. The biggest gain is in **specificity** (+0.082 on CodeSearchNet) — function signatures help the judge confirm that retrieved code is the right function, not just similar-looking code.

**Recommendation**: Keep both enrichment layers enabled. Pre-embedding enrichment improves recall (better embeddings), post-retrieval enrichment improves precision (better context for consumers).

---

## Latency Comparison

| Benchmark | synsc-context (avg) | Nia (avg) | Context7 (avg) | Fastest |
|-----------|:---:|:---:|:---:|:---:|
| Custom Retrieval | 3,713ms | 10,017ms | — | synsc (2.7x) |
| Multi-Hop | 3,575ms | 7,025ms | — | synsc (2.0x) |
| Code QA | 3,549ms | 7,137ms | — | synsc (2.0x) |
| Adversarial | 3,851ms | 7,233ms | — | synsc (1.9x) |
| CoSQA (Validated) | 4,036ms | 7,686ms | — | synsc (1.9x) |
| CodeSearchNet (Judge) | 5,365ms | — | 2,988ms | Context7 (1.8x) |
| CoSQA (Judge) | 5,014ms | — | 2,946ms | Context7 (1.7x) |

**Note**: synsc-context benchmarks were run locally (localhost:8742) with ~1.1s network latency per Supabase round-trip. Production latency on Render is significantly lower due to co-location with Supabase. Context7 and Nia are always remote.

---

## Methodology & Fairness

### Fairness Evolution

| Issue | How It Was Addressed |
|-------|---------------------|
| Custom datasets may favor synsc-context | Added industry-standard CoSQA + CodeSearchNet |
| Direct-to-DB injection gave unfair chunk alignment | Re-indexed through normal pipeline (deleted + re-indexed from GitHub) |
| Content-matching penalizes engines that transform text | Added hybrid matching (content OR file path) |
| File-path matching requires proper repo scoping | Added LLM-as-Judge (fully engine-agnostic) |
| Nia's `repositories` filter doesn't work | LLM judge doesn't depend on scoping |
| Only two engines compared | Added Context7 as a third independent engine |
| Small LLM judge sample size (50 queries) | Scaled to full datasets (497-500 queries) |

### Dataset Summary

| Benchmark | Dataset | Source | Standard? | Size |
|-----------|---------|--------|:---------:|------|
| Retrieval | `retrieval_ground_truth.json` | Hand-crafted | No | 10 queries |
| Multi-Hop | `multihop_test_cases.json` | Hand-crafted | No | 10 queries |
| Code QA | `code_qa_test_cases.json` | Hand-crafted | No | 15 queries |
| Adversarial | `adversarial_test_cases.json` | Hand-crafted | No | 10 pairs |
| Hallucination | `hallucination_test_cases.json` | Hand-crafted (Nia CAB method) | Partial | 10 tasks |
| CoSQA | HuggingFace `CoIR-Retrieval/cosqa` | Huang et al., 2021 | **Yes** | 500 queries |
| CodeSearchNet | HuggingFace `code-search-net` | Husain et al., 2019 | **Yes** | 497 queries |
| LLM Judge | CoSQA + CodeSearchNet + Claude Sonnet 4.6 | Novel | Novel | 997 queries |

### Reproducibility

All benchmarks can be reproduced with:

```bash
# Custom benchmarks (requires FastAPI/Pydantic/httpx indexed)
uv run python -m benchmarks --engines synsc nia

# Industry-standard benchmarks
uv run python -m benchmarks --validated-only --engines synsc --dataset cosqa
uv run python -m benchmarks --validated-only --engines synsc --dataset codesearchnet

# LLM-as-Judge (requires BENCH_LLM_* env vars)
uv run python -m benchmarks --judge-only --engines synsc context7 --skip-indexing

# Enhanced Judge with position debiasing (v4.0)
uv run python -m benchmarks --enhanced-judge-only --engines synsc context7 --max-queries 500

# Quick test (no debiasing, 50 queries)
uv run python -m benchmarks --enhanced-judge-only --engines synsc context7 --max-queries 50 --no-debiasing
```

---

## Ecological Validity

A critical consideration often overlooked in retrieval benchmarks: **do these benchmarks reflect how AI agents actually use context engines?**

### CoSQA Queries Are Not Agent Queries

CoSQA contains queries like "how to remove duplicates from list python" and "python read csv file into dictionary". In practice, **an LLM would never invoke a context engine for these queries** — it already knows the answer from training data. Agents invoke context engines when they encounter **unfamiliar codebases** where the LLM lacks knowledge:

| Agent query (real-world) | CoSQA query (benchmark) |
|--------------------------|------------------------|
| "Find where rate limiting is enforced in this repo" | "python check file is readonly" |
| "What's the schema for chunk_embeddings table?" | "how to parse json from string in python" |
| "How does the SSE streaming handler work?" | "python convert datetime to unix timestamp" |

The left column requires retrieval. The right column doesn't.

### What This Means for Interpretation

- **CodeSearchNet** (docstring → function) is closer to real agent use: "find the function that does X" mirrors how agents navigate codebases
- **CoSQA scores should be weighted lower** when evaluating context engines for agent workflows
- Context7's CoSQA advantage (1.67x) is real but applies to a use case where retrieval is unnecessary
- synsc-context's CodeSearchNet advantage (2.8x) applies to the use case where retrieval is essential

### Recommended Future Benchmark

A higher-validity evaluation would use:
1. **Unfamiliar repos** the LLM hasn't seen in training (private, niche, or freshly created)
2. **Task-completion metrics** (can the LLM fix a bug / add a feature with retrieved context?)
3. **SWE-Bench-style evaluation** where context engine contribution is isolated (same LLM, different context sources)

---

## Strengths & Weaknesses

### synsc-context

| Strengths | Weaknesses |
|-----------|-----------|
| Near-perfect Code QA (1.000 accuracy) | Adversarial discrimination only 0.55 |
| Excellent CodeSearchNet MRR (0.940) | Latency ~5s locally (target <500ms for production) |
| 2.8x better than Context7 on code retrieval (debiased) | Weaker on documentation-style queries (CoSQA) |
| 0% true hallucination rate | Single worker (memory-bound due to embedding model) |
| 2.5x better context precision than Context7 | Judge position sensitivity on code (PC=0.553) |
| Post-retrieval enrichment improves specificity | |
| Scoped search — results from the right repo | |

### Context7

| Strengths | Weaknesses |
|-----------|-----------|
| Strong on documentation queries (CoSQA 1.664 debiased) | Poor on code retrieval (CodeSearchNet 0.655 debiased) |
| Fast response times (~3s) | Returns docs, not code implementations |
| No indexing required (pre-crawled) | Cannot search private or un-crawled repos |
| Good for "how do I..." questions | 2.8x worse than synsc on code-to-code search |
| More position-stable (lower score drift) | CoSQA advantage applies to low-validity queries |

### Nia

| Strengths | Weaknesses |
|-----------|-----------|
| Global knowledge search (docs + code + papers) | Cannot scope search to a specific repository |
| Built-in reranking pipeline | MRR 0.004 on CoSQA (157x worse than synsc-context) |
| Package search (3,000+ packages, no indexing needed) | Zero adversarial capability |
| Web search and deep research endpoints | `repositories` filter has no effect on results |

### Architectural Differences

| Aspect | synsc-context | Context7 | Nia |
|--------|---------------|----------|-----|
| **Design goal** | Scoped code retrieval within a repo | Documentation retrieval | Universal knowledge search |
| **Search scope** | Specified repository only | Pre-crawled docs/libs | All indexed sources |
| **Indexing** | Per-repo with AST extraction + enrichment | Pre-crawled, no user indexing | Global with doc crawling |
| **Best use case** | "Find this function in my codebase" | "How does X work?" | "Find knowledge across the ecosystem" |
| **Embedding model** | Gemini (code) + sentence-transformers (papers) | Unknown | Unknown (proprietary) |

---

## Limitations & Overfitting Risk

These results should be interpreted with the following caveats:

1. **Custom datasets are not independent of the engine.** Hand-crafted queries were written by the team that built synsc-context. There's inherent bias toward queries the engine handles well.

2. **Small sample sizes in custom benchmarks.** 10-15 queries per benchmark is enough to spot large differences but not statistically robust. A 90% P@1 on 10 queries has a ±19% confidence interval (95% CI).

3. **3 repos don't represent all codebases.** FastAPI, Pydantic, and httpx are well-structured Python libraries. Performance may degrade on messy codebases with poor documentation, mixed languages, or generated code.

4. **Nia's API may not be optimally configured.** The `repositories` filter not working could be a bug, a missing parameter, or an API version issue. Nia's team may achieve better results with their MCP tools or a different endpoint configuration.

5. **Context7 was not tested on custom benchmarks.** The comparison is limited to LLM judge on industry-standard datasets. Custom benchmarks require indexing specific repos, which Context7's API doesn't support.

6. **Latency measured locally for synsc-context.** Production latency on Render will be significantly lower (co-located with Supabase). Only Context7 and Nia latencies reflect real-world network conditions.

7. **LLM judge may have biases.** Claude Sonnet 4.6 may prefer certain response formats. However, the judge sees raw retrieved context with no engine identification, minimizing this risk.

8. **CoSQA weakness is structural.** synsc-context indexes code, not documentation. CoSQA queries ("how to check if a file exists in python") are better answered by docs than code. This is a design choice, not a bug.

---

## Conclusions & Next Steps

### Key Takeaways

1. **synsc-context is the strongest engine for scoped code search** — winning CodeSearchNet by 2.8x (debiased), 436 wins vs 36, and all custom benchmarks.

2. **Context7 wins on documentation-style queries** — but these queries have low ecological validity for agent workflows. An LLM already knows "how to parse JSON in Python" and would never invoke a context engine for it.

3. **The engines solve different problems.** synsc-context excels at "find code in this repo" (the actual agent use case), Context7 at "how does X work" (a use case where retrieval is often unnecessary), and Nia at "find knowledge across the ecosystem."

4. **Position debiasing matters for code retrieval evaluation.** Code chunks are highly order-sensitive (PC=0.553 on CodeSearchNet vs 0.82 baseline for general text). Any LLM-as-judge evaluation of code retrieval should use position debiasing.

5. **Context quality metrics tell a nuanced story.** synsc-context has 2.5x better context precision even on CoSQA where it loses on judge scores — the issue isn't retrieval quality but content type mismatch (code vs docs).

6. **Post-retrieval enrichment provides measurable improvement** — +0.045 on CodeSearchNet total score, with specificity seeing the biggest gain (+0.082).

7. **Adversarial robustness remains the #1 code retrieval improvement area** — 0.55 discrimination is near-random.

### Recommended Next Steps

| Priority | Action | Expected Impact |
|:---:|--------|-----------------|
| 1 | **SWE-Bench-style task-completion evaluation** | Measure actual agent improvement — the only metric that matters |
| 2 | Add cross-encoder reranker (e.g., `ms-marco-MiniLM`) | Adversarial discrimination 0.55 → 0.80+ |
| 3 | HNSW index + embedding cache | Latency 5s → <500ms |
| 4 | Benchmark on real user queries from production logs | Organic quality validation |
| 5 | Test on diverse repos (monorepos, multi-language, poorly documented) | Generalization assessment |
| 6 | Evaluate code-specific embedding models (CodeSage, StarEncoder) | Potential quality improvement |
| 7 | Multi-judge evaluation (Claude + GPT-4 + Gemini) | Cross-validate judge reliability |

---

*Report generated by the synsc-context benchmarking harness. Raw results available in `benchmarks/results/`.*
