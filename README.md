# synsc-context-bench

Benchmark harness for head-to-head comparison of code context engines. Tests [synsc-context](https://github.com/InkVell/synsc-context), [Nia](https://trynia.ai), and [Context7](https://context7.com) across 8 benchmark suites using both automated IR metrics and LLM-as-judge evaluation.

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in API keys
cp benchmarks/.env.local.example benchmarks/.env.local
```

### Required environment variables

At minimum, configure one engine and (for judge/hallucination benchmarks) one LLM provider:

| Variable | Purpose |
|----------|---------|
| `SYNSC_API_URL` | synsc-context server URL (default: `http://localhost:8742`) |
| `SYNSC_API_KEY` | synsc-context API key |
| `NIA_API_KEY` | Nia API key |
| `CONTEXT7_ENABLED` | Set `true` to include Context7 (no API key needed for basic usage) |
| `BENCH_LLM_PROVIDER` | LLM provider for judge benchmarks (`anthropic`, `gemini`, or `openai`) |
| `BENCH_LLM_MODEL` | Model ID for judge benchmarks |
| `BENCH_LLM_API_KEY` | API key for the judge LLM |

## Usage

```bash
# Run all benchmarks (all configured engines)
uv run python -m benchmarks

# Run a specific benchmark suite
uv run python -m benchmarks --judge-only --engines synsc context7
uv run python -m benchmarks --retrieval-only --skip-indexing
uv run python -m benchmarks --hallucination-only
uv run python -m benchmarks --enhanced-judge-only

# Limit queries for quick iteration
uv run python -m benchmarks --judge-only --engines synsc --max-queries 50

# Download industry-standard datasets from HuggingFace
uv run python -m benchmarks --download-datasets

# Multi-model hallucination (test across all configured LLM tiers)
uv run python -m benchmarks --multi-model

# Index benchmark corpus directly into synsc-context DB
uv run python -m benchmarks.benchmark_indexer --dataset cosqa
```

### CLI flags

| Flag | Effect |
|------|--------|
| `--engines synsc nia context7` | Select which engines to test |
| `--skip-indexing` | Skip the repo indexing step (repos already indexed) |
| `--match-mode hybrid\|file\|content` | How to match results to ground truth corpus |
| `--no-debiasing` | Disable position debiasing in enhanced judge (2x faster) |
| `--no-significance` | Skip statistical significance analysis |
| `--bootstrap-n N` | Number of bootstrap resamples (default: 10,000) |
| `--significance-alpha F` | Significance level (default: 0.05) |
| `--dataset cosqa\|codesearchnet` | Run only a specific validated dataset |
| `--multi-model` | Hallucination benchmark across all configured model tiers |

Each benchmark suite has both a `--*-only` flag (run only that suite) and a `--skip-*` flag (skip that suite).

## Benchmark Suites

### 1. Retrieval Quality

Standard IR metrics computed against hand-crafted ground truth (10 queries with known relevant files/keywords).

**Metrics**: Precision@K, Recall@K, NDCG@K, MRR, MAP, Success@K, R-Precision.

### 2. Multi-Hop Retrieval

Queries that require combining information from 2+ files or repos. Tests whether the engine surfaces all required pieces of evidence, not just the most obvious one.

**Metrics**: Hop Coverage, Hop Recall@K, Hop MRR, Answer Completeness.

### 3. Code QA

Precise code questions that stress-test chunking quality and symbol extraction:
- "Where is function X defined?"
- "What class inherits from BaseClass?"
- "Where is function X called with argument Y?"

**Metrics**: Exact Match, Symbol Found, File Match, partial scoring by QA type.

### 4. Adversarial Near-Miss

The hardest retrieval test. Each query has known correct answers and known decoys (semantically similar but functionally different code). Categories:
- Same name, wrong context (`connect()` in DB vs HTTP module)
- Same file, wrong function
- Similar signature, different behavior
- Version confusion (v1 mixed with v2)
- Test/mock vs production code
- Comment/docstring vs actual implementation

**Metrics**: Discrimination Rate, Decoy Rank, Correct-Above-Decoy Rate.

### 5. Hallucination Rate

Measures how well engine context prevents LLM hallucinations. For known SDK/library APIs:
1. Query each engine for context about a task
2. Feed context to an LLM and ask it to generate code
3. Check generated code for invented methods, wrong parameters, deprecated APIs

**Metrics**: Hallucination Rate, Invented Method Count, Wrong Parameter Count, Deprecated API Usage.

### 6. Validated Datasets

Industry-standard benchmarks downloaded from HuggingFace:

| Dataset | Source | Description |
|---------|--------|-------------|
| **CoSQA** | Huang et al., 2021 | 20K human-annotated (web query, code) pairs. Binary relevance labels from 3+ annotators. |
| **CodeSearchNet** | Husain et al., 2019 | 2M (comment, code) pairs across 6 languages. The standard code retrieval benchmark. |
| **CodeSearchNet Challenge** | Husain et al., 2019 | Curated queries with expert graded relevance (0-3). Gold standard for code retrieval. |

**Metrics**: Same IR metrics as Retrieval Quality, computed against human-annotated ground truth.

### 7. LLM-as-Judge

Blind quality scoring — an LLM evaluates the *usefulness* of retrieved context without knowing which engine produced it. This is engine-format-agnostic: it doesn't penalize engines that transform or reformat text.

**Scoring dimensions**: Relevance (0-3), Completeness (0-3), Specificity (0-3).

### 8. Enhanced LLM-as-Judge

Research-grade evaluation with debiasing and consistency guarantees:

- **Position debiasing** (Zheng et al. 2023): Each query evaluated twice with swapped engine order, scores averaged. Eliminates the documented ~10% positional bias in LLM evaluations.
- **4-dimensional scoring**: Relevance, Completeness, Specificity, Faithfulness — each 0-3 with calibration anchors in the system prompt.
- **Judge consistency metrics**: Cohen's kappa, Krippendorff's alpha, Position Consistency (Shi et al. 2025).
- **RAGAS-inspired context quality**: Context Precision (position-weighted relevance), Context Density (useful tokens / total tokens), Noise Ratio (Es et al. 2024).

## Statistical Analysis

All pairwise engine comparisons include:
- **Paired t-tests** (Sakai 2006, recommended by Urbano et al. 2019)
- **Wilcoxon signed-rank tests** (non-parametric alternative)
- **Bootstrap confidence intervals** (10,000 resamples by default)
- **Effect sizes**: Cohen's d and Cliff's delta
- **Bonferroni correction** for multiple comparisons

## Semantic Metrics

Beyond exact-match IR metrics, the harness computes code-aware similarity:
- **Weighted CodeBLEU components** (Ren et al. 2020) — token-level n-gram overlap with code keyword weighting
- **Soft token overlap** — Jaccard with normalization, code-aware matching
- **AST-aware similarity** — identifier overlap and structural keyword matching
- **Success@K** and **MAP** — standard BEIR metrics

## Engine Adapters

Each engine implements the `ContextEngineAdapter` interface (`benchmarks/adapters/base.py`):

| Engine | Adapter | How it works |
|--------|---------|-------------|
| **synsc-context** | `benchmarks/adapters/synsc.py` | HTTP API. Requires a running server and indexed repos. |
| **Nia** | `benchmarks/adapters/nia.py` | REST API. Global knowledge search. |
| **Context7** | `benchmarks/adapters/context7.py` | HTTP API over pre-crawled docs. No indexing step. |

To add a new engine, implement `ContextEngineAdapter` and register it in `benchmarks/__main__.py`.

## Ground Truth Datasets

### Hand-crafted (`benchmarks/datasets/`)

| File | Description | Size |
|------|-------------|------|
| `retrieval_ground_truth.json` | Queries with known relevant files/keywords | 10 queries |
| `multihop_test_cases.json` | Queries requiring cross-file context | 10 queries |
| `code_qa_test_cases.json` | Code-specific QA (definitions, call sites, etc.) | 15 queries |
| `adversarial_test_cases.json` | Semantically similar but functionally different pairs | 10 pairs |
| `hallucination_test_cases.json` | Code generation tasks with known API surfaces | 10 tasks |

### Industry-standard (via `--download-datasets`)

Downloaded from HuggingFace and converted to internal format. Requires `pip install datasets`.

## Multi-Model Matrix

The `--multi-model` flag runs hallucination benchmarks across multiple LLMs to test whether context quality is model-dependent. Configure tiers per provider in `.env.local`:

```
BENCH_GEMINI_API_KEY=...
BENCH_GEMINI_LOW_MODEL=gemini-2.5-flash-lite
BENCH_GEMINI_MID_MODEL=gemini-2.5-flash
BENCH_GEMINI_HIGH_MODEL=gemini-2.5-pro

BENCH_ANTHROPIC_API_KEY=...
BENCH_ANTHROPIC_LOW_MODEL=claude-haiku-4-5-20251001
BENCH_ANTHROPIC_MID_MODEL=claude-sonnet-4-6
BENCH_ANTHROPIC_HIGH_MODEL=claude-opus-4-6

BENCH_OPENAI_API_KEY=...
BENCH_OPENAI_LOW_MODEL=gpt-5-nano-2025-08-07
BENCH_OPENAI_MID_MODEL=gpt-5-2025-08-07
BENCH_OPENAI_HIGH_MODEL=o3
```

Only providers with a set API key are included.

## Adding Test Cases

### Retrieval queries

Add entries to `benchmarks/datasets/retrieval_ground_truth.json`:
```json
{
  "id": "q011",
  "query": "Your search query",
  "category": "code_search",
  "difficulty": "easy|medium|hard",
  "relevant_files": ["path/to/expected/file.py"],
  "relevant_keywords": ["expected_function", "expected_class"],
  "total_relevant": 3
}
```

### Hallucination test cases

Add entries to `benchmarks/datasets/hallucination_test_cases.json`:
```json
{
  "id": "h011",
  "query": "Task description for code generation",
  "library": "library-name",
  "valid_methods": ["real_method_1", "real_method_2"],
  "valid_parameters": {"method_name": ["param1", "param2"]},
  "deprecated_apis": ["old_method", "removed_function"]
}
```

## Project Structure

```
benchmarks/
├── __main__.py              # CLI entry point
├── runner.py                # Orchestrates all benchmark suites
├── config.py                # Environment-based configuration
├── metrics.py               # NDCG, MRR, Precision@K, Recall@K, MAP, R-Precision
├── semantic_metrics.py      # CodeBLEU, soft token overlap, AST-aware similarity
├── statistical_analysis.py  # Paired tests, bootstrap CIs, effect sizes
├── llm_judge.py             # LLM-as-Judge (3-dimension blind scoring)
├── enhanced_judge.py        # Position-debiased judge + RAGAS metrics
├── validated_eval.py        # CoSQA / CodeSearchNet evaluation
├── hallucination.py         # Hallucination rate benchmark
├── multihop.py              # Multi-hop retrieval benchmark
├── code_qa.py               # Code-specific QA benchmark
├── adversarial.py           # Adversarial near-miss benchmark
├── dataset_loader.py        # HuggingFace dataset downloader
├── benchmark_indexer.py     # Direct-to-DB corpus indexer (requires synsc-context)
├── create_benchmark_repo.py # Create GitHub benchmark corpus repos
├── adapters/
│   ├── base.py              # Abstract adapter interface
│   ├── synsc.py             # synsc-context HTTP adapter
│   ├── nia.py               # Nia REST API adapter
│   └── context7.py          # Context7 adapter
├── datasets/                # Ground truth + downloaded datasets
└── results/                 # Auto-generated benchmark reports (JSON)
docs/
├── WHITEPAPER.md            # Technical whitepaper
├── BENCHMARK_REPORT.md      # Full analysis write-up
├── RESULTS.md               # Tabulated results
└── evaluation-methodology-survey.md  # Literature survey
scripts/
└── md_to_pdf.py             # Convert markdown docs to styled PDFs
```

## References

- Husain et al. (2019). "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search." arXiv:1909.09436.
- Huang et al. (2021). "CoSQA: 20,000+ Web Queries for Code Search and Question Answering." ACL 2021.
- Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."
- Shi et al. (2025). "Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges."
- Es et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation."
- Ren et al. (2020). "CodeBLEU: A Method for Automatic Evaluation of Code Synthesis."
- Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS.
- Sakai (2006). "Evaluating Evaluation Metrics based on the Bootstrap." SIGIR.
- Urbano, Marrero & Martin (2019). "On the Measurement of Test Collection Reliability." SIGIR.
