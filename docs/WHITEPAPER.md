# Synsc Context: Structure-Aware Code Retrieval for AI Agents

**A Technical Whitepaper**

*Synthetic Sciences Engineering Team*
*March 2026*

---

## Abstract

We present **synsc-context**, a code retrieval engine designed to provide AI agents with precise, scoped context from software repositories. Unlike documentation-oriented engines that surface human-readable explanations, synsc-context indexes source code at the chunk level with AST-extracted symbol metadata, producing embeddings that capture both semantic meaning and structural context. We introduce a two-layer enrichment pipeline — pre-embedding structural prefixes and post-retrieval symbol/docstring injection — that improves retrieval specificity without modifying stored content.

We evaluate synsc-context against two production context engines (Context7 and Nia) across 8 benchmark suites plus a position-debiased enhanced LLM judge, including industry-standard CodeSearchNet (Husain et al., 2019) and CoSQA (Huang et al., 2021) datasets. On code-to-code retrieval (CodeSearchNet), synsc-context achieves an MRR of 0.940 and outscores Context7 by 2.8x on position-debiased 4-dimensional LLM-as-judge evaluation (436 wins vs 36 across 497 queries). On natural-language documentation queries (CoSQA), Context7 leads by 1.67x — though we argue these queries have low ecological validity for agent workflows, as LLMs already know the answers from training data. We release the full benchmark harness, including position-debiasing, judge consistency metrics (Cohen's κ, Position Consistency), and RAGAS-inspired context quality analysis, for reproducibility.

---

## 1. Introduction

Large language models are increasingly used as software engineering assistants, but their effectiveness depends on the quality of context they receive. Two dominant approaches exist for providing code context to LLMs:

1. **Documentation retrieval**: Engines like Context7 index pre-crawled documentation sites, returning human-readable explanations. These excel at "how do I..." queries but cannot search private repositories or return specific implementations.

2. **Universal knowledge search**: Engines like Nia provide broad search across documentation, issues, and code, but cannot scope results to a specific repository.

3. **Scoped code retrieval**: synsc-context takes a third approach — indexing individual repositories at the chunk level with AST-extracted metadata, then performing scoped semantic search within a user's collection.

Each approach serves different needs. This paper describes synsc-context's architecture, the design decisions behind it, and a rigorous multi-engine evaluation.

### 1.1 Design Goals

- **Repository-scoped search**: Results come only from repositories the user has indexed, not from a global corpus
- **Chunk-level precision**: Return the specific function or class relevant to a query, not an entire file
- **Symbol awareness**: Leverage AST-extracted metadata (signatures, docstrings, scope chains) to improve embedding quality
- **Multi-tenant isolation**: Public repositories are deduplicated across users; private repositories are isolated
- **MCP integration**: Expose all capabilities as Model Context Protocol tools for direct agent consumption

---

## 2. System Architecture

synsc-context consists of four components: an indexing pipeline that processes GitHub repositories into searchable chunks, an embedding layer that produces vector representations, a retrieval pipeline that performs semantic search with post-processing, and an MCP/HTTP API layer that exposes these capabilities to AI agents.

### 2.1 Overview

```
GitHub Repository
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  INDEXING PIPELINE                                   │
│                                                      │
│  Clone → Filter → Chunk → AST Extract → Enrich →    │
│  Embed → Store                                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ GitClient │→│ Chunker  │→│ Context Enrichment│   │
│  │ (dulwich) │  │(tiktoken)│  │  (scope trees)   │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│        │              │               │              │
│        ▼              ▼               ▼              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │tree-sitter│  │  Gemini  │  │    Supabase      │   │
│  │  parsers  │  │ Embed API│  │ PostgreSQL+pgvec │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE                                  │
│                                                      │
│  Query Embed → pgvector ANN → Symbol Boost →        │
│  Metadata Score → Threshold → MMR → Enrich          │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  API LAYER                                           │
│                                                      │
│  MCP Server (stdio) ←→ HTTP Server (FastAPI)        │
│  31 tools: search, index, analyze, papers, datasets │
└─────────────────────────────────────────────────────┘
```

### 2.2 Data Model

The storage layer uses PostgreSQL with the pgvector extension. The schema is designed around five core entities:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `repositories` | Repository metadata | url, branch, commit_sha, is_public, indexed_by |
| `repository_files` | Per-file metadata | file_path, language, line_count, content_hash |
| `code_chunks` | Indexed code segments | content, start_line, end_line, chunk_type, symbol_names |
| `chunk_embeddings` | Vector representations | embedding (768-dim pgvector), chunk_id, repo_id |
| `symbols` | AST-extracted symbols | name, qualified_name, signature, docstring, symbol_type |
| `chunk_relationships` | Inter-chunk edges | source_chunk_id, target_chunk_id, relationship_type, weight |

**Deduplication**: Public repositories are indexed once and shared across all users via a `user_repositories` junction table. When a user adds a popular repository (e.g., React, FastAPI), the system checks if it's already indexed at the same commit SHA. If so, it adds a collection reference in ~100ms instead of re-indexing.

---

## 3. Indexing Pipeline

### 3.1 Repository Ingestion

Repositories are cloned via dulwich (pure-Python Git) with configurable branch selection. The system auto-detects the default branch via the GitHub API rather than assuming `main`.

**File filtering** applies three layers:

1. **Extension whitelist**: 50+ supported extensions across 30 languages, including documentation formats (.md, .mdx, .rst)
2. **Exclusion patterns**: node_modules, .git, __pycache__, minified/bundled files, lock files, media, binaries
3. **Fast mode** (default): Additionally skips test files, examples, and benchmark directories to reduce indexing time and noise

Files are read in parallel using a ThreadPoolExecutor, then processed in batches of 100 with periodic database flushes to minimize round-trips.

### 3.2 Chunking Strategy

Code is split into chunks using a token-based algorithm with AST-aware boundary selection:

**Parameters:**
- Max tokens per chunk: 2,048
- Overlap: 100 tokens
- Minimum chunk size: 50 tokens
- Tokenizer: `cl100k_base` (GPT-4 tokenizer via tiktoken)

**Algorithm:**

1. Extract symbol boundaries (function/class start lines) from the AST
2. Accumulate lines until reaching 75% of max tokens (soft limit)
3. Once past the soft limit, seek the next symbol boundary to split at
4. If no boundary found before reaching max tokens, hard-split at the limit
5. Carry overlap lines from the previous chunk into the next

This produces chunks that align with logical code units — a function or class method typically occupies one chunk, rather than being split at an arbitrary line.

**Documentation files** (.md, .mdx, .rst) use the same token-based chunker but skip AST parsing, set `chunk_type="documentation"`, and receive a documentation-specific embedding prefix.

### 3.3 Symbol Extraction

tree-sitter parsers extract structural metadata from source files:

| Field | Description |
|-------|-------------|
| `name` | Simple identifier (e.g., `validate_token`) |
| `qualified_name` | Fully qualified (e.g., `AuthMiddleware.validate_token`) |
| `symbol_type` | function, class, method, variable, constant, type, interface |
| `signature` | Full declaration line(s) |
| `docstring` | Associated documentation string |
| `parameters` | List of {name, type, default} |
| `return_type` | Return type annotation |
| `decorators` | Applied decorators |
| `start_line`, `end_line` | Source location (1-indexed) |

Symbols are extracted even in turbo mode (which skips AST-based chunking for speed), ensuring that context enrichment always has access to structural metadata.

### 3.4 Pre-Embedding Context Enrichment

Before generating embeddings, each chunk receives a structural context prefix. This prefix is not stored in the database — only the enriched text is sent to the embedding model.

**Prefix format:**
```
# auth/middleware.py
# Scope: AuthMiddleware > validate_token
# Defines: validate_token(self, token: str) -> bool
# Uses: jwt, datetime, hashlib
# After: refresh_token, revoke_token
# Before: AuthMiddleware.__init__
```

The prefix is constructed by:

1. **Scope tree construction**: Building a hierarchical tree from extracted symbols using line-range containment. If symbol A's range fully contains B's, B is a child of A.
2. **Scope chain lookup**: Finding the innermost-to-root scope chain for the chunk's start line.
3. **Sibling discovery**: Listing up to 3 symbols before and after the chunk's primary symbol in its parent scope.
4. **Import extraction**: Regex-based extraction of imported names from the full file (capped at 10).

This approach is inspired by [supermemoryai/code-chunk](https://github.com/supermemoryai/code-chunk) and ensures the embedding captures structural context (what scope this code lives in, what it defines, what it uses) alongside the raw code.

**Documentation enrichment** uses a simpler prefix:
```
# README.md
# Type: Documentation
# Section: Getting Started
```

### 3.5 Embedding Generation

| Content Type | Model | Dimensions | Batch Size | Task Type |
|-------------|-------|:----------:|:----------:|-----------|
| Code chunks | Gemini `gemini-embedding-001` | 768 | 100 | `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` |
| Research papers | `all-mpnet-base-v2` (local) | 768 | 32 | Sentence embedding |

All embeddings are L2-normalized for cosine similarity. The Gemini API uses dual task types — `RETRIEVAL_DOCUMENT` for indexing and `RETRIEVAL_QUERY` for search — which is critical for asymmetric retrieval where queries are natural language but documents are code.

Embeddings are stored in PostgreSQL via pgvector with batch inserts of 100 per transaction, using `ON CONFLICT DO UPDATE` for idempotent re-indexing.

### 3.6 Chunk Relationships

At index time, the system builds a directed graph of inter-chunk relationships:

| Relationship Type | Description | Weight |
|-------------------|-------------|:------:|
| `adjacent` | Consecutive chunks in the same file (chunk N → N+1) | 1.0 |
| `same_class` | Chunks overlapping a class symbol's line range (capped at 10 per class) | 1.0 |

These edges enable future graph-traversal search (e.g., "given this function, what's the next code block?") but are not yet used in the retrieval pipeline.

---

## 4. Retrieval Pipeline

Search queries pass through a 6-stage quality pipeline:

### 4.1 Vector Similarity Search

The query is embedded using Gemini with task type `RETRIEVAL_QUERY`, then matched against stored embeddings using pgvector's `<=>` (L2 distance) operator:

```sql
WITH user_repos AS MATERIALIZED (
    SELECT ur.repo_id FROM user_repositories ur
    WHERE ur.user_id = :user_id
)
SELECT cc.content, rf.file_path, cc.start_line, cc.end_line,
       cc.symbol_names, cc.chunk_index,
       1 - (ce.embedding <=> :query_vec::vector) AS similarity
FROM chunk_embeddings ce
INNER JOIN user_repos urp ON ce.repo_id = urp.repo_id
INNER JOIN repositories r ON ce.repo_id = r.repo_id
    AND (r.is_public = TRUE OR r.indexed_by = :user_id)
INNER JOIN code_chunks cc ON ce.chunk_id = cc.chunk_id
INNER JOIN repository_files rf ON cc.file_id = rf.file_id
ORDER BY ce.embedding <=> :query_vec::vector
LIMIT :top_k
```

The query over-fetches by `max(top_k × 3, 20)` to provide headroom for post-processing filters.

### 4.2 Symbol-Aware Score Boosting

Query text is parsed for code identifiers (camelCase, snake_case, PascalCase, dotted paths) using regex. Results containing matching symbols in their `symbol_names` JSON field receive a +0.15 score boost (capped at 1.0).

### 4.3 Metadata-Aware Scoring

Results from test, documentation, or example directories receive a -0.08 penalty. Chunks where >15% of lines contain assertion/mock patterns (assert, pytest, mock, describe, expect) receive a -0.04 penalty.

### 4.4 Dynamic Similarity Threshold

A minimum similarity floor filters low-quality results:

```
threshold = max(0.3, top_score × 0.6)
```

This adapts to query difficulty — high-confidence queries with a strong top result set a higher bar, while ambiguous queries with lower top scores remain permissive.

### 4.5 Maximal Marginal Relevance (MMR)

To prevent returning multiple chunks from the same function or file, MMR diversification balances relevance against redundancy:

```
score = λ × sim(candidate, query) − (1 − λ) × max_sim(candidate, selected)
```

Where λ = 0.7 (70% relevance, 30% diversity) and similarity is Jaccard on tokenized content.

### 4.6 Post-Retrieval Context Enrichment

After MMR, the final result set (typically ≤10 chunks) is enriched with structural context via two batched SQL queries:

1. **Symbol query**: Fetch all symbols for the result files, find the tightest enclosing symbol for each chunk, and prepend its signature and docstring
2. **Adjacent chunk query**: Fetch the preceding chunk for each result and append its last 5 lines as "preceding context"

**Enriched output format:**
```
# function: async def create_user(request: Request, db: Session) -> UserResponse:
# Docstring: Create a new user account with email verification.
# preceding context:
#     if not validate_email(email):
#         raise HTTPException(status_code=400, detail="Invalid email")

<actual code chunk>
```

This costs ~10-30ms on production (two indexed queries on ≤10 file IDs) and measurably improves LLM judge scores (Section 6.5).

### 4.7 Optional Cross-Encoder Reranking

An optional reranking stage uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB, ~10ms/pair) with blended scoring:

```
final_score = 0.4 × cross_encoder_score + 0.6 × vector_similarity
```

This is disabled by default — general-purpose cross-encoders can hurt code-specific retrieval. A code-trained cross-encoder is planned.

---

## 5. Benchmark Methodology

### 5.1 Engines Under Test

| Engine | Architecture | Scope | Indexing |
|--------|-------------|-------|----------|
| **synsc-context** | Chunk-level embeddings + AST metadata | User's indexed repos | Per-repo, with deduplication |
| **Context7** | Pre-crawled documentation | Popular libraries only | None (pre-indexed) |
| **Nia** | Universal knowledge search | Global corpus | Automatic crawling |

### 5.2 Benchmark Suite

We developed an 8-benchmark evaluation harness with progressive fairness guarantees:

| Phase | Benchmark | Ground Truth | Queries | Fairness |
|:-----:|-----------|-------------|:-------:|:--------:|
| 1 | Retrieval Quality | Hand-crafted file/keyword match | 10 | Medium |
| 1 | Multi-Hop Retrieval | Hand-crafted per-hop evidence | 10 | Medium |
| 1 | Code QA | Hand-crafted symbol match | 15 | Medium |
| 1 | Adversarial Near-Miss | Hand-crafted correct/decoy pairs | 10 | Medium |
| 1 | Hallucination Rate | Known API surfaces | 10 | Medium |
| 2 | CoSQA (Validated) | Human-annotated web queries (Huang et al.) | 500 | High |
| 2 | CodeSearchNet (Validated) | Human-annotated docstring pairs (Husain et al.) | 497 | High |
| 3 | LLM-as-Judge | Blind Claude Sonnet 4.6 scoring | 997 | High |
| 4 | Enhanced Judge | Position-debiased 4D scoring + consistency metrics | 997 | **Highest** |

### 5.3 Metrics

**Information Retrieval Metrics:**
- **Precision@K**: Fraction of top-K results that are relevant
- **Recall@K**: Fraction of total relevant items found in top-K
- **MRR** (Mean Reciprocal Rank): 1/rank of first relevant result
- **NDCG@K** (Normalized Discounted Cumulative Gain): Position-weighted relevance with ideal normalization. Uses graded relevance (0-2) with log₂(i+2) discount

**LLM-as-Judge Scoring (Basic — 3D):**

Claude Sonnet 4.6 evaluates each (query, retrieved context) pair on three dimensions (0-3 each): Relevance, Completeness, Specificity. The judge sees raw retrieved context with no engine identification. Temperature is set to 0 for deterministic scoring.

**Enhanced Judge Scoring (4D, position-debiased):**

The enhanced judge adds a 4th dimension (**faithfulness**) and implements position debiasing (Zheng et al., 2023):

| Dimension | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| **Relevance** | No snippet relates | Tangentially related | Partially relevant | Directly addresses query |
| **Completeness** | Cannot answer | Partial with gaps | Mostly complete | Fully answerable |
| **Specificity** | Generic boilerplate | Relevant + noise | Mostly targeted | Precisely targeted |
| **Faithfulness** | Misleading/wrong info | Some inaccuracies | Mostly accurate | Fully accurate |

Each query is evaluated **twice** — once in the original chunk order, once reversed — and scores are averaged. This eliminates positional bias where the judge favors content presented first. The consistency between passes is measured via Cohen's κ and Position Consistency (Shi et al., 2025).

**Context Quality Metrics (RAGAS-inspired, no LLM):**

| Metric | Definition |
|--------|------------|
| Context Precision | Position-weighted relevance of query terms in ranked chunks |
| Context Density | Fraction of tokens containing query-relevant terms |
| Signal-to-Noise | Useful content (code, identifiers) vs noise (whitespace, boilerplate) |
| Chunk Diversity | 1 − avg pairwise Jaccard similarity across result chunks |

### 5.4 Fairness Evolution

| Bias Identified | Mitigation |
|----------------|------------|
| Custom datasets may favor synsc-context | Added industry-standard CoSQA + CodeSearchNet |
| Content-matching penalizes text transformation | Added hybrid matching (content OR file path) |
| File-path matching requires repo scoping | Added LLM-as-Judge (engine-agnostic) |
| Nia's repository filter not working | LLM judge doesn't depend on scoping |
| Small sample size (50 queries) | Scaled to full datasets (497-500 queries) |
| Two-engine comparison | Added Context7 as third independent engine |

---

## 6. Results

### 6.1 Custom Benchmarks (synsc-context vs Nia)

**Test corpus**: FastAPI, Pydantic, httpx (indexed on both engines).

| Benchmark | synsc-context | Nia | Delta |
|-----------|:---:|:---:|:---:|
| Retrieval MRR | **0.950** | 0.175 | +443% |
| P@1 | **0.900** | 0.100 | +800% |
| Multi-Hop Coverage | **0.967** | 0.867 | +12% |
| Code QA Accuracy | **1.000** | 0.333 | +200% |
| Adversarial Accuracy | **0.800** | 0.000 | — |
| Hallucination Rate | **0%** | N/A | — |

synsc-context returns the correct result in position 1 for 90% of queries (P@1 = 0.900) vs 10% for Nia. The advantage is most pronounced on definition lookups, import tracing, and return type queries — tasks that require chunk-level precision with symbol awareness.

### 6.2 Industry-Standard Benchmarks

**CoSQA** (500 queries, human-annotated web search queries for Python code):

| Metric | synsc-context | Nia |
|--------|:---:|:---:|
| MRR | **0.629** | 0.004 |
| NDCG@10 | **0.634** | 0.004 |

**CodeSearchNet** (497 queries, docstring-to-function matching):

| Metric | synsc-context |
|--------|:---:|
| MRR | **0.940** |
| P@1 | **0.938** |
| NDCG@10 | **0.942** |

The CodeSearchNet MRR of 0.940 is notable — CodeBERT (Feng et al., 2020) reports ~0.70 and UniXcoder (Guo et al., 2022) ~0.75 on the full CodeSearchNet benchmark. Our higher score likely reflects corpus size differences (single-repo vs full dataset), but demonstrates strong retrieval quality on the same query format.

### 6.3 LLM-as-Judge (synsc-context vs Context7)

The fairest comparison — Claude Sonnet 4.6 blindly scores retrieved context without knowing which engine produced it.

**CodeSearchNet (497 queries):**

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| Relevance (0-3) | **2.861** | 0.612 | 4.7x |
| Completeness (0-3) | **2.813** | 0.318 | 8.8x |
| Specificity (0-3) | **2.382** | 0.330 | 7.2x |
| **Total (0-3)** | **2.685** | 0.420 | **6.4x** |

**CoSQA (500 queries):**

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| Relevance (0-3) | 1.522 | **1.762** | 0.86x |
| Completeness (0-3) | 1.032 | **1.402** | 0.74x |
| Specificity (0-3) | 0.988 | **1.330** | 0.74x |
| **Total (0-3)** | 1.181 | **1.498** | **0.79x** |

### 6.4 Enhanced Judge — Position-Debiased 4D Scoring (497-500 queries)

The most rigorous evaluation. Each query is judged twice (original and reversed chunk order) and scores averaged.

**CodeSearchNet (497 queries, debiased):**

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| Relevance (0-3) | **1.980** | 0.626 | 3.2x |
| Completeness (0-3) | **1.877** | 0.296 | 6.3x |
| Specificity (0-3) | **1.360** | 0.402 | 3.4x |
| Faithfulness (0-3) | **2.044** | 1.296 | 1.6x |
| **Total (0-3)** | **1.815** | 0.655 | **2.8x** |
| Wins | **436** | 36 | 12:1 |

**CoSQA (500 queries, debiased):**

| Metric | synsc-context | Context7 | Ratio |
|--------|:---:|:---:|:---:|
| Relevance (0-3) | 1.164 | **1.762** | 0.66x |
| Completeness (0-3) | 0.552 | **1.352** | 0.41x |
| Specificity (0-3) | 0.680 | **1.412** | 0.48x |
| Faithfulness (0-3) | 1.596 | **2.130** | 0.75x |
| **Total (0-3)** | 0.998 | **1.664** | **0.60x** |
| Wins | 109 | **341** | 1:3.1 |

### 6.5 Context Quality Metrics

RAGAS-inspired metrics computed without LLM calls:

| Metric | CSN synsc | CSN ctx7 | CoSQA synsc | CoSQA ctx7 |
|--------|:---:|:---:|:---:|:---:|
| Context Precision | **0.481** | 0.191 | **0.491** | 0.235 |
| Context Density | **0.101** | 0.058 | 0.030 | **0.034** |
| Signal-to-Noise | **0.595** | 0.389 | **0.643** | 0.517 |
| Chunk Diversity | **0.945** | 0.925 | 0.927 | **0.938** |

synsc-context has 2.5x better context precision even on CoSQA — the retrieval is precise, but code chunks don't answer "how do I..." questions as well as documentation.

### 6.6 Judge Consistency

| Metric | CodeSearchNet | CoSQA |
|--------|:---:|:---:|
| Position Consistency | 0.553 | 0.785 |
| Cohen's κ | 0.290 (fair) | 0.537 (moderate) |
| Avg Score Drift | 1.089 | 0.382 |

Code retrieval evaluation is more position-sensitive than general text (Shi et al. 2025 report PC=0.82±0.14 for general MT-Bench; our code-retrieval PC is 0.553). This confirms that **position debiasing is essential for evaluating code retrieval systems**.

### 6.7 Interpretation

The results reveal a clear architectural divide:

- **synsc-context dominates code retrieval** (CodeSearchNet: 2.8x debiased, 436 wins). When queries describe a function and the goal is to find its implementation, chunk-level indexing with symbol metadata is the right tool.

- **Context7 wins on documentation queries** (CoSQA: 1.67x debiased, 341 wins). When queries are natural-language "how do I..." questions, pre-crawled documentation returns more useful answers than raw code.

- **Nia struggles with scoped retrieval**. Its universal search mixes results from all indexed sources. MRR of 0.004 on CoSQA (157x worse than synsc-context) reflects this.

However, the **ecological validity** of these benchmarks differs significantly:

| Use Case | Best Engine | Ecological Validity | Why |
|----------|------------|:---:|-----|
| "Find the validation function in my codebase" | synsc-context | **High** | LLM doesn't know your code |
| "How does FastAPI dependency injection work?" | Context7 | **Low** | LLM already knows this |
| "Find knowledge across the ecosystem" | Nia | Medium | Depends on obscurity |

CoSQA queries ("how to parse json in python") are questions an LLM would never invoke a context engine for — it already knows the answer. The real value of a context engine is answering questions about **unfamiliar codebases** where the LLM lacks training data. CodeSearchNet (docstring → function lookup) is much closer to this use case.

### 6.8 Context Enrichment Impact

Post-retrieval enrichment improved scores across both datasets:

| Dataset | Without Enrichment | With Enrichment | Delta |
|---------|:---:|:---:|:---:|
| CodeSearchNet Total | 2.640 | **2.685** | +0.045 |
| CoSQA Total | 1.167 | **1.181** | +0.014 |

The largest gain was in **specificity** (+0.082 on CodeSearchNet). Function signatures prepended to chunks help the judge confirm that the retrieved code is the right function, not just similar-looking code.

### 6.9 Latency

| Benchmark | synsc-context | Nia | Context7 |
|-----------|:---:|:---:|:---:|
| Custom Retrieval | 3,713ms | 10,017ms | — |
| CodeSearchNet (Judge) | 5,365ms | — | 2,988ms |
| CoSQA (Judge) | 5,014ms | — | 2,946ms |

synsc-context benchmarks were run locally with ~1.1s geographic latency to Supabase (US-East). Production latency on co-located infrastructure is significantly lower. Context7 and Nia measurements reflect real-world network conditions.

---

## 7. Limitations

1. **Ecological validity gap**: CoSQA queries ("how to parse json in python") don't reflect real agent usage — LLMs already know these answers. CodeSearchNet is closer to real use but still uses well-known open-source repos. A SWE-Bench-style task-completion evaluation on unfamiliar repos would be more ecologically valid.

2. **Custom dataset bias**: Hand-crafted benchmarks (Phase 1) were written by the team that built synsc-context. Industry-standard datasets (Phase 2-4) mitigate but don't eliminate this.

3. **Corpus size**: CodeSearchNet evaluation uses a single-repo corpus, not the full 2M+ function dataset. MRR comparisons with CodeBERT/UniXcoder are directional, not apples-to-apples.

4. **Three repos don't represent all codebases**: FastAPI, Pydantic, and httpx are well-structured Python libraries. Performance may degrade on monorepos, multi-language projects, or codebases with poor documentation.

5. **Adversarial robustness**: 0.55 discrimination score is near-random. The embedding model struggles to distinguish semantically similar but functionally different code (e.g., `json.dumps()` vs `json.loads()`).

6. **Judge position sensitivity on code**: Position Consistency of 0.553 on CodeSearchNet (vs 0.82 baseline for general text per Shi et al. 2025) indicates that LLM judges are less reliable when evaluating code retrieval. Debiasing helps but doesn't fully resolve this.

7. **Single LLM judge**: Claude Sonnet 4.6 only. Multi-judge evaluation (Claude + GPT-4 + Gemini) with inter-rater agreement (Krippendorff's α) would strengthen confidence in judge scores.

8. **Single embedding model**: Gemini `gemini-embedding-001` is a general-purpose model. Code-specific embedding models (CodeSage, StarEncoder) may improve quality.

---

## 8. Related Work

**Code search engines**: GitHub Code Search uses trigram indexing for exact matches; Sourcegraph uses keyword + regex search with SCIP-based code intelligence. synsc-context differs by using semantic embeddings for natural-language queries.

**Embedding models for code**: CodeBERT (Feng et al., 2020), UniXcoder (Guo et al., 2022), and StarCoder (Li et al., 2023) produce code-aware embeddings. synsc-context uses Gemini's general embedding model with structural prefixes to inject code awareness.

**Context for LLMs**: RAG systems like LangChain and LlamaIndex provide generic document retrieval. synsc-context specializes in code with AST-aware chunking, symbol extraction, and scope-tree enrichment.

**Benchmarks**: CodeSearchNet (Husain et al., 2019) is the de facto standard for code search evaluation. CoSQA (Huang et al., 2021) adds real web queries. Our LLM-as-Judge approach extends these with a format-agnostic evaluation method.

**MCP integration**: The Model Context Protocol (Anthropic, 2024) standardizes tool interfaces for AI agents. synsc-context exposes 31 MCP tools covering code search, paper indexing, dataset exploration, and repository analysis.

---

## 9. Future Work

| Priority | Direction | Expected Impact |
|:--------:|-----------|-----------------|
| 1 | **SWE-Bench-style task-completion evaluation** | Measure whether synsc actually helps agents solve real coding tasks on unfamiliar repos |
| 2 | Code-specific cross-encoder reranker | Adversarial discrimination 0.55 → 0.80+ |
| 3 | HNSW index + embedding cache | Latency 5s → <500ms |
| 4 | Multi-judge evaluation (Claude + GPT-4 + Gemini) | Cross-validate judge reliability via Krippendorff's α |
| 5 | Code-specific embedding model (CodeSage, StarEncoder) | Quality improvement on code-to-code retrieval |
| 6 | Graph-traversal search using chunk relationships | Multi-hop retrieval without multiple queries |
| 7 | Evaluation on diverse repos (monorepos, multi-language) | Generalization assessment |

---

## 10. Conclusion

synsc-context demonstrates that structure-aware code retrieval — combining AST-extracted metadata with semantic embeddings — significantly outperforms documentation-oriented and universal search engines on code-to-code retrieval tasks. The 2.8x advantage over Context7 on CodeSearchNet (position-debiased 4D LLM judge, 436 wins vs 36) validates the approach of indexing source code at the chunk level with structural context enrichment.

The evaluation also reveals important methodological insights. Position debiasing is essential for code retrieval evaluation — code chunks exhibit stronger positional effects (PC=0.553) than general text (PC=0.82). More fundamentally, standard retrieval benchmarks like CoSQA have low ecological validity for evaluating context engines: the queries they contain ("how to parse json in python") are questions an LLM would never need to retrieve context for. The real value of a context engine is answering questions about unfamiliar codebases — a use case not well-captured by existing benchmarks.

The full benchmark harness, including position-debiased evaluation, judge consistency analysis, RAGAS-inspired context quality metrics, and statistical significance testing, is available for reproducibility.

---

## References

1. Husain, H., Wu, H., Gazit, T., Allamanis, M., & Brockschmidt, M. (2019). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. *arXiv:1909.09436*.

2. Huang, J., Tang, D., Shou, L., Gong, M., Xu, K., Jiang, D., Zhou, M., & Duan, N. (2021). CoSQA: 20,000+ Web Queries for Code Search and Question Answering. *ACL 2021*.

3. Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., & Zhou, M. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. *EMNLP 2020*.

4. Guo, D., Lu, S., Duan, N., Wang, Y., Zhou, M., & Yin, J. (2022). UniXcoder: Unified Cross-Modal Pre-training for Code Representation. *ACL 2022*.

5. Li, R., et al. (2023). StarCoder: May the Source Be With You! *arXiv:2305.06161*.

6. Anthropic. (2024). Model Context Protocol Specification. *https://modelcontextprotocol.io*.

7. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks Track*. arXiv:2306.05685.

8. Shi, W., et al. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. *arXiv:2406.07791*.

9. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.

10. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021*. arXiv:2104.08663.

11. Urbano, J., Lima, H., & Hanjalic, A. (2019). Statistical Significance Testing in Information Retrieval. *SIGIR 2019*.

---

*Synthetic Sciences — context.syntheticsciences.ai*
