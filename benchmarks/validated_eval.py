"""Evaluation runner for validated benchmark datasets.

Runs retrieval evaluation using CodeSearchNet and CoSQA datasets
with proper IR metrics (NDCG, MRR, Precision@K, Recall@K).

Key difference from the hand-crafted benchmarks:
- Ground truth comes from human annotations, not our keyword heuristics
- CodeSearchNet Challenge has graded relevance (0-3) from expert annotators
- CoSQA has binary labels from 3+ human annotators on real web queries

Evaluation protocol:
1. The corpus (code snippets) is pre-loaded from the downloaded dataset
2. For each query, we search the engine and check which returned results
   match the known relevant docs (by content similarity, file path, or both)
3. Metrics are computed against the ground-truth relevance labels

Match modes:
- content: Match by text similarity (Jaccard + SequenceMatcher)
- file: Match by file path in the benchmark repo
- hybrid: Match if either content or file path matches
- llm: LLM judge evaluates if retrieved results answer the query.
  Fairest for cross-engine comparison (engine-agnostic, format-agnostic).
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from .adapters.base import ContextEngineAdapter, SearchResult
from .metrics import (
    AggregateMetrics,
    QueryEvaluation,
    RetrievalResult,
    aggregate,
    evaluate_query,
)

MatchMode = Literal["content", "file", "hybrid", "llm"]


@dataclass
class ValidatedBenchmarkResult:
    """Result from running a validated dataset benchmark."""

    dataset_name: str
    engine: str
    num_queries: int = 0
    num_corpus: int = 0
    aggregate_metrics: dict = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)


def _content_similarity(a: str, b: str) -> float:
    """Fast content similarity between two code snippets.

    Uses a normalized token overlap + SequenceMatcher ratio for speed.
    """
    if not a or not b:
        return 0.0

    # Quick token overlap check first (fast reject)
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    if jaccard < 0.1:
        return 0.0  # fast reject

    # More precise ratio for candidates that pass token overlap
    # Use truncated strings for speed (first 500 chars)
    ratio = SequenceMatcher(None, a[:500], b[:500]).ratio()
    return ratio


def _build_corpus_file_map(corpus_list: list[dict]) -> dict[str, str]:
    """Build mapping from corpus doc ID to expected benchmark repo file path.

    Mirrors the file-naming logic in create_benchmark_repo.py so we can
    match search results by file path instead of content similarity.
    """
    file_map: dict[str, str] = {}
    for i, doc in enumerate(corpus_list):
        lang = doc.get("language", "python")
        ext = {"python": "py", "javascript": "js", "java": "java",
               "go": "go", "ruby": "rb", "php": "php"}.get(lang, "txt")

        content = doc.get("content", "")
        match = re.match(r"(?:def|class)\s+(\w+)", content.strip())
        name = match.group(1) if match else f"snippet_{doc['id']}"

        group = f"group_{i // 50:02d}"
        fname = f"{i:04d}_{name}.{ext}"
        # e.g. "python/group_00/0042_parse_datetime.py"
        file_map[doc["id"]] = f"{lang}/{group}/{fname}"

    return file_map


def _match_result_by_file(
    result: SearchResult,
    relevant_docs: list[dict],
) -> tuple[bool, int]:
    """Match a search result to relevant docs by file path.

    Checks if the result's file_path contains/ends with the expected
    benchmark repo file path for any relevant document.
    """
    if not result.file_path:
        return False, 0

    best_grade = 0
    result_path = result.file_path.replace("\\", "/")

    for doc in relevant_docs:
        expected = doc.get("_benchmark_file_path", "")
        if not expected:
            continue
        # Match if result path ends with or contains the expected path
        if result_path.endswith(expected) or expected in result_path:
            grade = doc.get("relevance", 1)
            best_grade = max(best_grade, grade)

    return best_grade > 0, best_grade


def _match_result_to_corpus(
    result: SearchResult,
    relevant_docs: list[dict],
    threshold: float = 0.5,
    match_mode: MatchMode = "hybrid",
) -> tuple[bool, int]:
    """Check if a search result matches any relevant document in the corpus.

    Args:
        match_mode: "content" (text similarity), "file" (file path),
                    or "hybrid" (either matches).

    Returns (is_relevant, relevance_grade).
    """
    content_match = False
    content_grade = 0
    file_match = False
    file_grade = 0

    if match_mode in ("content", "hybrid"):
        for doc in relevant_docs:
            sim = _content_similarity(result.content, doc["content"])
            if sim >= threshold:
                grade = doc.get("relevance", 1)
                if grade > content_grade:
                    content_grade = grade
                    content_match = True

    if match_mode in ("file", "hybrid"):
        file_match, file_grade = _match_result_by_file(result, relevant_docs)

    best_grade = max(content_grade, file_grade)
    is_relevant = content_match or file_match
    return is_relevant, best_grade


VALIDATED_JUDGE_PROMPT = """\
You are an expert code retrieval evaluator. You will be given:
1. A natural language query
2. The GROUND TRUTH code that answers the query
3. A retrieved result from a search engine

Does the retrieved result contain information that answers the query?
The result does NOT need to be an exact match — it could be documentation,
a different code snippet that achieves the same thing, or an explanation.

Score:
- 0: Completely irrelevant
- 1: Tangentially related but doesn't answer the query
- 2: Partially answers the query
- 3: Fully answers the query (even if format differs from ground truth)

Respond with ONLY a JSON object: {"relevance": <0-3>}"""


async def _llm_judge_match(
    query: str,
    ground_truth_content: str,
    result_content: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[bool, int]:
    """Use LLM judge to evaluate if a result matches the ground truth.

    Returns (is_relevant, relevance_grade).
    """
    from .llm_judge import _call_llm_judge_raw, _safe_parse_json

    user_prompt = (
        f"## Query\n{query}\n\n"
        f"## Ground Truth Code\n```\n{ground_truth_content[:2000]}\n```\n\n"
        f"## Retrieved Result\n```\n{result_content[:2000]}\n```\n\n"
        f"Does the retrieved result answer the query? "
        f'Respond with ONLY JSON: {{"relevance": <0-3>}}'
    )

    text = await _call_llm_judge_raw(
        system_prompt=VALIDATED_JUDGE_PROMPT,
        user_prompt=user_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    scores = _safe_parse_json(text, defaults={"relevance": 0})
    grade = min(3, max(0, int(scores.get("relevance", 0))))
    return grade >= 2, grade


async def run_validated_benchmark(
    engine: ContextEngineAdapter,
    dataset_path: str,
    k_values: list[int] | None = None,
    max_queries: int | None = None,
    repo_ids: list[str] | None = None,
    match_mode: MatchMode = "hybrid",
    llm_provider: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
) -> tuple[AggregateMetrics, list[QueryEvaluation]]:
    """Run retrieval evaluation using a validated dataset.

    Args:
        engine: The context engine adapter to evaluate
        dataset_path: Path to the downloaded dataset JSON
        k_values: Which K values to evaluate (default: [1, 3, 5, 10])
        max_queries: Limit queries for quick testing
        repo_ids: Scope search to specific repo IDs (e.g., benchmark corpus)
        match_mode: How to match results to corpus — "content" (text similarity),
                    "file" (benchmark repo file path), "hybrid" (either),
                    or "llm" (LLM judge, fairest for cross-engine comparison)
        llm_provider: Required for match_mode="llm"
        llm_model: Required for match_mode="llm"
        llm_api_key: Required for match_mode="llm"

    Returns:
        (aggregate_metrics, per_query_evaluations)
    """
    k_values = k_values or [1, 3, 5, 10]

    with open(dataset_path) as f:
        data = json.load(f)

    dataset_name = data.get("_description", Path(dataset_path).stem)
    queries = data["queries"]
    corpus_list = data["corpus"]
    corpus = {doc["id"]: doc for doc in corpus_list}
    qrels = data["qrels"]

    # Build file path mapping for file-level matching
    file_map: dict[str, str] = {}
    if match_mode in ("file", "hybrid"):
        file_map = _build_corpus_file_map(corpus_list)

    # Build query -> relevant docs mapping
    query_relevant: dict[str, list[dict]] = {}
    for qrel in qrels:
        qid = qrel["query_id"]
        doc_id = qrel["doc_id"]
        rel = qrel.get("relevance", 1)
        if rel > 0 and doc_id in corpus:
            doc_with_meta = {
                **corpus[doc_id],
                "relevance": rel,
                "_benchmark_file_path": file_map.get(doc_id, ""),
            }
            query_relevant.setdefault(qid, []).append(doc_with_meta)

    if max_queries:
        queries = queries[:max_queries]

    evaluations: list[QueryEvaluation] = []
    total = len(queries)

    for query_data in tqdm(queries, desc=f"  {engine.name} validated", unit="q"):
        qid = query_data["id"]
        query = query_data["query"]
        lang = query_data.get("language")
        relevant_docs = query_relevant.get(qid, [])
        total_relevant = len(relevant_docs)

        if total_relevant == 0:
            continue

        try:
            search_results, latency = await engine.search_code(
                query=query,
                top_k=max(k_values),
                repo_ids=repo_ids,
                language=lang,
            )
        except Exception as e:
            print(f"  [!] Query failed: {query[:50]}... — {type(e).__name__}: {e}")
            continue

        # Match results against ground truth
        retrieval_results = []

        if match_mode == "llm" and llm_api_key:
            # LLM judge: evaluate top result against ground truth
            # Only judge top-K results to limit LLM calls
            ground_truth_content = relevant_docs[0].get("content", "") if relevant_docs else ""
            for i, sr in enumerate(search_results):
                if i < 3:  # LLM judge top 3 only, rest assumed irrelevant
                    try:
                        is_rel, grade = await _llm_judge_match(
                            query, ground_truth_content, sr.content,
                            llm_provider, llm_model, llm_api_key,
                        )
                        await asyncio.sleep(0.3)  # rate limit
                    except Exception:
                        is_rel, grade = False, 0
                else:
                    is_rel, grade = False, 0

                retrieval_results.append(
                    RetrievalResult(
                        id=sr.id,
                        score=sr.score,
                        content=sr.content[:200],
                        is_relevant=is_rel,
                        relevance_grade=grade,
                    )
                )
        else:
            for sr in search_results:
                is_rel, grade = _match_result_to_corpus(
                    sr, relevant_docs, match_mode=match_mode,
                )
                retrieval_results.append(
                    RetrievalResult(
                        id=sr.id,
                        score=sr.score,
                        content=sr.content[:200],
                        is_relevant=is_rel,
                        relevance_grade=grade,
                    )
                )

        qe = QueryEvaluation(
            query=query,
            engine=engine.name,
            results=retrieval_results,
            latency_ms=latency,
        )
        evaluate_query(qe, k_values, total_relevant)
        evaluations.append(qe)

    print()  # newline after in-place counter
    agg = aggregate(evaluations, k_values)
    return agg, evaluations


def print_validated_summary(
    dataset_name: str,
    agg: AggregateMetrics,
    match_mode: MatchMode = "hybrid",
) -> None:
    """Pretty-print validated benchmark results."""
    print(f"  Dataset:           {dataset_name}")
    print(f"  Match mode:        {match_mode}")
    print(f"  Queries evaluated: {agg.num_queries}")
    print(f"  MRR:               {agg.avg_mrr:.3f}")
    for k, v in sorted(agg.avg_precision_at.items()):
        print(f"  Precision@{k}:      {v:.3f}")
    for k, v in sorted(agg.avg_recall_at.items()):
        print(f"  Recall@{k}:         {v:.3f}")
    for k, v in sorted(agg.avg_ndcg_at.items()):
        print(f"  NDCG@{k}:           {v:.3f}")
    print(f"  Avg latency:       {agg.avg_latency_ms:.0f}ms")
    print(f"  P95 latency:       {agg.p95_latency_ms:.0f}ms")
