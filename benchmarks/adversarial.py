"""Adversarial near-miss benchmark.

Tests whether the engine can distinguish between similar-but-wrong chunks
and the actually correct chunk. This is the hardest retrieval test because
naive embedding similarity will rank near-misses highly.

Categories of adversarial near-misses:
1. **Same name, wrong context**: Function `connect()` in DB module vs HTTP module
2. **Same file, wrong function**: Correct file but returns the wrong function
3. **Similar signature, different behavior**: `validate(data)` vs `validate(schema)`
4. **Version confusion**: v1 API mixed in with v2 API results
5. **Test vs production**: Test mock/fixture returned instead of real implementation
6. **Comment vs code**: Docstring/comment mentioning the concept vs actual implementation

Scoring modes:
- **structural** (default): File-path + keyword matching. Fast, no LLM cost.
  Fair for engines that return source code. Disadvantages doc-oriented engines.
- **llm**: LLM judge evaluates whether the correct result was returned and
  whether decoys were avoided. Fair across all engine types.
"""

from __future__ import annotations

import asyncio
import json

from tqdm import tqdm
from dataclasses import dataclass, field

from .adapters.base import ContextEngineAdapter, SearchResult


@dataclass
class AdversarialTestCase:
    """A test case with known correct and known wrong (near-miss) answers."""

    id: str
    query: str
    description: str
    adversarial_type: str  # same_name | same_file | similar_sig | version_confusion | test_vs_prod | comment_vs_code

    # The correct answer
    correct_file: str  # file path (partial match)
    correct_keywords: list[str]  # must appear in the correct chunk

    # Known near-misses (decoys) — if these appear ABOVE the correct answer, it's a failure
    decoys: list[DecoyDefinition] = field(default_factory=list)


@dataclass
class DecoyDefinition:
    """A known near-miss that should rank BELOW the correct answer."""

    id: str
    description: str
    # How to identify this decoy in results
    file_pattern: str = ""  # partial file path match
    keywords: list[str] = field(default_factory=list)  # content keywords


@dataclass
class AdversarialResult:
    """Result of evaluating a single adversarial test case."""

    test_case_id: str
    query: str
    engine: str
    adversarial_type: str
    latency_ms: float

    correct_found: bool = False
    correct_rank: int | None = None  # 1-indexed

    # Decoy analysis
    decoys_above_correct: int = 0   # how many decoys ranked above the correct answer
    decoy_ranks: dict[str, int | None] = field(default_factory=dict)  # decoy_id -> rank

    # Derived scores
    discrimination_score: float = 0.0  # 1.0 = perfect (correct above all decoys), 0.0 = all decoys above correct
    total_results: int = 0


@dataclass
class AdversarialAggregateMetrics:
    """Aggregate adversarial benchmark metrics."""

    engine: str
    num_queries: int = 0

    # Core metrics
    accuracy: float = 0.0             # fraction where correct answer found
    avg_discrimination: float = 0.0    # avg discrimination score
    avg_correct_rank: float = 0.0      # avg rank of correct answer
    decoy_confusion_rate: float = 0.0  # fraction where ANY decoy outranked correct answer

    # Per adversarial_type breakdown
    by_type: dict[str, dict] = field(default_factory=dict)

    avg_latency_ms: float = 0.0


def _identify_result(
    result: SearchResult,
    file_pattern: str,
    keywords: list[str],
) -> bool:
    """Check if a result matches a file pattern + keyword set."""
    file_match = file_pattern in result.file_path if file_pattern else True
    keyword_match = any(
        kw.lower() in result.content.lower() for kw in keywords
    ) if keywords else True
    return file_match and keyword_match


def evaluate_adversarial(
    test_case: AdversarialTestCase,
    results: list[SearchResult],
    latency_ms: float,
    engine_name: str,
) -> AdversarialResult:
    """Evaluate a single adversarial test case."""
    ar = AdversarialResult(
        test_case_id=test_case.id,
        query=test_case.query,
        engine=engine_name,
        adversarial_type=test_case.adversarial_type,
        latency_ms=latency_ms,
        total_results=len(results),
    )

    # Find the correct answer's rank
    for i, r in enumerate(results):
        if _identify_result(r, test_case.correct_file, test_case.correct_keywords):
            ar.correct_found = True
            ar.correct_rank = i + 1
            break

    # Find each decoy's rank
    for decoy in test_case.decoys:
        for i, r in enumerate(results):
            if _identify_result(r, decoy.file_pattern, decoy.keywords):
                ar.decoy_ranks[decoy.id] = i + 1
                break
        else:
            ar.decoy_ranks[decoy.id] = None

    # Count decoys that outranked the correct answer
    if ar.correct_rank is not None:
        for decoy_id, decoy_rank in ar.decoy_ranks.items():
            if decoy_rank is not None and decoy_rank < ar.correct_rank:
                ar.decoys_above_correct += 1

    # Discrimination score:
    # 1.0 if correct answer is above ALL decoys
    # 0.0 if all decoys are above correct answer
    # Scales linearly in between
    num_decoys = len(test_case.decoys)
    if num_decoys > 0 and ar.correct_found:
        ar.discrimination_score = 1.0 - (ar.decoys_above_correct / num_decoys)
    elif ar.correct_found:
        ar.discrimination_score = 1.0
    else:
        ar.discrimination_score = 0.0

    return ar


ADVERSARIAL_JUDGE_PROMPT = """\
You are an expert code retrieval evaluator. You will be given:
1. A query describing what code the user is looking for
2. A description of the CORRECT answer (what should be returned)
3. Descriptions of DECOY answers (similar but wrong results that should NOT be returned)
4. The actual retrieved context from a search engine

Evaluate:
1. **correct_found** (true/false): Does the retrieved context contain the correct answer?
2. **discrimination** (0.0-1.0): How well did the engine avoid returning decoys instead of the correct answer? 1.0 = no decoys present or correct is clearly above decoys. 0.0 = only decoys returned.

Respond with ONLY a JSON object:
{"correct_found": <true/false>, "discrimination": <0.0-1.0>}"""


async def _llm_judge_adversarial(
    test_case: AdversarialTestCase,
    results: list[SearchResult],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[bool, float]:
    """Use LLM judge to evaluate adversarial test case."""
    from .llm_judge import _call_llm_judge_raw, _safe_parse_json

    context = "\n---\n".join(
        f"[Result {i+1}] {r.file_path}\n{r.content[:1500]}"
        for i, r in enumerate(results[:5])
    )

    decoy_desc = "\n".join(
        f"- DECOY: {d.description} (file: {d.file_pattern}, keywords: {', '.join(d.keywords)})"
        for d in test_case.decoys
    )

    user_prompt = (
        f"## Query\n{test_case.query}\n\n"
        f"## Correct Answer\n"
        f"File: {test_case.correct_file}\n"
        f"Keywords: {', '.join(test_case.correct_keywords)}\n"
        f"Description: {test_case.description}\n\n"
        f"## Decoys (wrong answers to avoid)\n{decoy_desc}\n\n"
        f"## Retrieved Context\n```\n{context}\n```\n\n"
        f"Evaluate the retrieved context. Respond with ONLY JSON: "
        f'{{"correct_found": <true/false>, "discrimination": <0.0-1.0>}}'
    )

    text = await _call_llm_judge_raw(
        system_prompt=ADVERSARIAL_JUDGE_PROMPT,
        user_prompt=user_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    scores = _safe_parse_json(text, defaults={"correct_found": False, "discrimination": 0.0})
    return bool(scores.get("correct_found", False)), float(scores.get("discrimination", 0.0))


async def run_adversarial_benchmark(
    engine: ContextEngineAdapter,
    dataset_path: str,
    top_k: int = 10,
    max_queries: int | None = None,
    scoring_mode: str = "structural",
    llm_provider: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
    seed: int = 0,
) -> tuple[AdversarialAggregateMetrics, list[AdversarialResult]]:
    """Run adversarial near-miss benchmark against one engine.

    Args:
        scoring_mode: "structural" (file-path + keyword matching) or
                      "llm" (LLM judge evaluation, fair for doc-oriented engines).
        seed: RNG seed for query sub-sampling (deterministic across engines).
    """
    from .sampling import sample_seeded

    test_cases = load_adversarial_cases(dataset_path)
    test_cases = sample_seeded(test_cases, max_queries, seed=seed)
    results_list: list[AdversarialResult] = []

    for tc in tqdm(test_cases, desc=f"  {engine.name} adversarial", unit="q"):
        try:
            search_results, latency = await engine.search_code(
                query=tc.query, top_k=top_k
            )
        except Exception as e:
            print(f"  [!] Adversarial query failed for {engine.name}: {tc.query[:50]}... — {e}")
            continue

        if scoring_mode == "llm" and llm_api_key:
            try:
                correct_found, discrimination = await _llm_judge_adversarial(
                    tc, search_results, llm_provider, llm_model, llm_api_key
                )
                ar = AdversarialResult(
                    test_case_id=tc.id,
                    query=tc.query,
                    engine=engine.name,
                    adversarial_type=tc.adversarial_type,
                    latency_ms=latency,
                    total_results=len(search_results),
                    correct_found=correct_found,
                    correct_rank=1 if correct_found else None,
                    discrimination_score=discrimination,
                )
                await asyncio.sleep(0.5)  # rate limit for LLM calls
            except Exception as e:
                print(f"  [!] LLM judge failed for {engine.name}: {tc.query[:50]}... — {e}")
                ar = evaluate_adversarial(tc, search_results, latency, engine.name)
        else:
            ar = evaluate_adversarial(tc, search_results, latency, engine.name)

        results_list.append(ar)

    agg = aggregate_adversarial(results_list)
    return agg, results_list


def aggregate_adversarial(results: list[AdversarialResult]) -> AdversarialAggregateMetrics:
    """Aggregate adversarial results."""
    if not results:
        return AdversarialAggregateMetrics(engine="unknown")

    engine = results[0].engine
    n = len(results)
    agg = AdversarialAggregateMetrics(engine=engine, num_queries=n)

    found = [r for r in results if r.correct_found]
    agg.accuracy = len(found) / n
    agg.avg_discrimination = sum(r.discrimination_score for r in results) / n
    agg.avg_correct_rank = (
        sum(r.correct_rank for r in found) / len(found) if found else 0.0
    )
    agg.decoy_confusion_rate = sum(
        1 for r in results if r.decoys_above_correct > 0
    ) / n
    agg.avg_latency_ms = sum(r.latency_ms for r in results) / n

    # Per-type breakdown
    types: dict[str, list[AdversarialResult]] = {}
    for r in results:
        types.setdefault(r.adversarial_type, []).append(r)

    for adv_type, type_results in types.items():
        tn = len(type_results)
        type_found = [r for r in type_results if r.correct_found]
        agg.by_type[adv_type] = {
            "num_queries": tn,
            "accuracy": len(type_found) / tn,
            "avg_discrimination": sum(r.discrimination_score for r in type_results) / tn,
            "confusion_rate": sum(1 for r in type_results if r.decoys_above_correct > 0) / tn,
        }

    return agg


def load_adversarial_cases(path: str) -> list[AdversarialTestCase]:
    """Load adversarial test cases from JSON."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", []):
        decoys = [
            DecoyDefinition(
                id=d["id"],
                description=d.get("description", ""),
                file_pattern=d.get("file_pattern", ""),
                keywords=d.get("keywords", []),
            )
            for d in item.get("decoys", [])
        ]
        cases.append(
            AdversarialTestCase(
                id=item["id"],
                query=item["query"],
                description=item.get("description", ""),
                adversarial_type=item["adversarial_type"],
                correct_file=item["correct_file"],
                correct_keywords=item.get("correct_keywords", []),
                decoys=decoys,
            )
        )
    return cases
