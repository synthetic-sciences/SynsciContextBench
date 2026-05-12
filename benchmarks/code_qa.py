"""Code-specific QA benchmark.

Tests chunking quality + symbol extraction with precise code questions:
- "Where is function X defined?"
- "Where is function X called with argument Y?"
- "What class inherits from BaseClass?"
- "What does the return type of method Z look like?"

These queries stress-test:
1. Symbol extraction accuracy
2. Chunk boundary preservation (don't split a function in half)
3. Cross-reference resolution (call sites vs definitions)
4. Language-aware search (not just string matching)
"""

from __future__ import annotations

import asyncio
import json
import re

from tqdm import tqdm
from dataclasses import dataclass, field

from .adapters.base import ContextEngineAdapter, SearchResult


@dataclass
class CodeQATestCase:
    """A single code-specific QA test case."""

    id: str
    query: str
    description: str
    qa_type: str  # definition | call_site | inheritance | return_type | import | argument_usage
    language: str  # python | javascript | typescript | rust | go

    # The target we're looking for
    target_symbol: str  # e.g. "create_app", "BaseModel", "AsyncClient"
    target_file: str = ""  # expected file path (partial match)

    # For call_site queries: which argument pattern to find
    argument_pattern: str = ""  # e.g. "timeout=", "verify=False"

    # Ground truth: what MUST appear in a correct result
    must_contain: list[str] = field(default_factory=list)  # substrings in content
    must_not_contain: list[str] = field(default_factory=list)  # should NOT appear (adversarial)

    # Expected result properties
    expected_chunk_type: str = ""  # "code" | "import" | "comment"
    expected_line_range: tuple[int, int] | None = None  # (start, end) approximate


@dataclass
class CodeQAResult:
    """Result of evaluating a single code QA test case."""

    test_case_id: str
    query: str
    engine: str
    qa_type: str
    latency_ms: float

    # Did we find what we were looking for?
    found: bool = False
    found_at_rank: int | None = None  # 1-indexed rank of first correct result

    # Quality checks
    symbol_found: bool = False        # target symbol appears in results
    file_match: bool = False          # correct file found
    content_complete: bool = False    # all must_contain present in one chunk
    chunk_coherent: bool = False      # chunk contains full function/class (not split mid-definition)
    no_false_positives: bool = True   # must_not_contain items absent

    # Detailed
    matching_results: int = 0
    total_results: int = 0


@dataclass
class CodeQAAggregateMetrics:
    """Aggregate code QA metrics."""

    engine: str
    num_queries: int = 0

    # Overall accuracy
    accuracy: float = 0.0            # fraction of queries where answer was found
    avg_rank: float = 0.0            # average rank of correct answer (lower=better)
    mrr: float = 0.0                 # mean reciprocal rank

    # Quality breakdown
    symbol_accuracy: float = 0.0     # fraction where target symbol found
    file_accuracy: float = 0.0       # fraction where correct file found
    completeness: float = 0.0        # fraction where full content present in one chunk
    coherence: float = 0.0           # fraction where chunk wasn't split mid-definition
    false_positive_rate: float = 0.0 # fraction with must_not_contain violations

    # Per qa_type breakdown
    by_type: dict[str, dict] = field(default_factory=dict)

    avg_latency_ms: float = 0.0


def _check_chunk_coherence(content: str, language: str) -> bool:
    """Check if a code chunk is coherent (not split mid-definition).

    Heuristic: balanced braces/indentation, starts with a definition keyword.
    """
    lines = content.strip().split("\n")
    if not lines:
        return False

    if language in ("python",):
        # Check: does it start with def/class and has complete indentation?
        first_line = lines[0].strip()
        starts_def = first_line.startswith(("def ", "class ", "async def "))
        if starts_def:
            # Check the last line returns to base indentation or is a return/pass
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            last_meaningful = ""
            for line in reversed(lines):
                if line.strip():
                    last_meaningful = line
                    break
            if last_meaningful:
                last_indent = len(last_meaningful) - len(last_meaningful.lstrip())
                # Coherent if last line is at or near base indent
                return last_indent <= base_indent + 4
        return True  # non-definition chunks are considered coherent

    elif language in ("javascript", "typescript", "java", "go", "rust"):
        # Check brace balance
        open_braces = content.count("{")
        close_braces = content.count("}")
        return abs(open_braces - close_braces) <= 1

    return True  # default: assume coherent


def evaluate_code_qa(
    test_case: CodeQATestCase,
    results: list[SearchResult],
    latency_ms: float,
    engine_name: str,
) -> CodeQAResult:
    """Evaluate a single code QA test case against search results."""
    qr = CodeQAResult(
        test_case_id=test_case.id,
        query=test_case.query,
        engine=engine_name,
        qa_type=test_case.qa_type,
        latency_ms=latency_ms,
        total_results=len(results),
    )

    for i, r in enumerate(results):
        content = r.content
        content_lower = content.lower()

        # Check symbol presence
        symbol_in_content = test_case.target_symbol.lower() in content_lower
        # Check file match
        file_matches = test_case.target_file in r.file_path if test_case.target_file else True
        # Check must_contain
        all_must_contain = all(mc.lower() in content_lower for mc in test_case.must_contain)
        # Check must_not_contain
        has_false_positive = any(mnc.lower() in content_lower for mnc in test_case.must_not_contain)
        # Check argument pattern
        arg_matches = True
        if test_case.argument_pattern:
            arg_matches = test_case.argument_pattern.lower() in content_lower

        # A result is "correct" if symbol found + file matches + args match
        is_correct = symbol_in_content and file_matches and arg_matches

        if is_correct:
            qr.matching_results += 1

        if is_correct and not qr.found:
            qr.found = True
            qr.found_at_rank = i + 1
            qr.symbol_found = True
            qr.file_match = file_matches
            qr.content_complete = all_must_contain
            qr.chunk_coherent = _check_chunk_coherence(content, test_case.language)
            qr.no_false_positives = not has_false_positive

        # Track symbol found even if not full match
        if symbol_in_content and not qr.symbol_found:
            qr.symbol_found = True

        if file_matches and not qr.file_match:
            qr.file_match = True

        if has_false_positive:
            qr.no_false_positives = False

    return qr


CODE_QA_JUDGE_PROMPT = """\
You are an expert code retrieval evaluator. You will be given:
1. A code-specific question (e.g., "Where is function X defined?")
2. The target symbol and expected properties
3. The actual retrieved context from a search engine

Evaluate whether the retrieved context answers the question:
1. **found** (true/false): Does the context contain the answer to the question?
2. **symbol_found** (true/false): Is the target symbol present in the context?
3. **complete** (true/false): Does the context contain the full definition/usage (not truncated)?
4. **relevance** (0-3): 0=irrelevant, 1=tangential, 2=partially answers, 3=fully answers

Respond with ONLY a JSON object:
{"found": <true/false>, "symbol_found": <true/false>, "complete": <true/false>, "relevance": <0-3>}"""


async def _llm_judge_code_qa(
    test_case: CodeQATestCase,
    results: list[SearchResult],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[bool, bool, bool, int]:
    """Use LLM judge to evaluate code QA test case.

    Returns (found, symbol_found, complete, relevance).
    """
    from .llm_judge import _call_llm_judge_raw, _safe_parse_json

    context = "\n---\n".join(
        f"[Result {i+1}] {r.file_path}\n{r.content[:1500]}"
        for i, r in enumerate(results[:5])
    )

    must_contain_str = ", ".join(test_case.must_contain) if test_case.must_contain else "N/A"

    user_prompt = (
        f"## Question\n{test_case.query}\n\n"
        f"## Target\n"
        f"Symbol: {test_case.target_symbol}\n"
        f"File: {test_case.target_file or 'any'}\n"
        f"Type: {test_case.qa_type}\n"
        f"Language: {test_case.language}\n"
        f"Expected content: {must_contain_str}\n\n"
        f"## Retrieved Context\n```\n{context}\n```\n\n"
        f"Evaluate the retrieved context. Respond with ONLY JSON: "
        f'{{"found": <true/false>, "symbol_found": <true/false>, "complete": <true/false>, "relevance": <0-3>}}'
    )

    text = await _call_llm_judge_raw(
        system_prompt=CODE_QA_JUDGE_PROMPT,
        user_prompt=user_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    scores = _safe_parse_json(text, defaults={"found": False, "symbol_found": False, "complete": False, "relevance": 0})
    return (
        bool(scores.get("found", False)),
        bool(scores.get("symbol_found", False)),
        bool(scores.get("complete", False)),
        min(3, max(0, int(scores.get("relevance", 0)))),
    )


async def run_code_qa_benchmark(
    engine: ContextEngineAdapter,
    dataset_path: str,
    top_k: int = 10,
    max_queries: int | None = None,
    scoring_mode: str = "structural",
    llm_provider: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
    seed: int = 0,
) -> tuple[CodeQAAggregateMetrics, list[CodeQAResult]]:
    """Run code-specific QA benchmark against one engine.

    Args:
        scoring_mode: "structural" (file-path + keyword matching) or
                      "llm" (LLM judge evaluation, fair for doc-oriented engines).
        seed: RNG seed for query sub-sampling (deterministic across engines).
    """
    from .sampling import sample_seeded

    test_cases = load_code_qa_cases(dataset_path)
    test_cases = sample_seeded(test_cases, max_queries, seed=seed)
    results_list: list[CodeQAResult] = []

    for tc in tqdm(test_cases, desc=f"  {engine.name} code-qa", unit="q"):
        try:
            search_results, latency = await engine.search_code(
                query=tc.query,
                top_k=top_k,
                language=tc.language,
            )
        except Exception as e:
            print(f"  [!] Code QA query failed for {engine.name}: {tc.query[:50]}... — {e}")
            continue

        if scoring_mode == "llm" and llm_api_key:
            try:
                found, symbol_found, complete, relevance = await _llm_judge_code_qa(
                    tc, search_results, llm_provider, llm_model, llm_api_key
                )
                qr = CodeQAResult(
                    test_case_id=tc.id,
                    query=tc.query,
                    engine=engine.name,
                    qa_type=tc.qa_type,
                    latency_ms=latency,
                    total_results=len(search_results),
                    found=found,
                    found_at_rank=1 if found else None,
                    symbol_found=symbol_found,
                    file_match=found,  # LLM judge doesn't distinguish file vs content
                    content_complete=complete,
                    chunk_coherent=complete,
                    no_false_positives=True,  # LLM judge handles this implicitly
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"  [!] LLM judge failed for {engine.name}: {tc.query[:50]}... — {e}")
                qr = evaluate_code_qa(tc, search_results, latency, engine.name)
        else:
            qr = evaluate_code_qa(tc, search_results, latency, engine.name)

        results_list.append(qr)

    agg = aggregate_code_qa(results_list)
    return agg, results_list


def aggregate_code_qa(results: list[CodeQAResult]) -> CodeQAAggregateMetrics:
    """Aggregate code QA results."""
    if not results:
        return CodeQAAggregateMetrics(engine="unknown")

    engine = results[0].engine
    n = len(results)
    agg = CodeQAAggregateMetrics(engine=engine, num_queries=n)

    found_results = [r for r in results if r.found]
    agg.accuracy = len(found_results) / n
    agg.mrr = sum((1.0 / r.found_at_rank if r.found_at_rank else 0.0) for r in results) / n
    agg.avg_rank = (
        sum(r.found_at_rank for r in found_results) / len(found_results)
        if found_results else 0.0
    )

    agg.symbol_accuracy = sum(1 for r in results if r.symbol_found) / n
    agg.file_accuracy = sum(1 for r in results if r.file_match) / n
    agg.completeness = sum(1 for r in results if r.content_complete) / n
    agg.coherence = sum(1 for r in results if r.chunk_coherent) / n
    agg.false_positive_rate = sum(1 for r in results if not r.no_false_positives) / n
    agg.avg_latency_ms = sum(r.latency_ms for r in results) / n

    # Per qa_type breakdown
    types: dict[str, list[CodeQAResult]] = {}
    for r in results:
        types.setdefault(r.qa_type, []).append(r)

    for qa_type, type_results in types.items():
        tn = len(type_results)
        type_found = [r for r in type_results if r.found]
        agg.by_type[qa_type] = {
            "num_queries": tn,
            "accuracy": len(type_found) / tn,
            "mrr": sum((1.0 / r.found_at_rank if r.found_at_rank else 0.0) for r in type_results) / tn,
            "completeness": sum(1 for r in type_results if r.content_complete) / tn,
            "coherence": sum(1 for r in type_results if r.chunk_coherent) / tn,
        }

    return agg


def load_code_qa_cases(path: str) -> list[CodeQATestCase]:
    """Load code QA test cases from JSON."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", []):
        line_range = None
        if "expected_line_range" in item:
            lr = item["expected_line_range"]
            line_range = (lr[0], lr[1]) if lr else None
        cases.append(
            CodeQATestCase(
                id=item["id"],
                query=item["query"],
                description=item.get("description", ""),
                qa_type=item["qa_type"],
                language=item.get("language", "python"),
                target_symbol=item["target_symbol"],
                target_file=item.get("target_file", ""),
                argument_pattern=item.get("argument_pattern", ""),
                must_contain=item.get("must_contain", []),
                must_not_contain=item.get("must_not_contain", []),
                expected_chunk_type=item.get("expected_chunk_type", ""),
                expected_line_range=line_range,
            )
        )
    return cases
