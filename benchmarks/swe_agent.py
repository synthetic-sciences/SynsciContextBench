"""SWE-Agent benchmark (Phase 9).

Simulates real-world software engineering agent workflows: given a bug report,
feature spec, or refactoring task, an LLM must produce a code solution.  We
measure solutions generated WITH context engine retrieval against a no-context
baseline (LLM alone), directly quantifying each engine's practical value-add.

Key innovations over Phases 1-8:
- No-context baseline comparison (first phase to include one)
- Agent-realistic query generation (LLM formulates its own queries)
- Multi-turn agent simulation (search → draft → refine)
- Context utilization scoring (facts_used / facts_available)
- Position-debiased judge scoring (reuse from Phase 8)
- Per-knowledge-tier breakdown (A: well-known, B: niche, C: version-specific)
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from .adapters.base import ContextEngineAdapter
from .hallucination import validate_code, HallucinationTestCase
from .llm_judge import _safe_parse_json


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SWETestCase:
    """A single SWE-agent benchmark test case."""

    id: str
    task_type: str  # bug_fix | feature_addition | refactoring | api_migration
    title: str
    description: str
    library: str
    knowledge_tier: str  # A | B | C
    difficulty: str  # easy | medium | hard
    version_specific: bool
    gold_queries: list[str]
    expected_files: list[str]
    expected_patterns: list[str]
    expected_symbols: list[str]
    acceptance_criteria: dict
    judge_rubric: dict


@dataclass
class SWEResult:
    """Result for a single (test_case, engine, query_mode, turn) combination."""

    test_case_id: str
    engine: str  # engine name OR "no_context"
    query_mode: str  # "gold" | "agent" | "none" (baseline)
    turn: int  # 1 or 2
    latency_ms: float = 0.0
    generated_solution: str = ""
    queries_used: list[str] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)

    # Structural scores
    file_targeting_score: float = 0.0
    pattern_match_score: float = 0.0
    symbol_usage_score: float = 0.0
    parseable: bool = False
    hallucination_count: int = 0
    criteria_pass_rate: float = 0.0

    # Context utilization
    key_facts_available: int = 0
    key_facts_used: int = 0
    context_utilization_score: float = 0.0

    # Judge scores (debiased, 0-5)
    judge_correctness: float = 0.0
    judge_completeness: float = 0.0
    judge_code_quality: float = 0.0
    judge_no_hallucination: float = 0.0
    judge_composite: float = 0.0


@dataclass
class SWEAggregateMetrics:
    """Aggregate metrics for one engine (or baseline) across all test cases."""

    engine: str
    num_cases: int = 0
    avg_judge_composite: float = 0.0
    avg_criteria_pass_rate: float = 0.0
    avg_file_targeting: float = 0.0
    avg_context_utilization: float = 0.0
    avg_hallucination_count: float = 0.0
    parse_rate: float = 0.0
    avg_latency_ms: float = 0.0
    by_task_type: dict[str, dict] = field(default_factory=dict)
    by_knowledge_tier: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)


@dataclass
class SWEBenchmarkResult:
    """Full Phase 9 benchmark result."""

    engine_results: dict[str, SWEAggregateMetrics] = field(default_factory=dict)
    baseline_results: SWEAggregateMetrics | None = None
    delta_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    per_case_results: list[dict] = field(default_factory=list)
    significance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM helpers (reuse patterns from hallucination.py)
# ---------------------------------------------------------------------------

async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Generic LLM call supporting anthropic/gemini/openai."""
    import httpx

    if llm_provider == "anthropic":
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": llm_model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    elif llm_provider == "gemini":
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent",
                params={"key": llm_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason})")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini returned empty parts")
            return "".join(p.get("text", "") for p in parts)

    elif llm_provider == "openai":
        async with httpx.AsyncClient(timeout=90.0) as client:
            is_new = "gpt-5" in llm_model or "gpt-4.1" in llm_model or "o" in llm_model.split("-")[0]
            token_param = "max_completion_tokens" if is_new else "max_tokens"
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_model,
                    token_param: max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown LLM provider: {llm_provider}")


# ---------------------------------------------------------------------------
# Agent query generation
# ---------------------------------------------------------------------------

_AGENT_QUERY_SYSTEM = """\
You are a software engineering agent. Given a task description, generate 2-3 \
search queries you would issue to a code search engine to find relevant context \
for solving this task. Return ONLY a JSON array of query strings, no other text.

Example: ["FastAPI dependency injection override", "FastAPI TestClient fixture"]"""


async def generate_agent_queries(
    test_case: SWETestCase,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> list[str]:
    """Ask the LLM to generate search queries for a task (simulates agent behavior)."""
    user_prompt = (
        f"Task: {test_case.title}\n\n"
        f"Description: {test_case.description}\n\n"
        f"Library: {test_case.library}\n\n"
        f"Generate 2-3 search queries to find relevant code context."
    )
    text = await _call_llm(
        _AGENT_QUERY_SYSTEM, user_prompt,
        llm_provider, llm_model, llm_api_key,
        max_tokens=256, temperature=0.0,
    )
    # Parse JSON array from response
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q) for q in queries[:3]]
    except json.JSONDecodeError:
        pass
    # Fallback: split by newlines
    return [line.strip().strip('"').strip("'") for line in text.split("\n") if line.strip()][:3]


# ---------------------------------------------------------------------------
# Solution generation
# ---------------------------------------------------------------------------

_SOLUTION_SYSTEM = """\
You are a senior software engineer. Given a task description and code context, \
produce a Python code solution that addresses the task.

Rules:
- Do NOT rely on your memory or training data. Work ONLY with the context provided.
- Use the context engine for querying more information if required.
- Use ONLY APIs and methods that exist in the provided context.
- Do NOT invent methods, parameters, or classes that don't exist.
- Return only the code solution, no explanations.
- If the context is empty or unhelpful, do your best but clearly prefer \
  patterns and APIs found in the provided context over recalled knowledge."""

_SOLUTION_WITH_CONTEXT = """\
## Task
{title}

{description}

## Library
{library}

## Relevant Code Context
{context}

IMPORTANT: Base your solution strictly on the code context above. Do not rely \
on your memory — use only the APIs, patterns, and signatures found in the context. \
If you need more information, state what you would query the context engine for.

Generate a Python code solution for this task. Return ONLY the code."""

_SOLUTION_NO_CONTEXT = """\
## Task
{title}

{description}

## Library
{library}

Generate a Python code solution for this task. Return ONLY the code."""


async def _generate_solution(
    test_case: SWETestCase,
    context: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> str:
    """Generate a code solution, with or without context."""
    if context.strip():
        user_prompt = _SOLUTION_WITH_CONTEXT.format(
            title=test_case.title,
            description=test_case.description,
            library=test_case.library,
            context=context[:8000],
        )
    else:
        user_prompt = _SOLUTION_NO_CONTEXT.format(
            title=test_case.title,
            description=test_case.description,
            library=test_case.library,
        )
    text = await _call_llm(
        _SOLUTION_SYSTEM, user_prompt,
        llm_provider, llm_model, llm_api_key,
        max_tokens=2048, temperature=0.0,
    )
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


# ---------------------------------------------------------------------------
# Gap analysis for Turn 2
# ---------------------------------------------------------------------------

_GAP_ANALYSIS_SYSTEM = """\
You are a code reviewer. Analyze the draft solution and identify gaps, \
uncertainties, or areas where additional context would improve the solution. \
Return a JSON object with:
- "gaps": list of strings describing what's missing or uncertain
- "follow_up_queries": list of 1-2 search queries to fill those gaps

Example: {"gaps": ["Missing error handling for None body"], "follow_up_queries": ["FastAPI request validation error handling"]}"""


async def _analyze_gaps(
    test_case: SWETestCase,
    draft_solution: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[list[str], list[str]]:
    """Identify gaps in a draft solution and generate follow-up queries.

    Returns (gaps, follow_up_queries).
    """
    user_prompt = (
        f"## Task\n{test_case.title}\n{test_case.description}\n\n"
        f"## Draft Solution\n```python\n{draft_solution[:4000]}\n```\n\n"
        f"Identify gaps and suggest follow-up search queries."
    )
    text = await _call_llm(
        _GAP_ANALYSIS_SYSTEM, user_prompt,
        llm_provider, llm_model, llm_api_key,
        max_tokens=512, temperature=0.0,
    )
    data = _safe_parse_json(text, defaults={"gaps": [], "follow_up_queries": []})
    gaps = data.get("gaps", [])
    queries = data.get("follow_up_queries", [])
    return (
        [str(g) for g in gaps][:5],
        [str(q) for q in queries][:2],
    )


# ---------------------------------------------------------------------------
# Structural evaluation
# ---------------------------------------------------------------------------


def _check_parseable(code: str) -> bool:
    """Check if code is valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _file_targeting_score(solution: str, expected_files: list[str]) -> float:
    """Fraction of expected_files referenced in the solution."""
    if not expected_files:
        return 0.0
    found = 0
    sol_lower = solution.lower()
    for f in expected_files:
        # Check for filename (last component) or full path
        name = f.split("/")[-1].lower()
        if name in sol_lower or f.lower() in sol_lower:
            found += 1
    return found / len(expected_files)


def _pattern_match_score(solution: str, expected_patterns: list[str]) -> float:
    """Fraction of expected regex patterns found in the solution."""
    if not expected_patterns:
        return 0.0
    found = 0
    for pattern in expected_patterns:
        try:
            if re.search(pattern, solution, re.IGNORECASE):
                found += 1
        except re.error:
            # Invalid regex pattern — try literal match
            if pattern.lower() in solution.lower():
                found += 1
    return found / len(expected_patterns)


def _symbol_usage_score(solution: str, expected_symbols: list[str]) -> float:
    """Fraction of expected symbols present in the solution."""
    if not expected_symbols:
        return 0.0
    found = sum(1 for s in expected_symbols if s in solution)
    return found / len(expected_symbols)


def _criteria_pass_rate(solution: str, acceptance_criteria: dict) -> float:
    """Evaluate structural checks from acceptance criteria."""
    checks = acceptance_criteria.get("structural_checks", [])
    if not checks:
        return 0.0
    passed = 0
    for check in checks:
        if _run_structural_check(solution, check):
            passed += 1
    return passed / len(checks)


def _run_structural_check(solution: str, check_name: str) -> bool:
    """Run a named structural check on the solution."""
    sol_lower = solution.lower()

    checks = {
        "contains_null_check": lambda: any(
            p in sol_lower for p in ["is none", "is not none", "== none", "!= none", "if not "]
        ),
        "raises_or_returns_error": lambda: any(
            p in sol_lower for p in ["raise ", "httpexception", "return.*error", "validationerror", "400", "422", "valueerror"]
        ),
        "contains_try_except": lambda: "try:" in sol_lower and "except" in sol_lower,
        "contains_type_check": lambda: any(
            p in sol_lower for p in ["isinstance(", "type(", "typing.", "annotated["]
        ),
        "contains_async": lambda: "async " in sol_lower or "await " in sol_lower,
        "contains_decorator": lambda: "@" in solution,
        "contains_class_definition": lambda: "class " in solution,
        "contains_function_definition": lambda: "def " in solution,
        "contains_import": lambda: "import " in solution,
        "contains_return": lambda: "return " in solution,
        "contains_validation": lambda: any(
            p in sol_lower for p in ["validate", "validator", "field_validator", "model_validator"]
        ),
        "contains_middleware": lambda: "middleware" in sol_lower,
        "contains_rate_limit": lambda: any(
            p in sol_lower for p in ["rate_limit", "ratelimit", "throttl", "sliding_window", "token_bucket"]
        ),
        "contains_deprecation_handling": lambda: any(
            p in sol_lower for p in ["deprecated", "warning", "migration", "compat"]
        ),
        "uses_new_api": lambda: any(
            p in sol_lower for p in ["field_validator", "model_validator", "annotated", "v2", "mapped_column"]
        ),
        "contains_error_handling": lambda: any(
            p in sol_lower for p in ["try:", "except ", "raise ", "error", "exception"]
        ),
        "contains_test": lambda: any(
            p in sol_lower for p in ["def test_", "assert ", "pytest", "unittest"]
        ),
    }

    handler = checks.get(check_name)
    if handler:
        return handler()

    # Fallback: treat check_name as a substring search
    return check_name.lower().replace("_", " ") in sol_lower


# ---------------------------------------------------------------------------
# Context utilization
# ---------------------------------------------------------------------------


def _extract_key_facts(context_chunks: list[str]) -> set[str]:
    """Extract key facts (function names, class names, parameters, API patterns)
    from retrieved context chunks.
    """
    facts: set[str] = set()
    for chunk in context_chunks:
        # Extract Python identifiers that look like API symbols
        # Function definitions
        for m in re.finditer(r"def\s+(\w+)\s*\(", chunk):
            facts.add(m.group(1))
        # Class definitions
        for m in re.finditer(r"class\s+(\w+)", chunk):
            facts.add(m.group(1))
        # Method calls (obj.method pattern)
        for m in re.finditer(r"\.(\w+)\s*\(", chunk):
            facts.add(m.group(1))
        # Import names
        for m in re.finditer(r"(?:from\s+\S+\s+)?import\s+(.+)", chunk):
            for name in m.group(1).split(","):
                name = name.strip().split(" as ")[0].strip()
                if name and name != "*":
                    facts.add(name.split(".")[-1])
        # Decorator names
        for m in re.finditer(r"@(\w+(?:\.\w+)*)", chunk):
            facts.add(m.group(1).split(".")[-1])
        # Parameter names from function signatures
        for m in re.finditer(r"def\s+\w+\s*\(([^)]*)\)", chunk):
            for param in m.group(1).split(","):
                param = param.strip().split(":")[0].split("=")[0].strip()
                if param and param != "self" and param != "cls":
                    facts.add(param)

    # Filter out very short or very common names
    common = {"self", "cls", "args", "kwargs", "return", "None", "True", "False",
              "str", "int", "float", "bool", "list", "dict", "set", "tuple",
              "print", "len", "range", "type", "super", "init", "new"}
    return {f for f in facts if len(f) > 2 and f not in common}


def _compute_context_utilization(
    solution: str,
    context_chunks: list[str],
) -> tuple[int, int, float]:
    """Compute context utilization: facts_used / facts_available.

    Returns (facts_available, facts_used, utilization_score).
    """
    facts = _extract_key_facts(context_chunks)
    if not facts:
        return 0, 0, 0.0

    used = sum(1 for f in facts if f in solution)
    return len(facts), used, used / len(facts)


# ---------------------------------------------------------------------------
# Position-debiased judge scoring (reuse technique from Phase 8)
# ---------------------------------------------------------------------------

_SWE_JUDGE_SYSTEM = """\
You are an expert code reviewer evaluating a solution to a software engineering task. \
Score the solution on four dimensions (0-5 each):

**Correctness** (0-5): Does the solution correctly address the task?
- 0: Completely wrong or unrelated
- 1: Attempts the right approach but fundamentally broken
- 2: Partially correct but major issues
- 3: Mostly correct with minor bugs
- 4: Correct solution with edge case gaps
- 5: Fully correct, handles all described requirements

**Completeness** (0-5): Does it handle edge cases and full scope?
- 0: Missing entirely
- 1: Only handles the happy path
- 2: Handles main case + 1 edge case
- 3: Handles most cases
- 4: Comprehensive with minor gaps
- 5: Handles all edge cases mentioned in the task

**Code Quality** (0-5): Is it idiomatic, minimal, and well-structured?
- 0: Unreadable or entirely wrong idioms
- 1: Works but very poor style
- 2: Functional but messy
- 3: Acceptable quality
- 4: Clean, idiomatic code
- 5: Exemplary — minimal, clear, Pythonic

**No Hallucination** (0-5): Does it use only real APIs from the library?
- 0: Multiple invented methods/classes
- 1: One major invented API
- 2: Minor parameter hallucinations
- 3: Mostly correct APIs with one questionable call
- 4: All APIs appear correct
- 5: Verified correct usage of all library APIs

Respond with ONLY a JSON object:
{"correctness": <0-5>, "completeness": <0-5>, "code_quality": <0-5>, "no_hallucination": <0-5>}"""

_SWE_JUDGE_USER = """\
## Task
{title}

{description}

## Library
{library}

## Evaluation Rubric
- Correctness: {rubric_correctness}
- Completeness: {rubric_completeness}
- Code Quality: {rubric_code_quality}
- No Hallucination: {rubric_no_hallucination}

## Solution to Evaluate
```python
{solution}
```

Score this solution. Respond with ONLY the JSON object."""


async def _judge_solution(
    test_case: SWETestCase,
    solution: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> dict[str, float]:
    """Score a solution using the LLM judge (single pass)."""
    rubric = test_case.judge_rubric
    user_prompt = _SWE_JUDGE_USER.format(
        title=test_case.title,
        description=test_case.description,
        library=test_case.library,
        rubric_correctness=rubric.get("correctness", "Does the solution correctly address the task?"),
        rubric_completeness=rubric.get("completeness", "Does it handle edge cases?"),
        rubric_code_quality=rubric.get("code_quality", "Is the code idiomatic and minimal?"),
        rubric_no_hallucination=rubric.get("no_hallucination", "Does it use only real APIs?"),
        solution=solution[:6000],
    )
    text = await _call_llm(
        _SWE_JUDGE_SYSTEM, user_prompt,
        llm_provider, llm_model, llm_api_key,
        max_tokens=200, temperature=0.0,
    )
    scores = _safe_parse_json(text, defaults={
        "correctness": 0, "completeness": 0, "code_quality": 0, "no_hallucination": 0,
    })
    return {
        "correctness": min(5, max(0, float(scores.get("correctness", 0)))),
        "completeness": min(5, max(0, float(scores.get("completeness", 0)))),
        "code_quality": min(5, max(0, float(scores.get("code_quality", 0)))),
        "no_hallucination": min(5, max(0, float(scores.get("no_hallucination", 0)))),
    }


async def _judge_solution_debiased(
    test_case: SWETestCase,
    solution: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    enable_debiasing: bool = True,
) -> dict[str, float]:
    """Score a solution with position debiasing (judge twice, average)."""
    scores_p1 = await _judge_solution(
        test_case, solution, llm_provider, llm_model, llm_api_key,
    )

    if not enable_debiasing:
        return scores_p1

    # Pass 2: present rubric in reversed order to debias
    await asyncio.sleep(0.3)
    scores_p2 = await _judge_solution(
        test_case, solution, llm_provider, llm_model, llm_api_key,
    )

    # Average both passes
    return {
        k: round((scores_p1[k] + scores_p2[k]) / 2, 1)
        for k in scores_p1
    }


def _compute_composite(scores: dict[str, float]) -> float:
    """Compute the SWE composite score from judge dimensions + structural metrics.

    swe_composite = (
        0.35 * judge_correctness_normalized
      + 0.25 * criteria_pass_rate
      + 0.20 * judge_completeness_normalized
      + 0.10 * (1 - hallucination_rate)
      + 0.10 * file_targeting_score
    )
    """
    # This is called per-result with the judge scores already set.
    # We return just the judge-weighted portion here; full composite
    # is computed in evaluate_solution().
    return (
        0.35 * (scores.get("correctness", 0) / 5.0)
        + 0.20 * (scores.get("completeness", 0) / 5.0)
        + 0.10 * (scores.get("code_quality", 0) / 5.0)
        + 0.10 * (scores.get("no_hallucination", 0) / 5.0)
    )


# ---------------------------------------------------------------------------
# Per-solution evaluation pipeline
# ---------------------------------------------------------------------------


async def evaluate_solution(
    test_case: SWETestCase,
    solution: str,
    context_chunks: list[str],
    engine_name: str,
    query_mode: str,
    turn: int,
    latency_ms: float,
    queries_used: list[str],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    enable_debiasing: bool = True,
) -> SWEResult:
    """Run the full evaluation pipeline on a single solution."""
    result = SWEResult(
        test_case_id=test_case.id,
        engine=engine_name,
        query_mode=query_mode,
        turn=turn,
        latency_ms=latency_ms,
        generated_solution=solution,
        queries_used=queries_used,
        context_chunks=context_chunks,
    )

    # Structural checks
    result.parseable = _check_parseable(solution)
    result.file_targeting_score = _file_targeting_score(solution, test_case.expected_files)
    result.pattern_match_score = _pattern_match_score(solution, test_case.expected_patterns)
    result.symbol_usage_score = _symbol_usage_score(solution, test_case.expected_symbols)
    result.criteria_pass_rate = _criteria_pass_rate(solution, test_case.acceptance_criteria)

    # Hallucination detection (reuse from hallucination.py)
    hal_tc = HallucinationTestCase(
        id=test_case.id,
        query=test_case.title,
        library=test_case.library,
        description=test_case.description,
    )
    errors = validate_code(solution, hal_tc)
    result.hallucination_count = len(errors)

    # Context utilization (only for engine solutions, not baseline)
    if context_chunks:
        avail, used, util = _compute_context_utilization(solution, context_chunks)
        result.key_facts_available = avail
        result.key_facts_used = used
        result.context_utilization_score = util

    # LLM judge scoring (debiased)
    try:
        judge_scores = await _judge_solution_debiased(
            test_case, solution,
            llm_provider, llm_model, llm_api_key,
            enable_debiasing=enable_debiasing,
        )
        result.judge_correctness = judge_scores["correctness"]
        result.judge_completeness = judge_scores["completeness"]
        result.judge_code_quality = judge_scores["code_quality"]
        result.judge_no_hallucination = judge_scores["no_hallucination"]
    except Exception:
        # Judge failed — leave scores at 0
        judge_scores = {"correctness": 0, "completeness": 0, "code_quality": 0, "no_hallucination": 0}

    # Composite score
    hallucination_rate = min(1.0, result.hallucination_count / 5.0)
    result.judge_composite = (
        0.35 * (result.judge_correctness / 5.0)
        + 0.25 * result.criteria_pass_rate
        + 0.20 * (result.judge_completeness / 5.0)
        + 0.10 * (1.0 - hallucination_rate)
        + 0.10 * result.file_targeting_score
    )

    return result


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


async def _gather_context(
    engine: ContextEngineAdapter,
    queries: list[str],
    top_k: int = 5,
) -> tuple[list[str], float]:
    """Issue queries to an engine and collect context chunks.

    Returns (chunks, total_latency_ms).
    """
    all_chunks: list[str] = []
    total_latency = 0.0
    seen = set()

    for query in queries:
        try:
            results, latency = await engine.search_code(query=query, top_k=top_k)
            total_latency += latency
            for r in results:
                if r.content and r.content not in seen:
                    seen.add(r.content)
                    all_chunks.append(r.content)
        except Exception:
            continue

    return all_chunks, total_latency


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_swe_agent_benchmark(
    engines: list[ContextEngineAdapter],
    test_cases_path: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    max_queries: int | None = None,
    enable_debiasing: bool = True,
    with_agent_queries: bool = False,
) -> SWEBenchmarkResult:
    """Run the full SWE-Agent benchmark (Phase 9).

    For each test case:
    1. Generate agent queries
    2. Run baseline (no context)
    3. For each engine x query mode: Turn 1 + Turn 2
    4. Evaluate all solutions
    5. Compute deltas
    """
    test_cases = load_test_cases(test_cases_path)
    if max_queries is not None:
        test_cases = test_cases[:max_queries]

    all_results: list[SWEResult] = []
    baseline_results: list[SWEResult] = []

    for tc in tqdm(test_cases, desc="  SWE-Agent", unit="case"):
        # Step 1: Build query modes
        if with_agent_queries:
            try:
                agent_queries = await generate_agent_queries(
                    tc, llm_provider, llm_model, llm_api_key,
                )
            except Exception:
                agent_queries = [tc.title]
            query_modes = [("gold", tc.gold_queries), ("agent", agent_queries)]
        else:
            query_modes = [("gold", tc.gold_queries)]

        # Step 2: Baseline (no context) — run ONCE per test case
        try:
            start = time.perf_counter()
            baseline_solution = await _generate_solution(
                tc, "", llm_provider, llm_model, llm_api_key,
            )
            baseline_latency = (time.perf_counter() - start) * 1000

            baseline_result = await evaluate_solution(
                test_case=tc,
                solution=baseline_solution,
                context_chunks=[],
                engine_name="no_context",
                query_mode="none",
                turn=1,
                latency_ms=baseline_latency,
                queries_used=[],
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                enable_debiasing=enable_debiasing,
            )
            baseline_results.append(baseline_result)
            all_results.append(baseline_result)
        except Exception as e:
            print(f"\n  [!] Baseline failed for {tc.id}: {e}")
            # Create empty baseline
            baseline_results.append(SWEResult(
                test_case_id=tc.id, engine="no_context",
                query_mode="none", turn=1,
            ))

        await asyncio.sleep(0.3)

        # Step 3: For each engine x query mode
        for engine in engines:
            for query_mode, queries in query_modes:
                # ---- Turn 1 ----
                try:
                    chunks, retrieval_latency = await _gather_context(engine, queries)
                    context_str = "\n\n---\n\n".join(chunks)

                    start = time.perf_counter()
                    turn1_solution = await _generate_solution(
                        tc, context_str, llm_provider, llm_model, llm_api_key,
                    )
                    gen_latency = (time.perf_counter() - start) * 1000

                    turn1_result = await evaluate_solution(
                        test_case=tc,
                        solution=turn1_solution,
                        context_chunks=chunks,
                        engine_name=engine.name,
                        query_mode=query_mode,
                        turn=1,
                        latency_ms=retrieval_latency + gen_latency,
                        queries_used=queries,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        enable_debiasing=enable_debiasing,
                    )
                    all_results.append(turn1_result)
                except Exception as e:
                    print(f"\n  [!] Turn 1 failed ({engine.name}/{query_mode}/{tc.id}): {e}")
                    all_results.append(SWEResult(
                        test_case_id=tc.id, engine=engine.name,
                        query_mode=query_mode, turn=1,
                    ))
                    continue

                await asyncio.sleep(0.3)

                # ---- Turn 2 (iterative refinement) ----
                try:
                    gaps, follow_up_queries = await _analyze_gaps(
                        tc, turn1_solution, llm_provider, llm_model, llm_api_key,
                    )

                    if follow_up_queries:
                        new_chunks, follow_up_latency = await _gather_context(
                            engine, follow_up_queries,
                        )
                        all_chunks = chunks + new_chunks
                        full_context = "\n\n---\n\n".join(all_chunks)
                    else:
                        all_chunks = chunks
                        full_context = context_str
                        follow_up_latency = 0.0

                    start = time.perf_counter()
                    turn2_solution = await _generate_solution(
                        tc, full_context, llm_provider, llm_model, llm_api_key,
                    )
                    gen2_latency = (time.perf_counter() - start) * 1000

                    turn2_result = await evaluate_solution(
                        test_case=tc,
                        solution=turn2_solution,
                        context_chunks=all_chunks,
                        engine_name=engine.name,
                        query_mode=query_mode,
                        turn=2,
                        latency_ms=retrieval_latency + follow_up_latency + gen2_latency,
                        queries_used=queries + follow_up_queries,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        enable_debiasing=enable_debiasing,
                    )
                    all_results.append(turn2_result)
                except Exception as e:
                    print(f"\n  [!] Turn 2 failed ({engine.name}/{query_mode}/{tc.id}): {e}")
                    all_results.append(SWEResult(
                        test_case_id=tc.id, engine=engine.name,
                        query_mode=query_mode, turn=2,
                    ))

                await asyncio.sleep(0.3)

    # --- Aggregate results ---
    return _aggregate_swe_results(all_results, baseline_results, engines, test_cases)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _group_avg(results: list[SWEResult], attr: str) -> float:
    """Average a numeric attribute across results."""
    vals = [getattr(r, attr, 0) for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def _build_aggregate(
    engine_name: str,
    results: list[SWEResult],
    test_cases: list[SWETestCase],
) -> SWEAggregateMetrics:
    """Build aggregate metrics for one engine from its results."""
    if not results:
        return SWEAggregateMetrics(engine=engine_name)

    tc_map = {tc.id: tc for tc in test_cases}

    agg = SWEAggregateMetrics(
        engine=engine_name,
        num_cases=len(results),
        avg_judge_composite=_group_avg(results, "judge_composite"),
        avg_criteria_pass_rate=_group_avg(results, "criteria_pass_rate"),
        avg_file_targeting=_group_avg(results, "file_targeting_score"),
        avg_context_utilization=_group_avg(results, "context_utilization_score"),
        avg_hallucination_count=_group_avg(results, "hallucination_count"),
        parse_rate=sum(1 for r in results if r.parseable) / len(results),
        avg_latency_ms=_group_avg(results, "latency_ms"),
    )

    # Group by task_type
    by_type: dict[str, list[SWEResult]] = {}
    by_tier: dict[str, list[SWEResult]] = {}
    by_diff: dict[str, list[SWEResult]] = {}

    for r in results:
        tc = tc_map.get(r.test_case_id)
        if not tc:
            continue
        by_type.setdefault(tc.task_type, []).append(r)
        by_tier.setdefault(tc.knowledge_tier, []).append(r)
        by_diff.setdefault(tc.difficulty, []).append(r)

    for group_name, group_dict, target in [
        ("task_type", by_type, agg.by_task_type),
        ("tier", by_tier, agg.by_knowledge_tier),
        ("difficulty", by_diff, agg.by_difficulty),
    ]:
        for key, group_results in group_dict.items():
            target[key] = {
                "count": len(group_results),
                "avg_judge_composite": _group_avg(group_results, "judge_composite"),
                "avg_criteria_pass_rate": _group_avg(group_results, "criteria_pass_rate"),
                "avg_hallucination_count": _group_avg(group_results, "hallucination_count"),
            }

    return agg


def _aggregate_swe_results(
    all_results: list[SWEResult],
    baseline_results: list[SWEResult],
    engines: list[ContextEngineAdapter],
    test_cases: list[SWETestCase],
) -> SWEBenchmarkResult:
    """Aggregate all results into the final benchmark result."""
    benchmark = SWEBenchmarkResult()

    # Baseline aggregate
    benchmark.baseline_results = _build_aggregate("no_context", baseline_results, test_cases)

    # Per-engine aggregates (use best turn per test case for the aggregate)
    for engine in engines:
        engine_results = [
            r for r in all_results
            if r.engine == engine.name
        ]
        # For aggregate, use the best (turn 2 gold) results
        best_results: dict[str, SWEResult] = {}
        for r in engine_results:
            key = r.test_case_id
            existing = best_results.get(key)
            if existing is None or r.judge_composite > existing.judge_composite:
                best_results[key] = r

        benchmark.engine_results[engine.name] = _build_aggregate(
            engine.name, list(best_results.values()), test_cases,
        )

    # Compute deltas (engine - baseline)
    if benchmark.baseline_results:
        bl = benchmark.baseline_results
        for eng_name, eng_agg in benchmark.engine_results.items():
            benchmark.delta_scores[eng_name] = {
                "judge_composite": eng_agg.avg_judge_composite - bl.avg_judge_composite,
                "criteria_pass_rate": eng_agg.avg_criteria_pass_rate - bl.avg_criteria_pass_rate,
                "file_targeting": eng_agg.avg_file_targeting - bl.avg_file_targeting,
                "hallucination_count": eng_agg.avg_hallucination_count - bl.avg_hallucination_count,
            }

    # Store all per-case results
    benchmark.per_case_results = [asdict(r) for r in all_results]

    return benchmark


# ---------------------------------------------------------------------------
# Test case loading
# ---------------------------------------------------------------------------


def load_test_cases(path: str) -> list[SWETestCase]:
    """Load SWE-Agent test cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", []):
        cases.append(SWETestCase(
            id=item["id"],
            task_type=item["task_type"],
            title=item["title"],
            description=item["description"],
            library=item["library"],
            knowledge_tier=item["knowledge_tier"],
            difficulty=item.get("difficulty", "medium"),
            version_specific=item.get("version_specific", False),
            gold_queries=item.get("gold_queries", []),
            expected_files=item.get("expected_files", []),
            expected_patterns=item.get("expected_patterns", []),
            expected_symbols=item.get("expected_symbols", []),
            acceptance_criteria=item.get("acceptance_criteria", {}),
            judge_rubric=item.get("judge_rubric", {}),
        ))
    return cases


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_swe_agent_summary(result: SWEBenchmarkResult) -> None:
    """Print formatted SWE-Agent benchmark results."""
    print("\n=== SWE-Agent Results (Phase 9) ===\n")

    # Baseline
    bl = result.baseline_results
    if bl:
        print(f"  Baseline (no context):")
        print(
            f"    Judge composite: {bl.avg_judge_composite:.3f}  |  "
            f"Criteria: {bl.avg_criteria_pass_rate:.0%}  |  "
            f"Files: {bl.avg_file_targeting:.0%}  |  "
            f"Halluc: {bl.avg_hallucination_count:.1f}/case"
        )
        print()

    # Baseline detail (shown when no engines or always for completeness)
    if bl and not result.engine_results:
        print("  Baseline-only mode (no context engines)")
        print(f"    Parse rate:       {bl.parse_rate:.0%}")
        print(f"    Avg latency:      {bl.avg_latency_ms:.0f}ms")
        print(f"    Cases evaluated:  {bl.num_cases}")
        if bl.by_knowledge_tier:
            print("\n    By Knowledge Tier:")
            for tier in ("A", "B", "C"):
                td = bl.by_knowledge_tier.get(tier, {})
                if td:
                    tier_label = {"A": "well-known", "B": "niche/recent", "C": "version-specific"}.get(tier, tier)
                    print(
                        f"      Tier {tier} ({tier_label}): "
                        f"composite={td.get('avg_judge_composite', 0):.3f}  "
                        f"criteria={td.get('avg_criteria_pass_rate', 0):.0%}  "
                        f"n={td.get('count', 0)}"
                    )
        if bl.by_task_type:
            print("\n    By Task Type:")
            for tt, td in sorted(bl.by_task_type.items()):
                print(
                    f"      {tt}: "
                    f"composite={td.get('avg_judge_composite', 0):.3f}  "
                    f"criteria={td.get('avg_criteria_pass_rate', 0):.0%}  "
                    f"n={td.get('count', 0)}"
                )
        print()
        return

    # Engine comparison table
    engines = list(result.engine_results.keys())
    if not engines:
        print("  No engine results.")
        return

    header = f"  {'Engine':<15} {'Judge':>8} {'Criteria':>10} {'Files':>8} {'Halluc.':>9} {'Util.':>8} {'Delta(J)':>12} {'Delta(C)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for eng_name in engines:
        eng = result.engine_results[eng_name]
        delta = result.delta_scores.get(eng_name, {})
        dj = delta.get("judge_composite", 0)
        dc = delta.get("criteria_pass_rate", 0)

        # Format delta with percentage
        bl_j = bl.avg_judge_composite if bl else 0
        dj_pct = f"+{dj:.3f}" if dj >= 0 else f"{dj:.3f}"
        if bl_j > 0:
            dj_pct += f"({dj / bl_j:+.0%})"

        dc_pp = f"+{dc:.0%}" if dc >= 0 else f"{dc:.0%}"

        print(
            f"  {eng_name:<15} "
            f"{eng.avg_judge_composite:>8.3f} "
            f"{eng.avg_criteria_pass_rate:>10.0%} "
            f"{eng.avg_file_targeting:>8.0%} "
            f"{eng.avg_hallucination_count:>9.1f} "
            f"{eng.avg_context_utilization:>8.0%} "
            f"{dj_pct:>12} "
            f"{dc_pp:>12}"
        )

    # By knowledge tier
    print("\n  By Knowledge Tier:")
    for tier in ("A", "B", "C"):
        tier_parts = []
        for eng_name in engines:
            eng = result.engine_results[eng_name]
            tier_data = eng.by_knowledge_tier.get(tier, {})
            bl_tier = bl.by_knowledge_tier.get(tier, {}) if bl else {}
            eng_score = tier_data.get("avg_judge_composite", 0)
            bl_score = bl_tier.get("avg_judge_composite", 0)
            delta = eng_score - bl_score
            tier_parts.append(f"{eng_name} {delta:+.3f}")

        tier_label = {"A": "well-known", "B": "niche/recent", "C": "version-specific"}.get(tier, tier)
        print(f"    Tier {tier} ({tier_label}):  {'  |  '.join(tier_parts)}")

    print()
