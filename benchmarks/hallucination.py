"""Hallucination rate benchmark.

Replicates Nia's Context Augmentation Benchmark methodology:
1. Pick SDK/library with known API surface (ground truth)
2. Ask each engine for context about a specific task
3. Feed context to an LLM and ask it to generate code
4. Check generated code for: invented_method, wrong_parameter, outdated_api

This measures how well each engine prevents LLM hallucinations by
providing accurate, up-to-date context.
"""

from __future__ import annotations

import json
import re

from tqdm import tqdm
from dataclasses import dataclass, field

import httpx

from .adapters.base import ContextEngineAdapter
from .metrics import HallucinationResult


@dataclass
class HallucinationTestCase:
    """A single test case for the hallucination benchmark."""

    id: str
    query: str  # What to ask the LLM to generate
    library: str  # e.g. "fastapi", "pydantic-v2"
    description: str  # Human-readable description

    # Ground truth: known correct API surface
    valid_methods: list[str] = field(default_factory=list)
    valid_parameters: dict[str, list[str]] = field(default_factory=dict)  # method -> [params]
    deprecated_apis: list[str] = field(default_factory=list)

    # Optional: repo to search in (if already indexed)
    repo_id: str | None = None


@dataclass
class HallucinationBenchmarkResult:
    """Full result of running the hallucination benchmark."""

    engine: str
    test_cases: list[HallucinationResult]
    overall_rate: float = 0.0  # legacy: all failures / total
    true_hallucination_rate: float = 0.0  # wrong code when context was available
    abstention_rate: float = 0.0  # LLM correctly refused (no context)
    context_miss_rate: float = 0.0  # search returned no/weak context
    error_breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Code validator
# ---------------------------------------------------------------------------

def _extract_function_calls(code: str) -> list[dict]:
    """Extract function/method calls from generated code.

    Returns list of {name, args} dicts.
    Simple regex-based extraction — not a full AST parse, but sufficient
    for benchmark validation.
    """
    calls = []
    # Match patterns like: obj.method(args) or function(args)
    pattern = r'(\w+(?:\.\w+)*)\s*\((.*?)\)'
    for match in re.finditer(pattern, code, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2).strip()
        # Extract keyword arguments
        kwarg_pattern = r'(\w+)\s*='
        kwargs = re.findall(kwarg_pattern, args_str)
        calls.append({"name": name, "args": kwargs, "raw_args": args_str})
    return calls


def validate_code(
    code: str,
    test_case: HallucinationTestCase,
) -> list[dict]:
    """Validate generated code against ground truth.

    Returns list of errors found.
    """
    errors = []
    calls = _extract_function_calls(code)

    for call in calls:
        name = call["name"]

        # Check for invented methods (not in known API surface)
        if test_case.valid_methods:
            # Only check methods that look like they belong to the library
            parts = name.split(".")
            method_name = parts[-1] if len(parts) > 1 else name
            # Skip common builtins
            builtins = {
                "print", "len", "range", "str", "int", "float", "list",
                "dict", "set", "tuple", "type", "isinstance", "getattr",
                "setattr", "hasattr", "super", "open", "format",
            }
            if method_name not in builtins and method_name not in test_case.valid_methods:
                # Fuzzy check — only flag if it looks like a library call
                if any(lib_part in name.lower() for lib_part in test_case.library.lower().split("-")):
                    errors.append({
                        "type": "invented_method",
                        "detail": f"Method '{name}' not found in {test_case.library} API",
                        "call": name,
                    })

        # Check for wrong parameters
        if test_case.valid_parameters and name in test_case.valid_parameters:
            valid_params = test_case.valid_parameters[name]
            for kwarg in call["args"]:
                if kwarg not in valid_params:
                    errors.append({
                        "type": "wrong_parameter",
                        "detail": f"Parameter '{kwarg}' not valid for {name}()",
                        "call": name,
                        "parameter": kwarg,
                    })

        # Check for deprecated/outdated APIs
        if test_case.deprecated_apis:
            for deprecated in test_case.deprecated_apis:
                if deprecated in name:
                    errors.append({
                        "type": "outdated_api",
                        "detail": f"'{name}' uses deprecated API '{deprecated}'",
                        "call": name,
                    })

    return errors


# ---------------------------------------------------------------------------
# LLM code generation
# ---------------------------------------------------------------------------

async def _generate_code_with_context(
    query: str,
    context: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> str:
    """Ask an LLM to generate code using the provided context."""
    system_prompt = (
        "You are a coding assistant. Generate code based on the user's request. "
        "Use ONLY the provided context/documentation to write correct code. "
        "Do not invent APIs or parameters that aren't in the context. "
        "Return only the code, no explanations."
    )

    user_prompt = (
        f"Using the following documentation/context:\n\n"
        f"---\n{context}\n---\n\n"
        f"Generate code for: {query}\n\n"
        f"Return only Python code."
    )

    if llm_provider == "anthropic":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": llm_model,
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    elif llm_provider == "gemini":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent",
                params={"key": llm_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 2048},
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle blocked/empty responses from Gemini
            candidates = data.get("candidates", [])
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason})")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")

            # Extract text from parts, handling missing content
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise RuntimeError(f"Gemini returned empty parts (finishReason={finish_reason})")

            # Concatenate all text parts
            return "".join(p.get("text", "") for p in parts)

    elif llm_provider == "openai":
        async with httpx.AsyncClient(timeout=60.0) as client:
            # GPT-5+ models use max_completion_tokens instead of max_tokens
            is_gpt5 = "gpt-5" in llm_model or "gpt-4.1" in llm_model or "o" in llm_model.split("-")[0]
            token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
            body = {
                "model": llm_model,
                token_param: 2048,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}. Supported: anthropic, gemini, openai")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_hallucination_benchmark(
    engine: ContextEngineAdapter,
    test_cases: list[HallucinationTestCase],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> HallucinationBenchmarkResult:
    """Run the full hallucination benchmark against one engine.

    For each test case:
    1. Query the engine for relevant context
    2. Feed context to LLM for code generation
    3. Validate generated code against ground truth
    """
    results: list[HallucinationResult] = []

    for tc in tqdm(test_cases, desc=f"  {engine.name} hallucination", unit="q"):
        # Step 1: Get context from the engine
        try:
            search_results, _ = await engine.search_code(
                query=tc.query,
                top_k=5,
                repo_ids=[tc.repo_id] if tc.repo_id else None,
            )
        except Exception as e:
            results.append(
                HallucinationResult(
                    query=tc.query,
                    engine=engine.name,
                    generated_code="",
                    errors=[{"type": "search_failed", "detail": str(e)}],
                    hallucination_rate=1.0,
                )
            )
            continue

        context = "\n\n---\n\n".join(r.content for r in search_results)
        has_relevant_context = bool(context.strip())

        if not has_relevant_context:
            context = "(No relevant context found)"

        # Step 2: Generate code using LLM + context
        try:
            generated_code = await _generate_code_with_context(
                query=tc.query,
                context=context,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
            )
        except Exception as e:
            # Classify: if no context was available, this is an abstention (good)
            # If context was available but LLM failed, it's a generation error
            if not has_relevant_context:
                err_type = "context_miss_abstention"
            else:
                err_type = "generation_failed"
            results.append(
                HallucinationResult(
                    query=tc.query,
                    engine=engine.name,
                    generated_code="",
                    errors=[{"type": err_type, "detail": str(e)}],
                    hallucination_rate=1.0,
                )
            )
            continue

        # Check if LLM explicitly refused due to lack of context
        refusal_phrases = [
            "cannot fulfill", "cannot generate", "no relevant context",
            "don't have enough", "not contain any information",
            "sorry, i cannot", "i am sorry",
        ]
        is_refusal = any(p in generated_code.lower() for p in refusal_phrases)

        if is_refusal and not has_relevant_context:
            # LLM correctly refused — not a hallucination
            results.append(
                HallucinationResult(
                    query=tc.query,
                    engine=engine.name,
                    generated_code=generated_code,
                    errors=[{"type": "correct_abstention", "detail": "LLM refused due to insufficient context"}],
                    hallucination_rate=0.0,  # This is GOOD behavior
                )
            )
            continue

        # Step 3: Validate against ground truth
        errors = validate_code(generated_code, tc)

        # Classify errors by context availability
        if errors and not has_relevant_context:
            # Engine had no context but LLM tried anyway — context miss
            for err in errors:
                err["context_available"] = False
        elif errors:
            # Engine provided context but LLM still hallucinated — true hallucination
            for err in errors:
                err["context_available"] = True

        results.append(
            HallucinationResult(
                query=tc.query,
                engine=engine.name,
                generated_code=generated_code,
                errors=errors,
                hallucination_rate=1.0 if errors else 0.0,
            )
        )

    # Aggregate with classification
    total = len(results)
    true_hallucinations = 0  # LLM produced wrong code WITH context available
    correct_abstentions = 0  # LLM refused or failed WITHOUT context (good)
    context_misses = 0       # Engine had no relevant context for the query
    clean_generations = 0    # LLM produced correct code

    breakdown: dict[str, int] = {}
    for r in results:
        if not r.errors:
            clean_generations += 1
            continue

        for err in r.errors:
            t = err.get("type", "unknown")
            breakdown[t] = breakdown.get(t, 0) + 1

        err_types = {e.get("type") for e in r.errors}
        if "correct_abstention" in err_types or "context_miss_abstention" in err_types:
            correct_abstentions += 1
        elif any(e.get("context_available") is False for e in r.errors):
            context_misses += 1
        elif "search_failed" in err_types:
            context_misses += 1
        else:
            true_hallucinations += 1

    # True hallucination rate: only count failures where context WAS available
    cases_with_context = total - correct_abstentions - context_misses
    true_hal_rate = true_hallucinations / cases_with_context if cases_with_context > 0 else 0.0

    # Legacy overall rate (all failures / total) kept for backwards compatibility
    all_failures = sum(1 for r in results if r.errors and r.hallucination_rate > 0)
    overall_rate = all_failures / total if total > 0 else 0.0

    return HallucinationBenchmarkResult(
        engine=engine.name,
        test_cases=results,
        overall_rate=overall_rate,
        true_hallucination_rate=true_hal_rate,
        abstention_rate=correct_abstentions / total if total > 0 else 0.0,
        context_miss_rate=context_misses / total if total > 0 else 0.0,
        error_breakdown=breakdown,
    )


def load_test_cases(path: str) -> list[HallucinationTestCase]:
    """Load hallucination test cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", data if isinstance(data, list) else []):
        cases.append(
            HallucinationTestCase(
                id=item["id"],
                query=item["query"],
                library=item["library"],
                description=item.get("description", ""),
                valid_methods=item.get("valid_methods", []),
                valid_parameters=item.get("valid_parameters", {}),
                deprecated_apis=item.get("deprecated_apis", []),
                repo_id=item.get("repo_id"),
            )
        )
    return cases
