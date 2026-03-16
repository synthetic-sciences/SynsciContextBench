"""LLM-as-Judge benchmark for fair cross-engine comparison.

Instead of matching retrieved content to ground truth (which penalizes
engines that transform text), we ask an LLM judge to evaluate the
*quality* of retrieved context for answering each query.

This is engine-agnostic: we only care whether the retrieved context
is useful, not whether it matches a specific text format.

Flow:
1. Load queries from validated datasets (CoSQA, CodeSearchNet)
2. For each query, send to each engine → get retrieved context
3. Send (query, context) to LLM judge → get scores
4. Aggregate and compare scores across engines
"""

from __future__ import annotations

import asyncio
import json
import time

from tqdm import tqdm
from dataclasses import dataclass, field

import httpx

from .adapters.base import ContextEngineAdapter
import re


def _safe_parse_json(text: str, defaults: dict | None = None) -> dict:
    """Parse JSON from LLM output, handling markdown fences and malformed responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Try to extract JSON object if there's extra text around it
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if defaults is not None:
            return defaults
        raise

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JudgeScore:
    """LLM judge scores for a single (query, engine) pair."""

    relevance: int = 0  # 0-3: Does the context contain relevant code?
    completeness: int = 0  # 0-3: Could you answer the query fully from this context?
    specificity: int = 0  # 0-3: Is the context specific (not generic boilerplate)?

    @property
    def total(self) -> float:
        return (self.relevance + self.completeness + self.specificity) / 3.0


@dataclass
class JudgeQueryResult:
    """Result for a single query across engines."""

    query: str
    scores: dict[str, JudgeScore] = field(default_factory=dict)  # engine -> score
    contexts: dict[str, str] = field(default_factory=dict)  # engine -> raw context
    latencies: dict[str, float] = field(default_factory=dict)  # engine -> ms
    error: str = ""


@dataclass
class JudgeAggregateMetrics:
    """Aggregate judge metrics for one engine."""

    engine: str
    num_queries: int = 0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_specificity: float = 0.0
    avg_total: float = 0.0
    avg_latency_ms: float = 0.0
    win_count: int = 0  # Number of queries where this engine scored highest
    tie_count: int = 0


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert code retrieval evaluator. You will be given a natural language \
query and a set of code snippets retrieved by a search engine.

Score the retrieved context on three dimensions (0-3 each):

**Relevance** (0-3):
- 0: No retrieved snippet relates to the query at all
- 1: Tangentially related (same domain but wrong functionality)
- 2: Partially relevant (addresses part of the query)
- 3: Highly relevant (directly addresses what the query asks for)

**Completeness** (0-3):
- 0: Cannot answer the query at all from this context
- 1: Could partially answer with significant gaps
- 2: Could mostly answer with minor gaps
- 3: Could fully answer the query from this context alone

**Specificity** (0-3):
- 0: Only generic boilerplate or unrelated code
- 1: Some relevant code mixed with much noise
- 2: Mostly specific and useful code
- 3: Precisely targeted code with minimal noise

Respond with ONLY a JSON object, no other text:
{"relevance": <0-3>, "completeness": <0-3>, "specificity": <0-3>}"""


async def _call_llm_judge_raw(
    system_prompt: str,
    user_prompt: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    max_retries: int = 3,
) -> str:
    """Call an LLM with custom system/user prompts. Returns raw text response.

    Used by adversarial and code_qa modules for LLM-based scoring.
    Retries on empty responses or HTTP errors with exponential backoff.
    """
    import asyncio as _asyncio

    last_error = None
    for attempt in range(max_retries):
        try:
            text = await _call_llm_judge_raw_inner(
                system_prompt, user_prompt, llm_provider, llm_model, llm_api_key
            )
            if text and text.strip():
                return text
            last_error = ValueError(f"Empty LLM response (attempt {attempt + 1})")
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = e
        if attempt < max_retries - 1:
            await _asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"LLM judge failed after {max_retries} retries: {last_error}")


async def _call_llm_judge_raw_inner(
    system_prompt: str,
    user_prompt: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> str:
    """Inner LLM call without retry logic."""
    if llm_provider == "gemini":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent",
                params={"key": llm_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 200, "temperature": 0.0},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            return "".join(
                p.get("text", "")
                for p in candidates[0].get("content", {}).get("parts", [])
            )

    elif llm_provider == "anthropic":
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
                    "max_tokens": 200,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    elif llm_provider == "openai":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_model,
                    "max_tokens": 200,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown LLM provider: {llm_provider}")


async def _call_llm_judge(
    query: str,
    context: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> JudgeScore:
    """Ask the LLM judge to score retrieved context for a query."""
    user_prompt = (
        f"## Query\n{query}\n\n"
        f"## Retrieved Context\n```\n{context[:6000]}\n```\n\n"
        f"Score the above context for this query. "
        f"Respond with ONLY a JSON object: "
        f'{{"relevance": <0-3>, "completeness": <0-3>, "specificity": <0-3>}}'
    )

    if llm_provider == "gemini":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent",
                params={"key": llm_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 100,
                        "temperature": 0.0,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            text = "".join(
                p.get("text", "")
                for p in candidates[0].get("content", {}).get("parts", [])
            )

    elif llm_provider == "anthropic":
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
                    "max_tokens": 100,
                    "system": JUDGE_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"]

    elif llm_provider == "openai":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    scores = _safe_parse_json(text, defaults={"relevance": 0, "completeness": 0, "specificity": 0})
    return JudgeScore(
        relevance=min(3, max(0, int(scores.get("relevance", 0)))),
        completeness=min(3, max(0, int(scores.get("completeness", 0)))),
        specificity=min(3, max(0, int(scores.get("specificity", 0)))),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_judge_benchmark(
    engines: list[ContextEngineAdapter],
    dataset_path: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    max_queries: int | None = None,
    top_k: int = 5,
) -> tuple[dict[str, JudgeAggregateMetrics], list[JudgeQueryResult]]:
    """Run LLM-as-judge benchmark across multiple engines.

    Returns per-engine aggregate metrics and per-query results.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if max_queries:
        queries = queries[:max_queries]

    total = len(queries)
    results: list[JudgeQueryResult] = []

    for q in tqdm(queries, desc="  judge", unit="q"):
        query_text = q.get("query", q.get("text", ""))
        if not query_text:
            continue

        qr = JudgeQueryResult(query=query_text)

        # Step 1: Retrieve context from each engine
        for engine in engines:
            try:
                search_results, latency = await engine.search_code(
                    query=query_text, top_k=top_k,
                )
                # Combine retrieved snippets into a single context block
                context_parts = []
                for sr in search_results:
                    header = f"# {sr.file_path}" if sr.file_path else ""
                    context_parts.append(
                        f"{header}\n{sr.content}".strip()
                    )
                qr.contexts[engine.name] = "\n\n---\n\n".join(context_parts)
                qr.latencies[engine.name] = latency
            except Exception as e:
                qr.contexts[engine.name] = ""
                qr.latencies[engine.name] = 0
                print(f"\n  [!] {engine.name} search failed: {e}")

        # Step 2: Judge each engine's context
        for engine in engines:
            context = qr.contexts.get(engine.name, "")
            if not context:
                qr.scores[engine.name] = JudgeScore(0, 0, 0)
                continue

            try:
                score = await _call_llm_judge(
                    query=query_text,
                    context=context,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
                qr.scores[engine.name] = score
            except Exception as e:
                print(f"\n  [!] Judge failed for {engine.name}: {e}")
                qr.scores[engine.name] = JudgeScore(0, 0, 0)
                qr.error = str(e)

            # Small delay to avoid rate limits on judge LLM
            await asyncio.sleep(0.5)

        results.append(qr)

    print()  # newline after progress counter

    # Step 3: Aggregate
    engine_metrics: dict[str, JudgeAggregateMetrics] = {}
    for engine in engines:
        engine_results = [r for r in results if engine.name in r.scores]
        n = len(engine_results)
        if n == 0:
            engine_metrics[engine.name] = JudgeAggregateMetrics(engine=engine.name)
            continue

        agg = JudgeAggregateMetrics(
            engine=engine.name,
            num_queries=n,
            avg_relevance=sum(r.scores[engine.name].relevance for r in engine_results) / n,
            avg_completeness=sum(r.scores[engine.name].completeness for r in engine_results) / n,
            avg_specificity=sum(r.scores[engine.name].specificity for r in engine_results) / n,
            avg_total=sum(r.scores[engine.name].total for r in engine_results) / n,
            avg_latency_ms=sum(r.latencies.get(engine.name, 0) for r in engine_results) / n,
        )
        engine_metrics[engine.name] = agg

    # Compute win/tie counts
    for r in results:
        if len(r.scores) < 2:
            continue
        scores_by_engine = {eng: s.total for eng, s in r.scores.items()}
        max_score = max(scores_by_engine.values())
        winners = [eng for eng, s in scores_by_engine.items() if s == max_score]
        if len(winners) == 1:
            engine_metrics[winners[0]].win_count += 1
        else:
            for w in winners:
                engine_metrics[w].tie_count += 1

    return engine_metrics, results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_judge_summary(
    metrics: dict[str, JudgeAggregateMetrics],
    dataset_name: str,
) -> None:
    """Print a formatted judge benchmark summary."""
    print(f"  Dataset:           {dataset_name}")
    print(f"  Judge scoring:     relevance + completeness + specificity (0-3 each)")
    print()

    engines = list(metrics.keys())
    header = f"  {'Metric':<22}" + "".join(f"{e:>18}" for e in engines)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, attr in [
        ("Avg relevance", "avg_relevance"),
        ("Avg completeness", "avg_completeness"),
        ("Avg specificity", "avg_specificity"),
        ("Avg total (0-3)", "avg_total"),
        ("Avg latency (ms)", "avg_latency_ms"),
        ("Wins", "win_count"),
        ("Ties", "tie_count"),
        ("Queries", "num_queries"),
    ]:
        row = f"  {label:<22}"
        for eng in engines:
            val = getattr(metrics[eng], attr, 0)
            if attr == "avg_latency_ms":
                row += f"{val:>18.0f}"
            elif isinstance(val, int):
                row += f"{val:>18}"
            else:
                row += f"{val:>18.3f}"
        print(row)
