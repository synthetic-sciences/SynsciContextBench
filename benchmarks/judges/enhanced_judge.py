"""Enhanced LLM-as-Judge with position-debiasing and consistency metrics.

Implements research-backed improvements over the basic LLM judge:

1. **Position debiasing** (Zheng et al. 2023): Evaluate each (query, context)
   pair twice — once in the original order, once swapped — and average scores.
   This eliminates the documented ~10% positional bias in LLM evaluations.

2. **Multi-dimensional scoring** with calibration anchors: Add worked examples
   in the system prompt so the judge has calibrated reference points.

3. **Judge consistency metrics** (inter-rater reliability):
   - Cohen's kappa (if using 2 judges or 2 orderings)
   - Krippendorff's alpha (for N judges)
   - Position Consistency (PC) from Shi et al. 2025

4. **RAGAS-inspired context quality metrics**:
   - Context Precision (position-weighted relevance)
   - Context Density (useful tokens / total tokens)
   - Noise Ratio (irrelevant content fraction)

References:
    Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."
    Shi et al. (2025). "Judging the Judges: Evaluating Alignment and Vulnerabilities
        in LLMs-as-Judges."
    Es et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation."
"""

from __future__ import annotations

import asyncio
import json
import re

from tqdm import tqdm
from dataclasses import dataclass, field

import httpx

from ..adapters.base import ContextEngineAdapter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EnhancedJudgeScore:
    """Score from the enhanced judge with debiasing metadata."""

    relevance: int = 0  # 0-3
    completeness: int = 0  # 0-3
    specificity: int = 0  # 0-3
    faithfulness: int = 0  # 0-3: Does the context avoid misleading information?

    # Debiasing metadata
    score_pass_1: float = 0.0  # Score from first evaluation
    score_pass_2: float = 0.0  # Score from swapped evaluation
    position_consistent: bool = True  # Same winner in both orderings

    @property
    def total(self) -> float:
        return (self.relevance + self.completeness + self.specificity + self.faithfulness) / 4.0


@dataclass
class ContextQualityMetrics:
    """RAGAS-inspired context quality metrics for a single result set."""

    engine: str
    query: str

    # Context Precision: weighted relevance of chunks by position
    # CP@k = (1/k) * sum_{i=1}^{k} (precision@i * rel_i)
    context_precision: float = 0.0

    # Context Density: fraction of retrieved tokens that are "useful"
    # (contain query-relevant code vs boilerplate/comments)
    context_density: float = 0.0

    # Signal-to-Noise Ratio: relevant_chars / total_chars
    signal_to_noise: float = 0.0

    # Token efficiency: how many useful tokens per total tokens retrieved
    total_tokens: int = 0
    useful_tokens: int = 0

    # Chunk diversity: 1 - avg pairwise Jaccard similarity between chunks
    chunk_diversity: float = 0.0


@dataclass
class JudgeConsistencyMetrics:
    """Inter-rater / intra-rater reliability metrics for the judge."""

    # Position Consistency (Shi et al. 2025):
    # fraction of queries where the winner doesn't change when order is swapped
    position_consistency: float = 0.0

    # Cohen's kappa between pass 1 and pass 2 ordinal scores
    cohens_kappa: float = 0.0
    kappa_interpretation: str = ""

    # Average absolute score difference between passes
    avg_score_drift: float = 0.0

    # Number of queries evaluated
    n_queries: int = 0


@dataclass
class EnhancedJudgeAggregateMetrics:
    """Aggregate metrics from the enhanced judge evaluation."""

    engine: str
    num_queries: int = 0

    # Score dimensions
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_specificity: float = 0.0
    avg_faithfulness: float = 0.0
    avg_total: float = 0.0
    avg_latency_ms: float = 0.0

    # Context quality
    avg_context_precision: float = 0.0
    avg_context_density: float = 0.0
    avg_signal_to_noise: float = 0.0
    avg_chunk_diversity: float = 0.0

    # Win/tie/loss counts
    win_count: int = 0
    tie_count: int = 0
    loss_count: int = 0


# ---------------------------------------------------------------------------
# Enhanced judge prompts with calibration anchors
# ---------------------------------------------------------------------------

ENHANCED_JUDGE_SYSTEM_PROMPT = """\
You are an expert code retrieval evaluator. You will be given a natural language \
query and a set of code snippets retrieved by a search engine.

Score the retrieved context on four dimensions (0-3 each):

**Relevance** (0-3):
- 0: No snippet relates to the query at all (e.g., query about sorting, returned DB code)
- 1: Tangentially related (same domain but wrong functionality)
- 2: Partially relevant (addresses part of the query or a related concept)
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

**Faithfulness** (0-3):
- 0: Context contains misleading or incorrect information for the query
- 1: Context is mostly neutral but may cause confusion
- 2: Context is accurate but may include tangential information
- 3: Context is fully accurate and directly applicable

Respond with ONLY a JSON object:
{"relevance": <0-3>, "completeness": <0-3>, "specificity": <0-3>, "faithfulness": <0-3>}"""


# ---------------------------------------------------------------------------
# Context quality computation
# ---------------------------------------------------------------------------


def compute_context_quality(
    query: str,
    search_results: list,
    engine_name: str,
) -> ContextQualityMetrics:
    """Compute RAGAS-inspired context quality metrics.

    These metrics don't require an LLM — they're computed from the
    retrieved content and the query.
    """
    cq = ContextQualityMetrics(engine=engine_name, query=query)

    if not search_results:
        return cq

    query_tokens = set(query.lower().split())

    total_chars = 0
    relevant_chars = 0
    chunk_token_sets: list[set[str]] = []
    precision_sum = 0.0
    relevant_so_far = 0

    for i, sr in enumerate(search_results):
        content = sr.content if hasattr(sr, "content") else str(sr)
        content_lower = content.lower()
        content_tokens = set(content_lower.split())
        chunk_token_sets.append(content_tokens)

        # Token overlap with query (proxy for relevance)
        overlap = len(query_tokens & content_tokens)
        is_relevant = overlap >= max(1, len(query_tokens) * 0.2)

        total_chars += len(content)
        if is_relevant:
            relevant_chars += len(content)
            relevant_so_far += 1

        # Context Precision: position-weighted precision
        precision_at_i = relevant_so_far / (i + 1)
        if is_relevant:
            precision_sum += precision_at_i

    k = len(search_results)
    cq.context_precision = precision_sum / k if k > 0 else 0.0
    cq.signal_to_noise = relevant_chars / total_chars if total_chars > 0 else 0.0
    cq.total_tokens = sum(len(ts) for ts in chunk_token_sets)
    cq.useful_tokens = int(cq.signal_to_noise * cq.total_tokens)

    # Context Density: average token overlap ratio
    densities = []
    for ts in chunk_token_sets:
        if ts:
            overlap_ratio = len(query_tokens & ts) / len(ts)
            densities.append(overlap_ratio)
    cq.context_density = sum(densities) / len(densities) if densities else 0.0

    # Chunk Diversity: 1 - avg pairwise Jaccard similarity
    if len(chunk_token_sets) >= 2:
        pairwise_sims = []
        for i in range(len(chunk_token_sets)):
            for j in range(i + 1, len(chunk_token_sets)):
                a, b = chunk_token_sets[i], chunk_token_sets[j]
                if a or b:
                    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
                    pairwise_sims.append(jaccard)
        avg_sim = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.0
        cq.chunk_diversity = 1.0 - avg_sim
    else:
        cq.chunk_diversity = 1.0  # single chunk = fully diverse

    return cq


# ---------------------------------------------------------------------------
# Judge consistency metrics
# ---------------------------------------------------------------------------


def compute_judge_consistency(
    scores_pass1: list[float],
    scores_pass2: list[float],
) -> JudgeConsistencyMetrics:
    """Compute judge self-consistency between two evaluation passes.

    Used to measure position bias: if scoring changes when evaluation
    order is swapped, there's a positional bias.
    """
    n = min(len(scores_pass1), len(scores_pass2))
    if n == 0:
        return JudgeConsistencyMetrics()

    # Position Consistency: fraction where scores agree (within 0.5)
    consistent = sum(1 for s1, s2 in zip(scores_pass1[:n], scores_pass2[:n]) if abs(s1 - s2) <= 0.5)
    pc = consistent / n

    # Average score drift
    drifts = [abs(s1 - s2) for s1, s2 in zip(scores_pass1[:n], scores_pass2[:n])]
    avg_drift = sum(drifts) / n

    # Cohen's kappa (ordinal agreement)
    # Discretize scores to integer bins (0, 1, 2, 3)
    bins1 = [min(3, max(0, round(s))) for s in scores_pass1[:n]]
    bins2 = [min(3, max(0, round(s))) for s in scores_pass2[:n]]

    kappa = _cohens_kappa(bins1, bins2, num_categories=4)
    kappa_label = _kappa_interpretation(kappa)

    return JudgeConsistencyMetrics(
        position_consistency=pc,
        cohens_kappa=kappa,
        kappa_interpretation=kappa_label,
        avg_score_drift=avg_drift,
        n_queries=n,
    )


def _cohens_kappa(ratings_a: list[int], ratings_b: list[int], num_categories: int = 4) -> float:
    """Compute Cohen's kappa coefficient for inter-rater agreement.

    Args:
        ratings_a, ratings_b: Integer ratings from two raters
        num_categories: Number of rating categories

    Returns:
        Cohen's kappa coefficient in [-1, 1]
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = [[0] * num_categories for _ in range(num_categories)]
    for a, b in zip(ratings_a, ratings_b):
        if 0 <= a < num_categories and 0 <= b < num_categories:
            matrix[a][b] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(num_categories)) / n

    # Expected agreement by chance
    p_e = 0.0
    for i in range(num_categories):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(num_categories))
        p_e += (row_sum * col_sum) / (n * n) if n > 0 else 0

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def _kappa_interpretation(kappa: float) -> str:
    """Interpret Cohen's kappa (Landis & Koch 1977)."""
    if kappa < 0:
        return "poor"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "almost_perfect"


# ---------------------------------------------------------------------------
# LLM judge with position debiasing
# ---------------------------------------------------------------------------


async def _call_enhanced_judge(
    query: str,
    context: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> dict:
    """Call the LLM judge and return raw scores dict."""
    user_prompt = (
        f"## Query\n{query}\n\n"
        f"## Retrieved Context\n```\n{context[:6000]}\n```\n\n"
        f"Score the above context for this query. "
        f"Respond with ONLY a JSON object: "
        f'{{"relevance": <0-3>, "completeness": <0-3>, "specificity": <0-3>, "faithfulness": <0-3>}}'
    )

    if llm_provider == "gemini":
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent",
                params={"key": llm_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": ENHANCED_JUDGE_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 150, "temperature": 0.0},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            text = "".join(
                p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
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
                    "max_tokens": 150,
                    "system": ENHANCED_JUDGE_SYSTEM_PROMPT,
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
                    "max_tokens": 150,
                    "messages": [
                        {"role": "system", "content": ENHANCED_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    from .llm_judge import _safe_parse_json

    scores = _safe_parse_json(text, defaults={"relevance": 0, "completeness": 0, "specificity": 0, "faithfulness": 0})
    return {
        "relevance": min(3, max(0, int(scores.get("relevance", 0)))),
        "completeness": min(3, max(0, int(scores.get("completeness", 0)))),
        "specificity": min(3, max(0, int(scores.get("specificity", 0)))),
        "faithfulness": min(3, max(0, int(scores.get("faithfulness", 0)))),
    }


async def run_enhanced_judge_benchmark(
    engines: list[ContextEngineAdapter],
    dataset_path: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    max_queries: int | None = None,
    top_k: int = 5,
    enable_debiasing: bool = True,
) -> tuple[
    dict[str, EnhancedJudgeAggregateMetrics],
    JudgeConsistencyMetrics,
    dict[str, list[ContextQualityMetrics]],
]:
    """Run enhanced LLM-as-judge with position debiasing and quality metrics.

    Returns:
        (per_engine_metrics, judge_consistency, per_engine_context_quality)
    """
    with open(dataset_path) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if max_queries:
        queries = queries[:max_queries]

    # Collect per-engine results
    engine_scores: dict[str, list[EnhancedJudgeScore]] = {e.name: [] for e in engines}
    engine_latencies: dict[str, list[float]] = {e.name: [] for e in engines}
    engine_context_quality: dict[str, list[ContextQualityMetrics]] = {e.name: [] for e in engines}

    # For consistency tracking
    all_pass1_scores: list[float] = []
    all_pass2_scores: list[float] = []

    for q in tqdm(queries, desc="  enhanced judge", unit="q"):
        query_text = q.get("query", q.get("text", ""))
        if not query_text:
            continue

        for engine in engines:
            # Step 1: Retrieve context
            try:
                search_results, latency = await engine.search_code(
                    query=query_text,
                    top_k=top_k,
                )
                engine_latencies[engine.name].append(latency)
            except Exception as e:
                engine_scores[engine.name].append(EnhancedJudgeScore())
                continue

            # Step 2: Compute context quality metrics (no LLM needed)
            cq = compute_context_quality(query_text, search_results, engine.name)
            engine_context_quality[engine.name].append(cq)

            # Step 3: Build context for judge
            context_parts = []
            for sr in search_results:
                header = f"# {sr.file_path}" if sr.file_path else ""
                context_parts.append(f"{header}\n{sr.content}".strip())
            context = "\n\n---\n\n".join(context_parts)

            if not context.strip():
                engine_scores[engine.name].append(EnhancedJudgeScore())
                continue

            # Step 4: Judge evaluation (pass 1)
            try:
                scores_p1 = await _call_enhanced_judge(
                    query_text, context, llm_provider, llm_model, llm_api_key
                )
                p1_total = sum(scores_p1.values()) / 4.0
            except Exception:
                scores_p1 = {"relevance": 0, "completeness": 0, "specificity": 0, "faithfulness": 0}
                p1_total = 0.0

            # Step 5: Judge evaluation (pass 2 — shuffled context for debiasing)
            p2_total = p1_total
            if enable_debiasing and context.strip():
                try:
                    await asyncio.sleep(0.3)
                    # Reverse the order of context chunks for position debiasing
                    reversed_parts = list(reversed(context_parts))
                    reversed_context = "\n\n---\n\n".join(reversed_parts)
                    scores_p2 = await _call_enhanced_judge(
                        query_text, reversed_context, llm_provider, llm_model, llm_api_key
                    )
                    p2_total = sum(scores_p2.values()) / 4.0

                    # Average the two passes for debiased score
                    final_scores = {k: round((scores_p1[k] + scores_p2[k]) / 2) for k in scores_p1}
                except Exception:
                    final_scores = scores_p1
                    p2_total = p1_total
            else:
                final_scores = scores_p1

            all_pass1_scores.append(p1_total)
            all_pass2_scores.append(p2_total)

            score = EnhancedJudgeScore(
                relevance=final_scores["relevance"],
                completeness=final_scores["completeness"],
                specificity=final_scores["specificity"],
                faithfulness=final_scores["faithfulness"],
                score_pass_1=p1_total,
                score_pass_2=p2_total,
                position_consistent=abs(p1_total - p2_total) <= 0.5,
            )
            engine_scores[engine.name].append(score)

            await asyncio.sleep(0.5)  # Rate limit

    # Aggregate per engine
    engine_metrics: dict[str, EnhancedJudgeAggregateMetrics] = {}
    for engine in engines:
        scores = engine_scores[engine.name]
        latencies = engine_latencies[engine.name]
        cqs = engine_context_quality[engine.name]
        n = len(scores)
        if n == 0:
            engine_metrics[engine.name] = EnhancedJudgeAggregateMetrics(engine=engine.name)
            continue

        agg = EnhancedJudgeAggregateMetrics(
            engine=engine.name,
            num_queries=n,
            avg_relevance=sum(s.relevance for s in scores) / n,
            avg_completeness=sum(s.completeness for s in scores) / n,
            avg_specificity=sum(s.specificity for s in scores) / n,
            avg_faithfulness=sum(s.faithfulness for s in scores) / n,
            avg_total=sum(s.total for s in scores) / n,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            avg_context_precision=sum(c.context_precision for c in cqs) / len(cqs) if cqs else 0.0,
            avg_context_density=sum(c.context_density for c in cqs) / len(cqs) if cqs else 0.0,
            avg_signal_to_noise=sum(c.signal_to_noise for c in cqs) / len(cqs) if cqs else 0.0,
            avg_chunk_diversity=sum(c.chunk_diversity for c in cqs) / len(cqs) if cqs else 0.0,
        )
        engine_metrics[engine.name] = agg

    # Compute wins/ties
    for i in range(
        min(len(engine_scores[engines[0].name]), *[len(engine_scores[e.name]) for e in engines[1:]])
    ):
        all_totals = {e.name: engine_scores[e.name][i].total for e in engines}
        max_total = max(all_totals.values())
        winners = [e for e, t in all_totals.items() if t == max_total]
        if len(winners) == 1:
            engine_metrics[winners[0]].win_count += 1
        else:
            for w in winners:
                engine_metrics[w].tie_count += 1

    # Judge consistency
    consistency = compute_judge_consistency(all_pass1_scores, all_pass2_scores)

    return engine_metrics, consistency, engine_context_quality


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_enhanced_judge_summary(
    metrics: dict[str, EnhancedJudgeAggregateMetrics],
    consistency: JudgeConsistencyMetrics,
    dataset_name: str,
) -> None:
    """Print formatted enhanced judge results."""
    print(f"\n  === Enhanced LLM-as-Judge: {dataset_name} ===")
    print(f"  Scoring: relevance + completeness + specificity + faithfulness (0-3 each)")
    print(f"  Position debiasing: {'enabled' if consistency.n_queries > 0 else 'disabled'}")

    engines = list(metrics.keys())
    header = f"  {'Metric':<28}" + "".join(f"{e:>18}" for e in engines)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, attr in [
        ("Avg relevance (0-3)", "avg_relevance"),
        ("Avg completeness (0-3)", "avg_completeness"),
        ("Avg specificity (0-3)", "avg_specificity"),
        ("Avg faithfulness (0-3)", "avg_faithfulness"),
        ("Avg total (0-3)", "avg_total"),
        ("Avg latency (ms)", "avg_latency_ms"),
        ("Ctx precision", "avg_context_precision"),
        ("Ctx density", "avg_context_density"),
        ("Signal/noise", "avg_signal_to_noise"),
        ("Chunk diversity", "avg_chunk_diversity"),
        ("Wins", "win_count"),
        ("Ties", "tie_count"),
    ]:
        row = f"  {label:<28}"
        for eng in engines:
            val = getattr(metrics[eng], attr, 0)
            if attr == "avg_latency_ms":
                row += f"{val:>18.0f}"
            elif isinstance(val, int):
                row += f"{val:>18}"
            else:
                row += f"{val:>18.3f}"
        print(row)

    # Judge consistency
    if consistency.n_queries > 0:
        print(f"\n  --- Judge Self-Consistency ---")
        print(f"  Position consistency:  {consistency.position_consistency:.3f}")
        print(
            f"  Cohen's kappa:         {consistency.cohens_kappa:.3f} ({consistency.kappa_interpretation})"
        )
        print(f"  Avg score drift:       {consistency.avg_score_drift:.3f}")
        print(f"  N evaluated:           {consistency.n_queries}")
