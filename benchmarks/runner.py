"""Benchmark runner — orchestrates all benchmarks.

Runs both engines on the same queries, computes metrics, and produces
a comparison report.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .adapters.base import ContextEngineAdapter, SearchResult
from .adversarial import (
    AdversarialAggregateMetrics,
    run_adversarial_benchmark,
)
from .code_qa import (
    CodeQAAggregateMetrics,
    run_code_qa_benchmark,
)
from .config import BenchmarkConfig, LLMModelConfig
from .hallucination import (
    HallucinationBenchmarkResult,
    load_test_cases,
    run_hallucination_benchmark,
)
from .llm_judge import (
    JudgeAggregateMetrics,
    print_judge_summary,
    run_judge_benchmark,
)
from .metrics import (
    AggregateMetrics,
    QueryEvaluation,
    RetrievalResult,
    aggregate,
    aggregate_hallucinations,
    evaluate_query,
)
from .multihop import (
    MultiHopAggregateMetrics,
    run_multihop_benchmark,
)
from .validated_eval import (
    MatchMode,
    print_validated_summary,
    run_validated_benchmark,
)
from .statistical_analysis import (
    bootstrap_ci,
    print_significance_summary,
    run_pairwise_significance,
)
from .enhanced_judge import (
    print_enhanced_judge_summary,
    run_enhanced_judge_benchmark,
)
from .semantic_metrics import (
    ExtendedRetrievalMetrics,
    compute_extended_metrics,
    print_extended_metrics_summary,
)


@dataclass
class BenchmarkReport:
    """Full benchmark comparison report."""

    timestamp: str = ""
    engines: list[str] = field(default_factory=list)

    # Retrieval metrics per engine
    retrieval: dict[str, dict] = field(default_factory=dict)

    # Hallucination metrics per engine
    hallucination: dict[str, dict] = field(default_factory=dict)

    # Indexing performance per engine
    indexing: dict[str, dict] = field(default_factory=dict)

    # Multi-hop retrieval per engine
    multihop: dict[str, dict] = field(default_factory=dict)

    # Code QA per engine
    code_qa: dict[str, dict] = field(default_factory=dict)

    # Adversarial near-miss per engine
    adversarial: dict[str, dict] = field(default_factory=dict)

    # Validated datasets (CodeSearchNet, CoSQA) per engine
    validated: dict[str, dict] = field(default_factory=dict)

    # LLM-as-judge per engine
    judge: dict[str, dict] = field(default_factory=dict)

    # Enhanced judge (position-debiased, with context quality) per engine
    enhanced_judge: dict[str, dict] = field(default_factory=dict)

    # Extended retrieval metrics (MAP, Success@K, R-Precision, CodeBLEU)
    extended_metrics: dict[str, dict] = field(default_factory=dict)

    # Statistical significance analysis
    significance: dict[str, dict] = field(default_factory=dict)

    # Raw query-level results
    query_results: dict[str, list[dict]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval benchmark
# ---------------------------------------------------------------------------


def _match_relevance(
    result: SearchResult,
    relevant_files: list[str],
    relevant_keywords: list[str],
) -> tuple[bool, int]:
    """Determine if a search result is relevant based on ground truth.

    Returns (is_relevant, relevance_grade 0-2).
    """
    score = 0

    # File path match
    for rf in relevant_files:
        if rf in result.file_path:
            score += 1
            break

    # Keyword match
    content_lower = result.content.lower()
    for kw in relevant_keywords:
        if kw.lower() in content_lower:
            score += 1
            break

    is_relevant = score > 0
    grade = min(score, 2)
    return is_relevant, grade


async def run_retrieval_benchmark(
    engine: ContextEngineAdapter,
    ground_truth_path: str,
    k_values: list[int],
) -> tuple[AggregateMetrics, list[QueryEvaluation]]:
    """Run retrieval quality benchmark against one engine."""
    with open(ground_truth_path) as f:
        data = json.load(f)

    evaluations: list[QueryEvaluation] = []

    queries = data.get("queries", [])
    for query_data in tqdm(queries, desc=f"  {engine.name} retrieval", unit="q"):
        query = query_data["query"]
        relevant_files = query_data.get("relevant_files", [])
        relevant_keywords = query_data.get("relevant_keywords", [])
        total_relevant = query_data.get("total_relevant", len(relevant_files))

        try:
            search_results, latency = await engine.search_code(query=query, top_k=max(k_values))
        except Exception as e:
            print(f"  [!] Query failed for {engine.name}: {query[:50]}... — {e}")
            continue

        # Convert to RetrievalResults with relevance judgments
        retrieval_results = []
        for sr in search_results:
            is_rel, grade = _match_relevance(sr, relevant_files, relevant_keywords)
            retrieval_results.append(
                RetrievalResult(
                    id=sr.id,
                    score=sr.score,
                    content=sr.content[:200],  # truncate for report
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

    agg = aggregate(evaluations, k_values)
    return agg, evaluations


# ---------------------------------------------------------------------------
# Indexing benchmark
# ---------------------------------------------------------------------------


async def run_indexing_benchmark(
    engine: ContextEngineAdapter,
    repos: list[str],
    papers: list[str] | None = None,
) -> dict:
    """Measure indexing performance."""
    results = {"repos": [], "papers": []}

    for repo_url in repos:
        print(f"  Indexing repo: {repo_url} ({engine.name})")
        idx = await engine.index_repository(repo_url)
        results["repos"].append(
            {
                "url": repo_url,
                "success": idx.success,
                "duration_ms": idx.duration_ms,
                "resource_id": idx.resource_id,
                "error": idx.error,
            }
        )

    for arxiv_id in papers or []:
        print(f"  Indexing paper: {arxiv_id} ({engine.name})")
        idx = await engine.index_paper(arxiv_id)
        results["papers"].append(
            {
                "arxiv_id": arxiv_id,
                "success": idx.success,
                "duration_ms": idx.duration_ms,
                "resource_id": idx.resource_id,
                "error": idx.error,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


async def run_full_benchmark(
    engines: list[ContextEngineAdapter],
    config: BenchmarkConfig,
    skip_indexing: bool = False,
    skip_retrieval: bool = False,
    skip_hallucination: bool = False,
    skip_multihop: bool = False,
    skip_code_qa: bool = False,
    skip_adversarial: bool = False,
    skip_validated: bool = False,
    skip_judge: bool = False,
    skip_enhanced_judge: bool = False,
    dataset_filter: list[str] | None = None,
    match_mode: MatchMode = "hybrid",
    enable_debiasing: bool = True,
) -> BenchmarkReport:
    """Run all benchmarks across all engines and produce a comparison report."""
    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        engines=[e.name for e in engines],
    )

    gt_path = str(config.datasets_dir / "retrieval_ground_truth.json")
    hal_path = str(config.datasets_dir / "hallucination_test_cases.json")
    mh_path = str(config.datasets_dir / "multihop_test_cases.json")
    cq_path = str(config.datasets_dir / "code_qa_test_cases.json")
    adv_path = str(config.datasets_dir / "adversarial_test_cases.json")

    # --- Indexing ---
    if not skip_indexing:
        print("\n=== Indexing Benchmark ===")
        with open(gt_path) as f:
            gt_data = json.load(f)
        repos = [r["url"] for r in gt_data.get("test_repos", [])]

        for engine in engines:
            print(f"\n--- {engine.name} ---")
            idx_results = await run_indexing_benchmark(engine, repos)
            report.indexing[engine.name] = idx_results
            print(f"  Waiting 10s for indexing to complete...")
            await asyncio.sleep(10)

    # --- Retrieval ---
    if not skip_retrieval:
        print("\n=== Retrieval Quality Benchmark (Precision@K / NDCG / MRR) ===")
        print(f"  Running {len(engines)} engines concurrently...")

        async def _retrieval_task(eng: ContextEngineAdapter):
            return eng, await run_retrieval_benchmark(eng, gt_path, config.top_k_values)

        retrieval_results = await asyncio.gather(
            *[_retrieval_task(e) for e in engines], return_exceptions=True
        )
        for result in retrieval_results:
            if isinstance(result, Exception):
                print(f"\n  [!] Retrieval failed: {result}")
                continue
            engine, (agg, evals) = result
            print(f"\n--- {engine.name} ---")
            report.retrieval[engine.name] = asdict(agg)
            report.query_results[engine.name] = [
                {
                    "query": e.query,
                    "mrr": e.mrr,
                    "precision_at": e.precision_at,
                    "ndcg_at": e.ndcg_at,
                    "latency_ms": e.latency_ms,
                    "num_results": len(e.results),
                }
                for e in evals
            ]
            _print_retrieval_summary(agg)

    # --- Multi-hop Retrieval ---
    if not skip_multihop:
        print("\n=== Multi-Hop Retrieval Benchmark ===")
        print(f"  Running {len(engines)} engines concurrently...")

        async def _multihop_task(eng: ContextEngineAdapter):
            return eng, await run_multihop_benchmark(eng, mh_path)

        multihop_results = await asyncio.gather(
            *[_multihop_task(e) for e in engines], return_exceptions=True
        )
        for result in multihop_results:
            if isinstance(result, Exception):
                print(f"\n  [!] Multi-hop failed: {result}")
                continue
            engine, (mh_agg, mh_evals) = result
            print(f"\n--- {engine.name} ---")
            report.multihop[engine.name] = asdict(mh_agg)
            _print_multihop_summary(mh_agg)

    # --- Code QA ---
    if not skip_code_qa:
        print("\n=== Code-Specific QA Benchmark ===")
        print(f"  Running {len(engines)} engines concurrently...")

        async def _code_qa_task(eng: ContextEngineAdapter):
            return eng, await run_code_qa_benchmark(eng, cq_path)

        code_qa_results = await asyncio.gather(
            *[_code_qa_task(e) for e in engines], return_exceptions=True
        )
        for result in code_qa_results:
            if isinstance(result, Exception):
                print(f"\n  [!] Code QA failed: {result}")
                continue
            engine, (cq_agg, cq_results) = result
            print(f"\n--- {engine.name} ---")
            report.code_qa[engine.name] = asdict(cq_agg)
            _print_code_qa_summary(cq_agg)

    # --- Adversarial Near-miss ---
    if not skip_adversarial:
        print("\n=== Adversarial Near-Miss Benchmark ===")
        print(f"  Running {len(engines)} engines concurrently...")

        async def _adversarial_task(eng: ContextEngineAdapter):
            return eng, await run_adversarial_benchmark(eng, adv_path)

        adversarial_results = await asyncio.gather(
            *[_adversarial_task(e) for e in engines], return_exceptions=True
        )
        for result in adversarial_results:
            if isinstance(result, Exception):
                print(f"\n  [!] Adversarial failed: {result}")
                continue
            engine, (adv_agg, adv_results) = result
            print(f"\n--- {engine.name} ---")
            report.adversarial[engine.name] = asdict(adv_agg)
            _print_adversarial_summary(adv_agg)

    # --- Hallucination ---
    if not skip_hallucination:
        print("\n=== Hallucination Rate Benchmark ===")

        # Build model list: multi-model matrix if available, else legacy single model
        models_to_test: list[LLMModelConfig] = []
        if config.model_matrix:
            models_to_test = config.model_matrix
        elif config.llm_api_key:
            models_to_test = [
                LLMModelConfig(
                    provider=config.llm_provider,
                    model=config.llm_model,
                    tier="default",
                    api_key=config.llm_api_key,
                )
            ]

        if not models_to_test:
            print("  [!] Skipping — no LLM API keys configured")
        else:
            test_cases = load_test_cases(hal_path)
            print(f"  Models to test: {len(models_to_test)}")
            for m in models_to_test:
                print(f"    - {m.display_name} ({m.tier})")

            print(f"  Running {len(engines)} engines concurrently per model...")

            async def _hallucination_task(eng, mcfg):
                result = await run_hallucination_benchmark(
                    engine=eng,
                    test_cases=test_cases,
                    llm_provider=mcfg.provider,
                    llm_model=mcfg.model,
                    llm_api_key=mcfg.api_key,
                )
                return eng, mcfg, result

            for model_cfg in models_to_test:
                hal_tasks = await asyncio.gather(
                    *[_hallucination_task(e, model_cfg) for e in engines],
                    return_exceptions=True,
                )
                for task_result in hal_tasks:
                    if isinstance(task_result, Exception):
                        print(f"\n  [!] Hallucination failed: {task_result}")
                        continue
                    engine, mcfg, hal_result = task_result
                    label = f"{engine.name} + {mcfg.display_name} ({mcfg.tier})"
                    print(f"\n--- {label} ---")
                    report_key = f"{engine.name}__{mcfg.provider}_{mcfg.tier}"
                    report.hallucination[report_key] = {
                        "engine": engine.name,
                        "llm_provider": mcfg.provider,
                        "llm_model": mcfg.model,
                        "llm_tier": mcfg.tier,
                        "overall_rate": hal_result.overall_rate,
                        "true_hallucination_rate": hal_result.true_hallucination_rate,
                        "abstention_rate": hal_result.abstention_rate,
                        "context_miss_rate": hal_result.context_miss_rate,
                        "error_breakdown": hal_result.error_breakdown,
                        "num_cases": len(hal_result.test_cases),
                        "cases": [
                            {
                                "query": tc.query,
                                "hallucination_rate": tc.hallucination_rate,
                                "errors": tc.errors,
                                "code_snippet": tc.generated_code[:500],
                            }
                            for tc in hal_result.test_cases
                        ],
                    }
                    _print_hallucination_summary(hal_result)

    # --- Validated Datasets (CodeSearchNet, CoSQA) ---
    if not skip_validated:
        all_validated_datasets = [
            ("codesearchnet_benchmark.json", "CodeSearchNet", "codesearchnet"),
            ("cosqa_benchmark.json", "CoSQA", "cosqa"),
            (
                "codesearchnet_challenge_benchmark.json",
                "CodeSearchNet Challenge",
                "codesearchnet_challenge",
            ),
        ]
        # Apply dataset filter if specified
        validated_datasets = [
            (f, d, k)
            for f, d, k in all_validated_datasets
            if dataset_filter is None or k in dataset_filter
        ]
        found_any = False
        for filename, display_name, _key in validated_datasets:
            ds_path = config.datasets_dir / filename
            if not ds_path.exists():
                continue
            if not found_any:
                print("\n=== Validated Dataset Benchmarks ===")
                found_any = True

            print(f"\n--- {display_name} (running {len(engines)} engines concurrently) ---")

            async def _validated_task(eng, ds, kvals, mq, mm):
                return eng, await run_validated_benchmark(
                    eng,
                    ds,
                    kvals,
                    max_queries=mq,
                    match_mode=mm,
                )

            val_results = await asyncio.gather(
                *[
                    _validated_task(
                        e, str(ds_path), config.top_k_values, config.max_queries, match_mode
                    )
                    for e in engines
                ],
                return_exceptions=True,
            )
            for val_result in val_results:
                if isinstance(val_result, Exception):
                    print(f"\n  [!] Failed: {val_result}")
                    continue
                engine, (val_agg, val_evals) = val_result
                print(f"\n  [{engine.name}]")
                engine_key = f"{engine.name}_{filename.replace('.json', '')}"
                report.validated[engine_key] = {
                    "dataset": display_name,
                    "engine": engine.name,
                    "match_mode": match_mode,
                    **asdict(val_agg),
                }
                print_validated_summary(display_name, val_agg, match_mode)

        if not found_any:
            print(
                "\n  [!] No validated datasets found. Run: python -m benchmarks --download-datasets"
            )

    # --- LLM-as-Judge ---
    if not skip_judge:
        all_judge_datasets = [
            ("codesearchnet_benchmark.json", "CodeSearchNet", "codesearchnet"),
            ("cosqa_benchmark.json", "CoSQA", "cosqa"),
        ]
        judge_datasets = [
            (f, d, k)
            for f, d, k in all_judge_datasets
            if dataset_filter is None or k in dataset_filter
        ]

        # Determine judge LLM config
        judge_model = None
        if config.model_matrix:
            # Use lowest-tier model for judging (cost-effective)
            judge_model = config.model_matrix[0]
        elif config.llm_api_key:
            judge_model = LLMModelConfig(
                provider=config.llm_provider,
                model=config.llm_model,
                tier="default",
                api_key=config.llm_api_key,
            )

        if not judge_model:
            print("\n  [!] Skipping LLM judge — no LLM API keys configured")
        else:
            found_any_judge = False
            for filename, display_name, _key in judge_datasets:
                ds_path = config.datasets_dir / filename
                if not ds_path.exists():
                    continue
                if not found_any_judge:
                    print(f"\n=== LLM-as-Judge Benchmark (judge: {judge_model.display_name}) ===")
                    found_any_judge = True

                print(f"\n--- {display_name} ---")
                try:
                    judge_metrics, judge_results = await run_judge_benchmark(
                        engines=engines,
                        dataset_path=str(ds_path),
                        llm_provider=judge_model.provider,
                        llm_model=judge_model.model,
                        llm_api_key=judge_model.api_key,
                        max_queries=config.max_queries,
                    )
                    for eng_name, agg in judge_metrics.items():
                        report_key = f"{eng_name}_{_key}"
                        report.judge[report_key] = {
                            "dataset": display_name,
                            "engine": eng_name,
                            "judge_model": judge_model.display_name,
                            "num_queries": agg.num_queries,
                            "avg_relevance": agg.avg_relevance,
                            "avg_completeness": agg.avg_completeness,
                            "avg_specificity": agg.avg_specificity,
                            "avg_total": agg.avg_total,
                            "avg_latency_ms": agg.avg_latency_ms,
                            "win_count": agg.win_count,
                            "tie_count": agg.tie_count,
                        }
                    print_judge_summary(judge_metrics, display_name)
                except Exception as e:
                    print(f"  [!] Judge benchmark failed: {e}")
                    import traceback

                    traceback.print_exc()

            if not found_any_judge:
                print("\n  [!] No datasets found for judge benchmark")

    # --- Enhanced LLM-as-Judge (position-debiased, 4D scoring) ---
    if not skip_enhanced_judge:
        all_enh_datasets = [
            ("codesearchnet_benchmark.json", "CodeSearchNet", "codesearchnet"),
            ("cosqa_benchmark.json", "CoSQA", "cosqa"),
        ]
        enh_datasets = [
            (f, d, k)
            for f, d, k in all_enh_datasets
            if dataset_filter is None or k in dataset_filter
        ]

        # Determine judge LLM config
        enh_judge_model = None
        if config.model_matrix:
            enh_judge_model = config.model_matrix[0]
        elif config.llm_api_key:
            enh_judge_model = LLMModelConfig(
                provider=config.llm_provider,
                model=config.llm_model,
                tier="default",
                api_key=config.llm_api_key,
            )

        if not enh_judge_model:
            print("\n  [!] Skipping enhanced judge — no LLM API keys configured")
        else:
            # Filter to datasets that exist on disk
            valid_enh = [
                (f, d, k)
                for f, d, k in enh_datasets
                if (config.datasets_dir / f).exists()
            ]

            if not valid_enh:
                print(
                    "\n  [!] No datasets found for enhanced judge. "
                    "Run: python -m benchmarks --download-datasets"
                )
            else:
                print(
                    f"\n=== Enhanced LLM-as-Judge (judge: {enh_judge_model.display_name}, "
                    f"debiasing: {enable_debiasing}) ==="
                )
                print(
                    f"  Running {len(valid_enh)} dataset(s) concurrently: "
                    f"{', '.join(d for _, d, _ in valid_enh)}"
                )

                async def _run_enh_dataset(filename, display_name, _key):
                    ds_path = config.datasets_dir / filename
                    return _key, display_name, await run_enhanced_judge_benchmark(
                        engines=engines,
                        dataset_path=str(ds_path),
                        llm_provider=enh_judge_model.provider,
                        llm_model=enh_judge_model.model,
                        llm_api_key=enh_judge_model.api_key,
                        max_queries=config.max_queries,
                        enable_debiasing=enable_debiasing,
                    )

                enh_tasks = await asyncio.gather(
                    *[_run_enh_dataset(f, d, k) for f, d, k in valid_enh],
                    return_exceptions=True,
                )

                for task_result in enh_tasks:
                    if isinstance(task_result, Exception):
                        print(f"\n  [!] Enhanced judge failed: {task_result}")
                        import traceback

                        traceback.print_exc()
                        continue

                    _key, display_name, (enh_metrics, consistency, ctx_quality) = task_result

                    for eng_name, agg in enh_metrics.items():
                        report_key = f"{eng_name}_{_key}"
                        report.enhanced_judge[report_key] = {
                            "dataset": display_name,
                            "engine": eng_name,
                            "judge_model": enh_judge_model.display_name,
                            "debiasing_enabled": enable_debiasing,
                            "num_queries": agg.num_queries,
                            "avg_relevance": agg.avg_relevance,
                            "avg_completeness": agg.avg_completeness,
                            "avg_specificity": agg.avg_specificity,
                            "avg_faithfulness": agg.avg_faithfulness,
                            "avg_total": agg.avg_total,
                            "avg_latency_ms": agg.avg_latency_ms,
                            "avg_context_precision": agg.avg_context_precision,
                            "avg_context_density": agg.avg_context_density,
                            "avg_signal_to_noise": agg.avg_signal_to_noise,
                            "avg_chunk_diversity": agg.avg_chunk_diversity,
                            "win_count": agg.win_count,
                            "tie_count": agg.tie_count,
                        }
                    report.enhanced_judge[f"_consistency_{_key}"] = {
                        "cohens_kappa": consistency.cohens_kappa,
                        "position_consistency": consistency.position_consistency,
                        "avg_score_drift": consistency.avg_score_drift,
                        "n_queries": consistency.n_queries,
                    }
                    print_enhanced_judge_summary(enh_metrics, consistency, display_name)

    # --- Statistical Significance Analysis ---
    # Collect per-query scores for pairwise significance testing
    if len(engines) >= 2 and report.query_results:
        print("\n=== Statistical Significance Analysis ===")
        print(f"  (Paired t-test + Wilcoxon, Holm correction)")

        # Build per-query score maps: {metric: {engine: [scores]}}
        per_query_scores: dict[str, dict[str, list[float]]] = {}

        for eng_name, qrs in report.query_results.items():
            for qr in qrs:
                for metric_name in ("mrr",):
                    per_query_scores.setdefault(metric_name, {})
                    per_query_scores[metric_name].setdefault(eng_name, [])
                    per_query_scores[metric_name][eng_name].append(qr.get(metric_name, 0))

                # Precision@K and NDCG@K
                for k, v in qr.get("precision_at", {}).items():
                    mk = f"precision_at_{k}"
                    per_query_scores.setdefault(mk, {})
                    per_query_scores[mk].setdefault(eng_name, [])
                    per_query_scores[mk][eng_name].append(v)

                for k, v in qr.get("ndcg_at", {}).items():
                    mk = f"ndcg_at_{k}"
                    per_query_scores.setdefault(mk, {})
                    per_query_scores[mk].setdefault(eng_name, [])
                    per_query_scores[mk][eng_name].append(v)

        if per_query_scores:
            try:
                tests, cis, corrected = run_pairwise_significance(per_query_scores)
                print_significance_summary(tests, cis, corrected)

                # Store in report
                report.significance = {
                    "correction_method": corrected.correction_method,
                    "num_comparisons": corrected.num_comparisons,
                    "corrected_alpha": corrected.corrected_alpha,
                    "significant_pairs": [
                        {"metric": m, "engine_a": a, "engine_b": b}
                        for m, a, b in corrected.significant_pairs
                    ],
                    "tests": [
                        {
                            "test": t.test_name,
                            "metric": t.metric,
                            "engine_a": t.engine_a,
                            "engine_b": t.engine_b,
                            "mean_diff": t.mean_diff,
                            "p_value": t.p_value,
                            "effect_size": t.effect_size,
                            "effect_size_label": t.effect_size_label,
                            "ci_lower": t.ci_lower,
                            "ci_upper": t.ci_upper,
                            "significant": t.significant,
                        }
                        for t in tests
                    ],
                    "bootstrap_cis": [
                        {
                            "metric": ci.metric,
                            "engine": ci.engine,
                            "point_estimate": ci.point_estimate,
                            "ci_lower": ci.ci_lower,
                            "ci_upper": ci.ci_upper,
                            "std_error": ci.std_error,
                        }
                        for ci in cis
                    ],
                }
            except Exception as e:
                print(f"  [!] Significance analysis failed: {e}")

    # --- Save report ---
    report_path = config.results_dir / f"benchmark_{int(time.time())}.json"
    config.results_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n=== Report saved to {report_path} ===")

    # Print comparison table
    _print_comparison(report)

    return report


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _print_retrieval_summary(agg: AggregateMetrics) -> None:
    print(f"  Queries evaluated: {agg.num_queries}")
    print(f"  MRR:               {agg.avg_mrr:.3f}")
    for k, v in sorted(agg.avg_precision_at.items()):
        print(f"  Precision@{k}:      {v:.3f}")
    for k, v in sorted(agg.avg_ndcg_at.items()):
        print(f"  NDCG@{k}:           {v:.3f}")
    print(f"  Avg latency:       {agg.avg_latency_ms:.0f}ms")
    print(f"  P95 latency:       {agg.p95_latency_ms:.0f}ms")


def _print_multihop_summary(agg: MultiHopAggregateMetrics) -> None:
    print(f"  Queries evaluated: {agg.num_queries}")
    print(f"  Hop coverage:      {agg.avg_hop_coverage:.3f}")
    print(f"  Hop Recall@5:      {agg.avg_hop_recall_at_5:.3f}")
    print(f"  Hop Recall@10:     {agg.avg_hop_recall_at_10:.3f}")
    print(f"  Hop MRR:           {agg.avg_hop_mrr:.3f}")
    print(f"  Avg latency:       {agg.avg_latency_ms:.0f}ms")


def _print_code_qa_summary(agg: CodeQAAggregateMetrics) -> None:
    print(f"  Queries evaluated: {agg.num_queries}")
    print(f"  Accuracy:          {agg.accuracy:.3f}")
    print(f"  MRR:               {agg.mrr:.3f}")
    print(f"  Avg rank:          {agg.avg_rank:.1f}")
    print(f"  Symbol accuracy:   {agg.symbol_accuracy:.3f}")
    print(f"  File accuracy:     {agg.file_accuracy:.3f}")
    print(f"  Completeness:      {agg.completeness:.3f}")
    print(f"  Chunk coherence:   {agg.coherence:.3f}")
    print(f"  False positive %:  {agg.false_positive_rate:.1%}")
    if agg.by_type:
        print(f"  By QA type:")
        for qt, metrics in sorted(agg.by_type.items()):
            print(
                f"    {qt}: accuracy={metrics['accuracy']:.3f} mrr={metrics['mrr']:.3f} coherence={metrics['coherence']:.3f}"
            )
    print(f"  Avg latency:       {agg.avg_latency_ms:.0f}ms")


def _print_adversarial_summary(agg: AdversarialAggregateMetrics) -> None:
    print(f"  Queries evaluated: {agg.num_queries}")
    print(f"  Accuracy:          {agg.accuracy:.3f}")
    print(f"  Discrimination:    {agg.avg_discrimination:.3f}")
    print(f"  Avg correct rank:  {agg.avg_correct_rank:.1f}")
    print(f"  Decoy confusion:   {agg.decoy_confusion_rate:.1%}")
    if agg.by_type:
        print(f"  By adversarial type:")
        for at, metrics in sorted(agg.by_type.items()):
            print(
                f"    {at}: accuracy={metrics['accuracy']:.3f} discrimination={metrics['avg_discrimination']:.3f} confusion={metrics['confusion_rate']:.1%}"
            )
    print(f"  Avg latency:       {agg.avg_latency_ms:.0f}ms")


def _print_hallucination_summary(result: HallucinationBenchmarkResult) -> None:
    print(f"  Test cases:             {len(result.test_cases)}")
    print(
        f"  True hallucination rate: {result.true_hallucination_rate:.1%}  (wrong code WITH context)"
    )
    print(f"  Correct abstentions:     {result.abstention_rate:.1%}  (refused WITHOUT context)")
    print(f"  Context miss rate:       {result.context_miss_rate:.1%}  (no relevant context found)")
    print(f"  Overall failure rate:    {result.overall_rate:.1%}  (legacy: all failures / total)")
    if result.error_breakdown:
        print(f"  Error breakdown:")
        for err_type, count in sorted(result.error_breakdown.items()):
            print(f"    {err_type}: {count}")


def _print_comparison(report: BenchmarkReport) -> None:
    """Print a side-by-side comparison table."""
    if len(report.engines) < 2:
        return

    print("\n" + "=" * 70)
    print("COMPARISON: " + " vs ".join(report.engines))
    print("=" * 70)

    # Retrieval comparison
    if report.retrieval:
        print("\n--- Retrieval Quality (Precision@K / NDCG) ---")
        header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
        print(header)
        print("-" * len(header))

        row = f"{'MRR':<25}"
        for eng in report.engines:
            val = report.retrieval.get(eng, {}).get("avg_mrr", 0)
            row += f"{val:>15.3f}"
        print(row)

        k_values = set()
        for eng_data in report.retrieval.values():
            k_values.update(eng_data.get("avg_precision_at", {}).keys())
        for k in sorted(k_values):
            row = f"{'Precision@' + str(k):<25}"
            for eng in report.engines:
                val = report.retrieval.get(eng, {}).get("avg_precision_at", {}).get(str(k), 0)
                row += f"{val:>15.3f}"
            print(row)
        for k in sorted(k_values):
            row = f"{'NDCG@' + str(k):<25}"
            for eng in report.engines:
                val = report.retrieval.get(eng, {}).get("avg_ndcg_at", {}).get(str(k), 0)
                row += f"{val:>15.3f}"
            print(row)

        row = f"{'Avg latency (ms)':<25}"
        for eng in report.engines:
            val = report.retrieval.get(eng, {}).get("avg_latency_ms", 0)
            row += f"{val:>15.0f}"
        print(row)

        # MAP (BEIR standard)
        row = f"{'MAP':<25}"
        for eng in report.engines:
            val = report.retrieval.get(eng, {}).get("map_score", 0)
            row += f"{val:>15.3f}"
        print(row)

        # Success@K
        for k in sorted(k_values):
            row = f"{'Success@' + str(k):<25}"
            for eng in report.engines:
                val = report.retrieval.get(eng, {}).get("avg_success_at", {}).get(str(k), 0)
                row += f"{val:>15.3f}"
            print(row)

        # R-Precision
        row = f"{'R-Precision':<25}"
        for eng in report.engines:
            val = report.retrieval.get(eng, {}).get("avg_r_precision", 0)
            row += f"{val:>15.3f}"
        print(row)

    # Multi-hop comparison
    if report.multihop:
        print("\n--- Multi-Hop Retrieval ---")
        header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
        print(header)
        print("-" * len(header))

        for metric, label in [
            ("avg_hop_coverage", "Hop coverage"),
            ("avg_hop_recall_at_5", "Hop Recall@5"),
            ("avg_hop_recall_at_10", "Hop Recall@10"),
            ("avg_hop_mrr", "Hop MRR"),
        ]:
            row = f"{label:<25}"
            for eng in report.engines:
                val = report.multihop.get(eng, {}).get(metric, 0)
                row += f"{val:>15.3f}"
            print(row)

    # Code QA comparison
    if report.code_qa:
        print("\n--- Code-Specific QA ---")
        header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
        print(header)
        print("-" * len(header))

        for metric, label in [
            ("accuracy", "Accuracy"),
            ("mrr", "MRR"),
            ("symbol_accuracy", "Symbol accuracy"),
            ("file_accuracy", "File accuracy"),
            ("completeness", "Completeness"),
            ("coherence", "Chunk coherence"),
            ("false_positive_rate", "False positive rate"),
        ]:
            row = f"{label:<25}"
            for eng in report.engines:
                val = report.code_qa.get(eng, {}).get(metric, 0)
                if metric == "false_positive_rate":
                    row += f"{val:>14.1%} "
                else:
                    row += f"{val:>15.3f}"
            print(row)

    # Adversarial comparison
    if report.adversarial:
        print("\n--- Adversarial Near-Miss ---")
        header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
        print(header)
        print("-" * len(header))

        for metric, label in [
            ("accuracy", "Accuracy"),
            ("avg_discrimination", "Discrimination score"),
            ("avg_correct_rank", "Avg correct rank"),
            ("decoy_confusion_rate", "Decoy confusion rate"),
        ]:
            row = f"{label:<25}"
            for eng in report.engines:
                val = report.adversarial.get(eng, {}).get(metric, 0)
                if metric == "decoy_confusion_rate":
                    row += f"{val:>14.1%} "
                elif metric == "avg_correct_rank":
                    row += f"{val:>15.1f}"
                else:
                    row += f"{val:>15.3f}"
            print(row)

    # Validated dataset comparison
    if report.validated:
        # Group by dataset name
        datasets_seen: dict[str, list[str]] = {}  # dataset -> [engine_keys]
        for key, val_data in report.validated.items():
            ds_name = val_data.get("dataset", key)
            datasets_seen.setdefault(ds_name, []).append(key)

        for ds_name, keys in datasets_seen.items():
            print(f"\n--- {ds_name} (Validated) ---")
            header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
            print(header)
            print("-" * len(header))

            for metric, label in [
                ("avg_mrr", "MRR"),
                ("avg_latency_ms", "Avg latency (ms)"),
            ]:
                row = f"{label:<25}"
                for eng in report.engines:
                    eng_key = next((k for k in keys if k.startswith(eng)), None)
                    val = report.validated.get(eng_key, {}).get(metric, 0) if eng_key else 0
                    if "latency" in metric:
                        row += f"{val:>15.0f}"
                    else:
                        row += f"{val:>15.3f}"
                print(row)

            # Precision@K and NDCG@K
            sample_key = keys[0]
            k_values = set()
            for kk in keys:
                k_values.update(report.validated.get(kk, {}).get("avg_precision_at", {}).keys())
            for k in sorted(k_values):
                row = f"{'Precision@' + str(k):<25}"
                for eng in report.engines:
                    eng_key = next((kk for kk in keys if kk.startswith(eng)), None)
                    val = (
                        report.validated.get(eng_key, {}).get("avg_precision_at", {}).get(str(k), 0)
                        if eng_key
                        else 0
                    )
                    row += f"{val:>15.3f}"
                print(row)
            for k in sorted(k_values):
                row = f"{'NDCG@' + str(k):<25}"
                for eng in report.engines:
                    eng_key = next((kk for kk in keys if kk.startswith(eng)), None)
                    val = (
                        report.validated.get(eng_key, {}).get("avg_ndcg_at", {}).get(str(k), 0)
                        if eng_key
                        else 0
                    )
                    row += f"{val:>15.3f}"
                print(row)

    # LLM-as-Judge comparison
    if report.judge:
        datasets_seen_j: dict[str, list[str]] = {}
        for key, val_data in report.judge.items():
            ds_name = val_data.get("dataset", key)
            datasets_seen_j.setdefault(ds_name, []).append(key)

        for ds_name, keys in datasets_seen_j.items():
            print(f"\n--- {ds_name} (LLM Judge) ---")
            header = f"{'Metric':<25}" + "".join(f"{e:>15}" for e in report.engines)
            print(header)
            print("-" * len(header))

            for metric, label in [
                ("avg_relevance", "Avg relevance (0-3)"),
                ("avg_completeness", "Avg completeness (0-3)"),
                ("avg_specificity", "Avg specificity (0-3)"),
                ("avg_total", "Avg total (0-3)"),
                ("avg_latency_ms", "Avg latency (ms)"),
                ("win_count", "Wins"),
                ("tie_count", "Ties"),
            ]:
                row = f"{label:<25}"
                for eng in report.engines:
                    eng_key = next((k for k in keys if k.startswith(eng)), None)
                    val = report.judge.get(eng_key, {}).get(metric, 0) if eng_key else 0
                    if "latency" in metric:
                        row += f"{val:>15.0f}"
                    elif isinstance(val, int):
                        row += f"{val:>15}"
                    else:
                        row += f"{val:>15.3f}"
                print(row)

    # Hallucination comparison (multi-model aware)
    if report.hallucination:
        print("\n--- Hallucination Rate (by model) ---")

        # Group by engine
        engines_in_hal = sorted(
            set(v.get("engine", k.split("__")[0]) for k, v in report.hallucination.items())
        )

        for eng in engines_in_hal:
            print(f"\n  Engine: {eng}")
            # Find all model runs for this engine
            model_keys = [
                k
                for k, v in report.hallucination.items()
                if v.get("engine", k.split("__")[0]) == eng
            ]

            if not model_keys:
                continue

            # Build column headers from model info
            col_labels = []
            for k in model_keys:
                v = report.hallucination[k]
                provider = v.get("llm_provider", "?")
                tier = v.get("llm_tier", "?")
                model = v.get("llm_model", "?")
                col_labels.append(f"{provider}/{tier}")

            header = f"  {'Metric':<30}" + "".join(f"{c:>20}" for c in col_labels)
            print(header)
            print("  " + "-" * (len(header) - 2))

            # Print model names as sub-header
            model_row = f"  {'(model)':<30}"
            for k in model_keys:
                model_row += f"{report.hallucination[k].get('llm_model', '?'):>20}"
            print(model_row)

            for metric, label in [
                ("true_hallucination_rate", "True hallucination rate"),
                ("overall_rate", "Overall failure rate"),
                ("abstention_rate", "Correct abstentions"),
                ("context_miss_rate", "Context miss rate"),
            ]:
                row = f"  {label:<30}"
                for k in model_keys:
                    val = report.hallucination[k].get(metric, 0)
                    row += f"{val:>19.1%} "
                print(row)

            # Error breakdown
            all_error_types: set[str] = set()
            for k in model_keys:
                all_error_types.update(report.hallucination[k].get("error_breakdown", {}).keys())
            if all_error_types:
                print(f"  {'Error breakdown:':<30}")
                for err_type in sorted(all_error_types):
                    row = f"    {err_type:<28}"
                    for k in model_keys:
                        val = report.hallucination[k].get("error_breakdown", {}).get(err_type, 0)
                        row += f"{val:>20}"
                    print(row)

    print()
