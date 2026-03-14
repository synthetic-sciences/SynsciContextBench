"""CLI for running benchmarks.

Usage:
    python -m benchmarks                              # Run all benchmarks (synsc vs nia vs context7)
    python -m benchmarks --download-datasets           # Download CodeSearchNet + CoSQA
    python -m benchmarks --validated-only              # Only validated dataset benchmarks
    python -m benchmarks --retrieval-only              # Only Precision@K / NDCG / MRR
    python -m benchmarks --multihop-only               # Only multi-hop retrieval
    python -m benchmarks --code-qa-only                # Only code-specific QA
    python -m benchmarks --adversarial-only            # Only adversarial near-miss
    python -m benchmarks --hallucination-only          # Only hallucination rate
    python -m benchmarks --skip-indexing               # Skip repo indexing
    python -m benchmarks --engines synsc               # Only test synsc-context
    python -m benchmarks --engines nia                 # Only test Nia
    python -m benchmarks --engines context7            # Only test Context7
    python -m benchmarks --engines synsc nia context7  # All three engines
    python -m benchmarks --multi-model                 # Hallucination across all configured models
    python -m benchmarks --match-mode hybrid           # Match by content OR file path (default)
    python -m benchmarks --match-mode file             # Match by file path only (fair cross-engine)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .adapters import Context7Adapter, NiaAdapter, SynscAdapter
from .config import BenchmarkConfig
from .runner import run_full_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="synsc-context vs Nia vs Context7 benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["synsc", "nia", "context7", "all"],
        default=["all"],
        help="Which engines to benchmark (default: all)",
    )

    # Dataset download
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download validated datasets (CodeSearchNet, CoSQA) from HuggingFace",
    )
    parser.add_argument(
        "--dataset-max-samples",
        type=int,
        default=500,
        help="Max samples per dataset/language when downloading (default: 500)",
    )

    # --*-only flags (mutually exclusive group)
    only_group = parser.add_mutually_exclusive_group()
    only_group.add_argument(
        "--validated-only",
        action="store_true",
        help="Only run validated dataset benchmarks (CodeSearchNet, CoSQA)",
    )
    only_group.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval quality benchmark (Precision@K, NDCG, MRR)",
    )
    only_group.add_argument(
        "--multihop-only",
        action="store_true",
        help="Only run multi-hop retrieval benchmark",
    )
    only_group.add_argument(
        "--code-qa-only",
        action="store_true",
        help="Only run code-specific QA benchmark",
    )
    only_group.add_argument(
        "--adversarial-only",
        action="store_true",
        help="Only run adversarial near-miss benchmark",
    )
    only_group.add_argument(
        "--hallucination-only",
        action="store_true",
        help="Only run hallucination rate benchmark",
    )
    only_group.add_argument(
        "--judge-only",
        action="store_true",
        help="Only run LLM-as-judge benchmark (fair cross-engine comparison)",
    )
    only_group.add_argument(
        "--enhanced-judge-only",
        action="store_true",
        help="Only run enhanced LLM-as-judge (position-debiased, with context quality metrics)",
    )

    # --skip-* flags
    parser.add_argument("--skip-indexing", action="store_true", help="Skip indexing step")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval benchmark")
    parser.add_argument("--skip-multihop", action="store_true", help="Skip multi-hop benchmark")
    parser.add_argument("--skip-code-qa", action="store_true", help="Skip code QA benchmark")
    parser.add_argument(
        "--skip-adversarial", action="store_true", help="Skip adversarial benchmark"
    )
    parser.add_argument(
        "--skip-hallucination", action="store_true", help="Skip hallucination benchmark"
    )
    parser.add_argument(
        "--skip-validated", action="store_true", help="Skip validated dataset benchmarks"
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge benchmark")
    parser.add_argument(
        "--skip-enhanced-judge", action="store_true", help="Skip enhanced LLM-as-judge benchmark"
    )

    # Statistical analysis
    parser.add_argument(
        "--no-significance", action="store_true", help="Skip statistical significance analysis"
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for CI estimation (default: 10000)",
    )

    # Enhanced judge options
    parser.add_argument(
        "--no-debiasing",
        action="store_true",
        help="Disable position debiasing in enhanced judge (faster, 2x fewer LLM calls)",
    )

    # Multi-model
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Run hallucination benchmark across all configured model tiers (low/mid/high per provider)",
    )

    # Dataset filter (for validated benchmarks)
    parser.add_argument(
        "--dataset",
        choices=["cosqa", "codesearchnet", "codesearchnet_challenge"],
        default=None,
        help="Only run a specific validated dataset (e.g., --dataset cosqa)",
    )

    # Match mode for validated benchmarks
    parser.add_argument(
        "--match-mode",
        choices=["content", "file", "hybrid"],
        default="hybrid",
        help="How to match results to corpus: content (text similarity), "
        "file (file path), hybrid (either). Default: hybrid",
    )

    # Query limits
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Max queries per dataset (default: all). Use 50-100 for quick runs.",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    config = BenchmarkConfig()
    if args.max_queries is not None:
        config.max_queries = args.max_queries
    if args.multi_model:
        matrix = config.load_model_matrix()
        if not matrix:
            print(
                "[ERROR] --multi-model requires BENCH_{PROVIDER}_API_KEY env vars. See .env.local"
            )
            return 1
        print(f"Multi-model mode: {len(matrix)} models configured")
        for m in matrix:
            print(f"  {m.display_name} ({m.tier})")
        print()

    # --- Download datasets mode ---
    if args.download_datasets:
        from .dataset_loader import download_all

        download_all(
            csn_max_per_lang=args.dataset_max_samples,
            cosqa_max=args.dataset_max_samples,
        )
        return 0

    # Determine which engines to test
    engine_names = set(args.engines)
    if "all" in engine_names:
        engine_names = {"synsc", "nia", "context7"}

    # Validate config
    engines = []
    if "synsc" in engine_names:
        if not config.synsc_api_key:
            print("[!] SYNSC_API_KEY not set — skipping synsc-context")
        else:
            engines.append(SynscAdapter(config.synsc_api_url, config.synsc_api_key))

    if "nia" in engine_names:
        if not config.nia_api_key:
            print("[!] NIA_API_KEY not set — skipping Nia")
        else:
            engines.append(NiaAdapter(config.nia_api_url, config.nia_api_key))

    if "context7" in engine_names:
        if not config.context7_enabled:
            print("[!] CONTEXT7_ENABLED=false — skipping Context7")
        else:
            engines.append(
                Context7Adapter(
                    api_url=config.context7_api_url,
                    api_key=config.context7_api_key,
                    npx_command=config.context7_npx_command,
                    request_delay=config.context7_request_delay,
                )
            )

    if not engines:
        print("[ERROR] No engines configured. Set SYNSC_API_KEY / NIA_API_KEY / CONTEXT7_ENABLED.")
        return 1

    print(f"Benchmarking engines: {', '.join(e.name for e in engines)}")

    # Resolve skip flags
    all_skips = {
        "skip_indexing": args.skip_indexing,
        "skip_retrieval": args.skip_retrieval,
        "skip_multihop": args.skip_multihop,
        "skip_code_qa": args.skip_code_qa,
        "skip_adversarial": args.skip_adversarial,
        "skip_hallucination": args.skip_hallucination,
        "skip_validated": args.skip_validated,
        "skip_judge": args.skip_judge,
        "skip_enhanced_judge": args.skip_enhanced_judge,
    }

    # --*-only: skip everything except the chosen one
    only_map = {
        "validated_only": "skip_validated",
        "retrieval_only": "skip_retrieval",
        "multihop_only": "skip_multihop",
        "code_qa_only": "skip_code_qa",
        "adversarial_only": "skip_adversarial",
        "hallucination_only": "skip_hallucination",
        "judge_only": "skip_judge",
        "enhanced_judge_only": "skip_enhanced_judge",
    }
    for flag_name, keep_key in only_map.items():
        if getattr(args, flag_name, False):
            for k in all_skips:
                all_skips[k] = True
            all_skips["skip_indexing"] = True
            all_skips[keep_key] = False
            break

    # Dataset filter
    dataset_filter = None
    if args.dataset:
        dataset_filter = [args.dataset]

    try:
        report = await run_full_benchmark(
            engines=engines,
            config=config,
            dataset_filter=dataset_filter,
            match_mode=args.match_mode,
            enable_debiasing=not args.no_debiasing,
            **all_skips,
        )
    finally:
        for engine in engines:
            await engine.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
