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
    python -m benchmarks --swe-agent-only               # Only SWE-Agent benchmark (Phase 9)
    python -m benchmarks --diff-aware-only                  # Only Diff-aware indexing benchmark (Phase 10)
    python -m benchmarks --session-replay-only          # Only real-session replay (Phase 11)
    python -m benchmarks --skip-indexing               # Skip repo indexing
    python -m benchmarks --engines synsc               # Only the HTTP Delphi adapter
    python -m benchmarks --engines synsc-mcp           # Delphi through the MCP proxy (agent path)
    python -m benchmarks --engines nia context7        # Subset of engines
    python -m benchmarks --num-seeds 3 --max-queries 100   # 3-seed CI mode
    python -m benchmarks --judge-top-k 10 --match-mode llm # Fix old top-3 LLM judge cap
    python -m benchmarks --real-patch --swe-agent-only     # Real-patch SWE eval mode
    python -m benchmarks --multi-model                 # Hallucination across all configured models
    python -m benchmarks --match-mode hybrid           # Match by content OR file path (default)
    python -m benchmarks --match-mode file             # Match by file path only (fair cross-engine)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from .adapters import Context7Adapter, NiaAdapter, SynscAdapter, SynscMCPAdapter
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
        choices=["synsc", "synsc-mcp", "nia", "context7", "all", "none"],
        default=["all"],
        help="Which engines to benchmark (default: all). "
        "'synsc' uses the HTTP API; 'synsc-mcp' exercises the MCP proxy "
        "with quality_mode=agent (closer to real agent usage). "
        "Use 'none' with --swe-agent-only to run baseline-only.",
    )
    parser.add_argument(
        "--synsc-quality-mode",
        choices=["agent", "default"],
        default="agent",
        help="Pass-through to the Delphi adapter. 'agent' enables the "
        "agent-quality endpoints (build_context_pack, deep_index). "
        "Default: agent.",
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
    only_group.add_argument(
        "--swe-agent-only",
        action="store_true",
        help="Only run SWE-Agent benchmark (Phase 9: context engine value-add for real SWE tasks)",
    )
    only_group.add_argument(
        "--diff-aware-only",
        action="store_true",
        help="Only run the Diff-aware indexing benchmark (Phase 10).",
    )
    only_group.add_argument(
        "--session-replay-only",
        action="store_true",
        help="Only run the real-session replay benchmark (Phase 11).",
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
    parser.add_argument(
        "--skip-swe-agent", action="store_true", help="Skip SWE-Agent benchmark (Phase 9)"
    )
    parser.add_argument(
        "--skip-diff-aware", action="store_true",
        help="Skip the Diff-aware indexing benchmark (Phase 10).",
    )
    parser.add_argument(
        "--skip-session-replay", action="store_true",
        help="Skip the session-replay benchmark (Phase 11).",
    )
    parser.add_argument(
        "--real-patch", action="store_true",
        help="In SWE-Agent, also attempt real-patch evaluation when test cases "
        "include repo_url + test_command. Off by default because it shells out.",
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

    # SWE-Agent options
    parser.add_argument(
        "--with-agent-queries",
        action="store_true",
        help="Also run AI-generated queries alongside gold queries in SWE-Agent (default: gold only)",
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
        nargs="+",
        choices=["codesearchnet", "cosqa", "advtest", "codefeedback_st", "stackoverflow_qa", "apps"],
        default=None,
        help="Only run specific validated datasets (e.g., --dataset cosqa advtest)",
    )

    # Match mode for validated benchmarks
    parser.add_argument(
        "--match-mode",
        choices=["content", "file", "hybrid", "llm"],
        default="hybrid",
        help="How to match results to corpus: content (text similarity), "
        "file (file path), hybrid (either), llm (LLM judge, fairest). Default: hybrid",
    )

    # Query limits
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Max queries per dataset (default: all). Use 50-100 for quick runs.",
    )

    # Sampling / scoring controls
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for query sub-sampling. Use multiple values via --num-seeds "
        "(repeats the run, then aggregates).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Reserve N seeds in config (seed, seed+1, ..., seed+N-1). "
        "Only seeds[0] is consumed by the current single-run pipeline; "
        "the rest are recorded for downstream multi-seed aggregation. "
        "Default 1.",
    )
    parser.add_argument(
        "--judge-top-k",
        type=int,
        default=10,
        help="When --match-mode llm is set, judge top-K results per query. "
        "Default 10 (previously hard-coded to 3, which silently capped Recall@10).",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Resume from a previous run directory (e.g., benchmarks/results/run_20260322_115101). "
        "Completed phases are loaded from the latest checkpoint and skipped automatically.",
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging with per-query traces",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    config = BenchmarkConfig()

    # Set up structured logging
    from .infra.logging_config import TraceStore, setup_logging
    trace_store = TraceStore(config.results_dir)
    bench_logger = setup_logging(
        results_dir=config.results_dir,
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
    )
    if args.max_queries is not None:
        config.max_queries = args.max_queries

    # Threading sampling + judge controls through the config so every phase
    # picks them up. --num-seeds expands --seed into a list.
    base_seed = int(getattr(args, "seed", 0))
    n_seeds = max(1, int(getattr(args, "num_seeds", 1)))
    config.seeds = [base_seed + i for i in range(n_seeds)]
    config.judge_top_k = int(getattr(args, "judge_top_k", 10))
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
        from .utils.dataset_loader import download_all

        download_all(
            csn_max_per_lang=args.dataset_max_samples,
            cosqa_max=args.dataset_max_samples,
        )
        return 0

    # Determine which engines to test
    engine_names = set(args.engines)
    no_engine_mode = "none" in engine_names
    if "all" in engine_names:
        engine_names = {"synsc", "nia", "context7"}
    engine_names.discard("none")

    quality_mode = getattr(args, "synsc_quality_mode", "agent")

    # Validate config
    engines = []
    if "synsc" in engine_names:
        if not config.synsc_api_key:
            print("[!] SYNSC_API_KEY not set — skipping synsc-context")
        else:
            engines.append(SynscAdapter(
                config.synsc_api_url, config.synsc_api_key,
                quality_mode=quality_mode,
            ))

    if "synsc-mcp" in engine_names:
        engines.append(SynscMCPAdapter(
            api_url=config.synsc_api_url,
            api_key=config.synsc_api_key,
            quality_mode=quality_mode,
        ))

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

    if not engines and not no_engine_mode:
        print("[ERROR] No engines configured. Set SYNSC_API_KEY / NIA_API_KEY / CONTEXT7_ENABLED.")
        print("        Use '--engines none --swe-agent-only' to run baseline-only.")
        return 1

    if no_engine_mode and not getattr(args, "swe_agent_only", False):
        print("[ERROR] '--engines none' is only supported with --swe-agent-only (baseline-only mode).")
        return 1

    if engines:
        print(f"Benchmarking engines: {', '.join(e.name for e in engines)}")
    else:
        print("Baseline-only mode (no context engines)")

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
        "skip_swe_agent": args.skip_swe_agent,
        "skip_diff_aware": args.skip_diff_aware,
        "skip_session_replay": args.skip_session_replay,
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
        "swe_agent_only": "skip_swe_agent",
        "diff_aware_only": "skip_diff_aware",
        "session_replay_only": "skip_session_replay",
    }
    for flag_name, keep_key in only_map.items():
        if getattr(args, flag_name, False):
            for k in all_skips:
                all_skips[k] = True
            all_skips["skip_indexing"] = True
            all_skips[keep_key] = False
            break

    # Dataset filter
    dataset_filter = args.dataset  # already a list or None

    # Register graceful shutdown handler
    _interrupted = False
    _active_report = None  # set during run so emergency save can include it

    def _graceful_shutdown(signum, frame):
        nonlocal _interrupted
        if _interrupted:
            # Second Ctrl+C — force exit
            print("\n  [!] Force exit.")
            sys.exit(1)
        _interrupted = True
        print("\n  [!] Interrupted — saving progress before exit...")
        try:
            from dataclasses import asdict
            save_data = asdict(_active_report) if _active_report else None
            trace_store.save(report=save_data)
            print(f"  [checkpoint] Emergency save to {trace_store.results_dir}")
        except Exception as e:
            print(f"  [!] Failed to save: {e}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _graceful_shutdown)

    # Load checkpoint for resume
    resume_report = None
    if args.resume:
        from .runner import load_checkpoint
        resume_report = load_checkpoint(args.resume)
        if resume_report is None:
            print(f"[ERROR] No valid checkpoint found in {args.resume}")
            return 1

    report = None
    try:
        report = await run_full_benchmark(
            engines=engines,
            config=config,
            dataset_filter=dataset_filter,
            match_mode=args.match_mode,
            enable_debiasing=not args.no_debiasing,
            with_agent_queries=getattr(args, "with_agent_queries", False),
            trace_store=trace_store,
            resume_report=resume_report,
            **all_skips,
        )
    except KeyboardInterrupt:
        print("\n  [!] Interrupted — saving progress...")
        try:
            from dataclasses import asdict
            save_data = asdict(report) if report else None
            trace_store.save(report=save_data)
            print(f"  [checkpoint] Emergency save to {trace_store.results_dir}")
        except Exception as e:
            print(f"  [!] Failed to save: {e}")
    finally:
        for engine in engines:
            await engine.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
