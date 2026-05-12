# benchmarks/

The Python package that powers `synsci-context-bench`. The package is laid out
by responsibility so each subdirectory has one job:

```
benchmarks/
├── __main__.py        cli entry point — argparse + engine bootstrap
├── config.py          environment + path config (BenchmarkConfig)
├── runner.py          phase orchestrator (run_full_benchmark)
├── adapters/          engine adapters (synsc, synsc-mcp, nia, context7)
├── phases/            one module per benchmark phase
├── judges/            llm-as-judge implementations
├── scoring/           metrics, leaderboards, failure taxonomy, stats
├── infra/             logging, sampling, latency, consistency
├── utils/             dataset download + benchmark-repo fixture builder
├── datasets/
│   ├── curated/       hand-built test cases owned by this repo
│   └── validated/     downloaded standard datasets (CodeSearchNet etc.)
└── results/           run-* directories (traces, manifests, CSVs)
```

## Run something

```bash
uv sync
cp benchmarks/.env.local.example benchmarks/.env.local   # fill API keys
uv run python -m benchmarks --help
```

Common entry points:

```bash
# everything
uv run python -m benchmarks

# one phase
uv run python -m benchmarks --thesis-only --max-queries 20
uv run python -m benchmarks --session-replay-only
uv run python -m benchmarks --swe-agent-only --real-patch

# one engine
uv run python -m benchmarks --engines synsc-mcp

# multi-seed for confidence intervals
uv run python -m benchmarks --num-seeds 3 --max-queries 100
```

## Add something

| What | Where | Notes |
|------|-------|-------|
| New phase | `benchmarks/phases/` + wire into `runner.py` | Follow the `thesis.py` shape: dataset → `run_*_benchmark` → report dataclass |
| New engine | `benchmarks/adapters/` | Implement `ContextEngineAdapter` from `adapters/base.py` |
| New metric | `benchmarks/scoring/` | Pure function; add to phase aggregator |
| New judge | `benchmarks/judges/` | Reuse `_call_llm_judge_raw` + `_safe_parse_json` |
| Hand-built cases | `benchmarks/datasets/curated/<name>.json` | Document the schema in the file's `_description` |
| Downloaded dataset | extend `benchmarks/utils/dataset_loader.py` | Writes into `datasets/validated/` |

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root for the long form.

## Module map

Each subdirectory has its own `README.md` describing what's inside and why
it belongs there. Start with [`phases/`](./phases/README.md) if you want to
read along with a benchmark run.
