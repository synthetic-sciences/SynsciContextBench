# results/

One subdirectory per benchmark run, named `run_<YYYYMMDD_HHMMSS>` (or a
legacy unix timestamp like `run_1773603757`). The runner writes here
incrementally, so an interrupted run still leaves usable output.

## Per-run layout

```
results/run_20260318_211759/
├── logs/
│   └── bench_<ts>.jsonl          structured log lines
├── traces/
│   └── traces_<ts>.jsonl         per-query trace records
├── reports/
│   ├── benchmark_<ts>.json       final report dataclass
│   └── manifest_<ts>.json        run-level manifest (engines, config, report)
└── data/
    └── results_<ts>.csv          flat CSV — one row per query × engine × phase
```

Each `manifest_*.json` includes `engines`, `config`, and a snapshot of the
`BenchmarkReport`. The runner saves a fresh manifest at the end of every
phase, so the latest file is the canonical state.

## Reading a trace

Each line in `traces/*.jsonl` is one query trace. Useful fields:

| Field | What it tells you |
|-------|------------------|
| `engine` | which adapter produced this |
| `benchmark_type` | which phase (`retrieval`, `thesis`, `swe_agent`, ...) |
| `query_text`, `query_id` | the input |
| `latency_ms` | full user-visible wall-clock (includes retries/sleeps) |
| `num_results` | how many chunks the engine returned |
| `results[]` | per-rank chunk metadata |
| `relevance_judgments[]` | per-rank relevance + ground-truth match details |
| `scores` | MRR, P@K, NDCG@K, composite, etc. for this query |
| `error`, `error_category` | populated only on failure |

## Aggregating across runs

The CSVs under `data/` are easy to slurp with `pandas`:

```python
import pandas as pd, pathlib
df = pd.concat(
    pd.read_csv(p) for p in pathlib.Path("benchmarks/results").rglob("results_*.csv")
)
```

The `manifest_*.json` files give you the engines/config to join against.

## Resuming a run

Pass `--resume benchmarks/results/run_<ts>` to skip the phases the latest
manifest already records. Useful when one phase failed mid-run.

## Cleaning up

`results/` is not in `.gitignore` so old runs can sit here. Feel free to
delete subdirectories you no longer need; nothing else references them.
