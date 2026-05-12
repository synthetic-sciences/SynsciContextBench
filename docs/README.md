# docs/

Long-form documentation. The repo's top-level [README.md](../README.md) is the
entry point for "what is this and how do I run it"; the files in here are for
"how does it work" and "how is it scored".

| File | What's in it |
|------|--------------|
| [`PHASES.md`](./PHASES.md) | Per-phase deep dive: what each of the 11 phases measures, the dataset shape, the scoring rubric, and any known caveats. |
| [`METRICS.md`](./METRICS.md) | Per-metric reference: how MRR, NDCG, Recall@K, citation share, utilization, anchor-hit, composite, etc. are computed — including the diagnosis fixes. |
| [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md) | Last full-run report. Regenerate with `python -m benchmarks` and overwrite. |

For the system-level architecture (how `runner.py` orchestrates phases,
adapters, and judges), see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

For "how do I add a new phase / engine / dataset", see
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).
