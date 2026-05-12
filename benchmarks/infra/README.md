# infra/

Operational glue shared across phases.

```
infra/
├── logging_config.py    structured logging + QueryTrace/TraceStore
├── sampling.py          seeded + stratified query sub-sampling
├── latency.py           LatencyMeter (request / retry / sleep buckets)
└── consistency.py       repeat-run consistency checks
```

## logging_config.py

Wires the harness logger and persists per-query traces under
`benchmarks/results/run_<ts>/`:

```
results/run_20260512_154507/
├── logs/      structured JSONL log lines
├── traces/    per-query trace records (one per query × engine × phase)
├── reports/   final + intermediate report JSONs, manifests
└── data/      flat CSV exports
```

`QueryTrace.create(run_id, engine, benchmark_type)` is the standard pattern
phases follow. The TraceStore writes both incrementally (so a Ctrl-C still
preserves work) and at the end.

## sampling.py

Every phase calls `sample_seeded` or `stratified_sample` from here. The
previous `[:N]` truncation was order-dependent (which broke the diagnosis's
"randomize sampling" point) and provided no way to run multiple seeds.

```python
sample_seeded(items, max_items, seed=0)
stratified_sample(items, max_items, key=lambda x: x.category, seed=0)
```

A single `seed` produces the same draw across phases within a run.

## latency.py

`LatencyMeter` provides a wall-clock-correct breakdown:

```python
meter = LatencyMeter(); meter.start()
with meter.measure("request"): ...
await meter.sleep(0.5)            # counted as sleep_ms
with meter.measure("retry"): ...
breakdown = meter.stop()
# breakdown.total_ms, request_ms, retry_ms, sleep_ms, retry_count, sleep_count
```

The Nia and Context7 adapters already account for rate-limit sleeps in
their returned latency. New adapters should use `LatencyMeter` so the
breakdown is reportable.

## consistency.py

Repeat-run helpers used by the consistency report; not exercised by the
default `python -m benchmarks` flow.
