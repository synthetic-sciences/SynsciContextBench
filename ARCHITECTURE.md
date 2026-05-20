# Architecture

A short tour of how `synsci-context-bench` is put together. Read [`README.md`](./README.md)
first for the user-facing overview; this doc is for contributors who want to
make changes.

## Goals

1. **Compare context engines fairly.** All engines see the same queries,
   under the same latency accounting, scored by the same metrics.
2. **Separate code retrieval from agent context.** Real Atlas usage is
   not function-level retrieval. The harness measures both, on separate
   leaderboards, so an engine winning one and losing the other is visibly
   shown to do so.
3. **Make failures actionable.** Every score eventually flows into a
   per-engine failure-taxonomy report so "engine X lost" turns into
   "engine X lost N cases of bucket Y".

## High-level flow

```
python -m benchmarks <flags>
        │
        ▼
benchmarks/__main__.py
        │ argparse, env config, build engine adapters
        ▼
benchmarks/runner.py
        │ run_full_benchmark()
        │
        ├── Phase 1   run_retrieval_benchmark (inline)
        ├── Phase 2   phases.multihop.run_multihop_benchmark
        ├── Phase 3   phases.code_qa.run_code_qa_benchmark
        ├── Phase 4   phases.adversarial.run_adversarial_benchmark
        ├── Phase 5   phases.hallucination.run_hallucination_benchmark
        ├── Phase 6   phases.validated_eval.run_validated_benchmark
        ├── Phase 7   judges.llm_judge.run_judge_benchmark         (per dataset)
        ├── Phase 8   judges.enhanced_judge.run_enhanced_judge_benchmark
        ├── Phase 9   phases.swe_agent.run_swe_agent_benchmark     (+ swe_real_patch opt-in)
        ├── Phase 10  phases.atlas.run_atlas_benchmark
        ├── Phase 11  phases.session_replay.run_session_replay_benchmark
        │
        ├── scoring.leaderboards.build_leaderboards
        ├── scoring.failure_taxonomy.build_failure_taxonomy
        └── scoring.statistical_analysis.run_pairwise_significance
                │
                ▼
        BenchmarkReport
                │
                ▼ via infra.logging_config.TraceStore
        results/run_<ts>/{logs,traces,reports,data}
```

## Module taxonomy

| Subpackage | Responsibility |
|------------|----------------|
| `adapters/` | One file per engine. Common `ContextEngineAdapter` interface. Each adapter returns full user-visible latency, including any rate-limit sleeps and retries. |
| `phases/`   | One module per benchmark phase. Each loads a curated dataset, runs the engine(s), and returns a report dataclass. |
| `judges/`   | LLM-as-judge implementations. Position-debiased enhanced judge lives here; lighter 3-D judge is reused by phase-level fairness passes. |
| `scoring/`  | Deterministic scoring. Metrics, context-grounding signals, per-category leaderboards, failure taxonomy, statistical analysis. |
| `infra/`    | Operational glue: logging, structured tracing, seeded sampling, latency accounting. |
| `utils/`    | Standalone helpers: dataset downloader, fake-repo fixture builder. |
| `datasets/` | Test-case inputs. `curated/` holds in-repo cases; `validated/` holds downloaded standard datasets. |
| `results/`  | Per-run output trees. Not part of the package; the runner writes here. |

## Cross-cutting concerns

### Seeded sampling

Every phase uses `infra.sampling.sample_seeded` or `stratified_sample`
instead of `[:N]` truncation. A single `--seed` setting reproduces the same
draw across phases within a run; `--num-seeds N` reserves seeds for downstream
multi-seed aggregation.

### Latency

The benchmark reports full *user-visible* latency, including rate-limit
sleeps and retry backoffs. Adapters that throttle (Nia, Context7) must
include sleep+retry time in the latency they return; new adapters should
use `infra.latency.LatencyMeter` to get this right by construction.

### Tracing

`infra.logging_config.QueryTrace` is the unit of tracing. Every phase
creates one trace per query × engine and attaches it to the
`TraceStore`. The store persists incrementally to `results/run_<ts>/traces/`
so a Ctrl-C still leaves usable data.

### Phase-to-report wiring

Each phase populates one slot on `BenchmarkReport` (`retrieval`, `atlas`,
`session_replay`, ...) plus, when applicable, per-query rows on
`query_results`. After all phases finish, `scoring.leaderboards` and
`scoring.failure_taxonomy` consume the report as a dict and write back
into `report.leaderboards` / `report.failure_taxonomy`. The end manifest
captures the full picture.

## Why this layout

The previous flat layout made it hard to see what the harness actually
does. Reviewers asking "what's the difference between code retrieval and
Atlas context?" had to read every phase. With phases / judges / scoring /
infra / utils as separate subpackages, the answer is the layout itself.

It also made the diagnosis-driven changes physically visible:
`scoring/leaderboards.py` and `scoring/failure_taxonomy.py` are new files
that explicitly replace single-winner reporting, and they live in `scoring/`
next to the metrics that feed them.

## Where the diagnosis lands

| Diagnosis item | Where it's addressed |
|----------------|---------------------|
| Code-retrieval favors Delphi but Atlas usage exposes weaker axes | `phases/atlas.py` + `phases/session_replay.py` + `scoring/leaderboards.py` |
| Recall@K above 1.0 | `scoring/metrics.py:recall_at_k` (de-dup + clamp) |
| Validated LLM judge only scores top 3 | `phases/validated_eval.py:run_validated_benchmark` (`judge_top_k`) |
| Query sampling is first-N, not randomized | `infra/sampling.py` + every phase now uses it |
| Context7 sleeps not counted | `adapters/context7.py` (user-visible latency) |
| Nia retries not counted | `adapters/nia.py` (user-visible latency) |
| MCP proxy hides quality controls | `adapters/synsc_mcp.py` (build_context_pack + quality_mode) |
| No failure taxonomy | `scoring/failure_taxonomy.py` |
| Single "winner" leaderboard | `scoring/leaderboards.py` |

See [`CHANGELOG.md`](./CHANGELOG.md) for the chronological view.
