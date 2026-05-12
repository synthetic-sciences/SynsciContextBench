# phases/

One module per benchmark phase. The runner invokes them in the order below.
Each module exposes a `run_<phase>_benchmark(...)` function and a small
report dataclass; the runner persists the reports under `benchmarks/results/`.

```
phases/
├── multihop.py            P2 — cross-file retrieval
├── code_qa.py             P3 — function / call-site / inheritance QA
├── adversarial.py         P4 — same name, wrong context decoys
├── hallucination.py       P5 — does context prevent invented APIs
├── validated_eval.py      P6 — CodeSearchNet / CoSQA / AdvTest / APPS / SO-QA
├── swe_agent.py           P9 — code-gen w/ and w/o context, no-context baseline
├── swe_real_patch.py      P9b — opt-in: clone + apply patch + run tests
├── thesis.py              P10 — Thesis-workflow tasks (8 categories)
└── session_replay.py      P11 — replay real production losses
```

The Phase 1 retrieval benchmark itself currently lives inside `runner.py`
(`run_retrieval_benchmark`) so it has direct access to the trace-store
plumbing; it follows the same shape as the modules above.

## Phase contract

Every phase obeys the same shape so the runner can swap them:

```python
async def run_<phase>_benchmark(
    engine: ContextEngineAdapter,   # or list[ContextEngineAdapter] for multi-engine phases
    dataset_path: str,
    *,
    top_k: int = 10,
    max_queries: int | None = None,
    seed: int = 0,                  # seeded sampling for sub-sampled runs
    llm_provider: str = "",         # optional judge creds
    llm_model: str = "",
    llm_api_key: str = "",
) -> tuple[<Aggregate>, list[<Per-Query>]]
```

Newer phases (`thesis`, `session_replay`) return a single nested dataclass
(`ThesisEngineReport`, `ReplayEngineReport`) that contains both the aggregate
and the per-case list — this is the preferred shape for new phases.

## Sub-sampling

Every phase calls `infra.sampling.sample_seeded` / `stratified_sample` instead
of slicing the dataset list. This guarantees:

- The same `seed` produces the same draw across modules (so different
  engines see the same queries within a run).
- Stratified phases (e.g., SWE-Agent on knowledge tiers, Thesis on
  categories) keep their proportions when sub-sampled.

The previous `[:N]` truncation has been removed everywhere.

## Adding a new phase

1. Drop a new module here. Re-use `infra.sampling` and `scoring.*` instead
   of rolling your own.
2. Add a JSON test-case file under `benchmarks/datasets/curated/`.
3. Wire it into `benchmarks/runner.py`: import, add to `_PHASE_FIELDS`,
   add a `skip_<phase>` parameter, add a phase block in `run_full_benchmark`.
4. Add a CLI flag in `benchmarks/__main__.py` (`--<phase>-only`,
   `--skip-<phase>`).
5. Re-export from `benchmarks/phases/__init__.py`.
6. Document it in [`docs/PHASES.md`](../../docs/PHASES.md).
