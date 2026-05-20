# Contributing

How to add things to the benchmark without breaking it.

For the system-level layout see [`ARCHITECTURE.md`](./ARCHITECTURE.md);
the per-folder READMEs (e.g. [`benchmarks/phases/README.md`](./benchmarks/phases/README.md))
have the local conventions.

## Setup

```bash
uv sync
cp benchmarks/.env.local.example benchmarks/.env.local
# fill in keys for synsc-context, nia, and your judge LLM
```

Smoke-test the harness without hitting an engine:

```bash
SYNSC_API_KEY=x NIA_API_KEY=y CONTEXT7_ENABLED=false \
  uv run python -m benchmarks --diff-aware-only --max-queries 2 --skip-indexing
```

This should run, fail every search (no real engine), still build leaderboards,
and write a manifest under `benchmarks/results/run_<ts>/`.

## Add a new benchmark phase

1. **Author a dataset.** Drop a JSON file under
   `benchmarks/datasets/curated/<phase>_test_cases.json`. Start with
   `_description` and `_methodology` at the top. Look at
   `diff_aware_test_cases.json` for an example schema.
2. **Write the phase module.** Add `benchmarks/phases/<phase>.py`. Follow
   the contract in [`phases/README.md`](./benchmarks/phases/README.md):
   - Accept `engine`, `dataset_path`, `top_k`, `max_queries`, `seed`,
     and optional `llm_*` creds.
   - Pull sub-sampling from `benchmarks.infra.sampling.sample_seeded`
     (or `stratified_sample` if your cases bucket by category).
   - Return a single nested dataclass that includes both the aggregate
     and the per-case list.
3. **Wire it into `runner.py`.**
   - Import + re-export from `benchmarks.phases.__init__`.
   - Add the phase name to `_PHASE_FIELDS`.
   - Add `skip_<phase>` parameter to `run_full_benchmark`.
   - Add the phase block, including trace creation.
4. **Add CLI flags** in `benchmarks/__main__.py`: `--<phase>-only` and
   `--skip-<phase>`.
5. **Document it** in [`docs/PHASES.md`](./docs/PHASES.md) and update
   `benchmarks/datasets/curated/README.md` with the new file.

## Add a new engine

1. Subclass `ContextEngineAdapter` in `benchmarks/adapters/<engine>.py`.
   Implement every method; return empty lists / `IndexResult(success=False, ...)`
   for unsupported operations.
2. **Account for the full latency.** Include rate-limit sleeps and retry
   backoffs in the latency you return. New adapters should use
   `benchmarks.infra.latency.LatencyMeter` to get this right by
   construction.
3. Export from `benchmarks/adapters/__init__.py`.
4. Add to the `--engines` choices in `benchmarks/__main__.py` and the
   engine-bootstrap block in `main()`.
5. List required env vars in `benchmarks/.env.local.example` and in the
   top-level README's "Environment Variables" table.

## Add a new metric

Decide which subpackage it belongs in:

- Deterministic / formula-driven → `benchmarks/scoring/`. Pure function,
  no LLM, no I/O.
- LLM-driven → `benchmarks/judges/`. Reuse `_call_llm_judge_raw` and
  `_safe_parse_json`.

Plug the new metric into one or more phases' report dataclasses and
update [`docs/METRICS.md`](./docs/METRICS.md).

## Add a new validated dataset

1. Extend `benchmarks/utils/dataset_loader.py` with a downloader that
   writes to `DATASETS_DIR` (which already targets `datasets/validated/`).
2. Add the filename to the appropriate list in `benchmarks/runner.py`:
   `all_validated_datasets`, `judge_datasets`, or `enh_datasets`.
3. Note the source URL and license in
   `benchmarks/datasets/validated/README.md`.

## Tests / smoke

There is no formal test suite yet. The expectation is:

- The harness boots: `python -c "import benchmarks; import benchmarks.__main__"`
  must succeed.
- `python -m benchmarks --help` must list your new flag.
- A `--<phase>-only --max-queries 2 --skip-indexing` run must complete and
  produce a manifest, even with fake API keys (it will record connection
  errors as failed queries, which is the intended fallback behavior).

If you change scoring math, sanity-check by hand on a tiny synthetic case
the way the smoke at the bottom of the PR commit message did.

## Commit + branch

- Use `aayambansal/<short-topic>` for branch names if you're the
  maintainer. External contributors: any prefix is fine.
- Keep commits focused. The diagnosis + the diff-aware phase + session-replay work is
  one logical change; the package reorganization is another.
- Don't add `Co-Authored-By:` trailers; this repo has a single author of
  record.
