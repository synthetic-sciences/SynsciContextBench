# datasets/

```
datasets/
├── curated/      hand-built test cases owned by this repo
└── validated/    downloaded standard datasets (CodeSearchNet, CoSQA, ...)
```

Two categories with very different ownership:

- **`curated/`** — JSON files that this repo authors and ships. Editable by
  reviewers. Includes the Phase 10 (the diff-aware phase) and Phase 11 (session replay)
  cases that came out of the diagnosis.
- **`validated/`** — Data downloaded from third-party sources via
  `python -m benchmarks --download-datasets`. Larger files, regenerated on
  demand. Not edited by hand.

See each subfolder's README for the schema each file follows.

## Adding a new curated dataset

1. Drop a JSON file in `curated/` with a `_description` and `_methodology`
   field at the top — the existing files are the format reference.
2. Update the relevant phase under `benchmarks/phases/` to load it from
   `config.curated_dir`.
3. Document the schema in `curated/README.md`.

## Adding a new validated dataset

1. Add a downloader function in `benchmarks/utils/dataset_loader.py`. The
   output file must land under `validated/`.
2. Add the filename to the appropriate list in `benchmarks/runner.py`
   (`all_validated_datasets`, `judge_datasets`, or `enh_datasets`).
3. Update `validated/README.md` with the source URL and license note.
