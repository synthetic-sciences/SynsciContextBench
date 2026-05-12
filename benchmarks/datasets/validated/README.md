# datasets/validated/

Downloaded standard datasets, regenerable via:

```bash
uv run python -m benchmarks --download-datasets
uv run python -m benchmarks --download-datasets --dataset-max-samples 1000
```

| File | Source | Used by |
|------|--------|---------|
| `codesearchnet_benchmark.json`        | Husain et al. 2019 — CodeSearchNet                       | Phase 6 (validated retrieval), Phase 8 (enhanced judge) |
| `cosqa_benchmark.json`                | Huang et al. 2021 — CoSQA (ACL)                          | Phase 6, Phase 8 |
| `advtest_benchmark.json`              | CodeSearchNet AdvTest split                              | Phase 6, supplementary |
| `codefeedback_st_benchmark.json`      | CodeFeedback Single-Turn                                 | Phase 6 |
| `stackoverflow_qa_benchmark.json`     | StackOverflow code QA pairs                              | Phase 6 |
| `apps_benchmark.json`                 | APPS (introductory problems)                             | Phase 6 |

All files share the same internal shape:

```jsonc
{
  "_description": "...",
  "queries": [{ "id": "...", "query": "...", "language": "python" }, ...],
  "corpus":  [{ "id": "...", "content": "...", "language": "python" }, ...],
  "qrels":   [{ "query_id": "...", "doc_id": "...", "relevance": 0..3 }, ...]
}
```

`benchmarks/phases/validated_eval.py` handles all three lookups (file-path,
content-similarity, or LLM-judge) and respects the `--judge-top-k` setting
(default 10) — the previous hard-coded cap of 3 silently dropped any rank-4+
result from the recall calculation.

## Licensing

These datasets ship under their original licenses (CodeSearchNet — MIT;
CoSQA — research, see paper; AdvTest — MIT; APPS — MIT). The downloader
fetches them directly from HuggingFace; we do not redistribute the raw
corpora through this repo.
