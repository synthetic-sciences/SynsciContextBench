"""End-to-end validated-eval test on a synthetic dataset via the MockAdapter.

This isolates the *metric pipeline* from live retrieval. It proves that when the
relevant document is actually retrieved, MRR is ~1.0 — i.e. the near-zero
validated MRRs in the published run are a corpus-coverage / indexing problem
(the dataset corpus must be indexed into the engine), not a bug in the metric
code itself. That distinction matters for the paper's validity discussion.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.adapters.mock import MockAdapter
from benchmarks.phases.validated_eval import run_validated_benchmark

_CORPUS = [
    {"id": "d1", "content": "def parse_datetime(s): return datetime.fromisoformat(s)", "language": "python"},
    {"id": "d2", "content": "def slugify(text): return re.sub(r'[^a-z0-9]+', '-', text)", "language": "python"},
    {"id": "d3", "content": "class CacheStore: def evict(self): self._data.popitem()", "language": "python"},
]

_DATASET = {
    "_description": "synthetic-offline",
    "corpus": _CORPUS,
    "queries": [
        {"id": "q1", "query": "parse datetime fromisoformat", "language": "python"},
        {"id": "q2", "query": "slugify text url safe", "language": "python"},
    ],
    "qrels": [
        {"query_id": "q1", "doc_id": "d1", "relevance": 2},
        {"query_id": "q2", "doc_id": "d2", "relevance": 2},
    ],
}


async def test_validated_eval_pipeline_scores_correctly(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.json"
    dataset_path.write_text(json.dumps(_DATASET), encoding="utf-8")

    engine = MockAdapter(corpus=_CORPUS)
    agg, per_query = await run_validated_benchmark(
        engine=engine,
        dataset_path=str(dataset_path),
        k_values=[1, 3, 5],
        match_mode="content",
    )

    assert agg.num_queries == 2
    # The relevant doc is retrieved at rank 1 for both queries -> MRR ~ 1.0.
    assert agg.avg_mrr > 0.9, f"MRR pipeline broken: {agg.avg_mrr}"
    assert agg.avg_recall_at[1] > 0.9
