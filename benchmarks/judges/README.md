# judges/

LLM-as-judge implementations. Two variants:

| Module | Purpose |
|--------|---------|
| `llm_judge.py` | 3D blind scoring (relevance, completeness, faithfulness). Used by each per-phase fairness pass and reused by Thesis, SWE-Agent, and validated-eval through `_call_llm_judge_raw` + `_safe_parse_json`. |
| `enhanced_judge.py` | Position-debiased 4D scoring with faithfulness and RAGAS-style context-quality metrics. Each query is scored twice with shuffled chunk order to suppress positional bias documented in Zheng et al. (2023). |

## What goes here vs. `scoring/`

- **judges/**  → LLM-driven scoring. The output is a model judgment.
- **scoring/** → deterministic / formula-driven scoring. MRR, NDCG, anchor
  hit rates, citation share, etc.

Both are inputs to a phase's report dataclass — most phases blend them.

## Calling convention

```python
text = await _call_llm_judge_raw(
    system_prompt=...,
    user_prompt=...,
    llm_provider=...,    # "anthropic" | "gemini" | "openai"
    llm_model=...,
    llm_api_key=...,
)
parsed = _safe_parse_json(text, defaults={...})
```

Always pass `defaults` to `_safe_parse_json` so a partial/garbled response
falls back to a numeric zero rather than raising. Phases that depend on the
judge should still degrade gracefully when no API key is configured.

## Validated-eval `judge_top_k`

The validated-eval judge used to be hard-coded to score the top 3 results
only, which silently forced rank-4+ to be irrelevant and capped `Recall@10`
at `min(3, total_relevant) / total_relevant`. The judge now respects
`--judge-top-k` (default 10) per the diagnosis fix.
