"""LLM-as-judge implementations.

- ``llm_judge``       3D blind scoring used by the per-phase fairness pass.
- ``enhanced_judge``  Position-debiased 4D scoring with faithfulness and
                      RAGAS-style context-quality metrics. Each query is
                      scored twice with shuffled chunk order to suppress the
                      ~10% positional bias documented in Zheng et al. (2023).

Both judges accept a provider/model/api_key triple and return structured
scores. The runner picks between them via CLI flags.
"""

from .enhanced_judge import (
    print_enhanced_judge_summary,
    run_enhanced_judge_benchmark,
)
from .llm_judge import (
    JudgeAggregateMetrics,
    print_judge_summary,
    run_judge_benchmark,
)

__all__ = [
    "JudgeAggregateMetrics",
    "print_judge_summary",
    "print_enhanced_judge_summary",
    "run_judge_benchmark",
    "run_enhanced_judge_benchmark",
]
