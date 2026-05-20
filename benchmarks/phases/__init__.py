"""Benchmark phases.

Each module in this package implements one phase of the harness. Phases share
a common shape — they take a list of engine adapters and a path to a curated
dataset, run the engines concurrently, score per-query, and return a
``(aggregate, per_query)`` pair (or, for newer phases, a dataclass report).

Phases registered today (in the order the runner executes them):

1.  ``runner.run_retrieval_benchmark``      Precision/Recall/NDCG/MRR
2.  ``multihop``                            cross-file retrieval
3.  ``code_qa``                             function/symbol QA
4.  ``adversarial``                         decoys (same name, wrong context)
5.  ``hallucination``                       does context prevent invented APIs
6.  ``validated_eval``                      CodeSearchNet / CoSQA / AdvTest
7.  ``runner._enhanced_judge``              position-debiased 4D judge
8.  ``swe_agent``                           code generation with/without context
8b. ``swe_real_patch``                      opt-in: clone, apply, run tests
9.  ``atlas``                              tool contracts / graph / paper QA
10. ``session_replay``                      replay real production losses

Each phase is independently runnable through the CLI ``--<phase>-only`` flag.
"""

from .adversarial import run_adversarial_benchmark
from .code_qa import run_code_qa_benchmark
from .hallucination import run_hallucination_benchmark
from .multihop import run_multihop_benchmark
from .session_replay import run_session_replay_benchmark
from .swe_agent import run_swe_agent_benchmark
from .swe_real_patch import run_real_patch
from .atlas import run_atlas_benchmark
from .validated_eval import run_validated_benchmark

__all__ = [
    "run_adversarial_benchmark",
    "run_code_qa_benchmark",
    "run_hallucination_benchmark",
    "run_multihop_benchmark",
    "run_session_replay_benchmark",
    "run_swe_agent_benchmark",
    "run_real_patch",
    "run_atlas_benchmark",
    "run_validated_benchmark",
]
