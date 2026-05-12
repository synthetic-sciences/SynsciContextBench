"""Operational infrastructure shared across phases.

- ``logging_config``  Structured logging + per-query trace persistence
                      (``QueryTrace`` / ``TraceStore``). Trace files are
                      written under ``benchmarks/results/run_<ts>/traces/``.
- ``sampling``        Seeded + stratified query sub-sampling. Every phase
                      pulls from here instead of slicing ``[:N]`` so all
                      engines see the same draw within a run.
- ``latency``         End-to-end ``LatencyMeter`` that records request,
                      retry, and sleep buckets. Adapters use it to report
                      full user-visible latency instead of request-only.
- ``consistency``     Self-consistency checks across repeated runs.
"""

from .latency import LatencyBreakdown, LatencyMeter, empty_breakdown
from .logging_config import QueryTrace, TraceStore, get_logger, setup_logging
from .sampling import sample_seeded, stratified_sample

__all__ = [
    "LatencyBreakdown",
    "LatencyMeter",
    "QueryTrace",
    "TraceStore",
    "empty_breakdown",
    "get_logger",
    "sample_seeded",
    "setup_logging",
    "stratified_sample",
]
