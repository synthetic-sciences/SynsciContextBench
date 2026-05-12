"""Utility scripts shared by the harness.

- ``dataset_loader``         Downloader for the validated datasets
                             (CodeSearchNet, CoSQA, AdvTest, etc.). Writes
                             to ``benchmarks/datasets/validated/``.
- ``create_benchmark_repo``  Lays out a small fake repo on disk so the
                             retrieval phase can be exercised without
                             real GitHub clones.
"""

from .dataset_loader import download_all

__all__ = ["download_all"]
