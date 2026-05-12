"""Shared seeded-sampling helpers.

Benchmark modules historically truncated `queries[:max_queries]`, which has
two problems:

1. It always selects the first N items, so the easy/hard distribution depends
   on the dataset's authoring order rather than a representative draw.
2. It is impossible to run "another seed" without rewriting the file.

`sample_seeded` replaces the slice with a deterministic random sample. The
order returned is the random sample order (so traces remain reproducible).
"""

from __future__ import annotations

import random
from typing import Iterable, TypeVar

T = TypeVar("T")


def sample_seeded(
    items: Iterable[T],
    max_items: int | None,
    seed: int = 0,
) -> list[T]:
    """Return up to `max_items` from `items` using a seeded RNG.

    - If `max_items` is None or larger than `len(items)`, returns the input
      list as-is (no sampling).
    - Otherwise returns a deterministic sample of size `max_items`.

    The same `seed` produces the same draw across modules, so different
    engines see the same set of queries within one run.
    """
    pool = list(items)
    if max_items is None or max_items >= len(pool):
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, max_items)


def stratified_sample(
    items: list[T],
    max_items: int | None,
    *,
    key,
    seed: int = 0,
) -> list[T]:
    """Stratified sample: keep proportions of distinct `key(item)` values.

    Useful for keeping the difficulty/category mix stable when sub-sampling.
    Returns at most `max_items` items.
    """
    if max_items is None or max_items >= len(items):
        return items
    rng = random.Random(seed)
    buckets: dict[object, list[T]] = {}
    for it in items:
        buckets.setdefault(key(it), []).append(it)
    out: list[T] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
    # Round-robin draw across buckets so proportions stay close.
    while len(out) < max_items and any(buckets.values()):
        for k, bucket in buckets.items():
            if not bucket:
                continue
            out.append(bucket.pop())
            if len(out) >= max_items:
                break
    return out
