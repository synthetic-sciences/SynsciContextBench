"""End-to-end latency accounting.

Context7's request-delay sleeps and Nia's rate-limit backoffs are real user
time — silently dropping them from a benchmark makes those engines look
faster than they are. The previous adapters timed ``await client.post(...)``
only, which hides everything that happens around the request.

``LatencyMeter`` records:

- ``total_ms`` — wall-clock from ``start()`` to ``stop()``.
- ``request_ms`` — time spent in the actual HTTP / process call.
- ``retry_ms`` — time spent in re-tries.
- ``sleep_ms`` — explicit backoff / rate-limit sleeps.
- ``other_ms`` — anything else (auth refresh, parsing, etc.).

Adapters can wrap their request paths with ``with meter.measure("request"): ...``
and call ``meter.note_sleep(seconds)`` whenever they sleep. The runner then
records ``total_ms`` as the user-visible latency. The benchmark reports both
``avg_user_latency_ms`` (full) and ``avg_request_latency_ms`` (request-only)
so we can show the gap each engine introduces.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field


@dataclass
class LatencyBreakdown:
    total_ms: float = 0.0
    request_ms: float = 0.0
    retry_ms: float = 0.0
    sleep_ms: float = 0.0
    other_ms: float = 0.0
    retry_count: int = 0
    sleep_count: int = 0

    def as_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "request_ms": round(self.request_ms, 2),
            "retry_ms": round(self.retry_ms, 2),
            "sleep_ms": round(self.sleep_ms, 2),
            "other_ms": round(self.other_ms, 2),
            "retry_count": self.retry_count,
            "sleep_count": self.sleep_count,
        }


@dataclass
class LatencyMeter:
    """A small accumulator. Use within one logical 'user call'."""

    breakdown: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    _started_at: float | None = None

    def start(self) -> None:
        self._started_at = time.perf_counter()

    def stop(self) -> LatencyBreakdown:
        if self._started_at is None:
            return self.breakdown
        self.breakdown.total_ms = (time.perf_counter() - self._started_at) * 1000
        # 'other' is whatever wall-clock isn't attributable to a tracked bucket.
        tracked = (
            self.breakdown.request_ms
            + self.breakdown.retry_ms
            + self.breakdown.sleep_ms
        )
        self.breakdown.other_ms = max(0.0, self.breakdown.total_ms - tracked)
        return self.breakdown

    @contextlib.contextmanager
    def measure(self, bucket: str):
        """Time a block and add it to one of the buckets."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            if bucket == "request":
                self.breakdown.request_ms += elapsed
            elif bucket == "retry":
                self.breakdown.retry_ms += elapsed
                self.breakdown.retry_count += 1
            elif bucket == "sleep":
                self.breakdown.sleep_ms += elapsed
                self.breakdown.sleep_count += 1
            else:
                self.breakdown.other_ms += elapsed

    async def sleep(self, seconds: float) -> None:
        """Sleep helper that records into the sleep bucket."""
        if seconds <= 0:
            return
        with self.measure("sleep"):
            await asyncio.sleep(seconds)

    def note_sleep(self, seconds: float) -> None:
        """Record a sleep that already happened (sync code path)."""
        self.breakdown.sleep_ms += seconds * 1000
        self.breakdown.sleep_count += 1


def empty_breakdown() -> LatencyBreakdown:
    """Return a fresh zeroed breakdown."""
    return LatencyBreakdown()
