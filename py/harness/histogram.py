"""Latency recording.

The plan calls for an HDR histogram. HDR's advantage is bounded memory at sample
counts we will not reach: a benchmark run here is O(10^5)-O(10^6) samples, which
is a few MB of float64 and gives *exact* percentiles instead of bucketed ones.
We keep raw samples and sort. If a run ever gets large enough for this to hurt,
swap the storage — the interface is percentile-based either way.

There is deliberately no mean. A mean latency is not a number this project
reports; the tail is the whole point.
"""

from __future__ import annotations

import math

import numpy as np

PERCENTILES = (50, 95, 99, 99.9)


class LatencyRecorder:
    """Accumulates latency samples in microseconds and reports exact percentiles."""

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._sorted: np.ndarray | None = None

    def record(self, latency_us: float) -> None:
        self._samples.append(float(latency_us))
        self._sorted = None

    def record_many(self, latencies_us) -> None:
        self._samples.extend(float(v) for v in latencies_us)
        self._sorted = None

    def __len__(self) -> int:
        return len(self._samples)

    def _ordered(self) -> np.ndarray:
        if self._sorted is None:
            self._sorted = np.sort(np.asarray(self._samples, dtype=np.float64))
        return self._sorted

    def percentile(self, p: float) -> float:
        """Nearest-rank percentile: the smallest sample >= p% of the distribution.

        Nearest-rank rather than interpolated because an interpolated p99.9 invents
        a latency that no request actually experienced.
        """
        if not self._samples:
            raise ValueError("no samples recorded")
        ordered = self._ordered()
        # round() before ceil so 0.999 * 1000 doesn't land on 999.0000000001.
        rank = max(1, math.ceil(round(p / 100.0 * len(ordered), 9)))
        return float(ordered[min(rank, len(ordered)) - 1])

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {"count": 0}
        ordered = self._ordered()
        return {
            "count": len(ordered),
            "min_us": float(ordered[0]),
            "p50_us": self.percentile(50),
            "p95_us": self.percentile(95),
            "p99_us": self.percentile(99),
            "p999_us": self.percentile(99.9),
            "max_us": float(ordered[-1]),
        }
