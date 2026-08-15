"""Open-loop load generator.

Three things make this open-loop rather than closed-loop, and all three matter:

  1. Arrival times are computed *up front* as absolute deadlines. The generator
     never derives the next send time from when the previous response came back.
  2. Dispatch is non-blocking. The arrival thread hands work to a worker pool and
     immediately returns to waiting for the next deadline, so a slow system
     causes the queue to grow instead of causing the offered load to drop.
  3. Latency is measured from the *scheduled* arrival time, not from when a
     worker picked the request up. This is what makes coordinated omission
     visible: a request that sat in the queue for 200ms before a 20ms service
     took 220ms, and reporting 20ms would be a lie.

Reported separately, because the plan requires the tail to be attributable:
  latency      = done - scheduled   (what a client experiences)
  service      = done - start       (what the server spent)
  queue_delay  = start - scheduled  (what the backlog cost)
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from harness.histogram import LatencyRecorder

# Below this, a sleep costs more than it saves on macOS (~1ms granularity), so
# the arrival thread spins instead.
_SPIN_THRESHOLD_S = 0.002


def poisson_arrivals(qps: float, duration_s: float, seed: int) -> list[float]:
    """Absolute arrival offsets, in seconds from the start of the run.

    Inter-arrivals are exponential, so the process is Poisson: real query traffic
    arrives in bursts, and a fixed-interval generator would never produce the
    transient queues that dominate the tail.
    """
    if qps <= 0:
        raise ValueError("qps must be positive")
    rng = random.Random(seed)
    arrivals: list[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(qps)
        if t >= duration_s:
            return arrivals
        arrivals.append(t)


class ZipfQuerySampler:
    """Samples queries with Zipfian popularity, head of the distribution first.

    Query traffic is heavily skewed and that skew is what a result cache exploits,
    so the Zipf exponent is the knob that drives cache hit rate in Phase 2.
    """

    def __init__(self, queries: Sequence[str], s: float = 1.0, seed: int = 0) -> None:
        if not queries:
            raise ValueError("no queries to sample from")
        self._queries = list(queries)
        self._rng = random.Random(seed)
        weights = [1.0 / (rank**s) for rank in range(1, len(self._queries) + 1)]
        total = sum(weights)
        self._cumulative: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self._cumulative.append(acc)

    def sample(self) -> str:
        import bisect

        return self._queries[bisect.bisect_left(self._cumulative, self._rng.random())]


@dataclass
class LoadResult:
    """Outcome of one open-loop run, with the tail split by where it came from."""

    arrivals: list[float]
    latency: LatencyRecorder = field(default_factory=LatencyRecorder)
    service: LatencyRecorder = field(default_factory=LatencyRecorder)
    queue_delay: LatencyRecorder = field(default_factory=LatencyRecorder)
    completed: int = 0
    errors: int = 0
    offered_qps: float = 0.0
    achieved_qps: float = 0.0
    wall_s: float = 0.0

    def summary(self) -> dict:
        return {
            "offered_qps": self.offered_qps,
            "achieved_qps": self.achieved_qps,
            "requests": len(self.arrivals),
            "completed": self.completed,
            "errors": self.errors,
            "wall_s": self.wall_s,
            "latency": self.latency.summary(),
            "service": self.service.summary(),
            "queue_delay": self.queue_delay.summary(),
        }


def _wait_until(deadline_ns: int) -> None:
    """Sleep to just before the deadline, then spin — macOS sleep is ~1ms coarse."""
    while True:
        remaining_s = (deadline_ns - time.perf_counter_ns()) / 1e9
        if remaining_s <= 0:
            return
        if remaining_s > _SPIN_THRESHOLD_S:
            time.sleep(remaining_s - _SPIN_THRESHOLD_S)


def run_open_loop(
    dispatch: Callable[[str], object],
    queries: Sequence[str],
    qps: float,
    duration_s: float,
    workers: int = 8,
    seed: int = 0,
    zipf_s: float = 1.0,
) -> LoadResult:
    """Drive `dispatch` at a fixed arrival rate and record where the time went.

    `workers` bounds concurrency, not arrivals: once all workers are busy, further
    arrivals queue, which is exactly the effect being measured.
    """
    arrivals = poisson_arrivals(qps, duration_s, seed)
    sampler = ZipfQuerySampler(queries, s=zipf_s, seed=seed)
    sampled = [sampler.sample() for _ in arrivals]
    result = LoadResult(arrivals=arrivals, offered_qps=qps)

    records: list[tuple[int, int, int, bool]] = []

    def work(query: str, scheduled_ns: int) -> None:
        start_ns = time.perf_counter_ns()
        ok = True
        try:
            dispatch(query)
        except Exception:
            ok = False
        records.append((scheduled_ns, start_ns, time.perf_counter_ns(), ok))

    run_start_ns = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset_s, query in zip(arrivals, sampled):
            scheduled_ns = run_start_ns + int(offset_s * 1e9)
            _wait_until(scheduled_ns)
            # submit() does not block, so a backlog grows here rather than
            # throttling the arrival process.
            pool.submit(work, query, scheduled_ns)
    wall_s = (time.perf_counter_ns() - run_start_ns) / 1e9

    for scheduled_ns, start_ns, done_ns, ok in records:
        if ok:
            result.completed += 1
        else:
            result.errors += 1
        result.latency.record((done_ns - scheduled_ns) / 1000)
        result.service.record((done_ns - start_ns) / 1000)
        result.queue_delay.record((start_ns - scheduled_ns) / 1000)

    result.wall_s = wall_s
    result.achieved_qps = len(records) / wall_s if wall_s > 0 else 0.0
    return result
