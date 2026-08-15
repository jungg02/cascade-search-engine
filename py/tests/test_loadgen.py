"""Open-loop load generation.

The plan is explicit that a closed-loop generator invalidates the whole benchmark:
it cannot produce a queue, so it never shows the latency knee. The load-bearing
test here is test_latency_is_measured_from_the_scheduled_arrival_time — that is
the property that distinguishes open-loop from closed-loop, and it is the one
thing a plausible-looking implementation gets wrong.
"""

import statistics
import time

import pytest

from harness.loadgen import ZipfQuerySampler, poisson_arrivals, run_open_loop


def test_poisson_arrivals_have_the_requested_mean_rate():
    arrivals = poisson_arrivals(qps=1000, duration_s=10.0, seed=7)

    inter_arrivals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert statistics.mean(inter_arrivals) == pytest.approx(1 / 1000, rel=0.05)
    assert len(arrivals) == pytest.approx(10_000, rel=0.05)


def test_poisson_arrivals_are_exponential_not_uniform():
    # An exponential distribution has stdev equal to its mean; a fixed-interval
    # generator would have stdev 0 and would never produce bursts.
    arrivals = poisson_arrivals(qps=1000, duration_s=10.0, seed=7)

    inter_arrivals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert statistics.stdev(inter_arrivals) == pytest.approx(1 / 1000, rel=0.1)


def test_poisson_arrivals_are_deterministic_for_a_seed():
    assert poisson_arrivals(qps=50, duration_s=1.0, seed=3) == poisson_arrivals(
        qps=50, duration_s=1.0, seed=3
    )


def test_zipf_sampler_makes_head_queries_dominate():
    queries = [f"q{i}" for i in range(1000)]
    sampler = ZipfQuerySampler(queries, s=1.0, seed=11)

    draws = [sampler.sample() for _ in range(20_000)]

    assert draws.count("q0") > draws.count("q9") > draws.count("q999")


def test_zipf_skew_increases_with_s():
    queries = [f"q{i}" for i in range(1000)]

    flat_sampler = ZipfQuerySampler(queries, s=0.6, seed=2)
    peaked_sampler = ZipfQuerySampler(queries, s=1.4, seed=2)
    flat = [flat_sampler.sample() for _ in range(10_000)]
    peaked = [peaked_sampler.sample() for _ in range(10_000)]

    assert peaked.count("q0") > flat.count("q0")


def test_latency_is_measured_from_the_scheduled_arrival_time():
    # One worker, 20ms of service, offered load at 2x capacity. A closed-loop
    # generator would report ~20ms because it would only send the next request
    # after the previous returned. Open-loop must show the queue building.
    result = run_open_loop(
        dispatch=lambda query: time.sleep(0.020),
        queries=["a"],
        qps=100,
        duration_s=0.2,
        workers=1,
        seed=1,
    )

    assert result.completed == len(result.arrivals)
    assert result.service.percentile(50) == pytest.approx(20_000, rel=0.5)
    # Queueing must dominate: the reported tail is many times the service time.
    assert result.latency.percentile(99) > 5 * result.service.percentile(50)


def test_an_unloaded_run_reports_latency_close_to_service_time():
    # The complement of the test above: with no queueing the two must agree,
    # otherwise the scheduled-time accounting is adding a constant bias.
    # 1ms of service at 200 QPS across 8 workers is ~2% utilisation, so the only
    # gap between latency and service is scheduling noise.
    result = run_open_loop(
        dispatch=lambda query: time.sleep(0.001),
        queries=["a"],
        qps=200,
        duration_s=1.0,
        workers=8,
        seed=1,
    )

    assert result.latency.percentile(50) < 3 * result.service.percentile(50)


def test_dispatch_failures_are_counted_and_do_not_stop_the_run():
    def explode(query):
        raise RuntimeError("shard unavailable")

    result = run_open_loop(
        dispatch=explode, queries=["a"], qps=100, duration_s=0.1, workers=2, seed=1
    )

    assert result.errors == len(result.arrivals)
    assert result.completed == 0
