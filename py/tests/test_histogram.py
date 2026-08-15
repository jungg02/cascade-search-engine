"""The plan forbids reporting a mean latency, so the recorder must not offer one."""

import pytest

from harness.histogram import LatencyRecorder


def test_percentiles_use_nearest_rank_over_the_recorded_samples():
    recorder = LatencyRecorder()
    for value_us in range(1, 101):
        recorder.record(value_us)

    assert recorder.percentile(50) == 50
    assert recorder.percentile(99) == 99
    assert recorder.percentile(100) == 100


def test_p999_is_distinct_from_the_max_once_there_are_a_thousand_samples():
    # Nearest rank: 99.9% of 1000 samples is the 999th, which is not the worst.
    # If p99.9 ever equals max, the run is too short to resolve it.
    recorder = LatencyRecorder()
    for value_us in range(1, 1001):
        recorder.record(value_us)

    assert recorder.percentile(99.9) == 999
    assert recorder.percentile(100) == 1000


def test_summary_reports_the_tail_and_refuses_to_report_a_mean():
    recorder = LatencyRecorder()
    for value_us in range(1, 101):
        recorder.record(value_us)

    summary = recorder.summary()

    assert set(summary) == {"count", "min_us", "p50_us", "p95_us", "p99_us", "p999_us", "max_us"}


def test_summary_of_an_empty_recorder_is_a_zero_count_not_a_crash():
    assert LatencyRecorder().summary()["count"] == 0


def test_percentile_of_an_empty_recorder_raises_rather_than_inventing_a_number():
    with pytest.raises(ValueError):
        LatencyRecorder().percentile(99)
