"""TieredMergePolicy fires as soon as segment count exceeds max_segments,
returning the merge_factor smallest-by-doc_count segments to collapse into
one -- smallest-first keeps merge cost low and bounds it as re-tokenization
cost, per design spec §4a."""

from __future__ import annotations

from dataclasses import dataclass

from nrt.merge_policy import TieredMergePolicy


@dataclass
class _FakeSegment:
    doc_count: int


def test_no_merge_when_at_or_below_max_segments():
    policy = TieredMergePolicy(merge_factor=2, max_segments=3)
    segments = [_FakeSegment(doc_count=n) for n in (10, 20, 30)]
    assert policy.maybe_merge(segments) is None


def test_merges_the_smallest_segments_once_over_max():
    policy = TieredMergePolicy(merge_factor=2, max_segments=3)
    segments = [_FakeSegment(doc_count=n) for n in (50, 10, 30, 5)]
    plan = policy.maybe_merge(segments)
    assert plan is not None
    assert sorted(s.doc_count for s in plan) == [5, 10]


def test_merge_factor_larger_than_segment_count_merges_everything():
    policy = TieredMergePolicy(merge_factor=10, max_segments=1)
    segments = [_FakeSegment(doc_count=n) for n in (3, 1, 2)]
    plan = policy.maybe_merge(segments)
    assert sorted(s.doc_count for s in plan) == [1, 2, 3]


def test_default_thresholds():
    policy = TieredMergePolicy()
    assert policy.merge_factor == 4
    assert policy.max_segments == 8
