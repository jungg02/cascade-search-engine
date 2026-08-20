"""RRF and normalized score fusion are pure functions over per-channel
(docid -> score) dicts for a single query -- tested against small,
hand-computed examples, matching this repo's style for pure retrieval-math
functions (py/tests/test_metrics.py)."""

from __future__ import annotations

from dense.fusion import normalized_score_fusion, reciprocal_rank_fusion


def test_rrf_hand_computed_two_channels():
    # channel A ranks: d1 (1st), d2 (2nd); channel B ranks: d2 (1st), d3 (2nd)
    channel_a = {"d1": 10.0, "d2": 5.0}
    channel_b = {"d2": 0.9, "d3": 0.1}
    fused = reciprocal_rank_fusion([channel_a, channel_b], k=60)
    # d1: rank 1 in A only -> 1/61. d2: rank 2 in A, rank 1 in B -> 1/62 + 1/61.
    # d3: rank 2 in B only -> 1/62.
    assert fused["d1"] == 1 / 61
    assert fused["d2"] == 1 / 62 + 1 / 61
    assert fused["d3"] == 1 / 62
    # d2 should rank highest (present in both channels)
    assert fused["d2"] > fused["d1"] > fused["d3"]


def test_rrf_document_absent_from_a_channel_is_not_dropped():
    channel_a = {"only_in_a": 1.0}
    channel_b: dict[str, float] = {}
    fused = reciprocal_rank_fusion([channel_a, channel_b], k=60)
    assert "only_in_a" in fused
    assert fused["only_in_a"] == 1 / 61


def test_normalized_score_fusion_min_max_per_channel():
    channel_a = {"d1": 10.0, "d2": 0.0}  # normalizes to 1.0, 0.0
    channel_b = {"d1": 2.0, "d2": 4.0}  # normalizes to 0.0, 1.0
    fused = normalized_score_fusion([channel_a, channel_b])
    assert fused["d1"] == 1.0
    assert fused["d2"] == 1.0


def test_normalized_score_fusion_absent_document_contributes_zero():
    channel_a = {"d1": 10.0, "d2": 0.0}
    channel_b = {"d1": 5.0}
    fused = normalized_score_fusion([channel_a, channel_b])
    # d1 is the only doc in channel_b, so it min-max normalizes to 0.0 there
    # (min == max == 5.0 -> defined as 0.0, see implementation).
    assert fused["d1"] == 1.0 + 0.0
    assert fused["d2"] == 0.0 + 0.0


def test_normalized_score_fusion_single_value_channel_does_not_divide_by_zero():
    channel_a = {"only_doc": 7.0}
    fused = normalized_score_fusion([channel_a])
    assert fused["only_doc"] == 0.0
