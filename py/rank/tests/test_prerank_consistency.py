"""prerank_consistency is pure -- same shape as dense/ann_sweep.py's
recall_at_k, tested against hand-constructed examples."""

from __future__ import annotations

from rank.prerank_consistency import prerank_consistency


def test_full_survival_is_1():
    true_top_k = ["d1", "d2", "d3"]
    survivors = {"d1", "d2", "d3", "d4"}  # strict superset
    assert prerank_consistency(true_top_k, survivors) == 1.0


def test_partial_survival():
    true_top_k = ["d1", "d2", "d3", "d4"]
    survivors = {"d1", "d3"}  # strict subset, 2 of 4 survive
    assert prerank_consistency(true_top_k, survivors) == 0.5


def test_no_survival_is_0():
    true_top_k = ["d1", "d2"]
    survivors = {"d3", "d4"}
    assert prerank_consistency(true_top_k, survivors) == 0.0


def test_empty_true_top_k_is_0_not_a_zero_division():
    assert prerank_consistency([], {"d1"}) == 0.0
