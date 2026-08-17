"""select_subset_ids is pure (no corpus I/O) so these run against small
synthetic inputs, matching this repo's existing test style (py/tests/test_metrics.py
uses randomized synthetic qrels/runs rather than the real corpus)."""

from __future__ import annotations

from dense.subset import select_subset_ids


def test_every_qrels_docid_is_in_the_subset():
    qrels_ids = {5, 10, 999}
    subset = select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=100)
    assert qrels_ids <= subset


def test_subset_size_is_exact():
    qrels_ids = {5, 10, 999}
    subset = select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=100)
    assert len(subset) == 100


def test_same_seed_is_deterministic():
    qrels_ids = {5, 10, 999}
    first = select_subset_ids(qrels_ids, total_docs=5000, seed=42, subset_size=200)
    second = select_subset_ids(qrels_ids, total_docs=5000, seed=42, subset_size=200)
    assert first == second


def test_different_seeds_differ():
    qrels_ids = {5, 10, 999}
    first = select_subset_ids(qrels_ids, total_docs=5000, seed=1, subset_size=200)
    second = select_subset_ids(qrels_ids, total_docs=5000, seed=2, subset_size=200)
    assert first != second


def test_qrels_larger_than_subset_size_raises():
    qrels_ids = set(range(50))
    try:
        select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
