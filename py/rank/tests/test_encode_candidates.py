"""unique_candidate_docids is pure (no I/O), tested against small synthetic
run dicts rather than the real WAND run files."""

from __future__ import annotations

from rank.encode_candidates import unique_candidate_docids


def test_union_across_multiple_runs():
    run_a = {"q1": {"d1": 1.0, "d2": 0.5}}
    run_b = {"q2": {"d2": 0.9, "d3": 0.1}}
    result = unique_candidate_docids([run_a, run_b])
    assert result == {"d1", "d2", "d3"}


def test_empty_runs_list_is_empty_set():
    assert unique_candidate_docids([]) == set()


def test_single_run_single_query():
    run_a = {"q1": {"d1": 1.0}}
    assert unique_candidate_docids([run_a]) == {"d1"}
