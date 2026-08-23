"""unique_candidate_docids is pure (no I/O), tested against small synthetic
run dicts rather than the real WAND run files."""

from __future__ import annotations

from pathlib import Path

from harness.runfile import read_run
from rank.encode_candidates import read_run_truncated, truncate_to_top_k, unique_candidate_docids


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


def test_truncate_to_top_k_keeps_highest_scores():
    run = {"q1": {"d1": 1.0, "d2": 5.0, "d3": 3.0, "d4": 2.0}}
    result = truncate_to_top_k(run, k=2)
    assert result == {"q1": {"d2": 5.0, "d3": 3.0}}


def test_truncate_to_top_k_leaves_short_queries_unchanged():
    run = {"q1": {"d1": 1.0}}
    result = truncate_to_top_k(run, k=5)
    assert result == {"q1": {"d1": 1.0}}


def test_read_run_truncated_matches_read_run_then_truncate(tmp_path: Path):
    # A real run file, deliberately pre-sorted descending per query (the
    # invariant read_run_truncated relies on -- matches how
    # harness.runfile.write_run always writes, and how every WAND run in
    # this project is produced).
    run_file = tmp_path / "run.txt"
    run_file.write_text(
        "q1 Q0 d2 1 5.0 tag\n"
        "q1 Q0 d3 2 3.0 tag\n"
        "q1 Q0 d4 3 2.0 tag\n"
        "q1 Q0 d1 4 1.0 tag\n"
        "q2 Q0 d5 1 9.0 tag\n"
    )
    streamed = read_run_truncated(run_file, k=2)
    materialized = truncate_to_top_k(read_run(run_file), k=2)
    assert streamed == materialized
    assert streamed == {"q1": {"d2": 5.0, "d3": 3.0}, "q2": {"d5": 9.0}}
