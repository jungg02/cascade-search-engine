"""TREC run files are the interchange format between our engine, Anserini, and
trec_eval. If the ordering or formatting drifts, cross-engine comparison silently
compares different rankings."""

from harness.runfile import read_run, write_run


def test_run_file_is_written_in_descending_score_order_with_dense_ranks(tmp_path):
    path = tmp_path / "run.txt"
    write_run({"q1": {"dA": 1.0, "dB": 3.0, "dC": 2.0}}, path, tag="cascade")

    lines = path.read_text().strip().split("\n")

    assert lines[0].split() == ["q1", "Q0", "dB", "1", "3.0", "cascade"]
    assert [line.split()[3] for line in lines] == ["1", "2", "3"]


def test_run_file_round_trips(tmp_path):
    path = tmp_path / "run.txt"
    run = {"q1": {"dA": 1.5, "dB": 3.25}, "q2": {"dC": 0.5}}

    write_run(run, path, tag="cascade")

    assert read_run(path) == run


def test_write_run_respects_a_depth_cutoff(tmp_path):
    path = tmp_path / "run.txt"
    write_run({"q1": {f"d{i}": float(i) for i in range(50)}}, path, tag="t", depth=10)

    assert len(path.read_text().strip().split("\n")) == 10
