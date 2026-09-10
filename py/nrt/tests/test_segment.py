"""build_segment shells out to the real cpp/build/build_index binary (the
same unmodified binary every other phase uses) and opens the result via
cascade_index.Index -- this test needs the real binary and the real
compiled binding, so it skips (not a collection failure) if either is
missing, mirroring test_shard_cluster.py's convention.

SegmentSet.search fans out to every segment and merges by score -- this
test uses tiny real segments (a handful of synthetic docs each), not fakes,
because there is no fake for a real BM25-scored search result."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"
_READY = BUILD_INDEX_BIN.exists()

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="needs cpp/build/build_index (see README's Phase 1 setup)",
)

if _READY:
    from nrt.segment import Segment, SegmentSet, build_segment


def _write_tsv(path: Path, docs: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f"{docid}\t{text}\n" for docid, text in docs))
    return path


def test_build_segment_doc_count_matches_input_line_count(tmp_path):
    tsv = _write_tsv(tmp_path / "a.tsv", [
        ("d1", "alpha bravo charlie"),
        ("d2", "delta echo foxtrot"),
        ("d3", "golf hotel india"),
    ])
    segment = build_segment(tsv, tmp_path / "a_index")
    assert segment.doc_count == 3
    assert segment.source_tsv == tsv
    assert segment.index_dir == tmp_path / "a_index"


def test_build_segment_raises_if_binary_missing(tmp_path):
    tsv = _write_tsv(tmp_path / "a.tsv", [("d1", "x")])
    with pytest.raises(FileNotFoundError):
        build_segment(tsv, tmp_path / "a_index", build_index_bin=tmp_path / "no-such-binary")


def test_segment_set_returns_true_top_k_of_the_union(tmp_path):
    seg_a = build_segment(
        _write_tsv(tmp_path / "a.tsv", [
            ("d1", "quokka wombat quokka quokka"),
            ("d2", "unrelated passage about weather"),
        ]),
        tmp_path / "a_index",
    )
    seg_b = build_segment(
        _write_tsv(tmp_path / "b.tsv", [
            ("d3", "quokka sighting in tasmania"),
            ("d4", "another unrelated passage"),
        ]),
        tmp_path / "b_index",
    )

    segment_set = SegmentSet([seg_a, seg_b])
    results = segment_set.search("quokka", k=10)

    ids = [external_id for external_id, _ in results]
    assert ids[0] == "d1"  # highest term frequency for "quokka"
    assert "d3" in ids
    assert "d2" not in ids and "d4" not in ids
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_segment_set_truncates_to_k(tmp_path):
    seg = build_segment(
        _write_tsv(tmp_path / "a.tsv", [
            ("d1", "kiwi kiwi kiwi"),
            ("d2", "kiwi kiwi"),
            ("d3", "kiwi"),
        ]),
        tmp_path / "a_index",
    )
    results = SegmentSet([seg]).search("kiwi", k=2)
    assert [eid for eid, _ in results] == ["d1", "d2"]


def test_segment_set_with_no_segments_returns_empty():
    assert SegmentSet([]).search("anything", k=10) == []
