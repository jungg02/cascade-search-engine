"""SegmentSet filters tombstoned external ids out of the merged result and
over-fetches (k + len(tombstones)) from each segment so a tombstoned hit
doesn't shrink the merged top-k below k (spec §3, §5)."""

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
    from nrt.segment import SegmentSet, build_segment


def _write_tsv(path: Path, docs: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f"{docid}\t{text}\n" for docid, text in docs))
    return path


def test_tombstoned_doc_never_appears_even_if_top_1(tmp_path):
    seg = build_segment(
        _write_tsv(tmp_path / "a.tsv", [
            ("d1", "narwhal narwhal narwhal narwhal"),  # would be top-1 for "narwhal"
            ("d2", "narwhal sighting"),
            ("d3", "narwhal mentioned once"),
        ]),
        tmp_path / "a_index",
    )

    live = SegmentSet([seg]).search("narwhal", k=2)
    assert [eid for eid, _ in live] == ["d1", "d2"]

    tombstoned = SegmentSet([seg], tombstones={"d1"}).search("narwhal", k=2)
    ids = [eid for eid, _ in tombstoned]
    assert "d1" not in ids
    # Over-fetch means the k-th slot is backfilled by a different doc,
    # not silently shrunk to k-1.
    assert len(ids) == 2
    assert ids == ["d2", "d3"]
