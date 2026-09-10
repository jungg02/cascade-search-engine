"""NrtIndex buffers adds and flushes into a new segment once
flush_threshold_docs is reached; a doc is NOT searchable immediately after
add() but IS searchable immediately after the flush that contains it --
the core assumption the lag experiment (nrt.lag_experiment) times directly.
Needs the real build_index binary + cascade_index binding (no fake for a
real BM25 search), so it skips if either is missing."""

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
    from nrt.index import NrtIndex
    from nrt.segment import build_segment


def _ids(results: list[tuple[str, float]]) -> set[str]:
    return {eid for eid, _ in results}


def test_auto_flush_at_threshold_creates_a_segment(tmp_path):
    nrt = NrtIndex(tmp_path, flush_threshold_docs=3)
    for i in range(3):
        nrt.add(f"d{i}", f"jackfruit passage number {i}")
    assert nrt.segment_count == 1
    nrt.close()


def test_no_flush_before_threshold_and_manual_flush_creates_the_next_segment(tmp_path):
    nrt = NrtIndex(tmp_path, flush_threshold_docs=3)
    for i in range(3):
        nrt.add(f"d{i}", f"jackfruit passage number {i}")
    assert nrt.segment_count == 1

    for i in range(3, 5):
        nrt.add(f"d{i}", f"jackfruit passage number {i}")
    assert nrt.segment_count == 1  # 2 buffered docs, not yet at threshold

    nrt.flush()
    assert nrt.segment_count == 2
    nrt.close()


def test_doc_not_searchable_before_flush_but_searchable_after(tmp_path):
    nrt = NrtIndex(tmp_path, flush_threshold_docs=100)  # high -- only manual flush fires
    nrt.add("target", "mongoose unique passage text")

    before = nrt.search("mongoose", k=10)
    assert "target" not in _ids(before)

    nrt.flush()
    after = nrt.search("mongoose", k=10)
    assert "target" in _ids(after)
    nrt.close()


def test_initial_segments_are_searchable_immediately(tmp_path):
    base_tsv = tmp_path / "base.tsv"
    base_tsv.write_text("b0\tplatypus base corpus document\n")
    base_segment = build_segment(base_tsv, tmp_path / "base_index")

    nrt = NrtIndex(tmp_path, flush_threshold_docs=100, initial_segments=[base_segment])
    assert nrt.segment_count == 1
    results = nrt.search("platypus", k=10)
    assert "b0" in _ids(results)
    nrt.close()


def test_delete_makes_a_flushed_doc_unfindable(tmp_path):
    nrt = NrtIndex(tmp_path, flush_threshold_docs=100)
    nrt.add("victim", "capybara distinctive unique passage")
    nrt.add("other", "an unrelated second document")
    nrt.flush()

    assert "victim" in _ids(nrt.search("capybara", k=10))

    nrt.delete("victim")
    assert "victim" not in _ids(nrt.search("capybara", k=10))
    # The delete must not remove anything else.
    assert "other" in _ids(nrt.search("unrelated", k=10))
    nrt.close()


def test_delete_before_the_doc_is_ever_added_still_suppresses_it_once_added(tmp_path):
    # A delete is a standing tombstone on an external id, not scoped to a
    # segment that must already exist -- so a delete-before-add is legal
    # (if unusual) and still suppresses the doc once it lands.
    nrt = NrtIndex(tmp_path, flush_threshold_docs=100)
    nrt.delete("d0")
    nrt.add("d0", "wolverine early tombstone test")
    nrt.flush()
    assert "d0" not in _ids(nrt.search("wolverine", k=10))
    nrt.close()
