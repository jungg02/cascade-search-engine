"""A real merge fires on a background thread once segment count exceeds
the policy's max_segments; segment count drops back within the bound
afterward, and every document added before the merge is still findable
(merge must not lose documents). Polls with a bounded timeout since the
merge runs on a background thread, per design spec §7."""

from __future__ import annotations

import time
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
    from nrt.merge_policy import TieredMergePolicy


def _poll_until(predicate, timeout_s: float = 10.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_merge_fires_and_bounds_segment_count(tmp_path):
    policy = TieredMergePolicy(merge_factor=2, max_segments=3)
    nrt = NrtIndex(tmp_path, flush_threshold_docs=2, merge_policy=policy)

    all_ids: list[str] = []
    for batch in range(5):  # 5 flushes of 2 docs each = 10 docs, 5 segments -> merge must fire
        for i in range(2):
            doc_id = f"d{batch}_{i}"
            all_ids.append(doc_id)
            nrt.add(doc_id, f"aardvark passage {doc_id}")

    assert _poll_until(lambda: nrt.segment_count <= policy.max_segments)
    assert nrt.segment_count <= policy.max_segments

    # Every doc's text contains "aardvark", so one query against the merged
    # (and un-merged) segments together must return every doc -- this
    # checks "merge lost no documents" directly, without depending on
    # per-doc-id tokenization surviving the analyzer.
    found_ids = {eid for eid, _ in nrt.search("aardvark", k=len(all_ids) + 5)}
    assert found_ids == set(all_ids)

    nrt.close()


def test_no_merge_pool_created_when_merge_policy_is_none(tmp_path):
    nrt = NrtIndex(tmp_path, flush_threshold_docs=2)
    nrt.add("d0", "no merge policy configured")
    nrt.add("d1", "second doc triggers flush")
    assert nrt.segment_count == 1
    nrt.close()  # must not raise even though no merge pool exists
