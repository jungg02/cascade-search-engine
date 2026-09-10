"""Read side of near-real-time indexing: build a segment from a TSV via the
unmodified build_index binary, and fan a query out across a set of segments,
merging by score with tombstone filtering. See design spec §2, §3.

No writer here -- SegmentSet is usable standalone (nrt.segment_sweep's
fixed-corpus segment-count sweep only ever calls SegmentSet.search);
nrt.index.NrtIndex wraps a SegmentSet to add the write side.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cascade_index

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"


@dataclass
class Segment:
    index: "cascade_index.Index"
    source_tsv: Path
    index_dir: Path
    doc_count: int


def build_segment(
    tsv_path: Path, index_dir: Path, build_index_bin: Path = BUILD_INDEX_BIN
) -> Segment:
    """Shells out to build_index (unmodified), then opens the result via
    cascade_index.Index. index_dir is created if missing; callers are
    responsible for giving each segment its own fresh index_dir -- this
    function does not check for a pre-existing built index there."""
    if not build_index_bin.exists():
        raise FileNotFoundError(f"{build_index_bin} not built; run `make -C cpp all` first")
    index_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(build_index_bin), str(tsv_path), str(index_dir)], check=True)
    index = cascade_index.Index(str(index_dir))
    return Segment(index=index, source_tsv=tsv_path, index_dir=index_dir, doc_count=index.doc_count)


class SegmentSet:
    """Fans a query out to every segment concurrently and merges by score,
    filtering tombstoned external ids. Requesting k + len(tombstones) from
    each segment (rather than exactly k) is conservative but correct and
    simple -- documented as a known simplification in the design spec §3,
    not tuned further.

    `pool`, if given, is a caller-owned ThreadPoolExecutor reused across
    calls -- callers that search repeatedly (NrtIndex, nrt.segment_sweep)
    must pass one, since opening a fresh ThreadPoolExecutor per search call
    means N thread *spawns* per query, which would swamp the fan-out cost
    this class exists to measure (spec §6b). Tests that build a throwaway
    SegmentSet for one or two calls can omit it and get a call-owned,
    one-shot pool instead."""

    def __init__(
        self,
        segments: list[Segment],
        tombstones: set[str] | None = None,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
        self._segments = segments
        self._tombstones = tombstones if tombstones is not None else set()
        self._pool = pool

    def search(
        self, query: str, k: int = 10, algorithm=cascade_index.Algorithm.BLOCKMAX_WAND
    ) -> list[tuple[str, float]]:
        if not self._segments:
            return []
        over_fetch_k = k + len(self._tombstones)

        def _dispatch(pool: ThreadPoolExecutor) -> list[list[tuple[str, float]]]:
            futures = [
                pool.submit(seg.index.search, query, over_fetch_k, algorithm)
                for seg in self._segments
            ]
            # index.search returns (pairs, SearchStats); [0] discards stats,
            # which this layer never needs (spec §2's docid-remapping note).
            return [future.result()[0] for future in futures]

        if self._pool is not None:
            per_segment_pairs = _dispatch(self._pool)
        else:
            with ThreadPoolExecutor(max_workers=len(self._segments)) as pool:
                per_segment_pairs = _dispatch(pool)

        merged: list[tuple[str, float]] = []
        for pairs in per_segment_pairs:
            merged.extend((eid, score) for eid, score in pairs if eid not in self._tombstones)
        merged.sort(key=lambda r: r[1], reverse=True)
        return merged[:k]
