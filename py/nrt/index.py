"""Write side of near-real-time indexing: buffers documents in memory,
flushes to a new segment synchronously once a threshold is crossed, and
snapshots segment state under a lock so a concurrent flush or merge
swapping the segment list mid-query can't hand search() a
partially-updated list. See design spec §4.

This file starts with buffering + flush + search only, proven correct on
its own; delete() (Task 5) and background merge (Task 4) are added on top
of this same class in later changes to this file.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cascade_index

from nrt.segment import BUILD_INDEX_BIN, Segment, SegmentSet, build_segment


class NrtIndex:
    def __init__(
        self,
        base_dir: Path,
        flush_threshold_docs: int,
        flush_interval_s: float = 3600.0,
        initial_segments: list[Segment] | None = None,
        search_workers: int = 32,
        build_index_bin: Path = BUILD_INDEX_BIN,
    ) -> None:
        self._base_dir = base_dir
        self._flush_threshold_docs = flush_threshold_docs
        self._flush_interval_s = flush_interval_s
        self._build_index_bin = build_index_bin
        self._lock = threading.Lock()
        self._segments: list[Segment] = list(initial_segments) if initial_segments else []
        self._tombstones: set[str] = set()
        self._buffer: list[tuple[str, str]] = []
        self._next_segment_id = len(self._segments)
        self._last_flush_time = time.monotonic()
        # Reused across every search() call -- a fresh ThreadPoolExecutor
        # per call would spawn len(segments) threads per query, which would
        # dominate the fan-out cost this phase measures (spec §6b).
        self._search_pool = ThreadPoolExecutor(max_workers=search_workers)

    @property
    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    def add(self, external_id: str, text: str) -> None:
        with self._lock:
            self._buffer.append((external_id, text))
            should_flush = (
                len(self._buffer) >= self._flush_threshold_docs
                or time.monotonic() - self._last_flush_time >= self._flush_interval_s
            )
            if should_flush:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        seg_id = self._next_segment_id
        self._next_segment_id += 1
        segments_dir = self._base_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = segments_dir / f"seg_{seg_id}.tsv"
        tsv_path.write_text("".join(f"{eid}\t{text}\n" for eid, text in self._buffer))
        index_dir = segments_dir / f"seg_{seg_id}"
        segment = build_segment(tsv_path, index_dir, build_index_bin=self._build_index_bin)
        self._segments.append(segment)
        self._buffer = []
        self._last_flush_time = time.monotonic()

    def search(
        self, query: str, k: int = 10, algorithm=cascade_index.Algorithm.BLOCKMAX_WAND
    ) -> list[tuple[str, float]]:
        with self._lock:
            segments_snapshot = list(self._segments)
            tombstones_snapshot = set(self._tombstones)
        # The actual C++ search calls (which release the GIL) run outside
        # the lock, so a slow search never blocks add()/flush()/delete().
        return SegmentSet(segments_snapshot, tombstones_snapshot, pool=self._search_pool).search(
            query, k, algorithm
        )

    def close(self) -> None:
        self._search_pool.shutdown(wait=True)
