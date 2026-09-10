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

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cascade_index

from nrt.merge_policy import TieredMergePolicy
from nrt.segment import BUILD_INDEX_BIN, Segment, SegmentSet, build_segment


class NrtIndex:
    def __init__(
        self,
        base_dir: Path,
        flush_threshold_docs: int,
        flush_interval_s: float = 3600.0,
        initial_segments: list[Segment] | None = None,
        merge_policy: TieredMergePolicy | None = None,
        search_workers: int = 32,
        build_index_bin: Path = BUILD_INDEX_BIN,
    ) -> None:
        self._base_dir = base_dir
        self._flush_threshold_docs = flush_threshold_docs
        self._flush_interval_s = flush_interval_s
        self._build_index_bin = build_index_bin
        self._merge_policy = merge_policy
        # Sized to 1: merges are re-tokenization work (spec §4), not cheap,
        # and running them one at a time keeps the background thread count
        # bounded and predictable, matching py/server/report.py's
        # single-background-task convention.
        self._merge_pool = ThreadPoolExecutor(max_workers=1) if merge_policy is not None else None
        # Guards against submitting a second merge job for the same
        # over-budget segment list before the first one has run and swapped
        # its result in -- see _maybe_schedule_merge_locked.
        self._merge_in_progress = False
        self._lock = threading.Lock()
        self._segments: list[Segment] = list(initial_segments) if initial_segments else []
        self._tombstones: set[str] = set()
        self._buffer: list[tuple[str, str]] = []
        self._next_segment_id = len(self._segments)
        self._last_flush_time = time.monotonic()
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

    def delete(self, external_id: str) -> None:
        """Adds external_id to the tombstone set. Effective on the very
        next search() call, across every segment current and future -- a
        tombstone is never migrated or cleared at merge time, because it
        was never segment-scoped (spec §5)."""
        with self._lock:
            self._tombstones.add(external_id)

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

        self._maybe_schedule_merge_locked()

    def _maybe_schedule_merge_locked(self) -> None:
        """Called with self._lock held, from _flush_locked and again from
        _run_merge once a merge completes. Submits at most one merge job
        at a time: self._merge_in_progress guards this because, without
        it, two flushes (or a flush and a just-finished merge) that both
        observe an over-budget segment list before either merge has run
        would compute and submit the *same* smallest-segments plan twice
        -- the second job would then crash reading a source TSV the first
        job already deleted (confirmed empirically: this was Task 4's
        original defect, caught by the implementer before it reached
        review). Calling this again at the end of _run_merge -- not only
        from flush -- means convergence to at-or-under max_segments does
        not depend on further write traffic: a burst of flushes that
        pushes segment count well over budget is worked down by
        consecutive merge rounds on their own, chained back to back."""
        if self._merge_policy is None or self._merge_in_progress:
            return
        plan = self._merge_policy.maybe_merge(self._segments)
        if plan is None:
            return
        self._merge_in_progress = True
        merge_id = self._next_segment_id
        self._next_segment_id += 1
        self._merge_pool.submit(self._run_merge, plan, merge_id)

    def _run_merge(self, chosen: list[Segment], merge_id: int) -> None:
        """Runs on the background merge thread -- concatenates the chosen
        segments' source TSVs (this is why Segment retains source_tsv:
        merging needs the original text, not just the built index),
        rebuilds via build_segment, then swaps the merged segment in under
        the lock. This re-tokenizes from source text; it is not a
        postings-level merge (spec §4). If the build fails before the lock
        is acquired, the failure handler resets the _merge_in_progress flag
        and re-raises, so a failed merge does not permanently disable
        merging (important because a stale flag would prevent all future
        merges from being attempted, which is worse than the original
        duplicate-plan race this guard was added to prevent)."""
        try:
            segments_dir = self._base_dir / "segments"
            merged_tsv = segments_dir / f"seg_{merge_id}.tsv"
            merged_tsv.write_text("".join(seg.source_tsv.read_text() for seg in chosen))
            merged_dir = segments_dir / f"seg_{merge_id}"
            merged_segment = build_segment(merged_tsv, merged_dir, build_index_bin=self._build_index_bin)
        except Exception as exc:
            with self._lock:
                self._merge_in_progress = False
            print(f"nrt: merge {merge_id} failed, will retry on next flush: {exc!r}")
            raise

        # Identity-based filtering, not list.remove()/`in` -- Segment wraps
        # a pybind11 Index with no custom __eq__, and `chosen` holds the
        # exact same objects taken from self._segments, so comparing by
        # id() is both correct and avoids relying on dataclass-generated
        # equality over a C++-backed field.
        chosen_ids = {id(s) for s in chosen}
        with self._lock:
            self._segments = [s for s in self._segments if id(s) not in chosen_ids]
            self._segments.append(merged_segment)
            self._merge_in_progress = False
            # Chain: one merge round may not be enough to get back at or
            # under max_segments (e.g. a burst of flushes queued up while
            # this merge was running) -- re-evaluate immediately rather
            # than waiting for the next flush, which may never come.
            self._maybe_schedule_merge_locked()

        for seg in chosen:
            seg.source_tsv.unlink(missing_ok=True)
            shutil.rmtree(seg.index_dir, ignore_errors=True)

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
        if self._merge_pool is not None:
            self._merge_pool.shutdown(wait=True)
        self._search_pool.shutdown(wait=True)
