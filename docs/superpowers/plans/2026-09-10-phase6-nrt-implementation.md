# Phase 6: Near-Real-Time Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `py/nrt/`, a pure-Python in-process near-real-time indexing layer on top of the existing, unmodified `cascade_index` pybind11 binding: an in-memory write buffer that flushes to disk-backed segments, multi-segment query fan-out with score merge, tombstone deletes, and a background tiered merge policy. Produce `bench/phase6.md` with the freshness/latency tradeoff plot and the segment-count-vs-latency plot, completing the plan's Phase 6 stretch goal.

**Architecture:** `nrt.segment` builds a `Segment` by shelling out to the unmodified `cpp/build/build_index` binary and opening the result via `cascade_index.Index`; `SegmentSet` fans a query out to every segment (`ThreadPoolExecutor`), merges by score, and filters tombstoned external ids. `nrt.index.NrtIndex` wraps a `SegmentSet` with a write side: a `threading.Lock`-guarded buffer that flushes synchronously into a new segment once a doc-count or time threshold is crossed, plus a background-thread tiered merge (`nrt.merge_policy.TieredMergePolicy`) that keeps live segment count bounded. Two experiment drivers (`nrt.lag_experiment`, `nrt.segment_sweep`) measure index-to-searchable lag vs. flush threshold and query latency vs. live segment count at fixed corpus size; `nrt.report_6` renders both into `bench/phase6.md`.

**Tech Stack:** Python 3.12, the existing `cascade_index` binding (`py/cascade_index.cpython-312-darwin.so`, built from `cpp/bindings/module.cc`), `concurrent.futures`, `threading`, reusing `harness/loadgen.py`/`harness/histogram.py`/`harness/datasets.py`/`harness/runmeta.py` and `server/partition_corpus.py`'s line-split logic (Task 7 only). `matplotlib` (already present via the `dense` extra) for the two plots. No gRPC, no `cpp/server/`, no C++ changes anywhere in this plan.

**Spec:** `docs/superpowers/specs/2026-09-10-phase6-nrt-indexing-design.md`

## Global Constraints

- In-process only: this phase calls `cascade_index.Index`/`cascade_index.Algorithm` directly, never `cpp/server/`'s gRPC service — spawning a server per flushed segment would dominate the lag measurement with process-spawn cost (spec §2). In-process latency numbers are **not** comparable to `bench/phase2a.md`'s or `bench/phase2b.md`'s served p99s; every result table in `bench/phase6.md` states this.
- `cpp/index/builder.cc`, `cpp/query/searcher.h`, and `cpp/bindings/module.cc` are **not modified** — this phase is a pure Python orchestration layer over the existing, unmodified `build_index` binary and `Index` class (spec §10).
- Every segment resolves to the corpus's real MS MARCO passage id before Python sees a result (`Index.search` already returns `(external_id, score)` pairs) — no docid remapping anywhere in this layer (spec §2). `Index.search(query, k, algorithm)` returns a 2-tuple `(list[(external_id, score)], SearchStats)`; every call site in this plan unpacks index `[0]` and discards `SearchStats`.
- Tombstones key on **external id**, never on `(segment, local docid)` — merges renumber local docids, external ids are stable by construction (spec §5).
- All shared mutable state in `NrtIndex` (`self._segments`, `self._tombstones`, the write buffer) is guarded by one `threading.Lock`; `search()` holds the lock only long enough to copy references, never for the duration of the actual C++ search calls, so a search never blocks a flush/merge or vice versa beyond that copy (spec §4, mirrors `HedgedBroker`'s pattern in `py/server/broker.py`).
- No mean latency anywhere (project-wide rule; `harness/histogram.py`'s `LatencyRecorder` has no `.mean()`). Every reported number is a percentile from `LatencyRecorder`/`LoadResult.summary()`.
- Merge is re-tokenization from source TSV, not a postings-level merge — disclosed explicitly, not implemented as a sorted-run merge over existing postings (spec §4, out of scope per §10).
- Flush is synchronous and blocks new writes (not reads) for its duration — a real, disclosed limitation, not an oversight (spec §4).
- Python modules run with `PYTHONPATH=py` (`uv run python -m nrt.foo`), matching every existing phase's convention. Import the binding as `import cascade_index` (matching `py/baselines/cascade_bm25.py`'s convention), never `from cascade_index import ...`.
- This machine has 8GB RAM and no GPU (spec §9). Every corpus slice used here is small by construction (tens of thousands of docs, not the full 8.8M-line corpus) so that dozens of `build_index` calls per experiment run stay tractable.

---

## File Structure

```
py/nrt/__init__.py                    new — empty
py/nrt/segment.py                     new — Segment, build_segment, SegmentSet
py/nrt/tests/__init__.py              new — empty
py/nrt/tests/test_segment.py          new
py/nrt/tests/test_tombstones.py       new

py/nrt/merge_policy.py                new — TieredMergePolicy
py/nrt/tests/test_merge_policy.py     new

py/nrt/index.py                       new — NrtIndex: buffer/flush/search (Task 2), delete (Task 5), merge (Task 4)
py/nrt/tests/test_nrt_index.py        new
py/nrt/tests/test_nrt_index_merge.py  new

py/nrt/lag_experiment.py              new — experiment 6a driver
py/nrt/segment_sweep.py               new — experiment 6b driver

py/nrt/report_6.py                    new — renders bench/phase6.md + bench/plots/phase6-*.png

README.md                             modified — "Running Phase 6" + status table
```

---

## Task 1: `Segment`, `build_segment`, `SegmentSet`

**Files:**
- Create: `py/nrt/__init__.py` (empty)
- Create: `py/nrt/segment.py`
- Create: `py/nrt/tests/__init__.py` (empty)
- Test: `py/nrt/tests/test_segment.py`
- Test: `py/nrt/tests/test_tombstones.py`

**Interfaces:**
- Consumes: `cascade_index.Index`, `cascade_index.Algorithm` (existing binding, `py/cascade_index.cpython-312-darwin.so`); `cpp/build/build_index` binary (existing, unmodified).
- Produces (for Tasks 2, 4, 6, 7): `@dataclass class Segment(index, source_tsv: Path, index_dir: Path, doc_count: int)`; `build_segment(tsv_path: Path, index_dir: Path, build_index_bin: Path = BUILD_INDEX_BIN) -> Segment`; `class SegmentSet(segments: list[Segment], tombstones: set[str] | None = None, pool: ThreadPoolExecutor | None = None)` with `.search(query: str, k: int = 10, algorithm = cascade_index.Algorithm.BLOCKMAX_WAND) -> list[tuple[str, float]]`; module constants `REPO_ROOT`, `BUILD_INDEX_BIN`. `pool`, when given, is reused across calls instead of spawning a fresh `ThreadPoolExecutor` per `search()` — required by any caller that searches repeatedly (Tasks 3, 7), since per-call thread spawns would dominate the fan-out cost Task 7's experiment measures.

`Segment` carries `index_dir` in addition to the design spec's `index`/`source_tsv`/`doc_count` fields — deleting a merged-away segment's on-disk index directory (Task 4) needs its path, and `Segment` is the only place that path is known.

- [ ] **Step 1: Write the failing tests**

Create `py/nrt/tests/test_segment.py`:

```python
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
```

Create `py/nrt/tests/test_tombstones.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_segment.py nrt/tests/test_tombstones.py -v`
Expected: FAIL / collection error — `nrt.segment` doesn't exist yet (or SKIPPED for all tests, if `cpp/build/build_index` isn't present in this environment; build it per README's Phase 1 setup before continuing, then re-run to confirm real passes later).

- [ ] **Step 3: Write `py/nrt/__init__.py`, `py/nrt/tests/__init__.py`, and `py/nrt/segment.py`**

`py/nrt/__init__.py` and `py/nrt/tests/__init__.py`: empty files.

`py/nrt/segment.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_segment.py nrt/tests/test_tombstones.py -v`
Expected: 6 passed (or 6 skipped if `cpp/build/build_index` is absent — build it, then re-run to confirm a real pass before moving on).

- [ ] **Step 5: Commit**

```bash
git add py/nrt/__init__.py py/nrt/segment.py py/nrt/tests/__init__.py py/nrt/tests/test_segment.py py/nrt/tests/test_tombstones.py
git commit -m "$(cat <<'EOF'
phase6: Segment/build_segment/SegmentSet -- multi-segment fan-out with tombstones

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 2: `TieredMergePolicy`

**Files:**
- Create: `py/nrt/merge_policy.py`
- Test: `py/nrt/tests/test_merge_policy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — takes duck-typed objects with a `.doc_count` attribute (real `Segment`s in production, plain stand-ins in tests).
- Produces (for Task 4): `class TieredMergePolicy(merge_factor: int = 4, max_segments: int = 8)` with `.maybe_merge(segments: list) -> list | None`.

This task is independent of Task 1/3 (`NrtIndex`) — it is pure logic over a list of doc-count-bearing objects, deliberately ordered here so it can be written and tested before `NrtIndex` wires it in (Task 4).

- [ ] **Step 1: Write the failing tests**

Create `py/nrt/tests/test_merge_policy.py`:

```python
"""TieredMergePolicy fires as soon as segment count exceeds max_segments,
returning the merge_factor smallest-by-doc_count segments to collapse into
one -- smallest-first keeps merge cost low and bounds it as re-tokenization
cost, per design spec §4a."""

from __future__ import annotations

from dataclasses import dataclass

from nrt.merge_policy import TieredMergePolicy


@dataclass
class _FakeSegment:
    doc_count: int


def test_no_merge_when_at_or_below_max_segments():
    policy = TieredMergePolicy(merge_factor=2, max_segments=3)
    segments = [_FakeSegment(doc_count=n) for n in (10, 20, 30)]
    assert policy.maybe_merge(segments) is None


def test_merges_the_smallest_segments_once_over_max():
    policy = TieredMergePolicy(merge_factor=2, max_segments=3)
    segments = [_FakeSegment(doc_count=n) for n in (50, 10, 30, 5)]
    plan = policy.maybe_merge(segments)
    assert plan is not None
    assert sorted(s.doc_count for s in plan) == [5, 10]


def test_merge_factor_larger_than_segment_count_merges_everything():
    policy = TieredMergePolicy(merge_factor=10, max_segments=1)
    segments = [_FakeSegment(doc_count=n) for n in (3, 1, 2)]
    plan = policy.maybe_merge(segments)
    assert sorted(s.doc_count for s in plan) == [1, 2, 3]


def test_default_thresholds():
    policy = TieredMergePolicy()
    assert policy.merge_factor == 4
    assert policy.max_segments == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_merge_policy.py -v`
Expected: FAIL / collection error — `nrt.merge_policy` doesn't exist yet.

- [ ] **Step 3: Write `py/nrt/merge_policy.py`**

```python
"""A simplified tiered merge policy, not full Lucene multi-tier: bounds
live segment count from above at max_segments by collapsing the
merge_factor smallest segments into one as soon as the count exceeds it.
Smallest-first mirrors why real tiered policies merge small segments
before large ones -- amortized merge cost per document stays bounded.
See design spec §4a.
"""

from __future__ import annotations


class TieredMergePolicy:
    def __init__(self, merge_factor: int = 4, max_segments: int = 8) -> None:
        self.merge_factor = merge_factor
        self.max_segments = max_segments

    def maybe_merge(self, segments: list) -> list | None:
        if len(segments) <= self.max_segments:
            return None
        by_size = sorted(segments, key=lambda s: s.doc_count)
        return by_size[: min(self.merge_factor, len(by_size))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_merge_policy.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add py/nrt/merge_policy.py py/nrt/tests/test_merge_policy.py
git commit -m "$(cat <<'EOF'
phase6: TieredMergePolicy bounds live segment count from above

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 3: `NrtIndex` core (buffer, flush, search — no merge, no delete yet)

**Files:**
- Create: `py/nrt/index.py`
- Test: `py/nrt/tests/test_nrt_index.py`

**Interfaces:**
- Consumes: `nrt.segment.{Segment, SegmentSet, build_segment, BUILD_INDEX_BIN}` (Task 1).
- Produces (for Tasks 4, 5, 6, 7): `class NrtIndex`: `__init__(self, base_dir: Path, flush_threshold_docs: int, flush_interval_s: float = 3600.0, initial_segments: list[Segment] | None = None, search_workers: int = 32, build_index_bin: Path = BUILD_INDEX_BIN)`, `.add(external_id: str, text: str) -> None`, `.flush() -> None`, `.search(query: str, k: int = 10, algorithm = cascade_index.Algorithm.BLOCKMAX_WAND) -> list[tuple[str, float]]`, `.segment_count -> int` (property), `.close() -> None`. Internal `self._lock`, `self._segments`, `self._tombstones` (present but unused until Task 5), `self._buffer`, `self._next_segment_id`, `self._search_pool` are used by Tasks 4 and 5's modifications to this same file.

`initial_segments` is an addition beyond the design spec's literal `__init__` signature: the lag experiment (Task 6) needs to seed a "already indexed" base segment built once via `build_segment` directly (not through the auto-flush path, which would fragment a 50,000-doc bulk load into hundreds of tiny segments at a small `flush_threshold_docs`). Passing pre-built segments in at construction is the minimal way to support that without changing `add`'s auto-flush semantics.

`search_workers` backs a **persistent** `ThreadPoolExecutor` that every `search()` call reuses (via `SegmentSet`'s `pool` argument from Task 1) instead of spawning a fresh pool per call — spawning N threads per query would dominate the fan-out cost this phase measures, especially in Task 7's segment-count sweep. `close()` shuts this pool down; Task 4 extends it to also shut down the merge pool.

- [ ] **Step 1: Write the failing tests**

Create `py/nrt/tests/test_nrt_index.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index.py -v`
Expected: FAIL / collection error — `nrt.index` doesn't exist yet.

- [ ] **Step 3: Write `py/nrt/index.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index.py -v`
Expected: 4 passed (or 4 skipped if `cpp/build/build_index` is absent — build it, then re-run to confirm a real pass before moving on).

- [ ] **Step 5: Commit**

```bash
git add py/nrt/index.py py/nrt/tests/test_nrt_index.py
git commit -m "$(cat <<'EOF'
phase6: NrtIndex core -- buffered add, threshold-triggered flush, snapshot search

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 4: Background tiered merge integration

**Files:**
- Modify: `py/nrt/index.py` (add merge integration to `NrtIndex`)
- Test: `py/nrt/tests/test_nrt_index_merge.py`

**Interfaces:**
- Consumes: `NrtIndex` (Task 3), `nrt.merge_policy.TieredMergePolicy` (Task 2).
- Produces (for Task 6): `NrtIndex.__init__` gains `merge_policy: TieredMergePolicy | None = None`; `NrtIndex.close()` (already exists from Task 3, shutting down the search pool) is extended to also shut down the background merge pool, `wait=True`.

- [ ] **Step 1: Write the failing test**

Create `py/nrt/tests/test_nrt_index_merge.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index_merge.py -v`
Expected: FAIL — `NrtIndex.__init__() got an unexpected keyword argument 'merge_policy'`.

- [ ] **Step 3: Modify `py/nrt/index.py`**

Add `shutil` to the imports and add `from nrt.merge_policy import TieredMergePolicy`, so the file's import block (everything from `from __future__ import annotations` down to the `from nrt.segment import ...` line) reads:

```python
from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cascade_index

from nrt.merge_policy import TieredMergePolicy
from nrt.segment import BUILD_INDEX_BIN, Segment, SegmentSet, build_segment
```

Change `__init__`'s signature and body to add the merge policy and pool:

```python
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
        self._lock = threading.Lock()
        self._segments: list[Segment] = list(initial_segments) if initial_segments else []
        self._tombstones: set[str] = set()
        self._buffer: list[tuple[str, str]] = []
        self._next_segment_id = len(self._segments)
        self._last_flush_time = time.monotonic()
        self._search_pool = ThreadPoolExecutor(max_workers=search_workers)
```

(`merge_policy` is inserted before `search_workers`/`build_index_bin` in the parameter list; every call site in this plan passes these by keyword, so the reordering is safe.)

Add the merge trigger to the end of `_flush_locked` (still inside the `with self._lock` context that called it):

```python
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

        if self._merge_policy is not None:
            plan = self._merge_policy.maybe_merge(self._segments)
            if plan is not None:
                merge_id = self._next_segment_id
                self._next_segment_id += 1
                self._merge_pool.submit(self._run_merge, plan, merge_id)
```

Add `_run_merge` as a new method on `NrtIndex`, and replace the existing `close` method (from Task 3) with a version that also shuts down the merge pool:

```python
    def _run_merge(self, chosen: list[Segment], merge_id: int) -> None:
        """Runs on the background merge thread -- concatenates the chosen
        segments' source TSVs (this is why Segment retains source_tsv:
        merging needs the original text, not just the built index),
        rebuilds via build_segment, then swaps the merged segment in under
        the lock. This re-tokenizes from source text; it is not a
        postings-level merge (spec §4)."""
        segments_dir = self._base_dir / "segments"
        merged_tsv = segments_dir / f"seg_{merge_id}.tsv"
        merged_tsv.write_text("".join(seg.source_tsv.read_text() for seg in chosen))
        merged_dir = segments_dir / f"seg_{merge_id}"
        merged_segment = build_segment(merged_tsv, merged_dir, build_index_bin=self._build_index_bin)

        # Identity-based filtering, not list.remove()/`in` -- Segment wraps
        # a pybind11 Index with no custom __eq__, and `chosen` holds the
        # exact same objects taken from self._segments, so comparing by
        # id() is both correct and avoids relying on dataclass-generated
        # equality over a C++-backed field.
        chosen_ids = {id(s) for s in chosen}
        with self._lock:
            self._segments = [s for s in self._segments if id(s) not in chosen_ids]
            self._segments.append(merged_segment)

        for seg in chosen:
            seg.source_tsv.unlink(missing_ok=True)
            shutil.rmtree(seg.index_dir, ignore_errors=True)

    def close(self) -> None:
        if self._merge_pool is not None:
            self._merge_pool.shutdown(wait=True)
        self._search_pool.shutdown(wait=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index_merge.py nrt/tests/test_nrt_index.py -v`
Expected: 6 passed (or all skipped if `cpp/build/build_index` is absent — build it, then re-run to confirm a real pass before moving on).

- [ ] **Step 5: Commit**

```bash
git add py/nrt/index.py py/nrt/tests/test_nrt_index_merge.py
git commit -m "$(cat <<'EOF'
phase6: background tiered merge -- flush triggers a bounded, lock-safe segment merge

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 5: Tombstone deletes on `NrtIndex`

**Files:**
- Modify: `py/nrt/index.py` (add `delete`)
- Modify: `py/nrt/tests/test_nrt_index.py` (add the correctness test)

**Interfaces:**
- Consumes: `NrtIndex` (Tasks 3, 4) — `self._tombstones` (already present, unused until now), `self._lock`.
- Produces (for Task 6): `NrtIndex.delete(external_id: str) -> None`.

`SegmentSet` has supported tombstone filtering since Task 1 and `NrtIndex.search` has snapshotted `self._tombstones` into every `SegmentSet` call since Task 3 — the only missing piece is the mutator. This task's real content is the correctness test: prove a delete makes a document unfindable through the full `NrtIndex` (not just `SegmentSet` in isolation, already proven by `test_tombstones.py` in Task 1).

- [ ] **Step 1: Write the failing test**

Append to `py/nrt/tests/test_nrt_index.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index.py -v`
Expected: FAIL — `AttributeError: 'NrtIndex' object has no attribute 'delete'`.

- [ ] **Step 3: Modify `py/nrt/index.py`**

Add `delete` as a new method on `NrtIndex` (place it next to `flush`):

```python
    def delete(self, external_id: str) -> None:
        """Adds external_id to the tombstone set. Effective on the very
        next search() call, across every segment current and future -- a
        tombstone is never migrated or cleared at merge time, because it
        was never segment-scoped (spec §5)."""
        with self._lock:
            self._tombstones.add(external_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/tests/test_nrt_index.py -v`
Expected: 6 passed (or 6 skipped if `cpp/build/build_index` is absent — build it, then re-run to confirm a real pass before moving on).

- [ ] **Step 5: Commit**

```bash
git add py/nrt/index.py py/nrt/tests/test_nrt_index.py
git commit -m "$(cat <<'EOF'
phase6: NrtIndex.delete -- tombstone by external id, effective on the next search

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 6: Index-to-searchable lag experiment driver (6a, folding in 6c)

**Files:**
- Create: `py/nrt/lag_experiment.py`

**Interfaces:**
- Consumes: `harness.datasets.load_queries`, `harness.loadgen.run_open_loop`, `harness.runmeta.run_metadata` (existing), `nrt.index.NrtIndex` (Tasks 3-5), `nrt.merge_policy.TieredMergePolicy` (Task 2), `nrt.segment.build_segment` (Task 1).
- Produces (for Task 8): `bench/results/nrt-lag.json` with top-level `points: list[{flush_threshold_docs, lag_p99_us, query_p99_us, final_segment_count, collisions_skipped, merge_before_after: {before, after} | null}]` plus provenance.

No unit test file for this task — an experiment driver against a real (small-slice) corpus and real `build_index` calls, in the same category as `server.tail_latency`/`server.hedging` (neither of which has a test file either); verified by running it (Task 9) and inspecting the output.

- [ ] **Step 1: Write `py/nrt/lag_experiment.py`**

```python
"""Index-to-searchable lag vs. flush threshold (design spec §6a), with the
merge-in-action check (§6c) folded into the smallest-threshold run.

Fixed base corpus: data/msmarco-passage.tsv lines 0-49,999 (50,000 docs),
built once via build_segment into a static base segment -- "already
indexed" content, not part of the write stream. Held-back incoming stream:
lines 50,000-54,999 (5,000 docs) from the same file, fed one at a time via
NrtIndex.add. For each swept flush_threshold_docs, a probe query (the
doc's longest whitespace token) is checked against the current index
before the add (must not already match -- a base-corpus collision); docs
are added one at a time but *drained* (polled for searchability and their
lag recorded) in batches of flush_threshold_docs, right after each batch's
add() has triggered NrtIndex's own synchronous auto-flush -- see
run_threshold's docstring for why polling after every single add() instead
would simply never terminate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness.datasets import load_queries
from harness.histogram import LatencyRecorder
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from nrt.index import NrtIndex
from nrt.merge_policy import TieredMergePolicy
from nrt.segment import build_segment

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "msmarco-passage.tsv"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

BASE_DOC_COUNT = 50_000
STREAM_DOC_COUNT = 5_000
POLL_INTERVAL_S = 0.001


def _read_tsv_slice(path: Path, start: int, count: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + count:
                break
            docid, text = line.rstrip("\n").split("\t", 1)
            docs.append((docid, text))
    return docs


def _longest_token(text: str) -> str | None:
    tokens = text.split()
    if not tokens:
        return None
    # max() over a list scan only updates on strictly-greater length, so the
    # first token achieving the maximum length wins -- "ties broken by
    # first occurrence" (spec §6a).
    return max(tokens, key=len)


def _drain_pending(
    nrt: NrtIndex,
    pending: list[tuple[str, str, float]],
    lag_recorder: LatencyRecorder,
    timeout_s: float = 30.0,
) -> int:
    """Poll each (external_id, probe, write_start) in `pending` until it's
    searchable, recording its lag. Called right after a flush has just made
    the whole batch's segment durable, so in the common case every entry
    resolves on the very first poll -- the bounded timeout only matters if
    something is actually wrong (e.g. a probe collision the pre-condition
    check missed). Returns the count that timed out without ever becoming
    searchable, rather than raising, so one bad probe doesn't abort the
    whole threshold's run."""
    remaining = list(pending)
    deadline = time.perf_counter() + timeout_s
    while remaining and time.perf_counter() < deadline:
        # One timestamp for the whole pass, not one per entry: every entry
        # still in `remaining` became searchable at the same instant (the
        # flush that made them durable already completed before this
        # function was called), so timing each entry's own search call
        # separately would load the cumulative search time of every entry
        # ahead of it onto the tail of the batch -- an error that grows
        # with threshold and would distort exactly the tradeoff curve this
        # experiment exists to plot.
        found_at = time.perf_counter()
        still_remaining = []
        for external_id, probe, write_start in remaining:
            hits = {eid for eid, _ in nrt.search(probe, k=10)}
            if external_id in hits:
                lag_recorder.record((found_at - write_start) * 1e6)
            else:
                still_remaining.append((external_id, probe, write_start))
        remaining = still_remaining
        if remaining:
            time.sleep(POLL_INTERVAL_S)
    return len(remaining)


def run_threshold(
    threshold: int, base_dir: Path, base_segment, stream_docs: list[tuple[str, str]],
    max_segments: int, merge_factor: int, queries: list[str],
) -> dict:
    """Streams stream_docs through a fresh NrtIndex seeded with base_segment,
    batching adds into groups of `threshold` (the exact size that triggers
    NrtIndex's own auto-flush) and draining each batch's lag samples right
    after its flush -- NOT polling after every single add(), which would
    never terminate: add() only flushes once flush_threshold_docs docs are
    buffered (flush_interval_s is fixed at 1e9 specifically to disable the
    time-based trigger for this single-variable sweep), so a poll loop
    keyed to one doc at a time would spin forever on docs 1..threshold-1 of
    every batch. Batching also gives the *right* distribution, not just a
    terminating one: the first doc in a batch waits ~(threshold-1) inter-add
    gaps plus build time, the last waits only build time -- exactly the
    freshness cost a large threshold imposes, which is the thing this
    experiment measures.
    """
    lag_recorder = LatencyRecorder()
    policy = TieredMergePolicy(merge_factor=merge_factor, max_segments=max_segments)
    nrt = NrtIndex(
        base_dir, flush_threshold_docs=threshold, flush_interval_s=1e9,
        initial_segments=[base_segment], merge_policy=policy,
    )

    collisions_skipped = 0
    timed_out_total = 0
    segment_count_trace: list[int] = [nrt.segment_count]
    merge_before_after: dict | None = None
    pending: list[tuple[str, str, float]] = []

    def note_segment_count() -> None:
        nonlocal merge_before_after
        segment_count_trace.append(nrt.segment_count)
        if merge_before_after is None and segment_count_trace[-1] < segment_count_trace[-2]:
            merge_before_after = {"before": segment_count_trace[-2], "after": segment_count_trace[-1]}

    for external_id, text in stream_docs:
        note_segment_count()  # cheap (one lock acquire); catches a merge landing between docs
        probe = _longest_token(text)
        if probe is None:
            continue
        pre_hits = {eid for eid, _ in nrt.search(probe, k=10)}
        # external_id itself can never be in pre_hits (base and stream doc
        # ids are disjoint slices of the corpus, and this doc hasn't been
        # added yet) -- the real failure mode isn't the target already
        # matching, it's the probe being too common: if the base corpus
        # already fills every one of the k=10 slots, the target may not
        # crack the top-10 once added, and the drain below would then time
        # out on it rather than finding a false "already present" case
        # here. len(pre_hits) >= 10 is exactly that "probe isn't
        # discriminative enough" signal, so it's what's skipped on.
        if len(pre_hits) >= 10:
            collisions_skipped += 1
            continue

        write_start = time.perf_counter()
        nrt.add(external_id, text)  # synchronously flushes once `threshold` docs are buffered
        pending.append((external_id, probe, write_start))

        if len(pending) >= threshold:
            # The add() just above triggered NrtIndex's own auto-flush, so
            # this whole batch's segment is now durable -- drain it now.
            timed_out_total += _drain_pending(nrt, pending, lag_recorder)
            pending = []
            note_segment_count()

    nrt.flush()  # force-flush the tail batch (fewer than `threshold` docs)
    timed_out_total += _drain_pending(nrt, pending, lag_recorder)
    note_segment_count()

    def dispatch(query: str) -> None:
        nrt.search(query, k=10)

    query_result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=20.0, duration_s=5.0, workers=8, seed=0,
    )
    final_segment_count = nrt.segment_count
    nrt.close()

    if timed_out_total:
        print(f"  [threshold={threshold}] WARNING: {timed_out_total} doc(s) never became searchable")

    return {
        "flush_threshold_docs": threshold,
        "lag_p99_us": lag_recorder.percentile(99),
        "lag_samples": len(lag_recorder),
        "query_p99_us": query_result.latency.percentile(99),
        "query_summary": query_result.summary(),
        "final_segment_count": final_segment_count,
        "collisions_skipped": collisions_skipped,
        "timed_out": timed_out_total,
        "merge_before_after": merge_before_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--merge-factor", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "runs" / "nrt-lag")
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())][:50]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    base_tsv = args.work_dir / "base.tsv"
    base_docs = _read_tsv_slice(CORPUS_PATH, 0, BASE_DOC_COUNT)
    base_tsv.write_text("".join(f"{eid}\t{text}\n" for eid, text in base_docs))
    # This one base_segment object is shared (read-only) across every
    # threshold's run below. That's only safe because _run_merge always
    # picks the merge_factor *smallest* segments (nrt.merge_policy) and the
    # 50,000-doc base segment is always the largest in every run this
    # module drives -- so it can never be chosen for a merge, which is the
    # only thing that deletes a segment's on-disk files. If a future change
    # ever makes merge_factor >= the live segment count at default
    # max_segments, or otherwise changes which segments get merged, this
    # sharing assumption needs re-checking.
    base_segment = build_segment(base_tsv, args.work_dir / "base_index")

    stream_docs = _read_tsv_slice(CORPUS_PATH, BASE_DOC_COUNT, STREAM_DOC_COUNT)

    points = []
    for threshold in args.thresholds:
        threshold_dir = args.work_dir / f"threshold_{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        point = run_threshold(
            threshold, threshold_dir, base_segment, stream_docs,
            args.max_segments, args.merge_factor, queries,
        )
        points.append(point)
        print(
            f"threshold={threshold:>5}  lag_p99={point['lag_p99_us']/1000:7.2f}ms  "
            f"query_p99={point['query_p99_us']/1000:6.2f}ms  "
            f"segments={point['final_segment_count']}  "
            f"collisions_skipped={point['collisions_skipped']}"
        )

    output = {
        "corpus": "msmarco-passage",
        "base_doc_count": BASE_DOC_COUNT,
        "stream_doc_count": STREAM_DOC_COUNT,
        "max_segments": args.max_segments,
        "merge_factor": args.merge_factor,
        "num_query_set_queries": len(queries),
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "nrt-lag.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check on a tiny slice (fast thresholds only, not the full sweep)**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.lag_experiment --thresholds 50 --work-dir /tmp/nrt-lag-smoke`

Expected: prints one `threshold=   50  lag_p99=...ms  query_p99=...ms  segments=...  collisions_skipped=...` line and writes `bench/results/nrt-lag.json`. This builds a real 50,000-doc base segment (takes on the order of a minute the first time) and streams 5,000 docs through — the full multi-threshold sweep runs in Task 9. Use `--work-dir /tmp/nrt-lag-smoke` here specifically so this sanity check doesn't leave partial state under the real `runs/nrt-lag` directory Task 9 will use.

- [ ] **Step 3: Commit**

```bash
git add py/nrt/lag_experiment.py
git commit -m "$(cat <<'EOF'
phase6: index-to-searchable lag experiment driver (6a), folding in the merge-in-action check (6c)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 7: Query-latency-vs-segment-count sweep driver (6b)

**Files:**
- Create: `py/nrt/segment_sweep.py`

**Interfaces:**
- Consumes: `harness.datasets.load_queries`, `harness.loadgen.run_open_loop`, `harness.runmeta.run_metadata` (existing), `server.partition_corpus.partition` (existing, Phase 2B), `nrt.segment.{build_segment, SegmentSet}` (Task 1).
- Produces (for Task 8): `bench/results/nrt-segment-sweep.json` with top-level `points: list[{n, p50_us, p99_us}]` plus provenance.

No unit test file — same experiment-driver category as Task 6.

- [ ] **Step 1: Write `py/nrt/segment_sweep.py`**

```python
"""Query latency vs. live segment count at fixed corpus size (design spec
§6b). Fixed D=200,000 docs (data/msmarco-passage.tsv lines 0-199,999):
partitioning the identical 200,000 docs into N in {1, 2, 4, 8} segments
(reusing server.partition_corpus.partition's line-count-split logic
against this fixed 200,000-line slice, then build_segment per resulting
file) isolates in-process fan-out/merge cost at fixed corpus size -- see
spec §6b for what this does and does not fix about the corpus-shrinkage
confound bench/phase2b.md already disclosed for its shards.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cascade_index

from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from nrt.segment import Segment, SegmentSet, build_segment
from server.partition_corpus import partition

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "msmarco-passage.tsv"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
FIXED_CORPUS_DOCS = 200_000


def _write_fixed_slice(work_dir: Path) -> Path:
    slice_path = work_dir / "fixed-200k.tsv"
    if slice_path.exists():
        return slice_path
    with CORPUS_PATH.open("r", encoding="utf-8") as src, slice_path.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= FIXED_CORPUS_DOCS:
                break
            dst.write(line)
    return slice_path


def _is_built(index_dir: Path) -> bool:
    # Mirrors server.build_shards.is_built: index.terms is build_index's
    # last write, so its presence means a previous run finished rather
    # than crashed partway through.
    return (index_dir / "index.terms").exists()


def _build_segment_resumable(tsv: Path, index_dir: Path):
    # These are multi-minute builds at N up to 8 * 200,000/N docs each, so
    # a crashed or interrupted run is likely to be re-invoked -- unlike
    # NrtIndex's monotonically-numbered segments (which never reuse a
    # directory), this sweep always writes to the same n{n}/seg{i} paths
    # across re-runs, so a partially-built directory from a prior crash
    # must be cleared before rebuilding into it, and a *complete* one
    # should be reused rather than rebuilt.
    if _is_built(index_dir):
        index = cascade_index.Index(str(index_dir))
        return Segment(index=index, source_tsv=tsv, index_dir=index_dir, doc_count=index.doc_count)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    return build_segment(tsv, index_dir)


def run_point(n: int, slice_path: Path, work_dir: Path, queries: list[str]) -> dict:
    shard_tsvs = partition(slice_path, n, shards_dir=work_dir / "shards")
    segments = [
        _build_segment_resumable(tsv, work_dir / "indexes" / f"n{n}" / f"seg{i}")
        for i, tsv in enumerate(shard_tsvs)
    ]
    # One persistent pool reused across every query in this point's whole
    # run_open_loop run -- SegmentSet.search opens a fresh, one-shot pool
    # only when `pool` is omitted, and per-query thread *spawns* would
    # dominate the N-segment fan-out cost this sweep exists to measure
    # (spec §6b's "the only things that scale with N" claim). Sized fixed
    # at 32 (matching NrtIndex's own default), NOT at n: run_open_loop
    # drives with workers=8 concurrent dispatches, each submitting n
    # subtasks into this pool -- sizing the pool to n would starve it at
    # small N (one worker thread serving up to 8 concurrent searches at
    # N=1) and be generously sized at large N, an N-dependent queueing
    # artifact with nothing to do with fan-out cost.
    with ThreadPoolExecutor(max_workers=32) as pool:
        segment_set = SegmentSet(segments, pool=pool)

        def dispatch(query: str) -> None:
            segment_set.search(query, k=10)

        result = run_open_loop(
            dispatch=dispatch, queries=queries, qps=20.0, duration_s=5.0, workers=8, seed=0,
        )

    return {
        "n": n,
        "p50_us": result.latency.percentile(50),
        "p99_us": result.latency.percentile(99),
        "requests": len(result.arrivals),
        "summary": result.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "runs" / "nrt-segment-sweep")
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())][:50]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    slice_path = _write_fixed_slice(args.work_dir)

    points = []
    for n in args.n:
        point = run_point(n, slice_path, args.work_dir, queries)
        points.append(point)
        print(f"N={n:>2}  p50={point['p50_us']/1000:6.2f}ms  p99={point['p99_us']/1000:6.2f}ms")

    output = {
        "corpus": "msmarco-passage",
        "fixed_corpus_docs": FIXED_CORPUS_DOCS,
        "num_query_set_queries": len(queries),
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "nrt-segment-sweep.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check on the two smallest N (not the full sweep)**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.segment_sweep --n 1 2 --work-dir /tmp/nrt-sweep-smoke`

Expected: prints `N= 1  p50=...ms  p99=...ms` and `N= 2  ...` lines and writes `bench/results/nrt-segment-sweep.json`. This builds real 200,000-doc-scale segments (a few minutes the first time) — the full N sweep runs in Task 9. Use `--work-dir /tmp/nrt-sweep-smoke` so this doesn't leave partial state under the real `runs/nrt-segment-sweep` directory Task 9 will use.

- [ ] **Step 3: Commit**

```bash
git add py/nrt/segment_sweep.py
git commit -m "$(cat <<'EOF'
phase6: query-latency-vs-segment-count sweep at fixed corpus size (6b)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 8: `report_6.py` — renders `bench/phase6.md` and both plots

**Files:**
- Create: `py/nrt/report_6.py`

**Interfaces:**
- Consumes: `bench/results/nrt-lag.json` (Task 6), `bench/results/nrt-segment-sweep.json` (Task 7).
- Produces: `bench/phase6.md`, `bench/plots/phase6-freshness.png`, `bench/plots/phase6-segment-count.png`.

Every analysis sentence is computed from the committed result JSON, not a fixed string describing one run's numbers — matching `py/server/report_2b.py`'s established convention so re-running this module against new numbers reproduces the same conclusions rather than going stale.

- [ ] **Step 1: Write `py/nrt/report_6.py`**

```python
"""Renders bench/phase6.md from nrt-lag.json (6a + 6c) and
nrt-segment-sweep.json (6b). Every analysis sentence below is computed
from the committed result JSON -- nothing here is a fixed string
describing a specific run's numbers, matching py/server/report_2b.py's
established practice.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"
LAG_PATH = RESULTS_DIR / "nrt-lag.json"
SWEEP_PATH = RESULTS_DIR / "nrt-segment-sweep.json"
REPORT_PATH = REPO_ROOT / "bench" / "phase6.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_freshness(points: list[dict]) -> Path:
    thresholds = [p["flush_threshold_docs"] for p in points]
    lag_ms = [p["lag_p99_us"] / 1000 for p in points]
    query_ms = [p["query_p99_us"] / 1000 for p in points]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(lag_ms, query_ms, marker="o")
    for t, x, y in zip(thresholds, lag_ms, query_ms):
        ax.annotate(f"t={t}", (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("index-to-searchable lag, p99 (ms)")
    ax.set_ylabel("query latency, p99 (ms)")
    ax.set_title("Phase 6: freshness/latency tradeoff")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phase6-freshness.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_segment_count(points: list[dict]) -> Path:
    ns = [p["n"] for p in points]
    p50_ms = [p["p50_us"] / 1000 for p in points]
    p99_ms = [p["p99_us"] / 1000 for p in points]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ns, p50_ms, marker="o", label="p50")
    ax.plot(ns, p99_ms, marker="o", label="p99")
    ax.set_xlabel("live segment count (N), fixed 200,000-doc corpus")
    ax.set_ylabel("query latency (ms)")
    ax.set_title("Phase 6: query latency vs. segment count")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phase6-segment-count.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _freshness_table(points: list[dict]) -> str:
    lines = ["| flush threshold (docs) | lag p99 (ms) | query p99 (ms) | final segments | collisions skipped |",
             "|---|---|---|---|---|"]
    for p in points:
        lines.append(
            f"| {p['flush_threshold_docs']} | {p['lag_p99_us']/1000:.2f} | "
            f"{p['query_p99_us']/1000:.2f} | {p['final_segment_count']} | {p['collisions_skipped']} |"
        )
    return "\n".join(lines)


def _sweep_table(points: list[dict]) -> str:
    lines = ["| N | p50 (ms) | p99 (ms) |", "|---|---|---|"]
    for p in points:
        lines.append(f"| {p['n']} | {p['p50_us']/1000:.2f} | {p['p99_us']/1000:.2f} |")
    return "\n".join(lines)


def _merge_evidence(points: list[dict]) -> str:
    for p in points:
        if p.get("merge_before_after"):
            before = p["merge_before_after"]["before"]
            after = p["merge_before_after"]["after"]
            return (
                f"At `flush_threshold_docs={p['flush_threshold_docs']}`, a merge fired "
                f"during the run: segment count went from {before} to {after}, "
                f"consistent with the configured merge policy actually bounding growth."
            )
    return "No merge event was captured in this run's trace (re-run with a smaller threshold if this section is empty)."


def render() -> None:
    lag = _load(LAG_PATH)
    sweep = _load(SWEEP_PATH)

    freshness_plot = _plot_freshness(lag["points"])
    segment_plot = _plot_segment_count(sweep["points"])

    report = f"""# Phase 6: Near-Real-Time Indexing

## Freshness/latency tradeoff

{_freshness_table(lag["points"])}

![freshness/latency tradeoff]({freshness_plot.relative_to(REPO_ROOT)})

Base corpus: {lag['base_doc_count']:,} docs (already indexed). Write stream: \
{lag['stream_doc_count']:,} docs, added one at a time. `merge_factor={lag['merge_factor']}`, \
`max_segments={lag['max_segments']}`.

## Segment-count effect (fixed corpus size)

{_sweep_table(sweep["points"])}

![query latency vs. segment count]({segment_plot.relative_to(REPO_ROOT)})

Fixed corpus: {sweep['fixed_corpus_docs']:,} docs, partitioned into N segments for each point \
-- the same corpus-shrinkage-per-segment effect `bench/phase2b.md` disclosed for its shards is \
still present here. What isolates the fan-out/merge cost specifically is that all N segments \
live in one process with no network hop and no per-shard worker pool, so the only things that \
scale with N are Python-level fan-out overhead and the merge/sort/tombstone-filter step, not \
shard process scheduling or broker-pool sizing. This is a different, valid question from \
Phase 2B's "does more shards help tail latency," not a claim to have fixed Phase 2B's confound.

## Merge-in-action evidence

{_merge_evidence(lag["points"])}

## Known limitations

- **Merge is re-tokenization, not a postings-level merge.** Every merge rebuilds from source \
TSV text via the unmodified `build_index` binary; it is not a sorted-run merge over already-built \
postings (which would require new C++ in `cpp/index/builder.cc`, out of scope here). Merge cost \
scales with merged segment size the same way a fresh build does.
- **In-process latency is not served latency.** There is no gRPC, no worker pool, no request \
queueing in this layer -- these numbers are not comparable to `bench/phase2a.md`'s or \
`bench/phase2b.md`'s served p99s.
- **No NDCG/quality claim for the multi-segment configuration.** Per-segment BM25 statistics \
mean the multi-segment configuration's relevance is not evaluated here, the same framing \
`bench/phase2b.md` established for shards.
- **Flush is synchronous and blocks new writes for its duration.** This design does not overlap \
flush with the next write batch, unlike Lucene's concurrent in-memory segment writer.
- **The tombstone over-fetch bound (`k + len(tombstones)`) is a fixed heuristic**, not a tuned \
bound against how many tombstones plausibly cluster near the top-k of a single segment.
"""
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {freshness_plot.relative_to(REPO_ROOT)}")
    print(f"wrote {segment_plot.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    render()
```

- [ ] **Step 2: Run against whatever result JSON already exists from Tasks 6/7's sanity checks**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.report_6`
Expected: writes `bench/phase6.md`, `bench/plots/phase6-freshness.png`, `bench/plots/phase6-segment-count.png` without raising. (These will be regenerated for real in Task 9 once the full sweeps have run — this step just proves the renderer doesn't crash on whatever JSON shape Tasks 6/7 already produced.)

- [ ] **Step 3: Commit**

```bash
git add py/nrt/report_6.py
git commit -m "$(cat <<'EOF'
phase6: report_6 renders bench/phase6.md and both plots from committed result JSON

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```

---

## Task 9: Full local run against the real corpus, README update

**Files:**
- Modify: `README.md` (status table + "Running Phase 6" section)
- Regenerate: `bench/results/nrt-lag.json`, `bench/results/nrt-segment-sweep.json`, `bench/phase6.md`, `bench/plots/phase6-freshness.png`, `bench/plots/phase6-segment-count.png`

- [ ] **Step 1: Run the full lag sweep**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.lag_experiment` (uses the default `--thresholds 50 200 1000`, `--work-dir runs/nrt-lag`)

Expected: one line per threshold, then `wrote bench/results/nrt-lag.json`. Confirm the `threshold=50` line's run captured a `merge_before_after` (spec §6c expects a merge to fire at this threshold with `max_segments=8`); if it shows `null`, re-run with `--max-segments 4` for just that point and note the change, since it means fewer than 8 segments existed at once during that run.

- [ ] **Step 2: Run the full segment-count sweep**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.segment_sweep` (uses the default `--n 1 2 4 8`, `--work-dir runs/nrt-segment-sweep`)

Expected: one line per N, then `wrote bench/results/nrt-segment-sweep.json`.

- [ ] **Step 3: Render the report**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m nrt.report_6`

Expected: `bench/phase6.md` and both PNGs written. Open `bench/phase6.md` and sanity-check the freshness table's trend: a smaller `flush_threshold_docs` flushes more often, so it should show **lower** lag p99 alongside **higher** query p99 and a **higher** final segment count than a larger threshold. If the table shows the opposite trend, something is wrong with the experiment — investigate before proceeding, don't just write up whatever the numbers say.

- [ ] **Step 4: Run the full nrt test suite once more against real artifacts**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest nrt/ -v`
Expected: all tests pass (none skipped, since `cpp/build/build_index` must exist for the experiment runs above to have succeeded).

- [ ] **Step 5: Update `README.md`**

Change the status table row (README.md around line 19) from:

```markdown
| 6 — Near-real-time indexing | freshness/latency tradeoff | not started |
```

to:

```markdown
| 6 — Near-real-time indexing | `bench/phase6.md` | **done** |
```

Add a new section after the existing "Running Phase 4" section (matching the existing sections' structure — check `README.md`'s current "Running Phase 2 (sub-project B: ...)" section for the exact heading/style to mirror):

```markdown
## Running Phase 6 (near-real-time indexing)

Needs the Phase 1 index build tooling (`cpp/build/build_index`, see Phase 1 setup
above) — Phase 6 runs entirely in-process against the pybind11 binding, with no
`cpp/server/` involved.

```bash
cd py
PYTHONPATH=. uv run --project .. pytest nrt/ -v

PYTHONPATH=. uv run --project .. python -m nrt.lag_experiment
PYTHONPATH=. uv run --project .. python -m nrt.segment_sweep
PYTHONPATH=. uv run --project .. python -m nrt.report_6
```

Writes `bench/results/nrt-lag.json`, `bench/results/nrt-segment-sweep.json`,
`bench/phase6.md`, and `bench/plots/phase6-*.png`. See
`docs/superpowers/specs/2026-09-10-phase6-nrt-indexing-design.md` for the design
and `bench/phase6.md`'s own "Known limitations" section for what these numbers
do and don't claim.
```

- [ ] **Step 6: Commit**

```bash
git add README.md bench/phase6.md bench/results/nrt-lag.json bench/results/nrt-segment-sweep.json bench/plots/phase6-freshness.png bench/plots/phase6-segment-count.png
git commit -m "$(cat <<'EOF'
phase6: real freshness/latency and segment-count sweeps; mark Phase 6 done in README

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DPbBqRRXRno9CEsEr7ddgX
EOF
)"
```
