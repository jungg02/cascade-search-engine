# Phase 6: near-real-time indexing

Status: approved for implementation planning
Scope: `CASCADE_SEARCH_PLAN.md` §"Phase 6 (stretch) — Near-real-time
indexing": "Almost no portfolio project does this, and it is directly
relevant to TikTok, where content freshness is the product." Segment-based
indexing, in-memory buffer flushed on a threshold, tiered background merge,
multi-segment query fan-out with merge, tombstone deletes. Exit artifact:
freshness/latency tradeoff plot.

## 1. Purpose

Every prior phase serves a **static** index, built once offline by
`cpp/index/builder.cc` and never touched again. Phase 6 asks the plan's
freshness question: how long after a document is written does it become
findable, and what does keeping that lag low cost in query latency? Two
measurements answer it:

- **Index-to-searchable lag (p99)**: wall-clock time from "document handed
  to the index" to "a query that matches it returns it," swept across
  different flush thresholds.
- **Query latency vs. live segment count**, at a **fixed total corpus
  size** — isolating the fan-out/merge cost of having many small live
  segments instead of one big one, with no corpus-size confound (see §6,
  and contrast with `bench/phase2b.md`'s disclosed corpus-shrinkage
  confound, which this experiment is deliberately designed to avoid).

Both numbers trade against each other through one knob: the flush
threshold. Small threshold → frequent flush → low lag, many live segments,
slower queries. Large threshold → the opposite. That tension is the
"freshness/latency tradeoff plot" the plan asks for.

## 2. Architecture: in-process, not a new server

`cpp/server/` (gRPC service, bounded queue, worker pool, LRU cache) is
sub-project A's answer to "how many concurrent clients can one process
serve." That is not this phase's question, and reusing it would corrupt
the lag measurement: spawning a new `server_bin` process per flushed
segment and waiting for gRPC readiness (sub-project A's own
`ready_timeout_s=120` on this machine) would dominate the lag number with
process-spawn cost having nothing to do with indexing.

Instead, Phase 6 calls directly into the existing pybind11 binding
(`cpp/bindings/module.cc`), the same path Phases 0/1 used before Phase 2
introduced the server. `cascade_index.Index(directory)` mmaps a built index
directory; opening a freshly-built segment is milliseconds, not a process
spawn. `Index.search(query, k, algorithm)` already returns
`(external_id, score)` pairs (module.cc lines 44-65) — **no docid
remapping is needed anywhere in this design**, because every segment
already resolves to the corpus's real MS MARCO passage id before Python
ever sees a result. This also settles tombstones (§5): they key on
`external_id` because that is the only id this layer ever handles.

**Disclosure, stated here and repeated in the exit artifact:** in-process
latency is not served latency. There is no gRPC, no worker pool, no
request queueing — so Phase 6's query-latency numbers are not comparable
to `bench/phase2a.md`'s or `bench/phase2b.md`'s served p99s. This is a
deliberate simplification, not an oversight: it isolates the multi-segment
fan-out/merge cost this phase is measuring, and removes the broker-pool
confound `bench/phase2b.md` §"Known limitations" already disclosed.

```
writer thread(s)          NrtIndex (py/nrt/index.py)
  add(id, text)  ───────▶  in-memory buffer
                              │  buffer size ≥ flush_threshold_docs
                              │  OR time since last flush ≥ flush_interval_s
                              ▼
                     flush(): write buffer to segment TSV,
                     shell out to build_index (unmodified),
                     open as cascade_index.Index, publish segment

query threads              SegmentSet (py/nrt/segment.py)
  search(query, k) ─────▶  fan out to every live segment's Index.search()
                            (ThreadPoolExecutor, GIL released per-call —
                             same fan-out shape as py/server/broker.py)
                            merge by score, drop tombstoned external_ids,
                            over-fetch before truncating to k (§5)

background merge            MergePolicy (py/nrt/merge_policy.py)
  after each flush   ────▶  if segment count exceeds the policy's bound,
                            pick segments to merge, concatenate their
                            source TSVs, re-run build_index (§4), swap the
                            merged segment in and the old ones out
```

## 3. Component: `SegmentSet` (`py/nrt/segment.py`)

The read/query side, with no knowledge of writing:

```python
class Segment:
    index: "cascade_index.Index"
    source_tsv: Path       # retained for future merges (§4)
    doc_count: int

def build_segment(tsv_path: Path, index_dir: Path,
                   build_index_bin: Path = BUILD_INDEX_BIN) -> Segment:
    """Shells out to the unmodified build_index binary, then opens the
    result via cascade_index.Index. Mirrors build_shards.py's subprocess
    convention exactly (same binary, same argv shape)."""

class SegmentSet:
    def __init__(self, segments: list[Segment], tombstones: set[str]): ...
    def search(self, query: str, k: int = 10,
               algorithm: Algorithm = Algorithm.BLOCKMAX_WAND) -> list[tuple[str, float]]:
        """Fans out to every segment concurrently (ThreadPoolExecutor,
        max_workers=len(segments)), requesting k + len(tombstones) from
        each segment so a tombstoned hit doesn't shrink the merged top-k
        below k, merges by score descending, filters out any external_id
        in tombstones, truncates to k."""
```

`SegmentSet` is deliberately usable on its own, with no writer attached —
the fixed-corpus segment-count sweep (§6) builds N segments once with
`build_segment` and only ever calls `SegmentSet.search`; it never touches
`NrtIndex`.

**Over-fetch bound:** requesting `k + len(tombstones)` from every segment
is conservative (a real system would only need `k` plus a bound on how
many tombstones plausibly cluster near the top-k of a single segment) but
correct and simple, and `len(tombstones)` stays small in every experiment
run here (single-digit to low-hundreds). Documented as a known
simplification, not tuned further.

## 4. Component: `NrtIndex` (`py/nrt/index.py`)

The write side, wrapping a `SegmentSet`:

```python
class NrtIndex:
    def __init__(self, base_dir: Path, flush_threshold_docs: int,
                 flush_interval_s: float = 3600.0,
                 merge_policy: "MergePolicy | None" = None,
                 build_index_bin: Path = BUILD_INDEX_BIN) -> None: ...

    def add(self, external_id: str, text: str) -> None:
        """Appends to the in-memory buffer. Flushes synchronously if the
        buffer has reached flush_threshold_docs or flush_interval_s has
        elapsed since the last flush -- whichever comes first. The
        buffered document is NOT searchable until flush completes; that
        gap is exactly what the lag experiment (§6) measures."""

    def delete(self, external_id: str) -> None:
        """Adds external_id to the tombstone set under self._lock.
        Effective on the very next search() call, across every segment
        (current and future -- a tombstone is never scoped to one
        segment, since merges don't clear it; see below)."""

    def search(self, query: str, k: int = 10,
               algorithm: Algorithm = Algorithm.BLOCKMAX_WAND) -> list[tuple[str, float]]:
        """Snapshots self._segments and self._tombstones under self._lock,
        then delegates to SegmentSet.search on the snapshot -- so a
        concurrent flush or merge swapping the segment list mid-query
        can't hand a search a partially-updated list, and the actual
        C++ search calls (which release the GIL) run outside the lock."""

    def flush(self) -> None:
        """Force-flushes the current buffer into a new segment now,
        regardless of threshold. Used by the lag experiment to bound how
        long a poll loop should wait, and by tests."""

    @property
    def segment_count(self) -> int: ...
```

**Thread safety:** `add`, `delete`, `flush`, and the segment-list swap at
the end of a background merge all mutate shared state (`self._segments`,
`self._tombstones`, the write buffer) and must hold `self._lock`
(`threading.Lock`) while doing so — the same pattern `HedgedBroker` uses
for its counters (`py/server/broker.py`). `search` holds the lock only
long enough to copy references, not for the duration of the search calls
themselves, so queries never block on a flush or merge in progress (and
vice versa) beyond that copy.

**Flush** (`_flush_locked`, called with the lock held):
1. Write the buffered `(external_id, text)` pairs to
   `base_dir/segments/seg_{n}.tsv` (`docid\ttext`, the same TSV shape
   every other phase's corpus uses).
2. `build_segment(tsv_path, base_dir/segments/seg_{n})` (§3) — this shells
   out to `build_index`, which is synchronous and blocks the calling
   thread. **This is a real, disclosed limitation**: a flush pauses new
   writes (not reads — search still runs against the previous segment
   list) for however long `build_index` takes on `flush_threshold_docs`
   documents. At the small thresholds this phase uses (tens to low
   thousands of docs), that's sub-second; disclosed in the exit artifact
   as "this design does not overlap flush with the next write batch,
   unlike Lucene's concurrent in-memory segment writer."
3. Append the new `Segment` to `self._segments`, clear the buffer, record
   the flush timestamp.
4. Call `self._merge_policy.maybe_merge(self._segments)` (§4a) — if it
   returns a merge plan, run it in a background thread
   (`ThreadPoolExecutor(max_workers=1)`, mirroring `py/server/report.py`'s
   convention for the one background task this file needs) so the flush
   call itself returns immediately after publishing the new segment; the
   merge's eventual segment-list swap happens later, still under the lock.

**Merge** (`_run_merge`, on the background thread):
1. Concatenate the chosen segments' `source_tsv` files (this is why
   `Segment` retains `source_tsv` — merging needs the original text, not
   just the built index).
2. `build_segment` on the concatenated TSV into a new segment directory.
3. Under `self._lock`: replace the merged segments with the one new
   segment in `self._segments`; delete the old segments' TSV and index
   directories from disk.

**Disclosure, stated here and in the exit artifact:** this merge
re-tokenizes and rebuilds from source text — it is not a postings-level
merge (Lucene merges are a sorted-run merge over existing postings, no
re-analysis). Re-tokenizing is why every segment must retain its source
TSV, and why merge cost scales with merged segment size the same way a
fresh build does, not with the (much cheaper) cost of merging postings
that are already built. This is disclosed explicitly rather than
implemented as a postings merge, which would mean extending
`cpp/index/builder.cc` — out of scope (§8).

### 4a. `MergePolicy` (`py/nrt/merge_policy.py`)

A simplified tiered policy, not full Lucene multi-tier:

```python
class TieredMergePolicy:
    def __init__(self, merge_factor: int = 4, max_segments: int = 8): ...
    def maybe_merge(self, segments: list[Segment]) -> list[Segment] | None:
        """If len(segments) > max_segments, returns the merge_factor
        smallest-by-doc_count segments to merge (smallest-first keeps
        merge cost low and mirrors why real tiered policies merge small
        segments before large ones -- amortized merge cost per document
        stays bounded). Returns None if no merge is needed."""
```

This bounds live segment count from above at `max_segments` (a merge
fires as soon as the count exceeds it, collapsing `merge_factor` segments
into one) — stated plainly as the mechanism, not hidden behind "tiered
merge policy" as an unexplained black box.

## 5. Tombstone deletes

Keyed on **external id**, not docid. Two independent reasons, both stated
here because both would silently corrupt results if missed:

1. **Every segment already returns external ids** (§2) — there is no
   internal docid this layer ever sees, so keying on anything else would
   require plumbing docids through `SegmentSet.search` for no reason.
2. **Merges renumber local docids.** Even if this layer did track docids,
   a tombstone keyed on `(segment, local_docid)` would silently point at
   the wrong document the moment that segment gets merged into a new one
   with different local numbering. External ids are stable across merges
   by construction (MS MARCO passage ids never change), so they are the
   only correct key.

`NrtIndex.delete(external_id)` adds to a single `set[str]` shared across
every segment, current and future — a tombstone is never migrated,
re-applied, or cleared at merge time, because it was never
segment-scoped. `SegmentSet.search` filters against it uniformly (§3).

**Correctness test, not just a unit test on the set:** delete a document,
then run a query that the design spec's own held-back-doc lag test (§6)
already knows matches only that document, and assert zero results — the
same probe-query technique used to prove the doc was findable in the
first place proves it stops being findable after delete.

## 6. Experiments

### 6a. Index-to-searchable lag vs. flush threshold (`py/nrt/lag_experiment.py`)

Fixed base corpus: `data/msmarco-passage.tsv` lines 0-49,999 (50,000
docs), built once via `build_segment` into a static base segment — this
represents "already indexed" content, not part of the write stream.

Held-back incoming stream: lines 50,000-54,999 (5,000 docs) from the same
file, fed one at a time via `NrtIndex.add`.

For each `flush_threshold_docs` in a swept set (e.g. `{50, 200, 1000}`,
`flush_interval_s` set high enough not to fire during a run — this keeps
the sweep a single-variable one, per §1):
1. Construct a fresh `NrtIndex` over the base segment plus this threshold.
2. For each held-back doc, in order:
   a. Build a **probe query**: the longest whitespace-token in the doc's
      text (ties broken by first occurrence) — a token likely to be rare
      enough that it doesn't already match base-corpus documents.
   b. **Pre-condition check**: `NrtIndex.search(probe)` before `add()`
      returns zero results containing this doc's external_id (if it
      already does, the probe token collided with a base-corpus document
      — log and skip this doc rather than record a meaningless lag).
   c. `write_start = time.perf_counter()`; `NrtIndex.add(external_id, text)`.
   d. Poll `NrtIndex.search(probe)` (small sleep between polls, e.g. 1ms)
      until the doc's external_id appears; record
      `found_at - write_start` as this doc's lag sample.
3. Record the full lag distribution's **p99** (never a mean — per
   `py/harness/histogram.py`'s convention, reused here) for this
   threshold, plus the resulting average segment count and a
   `SegmentSet.search` p99 query-latency sample (fixed small query set,
   open-loop-style timing) against the index's state at the end of the
   run.
4. Output: `bench/results/nrt-lag.json` — one row per threshold with lag
   p99, query p99, and final segment count.

### 6b. Query latency vs. segment count at fixed corpus size (`py/nrt/segment_sweep.py`)

Fixed `D = 200,000` docs: `data/msmarco-passage.tsv` lines 0-199,999.
Deliberately the same corpus for every point — unlike
`bench/phase2b.md`'s shards, per-segment corpus size shrinks with N here
too, but that's *why* this experiment exists: partitioning the identical
200,000 docs into N ∈ {1, 2, 4, 8} segments (reusing
`server.partition_corpus.partition`'s line-count-split logic against this
fixed 200,000-line slice, then `build_segment` per resulting file, exactly
`py/server/build_shards.py`'s pattern but producing `Segment`/`SegmentSet`
objects instead of `ServerProcess`es) still shrinks per-segment corpus
size as N grows, the same confound `bench/phase2b.md` disclosed.

**What isolates the fan-out/merge effect here that Phase 2B couldn't
isolate:** all N segments live in **one process** with **no network hop
and no per-shard worker pool** (§2's disclosure) — so the *only* things
that scale with N are Python-level fan-out overhead (thread submission,
result collection) and the merge/sort/tombstone-filter step, not shard
process scheduling or broker-pool sizing. The corpus-shrinkage effect is
still present and must still be disclosed (§8), but this experiment's
framing is explicitly "in-process fan-out overhead at fixed corpus size,"
which is a different, valid question from Phase 2B's "does more shards
help tail latency" — not a claim to have fixed Phase 2B's confound.

For each N: open-loop-style fixed query set (reuse `harness.loadgen`'s
query sampling against the same dev query set other phases use, or the
dev queries already present in `runs/manifest.json`-adjacent files),
measure `SegmentSet.search` latency distribution, record p50/p99.

Output: `bench/results/nrt-segment-sweep.json` — one row per N.

### 6c. Merge-in-action check (folded into `lag_experiment.py`'s smallest-threshold run)

At `flush_threshold_docs=50` over a 5,000-doc stream, `max_segments=8`
guarantees several merges fire during the run. Record segment count right
before and right after each merge event (the policy already returns the
merge plan; log its effect) and include one before/after segment-count
data point in the exit artifact as evidence the policy is actually
bounding growth, not just configured to.

## 7. Testing

- `py/nrt/tests/test_segment.py`: `build_segment` on a tiny synthetic TSV
  in a tmp dir produces a `Segment` whose `.index.doc_count` matches the
  input line count; `SegmentSet.search` over 2 synthetic segments returns
  the true top-k of the union (mirrors `test_broker.py`'s merge assertion
  exactly, in-process instead of over stub network clients).
- `py/nrt/tests/test_tombstones.py`: build a 1-segment `SegmentSet`,
  delete one doc's external_id, assert it never appears in results even
  when it would otherwise be top-1 for a query that exactly matches it;
  assert the over-fetch (§3) means a *different* doc still fills the
  vacated k-th slot rather than the result list silently shrinking to k-1.
- `py/nrt/tests/test_nrt_index.py`: a synthetic corpus in a tmp dir,
  `flush_threshold_docs=3`; add 5 docs, assert `segment_count == 2` after
  the 3rd add (auto-flush) and a 3rd segment appears only after
  `.flush()` is called explicitly for the remaining 2; assert a doc is
  NOT searchable immediately after `add()` but IS searchable immediately
  after the flush that contains it (the core lag-experiment assumption,
  tested directly rather than only inferred from §6a's timing numbers).
- `py/nrt/tests/test_merge_policy.py`: `TieredMergePolicy(merge_factor=2,
  max_segments=3)` fed a synthetic list of 4 segments returns a plan to
  merge the 2 smallest; fed a list of 3 returns `None`.
- `py/nrt/tests/test_nrt_index_merge.py`: `flush_threshold_docs` small
  enough and `max_segments` small enough that a real merge fires within
  the test; assert segment count drops back to within the policy's bound
  after the merge completes (poll with a bounded timeout, since the merge
  runs on a background thread) and that every document added before the
  merge is still findable afterward (merge must not lose documents).
- No re-test of WAND/BlockMax-WAND correctness, BM25 scoring, or
  `build_index`'s own segment/merge logic — all proven by Phase 1 and
  untouched; this layer only adds buffering, flush-triggered rebuilds,
  fan-out, tombstones, and a merge policy on top of components already
  known correct.

## 8. Exit artifact

`bench/phase6.md`:
- Freshness/latency tradeoff: the §6a table (threshold → lag p99, query
  p99, final segment count) plus `bench/plots/phase6-freshness.png`
  (lag p99 vs. query p99, one point per threshold, connected to show the
  tradeoff curve directly — the plan's explicit ask).
- Segment-count effect: the §6b table plus
  `bench/plots/phase6-segment-count.png` (query p50/p99 vs. N at fixed
  D=200,000), with an explicit cross-reference to `bench/phase2b.md`'s
  disclosed corpus-shrinkage confound and a statement of what this
  experiment's in-process, single-process design does and does not fix
  about that confound (§6b).
- Merge-in-action evidence (§6c): one before/after segment-count data
  point.
- **Known limitations**, written up front rather than left for a
  reviewer to find (mirroring `bench/phase2b.md`'s final-fix-wave
  lesson): merge is re-tokenization not a postings merge (§4); in-process
  latency isn't served latency (§2); per-segment BM25 statistics mean no
  NDCG/quality claim for the multi-segment configuration (§8, inherits
  `bench/phase2b.md` §"Docid-semantics limitation" framing verbatim,
  adapted from "shard" to "segment"); flush is synchronous and blocks new
  writes for its duration (§4); the over-fetch bound on tombstoned
  results is a fixed `len(tombstones)` heuristic, not a tuned bound (§3).

## 9. Resource constraints

This machine has 8GB RAM and no GPU (established in Phases 3/4). Every
corpus slice here is small by construction: the lag experiment's base
segment is 50,000 docs (~1/175th of the full corpus), its write stream is
5,000 docs, and the segment-count sweep's fixed corpus is 200,000 docs
(~1/44th) — chosen specifically so that dozens of `build_index` calls
(one per flush, one per merge, one per sweep point) each complete in low
single-digit seconds rather than the minutes a full-corpus build takes,
keeping the whole phase's wall-clock time tractable for repeated
experiment runs during development. No concurrent process fan-out (§2) —
this phase runs single-process throughout, so there is no analog to
Phase 2B's "16 co-resident `server_bin` processes on 8GB" pressure point.

## 10. Out of scope

- Any change to `cpp/index/builder.cc`, `cpp/query/searcher.h`, or
  `cpp/bindings/module.cc` — all proven and frozen; this phase is a pure
  Python orchestration layer over the existing, unmodified `build_index`
  binary and `Index` class, the same posture Phase 2B took toward
  `cpp/server/`.
- A true postings-level merge (no re-tokenization) — would require new
  C++ to merge two `index.post`/`index.terms` files directly; disclosed
  as a limitation (§4, §8) rather than implemented.
- Concurrent (overlapped) flush — a flush blocks new writes for its
  duration (§4); making writes and flush truly concurrent (Lucene's
  approach) is real added complexity with its own concurrency-correctness
  surface, out of scope for a stretch phase already adding buffering,
  flush, merge, fan-out, and tombstones in one pass.
- Global (cross-segment) BM25 statistics or any NDCG/quality claim for
  the multi-segment configuration — same framing Phase 2B already
  established for shards (§8).
- Serving this over `cpp/server/`'s gRPC path, or any load-generator
  throughput/capacity claim — §2 explains why in-process is the right
  choice for measuring lag and fan-out cost specifically; a served NRT
  index is future scope, not this phase's.
- Distributed/multi-node NRT (this is one process, one machine, matching
  every other phase's scope).
