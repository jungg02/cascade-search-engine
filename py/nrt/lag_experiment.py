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
