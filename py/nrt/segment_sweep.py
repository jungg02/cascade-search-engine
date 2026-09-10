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
