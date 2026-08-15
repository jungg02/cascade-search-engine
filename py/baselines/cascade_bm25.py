"""Cascade C++ index — Phase 1's own BM25 baseline, run three ways.

Unlike Lucene and Tantivy, this "baseline" is the thing under test: the same
`cascade_index` binding (built from cpp/query/searcher.cc) that the correctness
suite in cpp/tests/test_index.cc checks WAND and BlockMax-WAND against exhaustive
DAAT-OR on a synthetic corpus. Running it here, over the same TREC-DL queries and
against the same qrels as the Phase 0 baselines, is what turns "the top-10s
matched on 20k synthetic docs" into "the NDCG@10 matches Lucene's within ~0.01 on
the real 8.8M-passage corpus" — the plan's actual Phase 1 exit bar.

k1/b are Anserini's msmarco defaults (0.9/0.4), matched deliberately — see
BM25Params in cpp/index/index_format.h. The analyzer targets Lucene's
EnglishAnalyzer (Porter stemming + stopwords); see cpp/index/analyzer.h for the
known divergence (an ASCII approximation of UAX#29 segmentation) if NDCG@10
misses Lucene's by more than the plan's tolerance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cascade_index

from baselines import append_manifest
from harness.datasets import DATA_DIR, QUERY_SETS, load_queries
from harness.histogram import LatencyRecorder
from harness.runfile import write_run

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX = REPO_ROOT / "cpp" / "build" / "build_index"
CORPUS_TSV = DATA_DIR / "msmarco-passage.tsv"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"
RUNS_DIR = REPO_ROOT / "runs"

K1, B = 0.9, 0.4
ANALYZER = "EnglishAnalyzer (Porter stemming + stopwords), targets Anserini default"

ALGORITHMS = {
    "cascade-daat-or": cascade_index.Algorithm.DAAT_OR,
    "cascade-wand": cascade_index.Algorithm.WAND,
    "cascade-blockmax-wand": cascade_index.Algorithm.BLOCKMAX_WAND,
}


def build_index(force: bool = False) -> dict:
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()) and not force:
        print(f"index already present at {INDEX_DIR}")
        return {"skipped": True}

    if not BUILD_INDEX.exists():
        raise SystemExit(f"{BUILD_INDEX} not built; run `make all` in cpp/ first")

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subprocess.run([str(BUILD_INDEX), str(CORPUS_TSV), str(INDEX_DIR)], check=True)
    elapsed = time.perf_counter() - started
    return {
        "skipped": False,
        "build_seconds": elapsed,
        "index_bytes": sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file()),
    }


def search(engine: str, query_set: str, hits: int = 1000, record_stats: bool = False) -> dict:
    """Run one (algorithm, query set) pair. Returns the manifest entry.

    `record_stats` also aggregates SearchStats and per-query latency — the
    columns the plan's exit table wants beyond NDCG. It is off for the depth-1000
    NDCG runs (matching Lucene/Tantivy's operating point) because pruning
    behavior is a different regime at depth 1000 than at the depth-10 a real
    query would use; see cascade_query_cost.py for that measurement instead.
    """
    index = cascade_index.Index(str(INDEX_DIR))
    algorithm = ALGORITHMS[engine]
    queries = load_queries(query_set)

    latency = LatencyRecorder()
    postings_scored = 0
    full_evaluations = 0
    run: dict[str, dict[str, float]] = {}
    wall_started = time.perf_counter()
    for qid, text in sorted(queries.items()):
        started = time.perf_counter_ns()
        results, stats = index.search(text, hits, algorithm)
        latency.record((time.perf_counter_ns() - started) / 1000)
        run[qid] = {docid: score for docid, score in results}
        postings_scored += stats.postings_scored
        full_evaluations += stats.full_evaluations
    wall_seconds = time.perf_counter() - wall_started

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{engine}.{query_set}.txt"
    write_run(run, run_path, tag=engine, depth=hits)

    entry = {
        "engine": engine,
        "query_set": query_set,
        "run_path": str(run_path.relative_to(REPO_ROOT)),
        "k1": K1,
        "b": B,
        "hits": hits,
        "analyzer": ANALYZER,
        "num_queries": len(queries),
        "wall_seconds": wall_seconds,
    }
    if record_stats:
        entry["mean_postings_scored"] = postings_scored / len(queries)
        entry["mean_full_evaluations"] = full_evaluations / len(queries)
        entry["serial_latency_us"] = latency.summary()
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--hits", type=int, default=1000)
    args = parser.parse_args()

    print(json.dumps(build_index(force=args.force_index), indent=2))

    # WAND and BlockMax-WAND run the full NDCG sweep. DAAT-OR is correct by
    # construction but scores every posting for every query term regardless of
    # depth, so running it over dev's 6,980 queries would be expensive for a
    # comparison the synthetic-corpus suite (cpp/tests/test_index.cc) already
    # covers structurally. It still runs on dl19+dl20 (97 queries, seconds, not
    # skipped): that's the actual real-corpus check that WAND/BMW's pruning
    # didn't drop anything Lucene's own NDCG numbers would have caught, and
    # phase1_report.py's equivalence claim is only true because this runs.
    engine_query_sets = {
        "cascade-daat-or": ["dl19", "dl20"],
        "cascade-wand": list(QUERY_SETS),
        "cascade-blockmax-wand": list(QUERY_SETS),
    }
    for engine, query_sets in engine_query_sets.items():
        for query_set in query_sets:
            entry = search(engine, query_set, hits=args.hits)
            append_manifest(entry)
            print(f"{engine:>24} {query_set:>5}  wrote {entry['run_path']}  "
                  f"({entry['wall_seconds']:.1f}s wall)")


if __name__ == "__main__":
    main()
