"""Phase 1's money table: postings scored, full evaluations, and p99 latency
for DAAT-OR, WAND, and BlockMax-WAND at a realistic top-10 depth.

Depth matters here in a way it doesn't for the NDCG runs in cascade_bm25.py:
WAND/BlockMax-WAND's pruning strength comes from how fast the top-k threshold
rises, and a threshold from only 10 slots rises far faster than one from 1000 —
so this is a different, and more representative, operating point from the
depth-1000 runs used for NDCG.

Serial and single-threaded, not open-loop: this is "how much work does one
query cost," the same thing cpp/tests/test_index.cc measures on the synthetic
corpus, just at the real 8.8M-document scale and over real TREC queries instead
of a Zipfian vocabulary of 400 made-up terms. Concurrent load and queueing are
Phase 2's subject (see harness/loadgen.py), not this one.

DAAT-OR's cost does not shrink with depth — every posting for every query term
gets scored regardless of how many results are kept — so it is the expensive
one here, deliberately run over the same fixed query sample as WAND and
BlockMax-WAND rather than the full query sets cascade_bm25.py uses, to keep the
whole comparison tractable while still being apples-to-apples across algorithms.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cascade_index

from baselines.cascade_bm25 import ALGORITHMS, ANALYZER, B, INDEX_DIR, K1
from harness.datasets import load_queries
from harness.histogram import LatencyRecorder
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def sample_queries(query_set: str, n: int, seed: int) -> list[str]:
    queries = load_queries(query_set)
    texts = [text for _, text in sorted(queries.items())]
    if n >= len(texts):
        return texts
    return random.Random(seed).sample(texts, n)


def measure(engine: str, queries: list[str], hits: int) -> dict:
    index = cascade_index.Index(str(INDEX_DIR))
    algorithm = ALGORITHMS[engine]

    latency = LatencyRecorder()
    postings_scored = 0
    full_evaluations = 0
    blocks_decoded = 0
    candidates_seen = 0

    for text in queries:
        started = time.perf_counter_ns()
        _, stats = index.search(text, hits, algorithm)
        latency.record((time.perf_counter_ns() - started) / 1000)
        postings_scored += stats.postings_scored
        full_evaluations += stats.full_evaluations
        blocks_decoded += stats.blocks_decoded
        candidates_seen += stats.candidates_seen

    n = len(queries)
    return {
        "engine": engine,
        "num_queries": n,
        "hits": hits,
        "mean_postings_scored": postings_scored / n,
        "mean_full_evaluations": full_evaluations / n,
        "mean_blocks_decoded": blocks_decoded / n,
        "mean_candidates_seen": candidates_seen / n,
        "serial_latency_us": latency.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-set", default="dev")
    # DAAT-OR runs ~23ms/query on the full 8.8M-passage index (measured), so the
    # full dev set (~6,980 queries) costs a few minutes total across all three
    # algorithms and is well worth it: p99 off 500 samples is noisy (the 99th
    # percentile is only the ~5th-largest value), off 6,980 it isn't.
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = sample_queries(args.query_set, args.sample_size, args.seed)
    print(f"sampled {len(queries)} queries from {args.query_set}")

    results = []
    for engine in ["cascade-daat-or", "cascade-wand", "cascade-blockmax-wand"]:
        result = measure(engine, queries, args.hits)
        results.append(result)
        lat = result["serial_latency_us"]
        print(
            f"{engine:>24}  postings={result['mean_postings_scored']:>12,.0f}  "
            f"evals={result['mean_full_evaluations']:>10,.0f}  "
            f"p99={lat['p99_us']/1000:>8.3f}ms"
        )

    output = {
        "query_set": args.query_set,
        "sample_size": len(queries),
        "seed": args.seed,
        "k1": K1,
        "b": B,
        "analyzer": ANALYZER,
        "results": results,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "cascade-query-cost.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
