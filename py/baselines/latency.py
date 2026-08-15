"""Open-loop latency measurement against a real BM25 engine.

Phase 0 asks for the baselines' latency, not just their NDCG. It also asks for
an open-loop generator, and until now `run_open_loop` had only ever been driven
against `time.sleep`. This drives it against Tantivy, which measures the engine
and exercises the generator at the same time.

Two deliberate choices:
  - The query stream is Zipfian over the dev queries, not uniform. Real traffic
    is skewed, and the skew is what a result cache exploits in Phase 2.
  - Each QPS point is run three times. The plan's pitfall list calls out laptop
    thermal throttling; reporting the spread across repeats is how we show
    whether a number is stable rather than asserting that it is.

Lucene is deliberately absent here. Driving Anserini per query would mean a JVM
subprocess per request, which would measure JVM startup. Lucene per-query
latency is deferred to Phase 2, where the index is behind a persistent server.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import tantivy

from baselines.tantivy_bm25 import INDEX_DIR, sanitize
from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def measure(qps: float, duration_s: float, workers: int, hits: int, queries: list[str],
            zipf_s: float, seed: int) -> dict:
    index = tantivy.Index.open(str(INDEX_DIR))
    index.reload()
    searcher = index.searcher()

    def dispatch(query: str) -> None:
        searcher.search(index.parse_query(query, ["contents"]), hits)

    result = run_open_loop(
        dispatch=dispatch,
        queries=queries,
        qps=qps,
        duration_s=duration_s,
        workers=workers,
        seed=seed,
        zipf_s=zipf_s,
    )
    if result.errors:
        raise RuntimeError(f"{result.errors} dispatch failures at {qps} QPS")
    if len(result.latency) == 0:
        raise RuntimeError(f"no samples recorded at {qps} QPS")
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", default=[10, 25, 50])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    args = parser.parse_args()

    # Ordered by query id so the Zipfian head is a stable, reproducible subset
    # rather than whatever order the loader happened to yield.
    queries = [sanitize(text) for _, text in sorted(load_queries("dev").items())]
    queries = [q for q in queries if q]

    points = []
    for qps in args.qps:
        repeats = [
            measure(qps, args.duration, args.workers, args.hits, queries, args.zipf_s, seed)
            for seed in range(args.repeats)
        ]
        p99s = [r["latency"]["p99_us"] for r in repeats]
        point = {
            "offered_qps": qps,
            "repeats": repeats,
            "p99_us_across_repeats": {
                "min": min(p99s),
                "median": statistics.median(p99s),
                "max": max(p99s),
            },
        }
        points.append(point)
        median = statistics.median([r["latency"]["p50_us"] for r in repeats])
        print(f"{qps:>6.0f} QPS   p50={median/1000:7.2f}ms   "
              f"p99={statistics.median(p99s)/1000:7.2f}ms   "
              f"(p99 spread {min(p99s)/1000:.2f}-{max(p99s)/1000:.2f}ms)")

    output = {
        "engine": "tantivy",
        "generator": "open-loop, Poisson arrivals, latency from scheduled arrival",
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "zipf_s": args.zipf_s,
        "workers": args.workers,
        "hits": args.hits,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "latency.tantivy.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
