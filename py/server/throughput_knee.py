"""Throughput-knee experiment: QPS on the x-axis, p99 on the y-axis, against
the real gRPC server (not an in-process pybind call, unlike Phase 1's
query-cost table) — the whole point of Phase 2 is measuring what a server's
own concurrency policy (fixed worker pool + bounded queue) does under load.

Structure mirrors baselines/latency.py: the same open-loop generator, the
same Zipfian query sampler over dev queries, the same 3-repeats-per-point
protocol (a laptop's thermal state isn't stable enough to trust one run).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.client import SearchClient
from server.process import ServerProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(client: SearchClient, qps: float, duration_s: float, workers: int,
            hits: int, queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        client.dispatch(query, k=hits)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    # RESOURCE_EXHAUSTED under load is an *expected* outcome at high offered
    # QPS, not a broken run: run_open_loop's own dispatch wrapper already
    # catches any exception and records it as an error rather than raising,
    # so a high error count at high QPS is exactly the knee showing up.
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=8)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--port", type=int, default=50161)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    with ServerProcess(
        INDEX_DIR, port=args.port, workers=args.server_workers,
        queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
        algorithm=args.algorithm,
    ) as server:
        client = SearchClient(server.address)

        points = []
        for qps in args.qps:
            repeats = [
                measure(client, qps, args.duration, args.client_workers, args.hits,
                        queries, args.zipf_s, seed)
                for seed in range(args.repeats)
            ]
            p99s = [r["latency"]["p99_us"] for r in repeats]
            point = {
                "offered_qps": qps,
                "repeats": repeats,
                "p99_us_across_repeats": {
                    "min": min(p99s), "median": statistics.median(p99s), "max": max(p99s),
                },
            }
            points.append(point)
            median = statistics.median([r["latency"]["p50_us"] for r in repeats])
            errors = sum(r["errors"] for r in repeats)
            print(f"{qps:>6.0f} QPS   p50={median/1000:7.2f}ms   "
                  f"p99={statistics.median(p99s)/1000:7.2f}ms   errors={errors}")

        client.close()

    output = {
        "server": "cascade-grpc",
        "generator": "open-loop, Poisson arrivals, latency from scheduled arrival",
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "zipf_s": args.zipf_s,
        "client_workers": args.client_workers,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "cache_capacity": args.cache_capacity,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-throughput-knee.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
