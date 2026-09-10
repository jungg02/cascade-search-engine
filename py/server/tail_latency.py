"""Tail-latency-amplification experiment: single-shard p99 (through the
broker, so every point pays the same broker overhead -- see design spec
§6) vs. N-shard fan-out p99, for N in {1, 4, 8, 16}. Mirrors
throughput_knee.py's structure: same open-loop generator, same dev-query
Zipfian sampler, same repeats-per-point protocol.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.broker import Broker
from server.build_shards import build_shards
from server.partition_corpus import CORPUS_PATH, partition
from server.shard_cluster import ShardCluster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
SINGLE_INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(broker: Broker, qps: float, duration_s: float, workers: int,
            hits: int, queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        broker.dispatch(query, k=hits)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    return result.summary()


def run_point(n: int, index_dirs: list[Path], base_port: int, args, queries: list[str]) -> dict:
    with ShardCluster(
        index_dirs, base_port=base_port, workers=args.server_workers,
        queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
        algorithm=args.algorithm, ready_timeout_s=120.0,
    ) as cluster:
        broker = Broker.for_addresses(cluster.addresses)
        repeats = [
            measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                    queries, args.zipf_s, seed)
            for seed in range(args.repeats)
        ]
        broker.close()
    p99s = [r["latency"]["p99_us"] for r in repeats]
    return {
        "n": n,
        "repeats": repeats,
        "p99_us_across_repeats": {
            "min": min(p99s), "median": statistics.median(p99s), "max": max(p99s),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--qps", type=float, default=20.0,
                         help="fixed, moderate QPS below every configuration's own knee")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--base-port", type=int, default=50300)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    points = []
    for n in args.n:
        if n == 1:
            index_dirs = [SINGLE_INDEX_DIR]
        else:
            shard_tsvs = partition(CORPUS_PATH, n)
            index_dirs = build_shards(shard_tsvs, n)
        point = run_point(n, index_dirs, args.base_port, args, queries)
        points.append(point)
        p99 = point["p99_us_across_repeats"]["median"]
        print(f"N={n:>3}   p99={p99/1000:7.2f}ms")

    output = {
        "server": "cascade-grpc-broker",
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
        "qps": args.qps,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-tail-latency.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
