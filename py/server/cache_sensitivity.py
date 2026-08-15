"""Cache-sensitivity experiment: sweep the Zipfian skew, plot hit rate
against p99, and split cache-hit and cache-miss latency (a blended number
hides which one actually matters — the same principle bench/baselines.md
applies to queueing).

Run at one fixed, moderate QPS, below the throughput knee found by
throughput_knee.py (Task 8), so the numbers reflect cache behavior rather
than queueing on top of it.

Each swept zipf_s point gets its own fresh ServerProcess (and therefore a
cold cache). Task 8's throughput_knee.py originally shared one server across
its whole QPS sweep and had to be fixed for exactly this reason: the Zipfian
sampler is rank-based, so it favors the same head queries regardless of
seed, and a shared cache means later sweep points inherit warmth from every
earlier point's traffic. That's tolerable for a QPS sweep (queueing
behavior doesn't depend on cache history) but not here — hit rate as a
function of skew is the entire point of this experiment, so a shared cache
would inflate later points and distort the hit-rate-vs-s curve.
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
from server.client import SearchClient
from server.process import ServerProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(client: SearchClient, zipf_s: float, qps: float, duration_s: float,
            workers: int, hits: int, queries: list[str], seed: int) -> dict:
    hit_latency = LatencyRecorder()
    miss_latency = LatencyRecorder()
    hits_count = 0
    misses_count = 0

    def dispatch(query: str) -> None:
        nonlocal hits_count, misses_count
        # This times the same dispatch() call run_open_loop's own `service`
        # recorder already times; the duplication is necessary because that
        # recorder has no hit/miss split, and adding one would mean changing
        # harness/loadgen.py for every other caller (Phase 0's baselines
        # included) rather than just this one experiment.
        started = time.perf_counter_ns()
        response = client.dispatch(query, k=hits)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        if response.cache_hit:
            hits_count += 1
            hit_latency.record(elapsed_us)
        else:
            misses_count += 1
            miss_latency.record(elapsed_us)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    total = hits_count + misses_count
    return {
        "zipf_s": zipf_s,
        "hit_rate": hits_count / total if total else 0.0,
        "hits": hits_count,
        "misses": misses_count,
        "hit_latency": hit_latency.summary(),
        "miss_latency": miss_latency.summary(),
        "overall": result.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipf-s", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--qps", type=float, default=20.0,
                         help="fixed QPS, chosen below the throughput knee")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--port", type=int, default=50162)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    points = []
    for zipf_s in args.zipf_s:
        with ServerProcess(
            INDEX_DIR, port=args.port, workers=args.server_workers,
            queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
            algorithm=args.algorithm,
        ) as server:
            client = SearchClient(server.address)
            point = measure(client, zipf_s, args.qps, args.duration, args.client_workers,
                             args.hits, queries, args.seed)
            client.close()
        points.append(point)
        hit_p99 = point["hit_latency"].get("p99_us", 0) / 1000
        miss_p99 = point["miss_latency"].get("p99_us", 0) / 1000
        print(f"s={zipf_s:>4.1f}  hit_rate={point['hit_rate']:.1%}  "
              f"hit_p99={hit_p99:.2f}ms  miss_p99={miss_p99:.2f}ms")

    output = {
        "server": "cascade-grpc",
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "qps": args.qps,
        "duration_s": args.duration,
        "cache_capacity": args.cache_capacity,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-cache-sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
