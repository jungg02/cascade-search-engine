"""Hedged-requests experiment, at one representative N (default 8, matching
the plan's own worked capacity-statement example): warm-up to measure each
shard's own p95, then baseline (no hedging) vs. hedged runs at the same
fixed QPS, reporting both the p99 delta and the extra-request cost -- see
design spec §6.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from harness.datasets import load_queries
from harness.histogram import LatencyRecorder
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.broker import Broker, HedgedBroker
from server.build_shards import build_shards
from server.client import SearchClient
from server.partition_corpus import CORPUS_PATH, partition
from server.shard_cluster import ShardCluster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def measure_shard_p95(client: SearchClient, qps: float, duration_s: float, workers: int,
                       hits: int, queries: list[str], zipf_s: float, seed: int) -> float:
    recorder = LatencyRecorder()

    def dispatch(query: str) -> None:
        started = time.perf_counter_ns()
        client.dispatch(query, k=hits)
        recorder.record((time.perf_counter_ns() - started) / 1000)

    run_open_loop(dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
                   workers=workers, seed=seed, zipf_s=zipf_s)
    return recorder.percentile(95)


def measure(broker, qps: float, duration_s: float, workers: int, hits: int,
            queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        broker.dispatch(query, k=hits)

    result = run_open_loop(dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
                            workers=workers, seed=seed, zipf_s=zipf_s)
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--warmup-duration", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--base-port", type=int, default=50400)
    parser.add_argument("--replica-base-port", type=int, default=50500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]
    shard_tsvs = partition(CORPUS_PATH, args.n)
    index_dirs = build_shards(shard_tsvs, args.n)
    server_kwargs = dict(
        workers=args.server_workers, queue_depth=args.queue_depth,
        cache_capacity=args.cache_capacity, algorithm=args.algorithm,
        ready_timeout_s=120.0,
    )

    # 1. Warm-up: measure each shard's own p95 service latency,
    #    client-side, one shard at a time -- Broker doesn't expose
    #    per-shard timing on its own merged view, so this wraps each
    #    shard's SearchClient individually instead.
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as cluster:
        shard_p95s = []
        for address in cluster.addresses:
            client = SearchClient(address)
            shard_p95s.append(
                measure_shard_p95(client, args.qps, args.warmup_duration, args.client_workers,
                                   args.hits, queries, args.zipf_s, args.seed)
            )
            client.close()
    hedge_delay_us = max(shard_p95s)
    print(f"warm-up shard p95s (us): {[round(p) for p in shard_p95s]}, "
          f"hedge_delay={hedge_delay_us/1000:.2f}ms")

    # 2. Baseline: no hedging.
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as cluster:
        broker = Broker.for_addresses(cluster.addresses)
        baseline_repeats = [
            measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                    queries, args.zipf_s, seed)
            for seed in range(args.repeats)
        ]
        broker.close()

    # 3. Hedged: replica cluster serves the same index directories as a
    #    second set of processes on a disjoint port range -- a replica
    #    duplicates the process, not the on-disk index (design spec §5).
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as primary_cluster, \
            ShardCluster(index_dirs, base_port=args.replica_base_port, **server_kwargs) as replica_cluster:
        hedge_delay_s = hedge_delay_us / 1e6
        hedged_repeats = []
        for seed in range(args.repeats):
            broker = HedgedBroker.for_addresses(
                primary_cluster.addresses, replica_cluster.addresses, hedge_delay_s=hedge_delay_s,
            )
            summary = measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                               queries, args.zipf_s, seed)
            summary["backup_calls_sent"] = broker.backup_calls_sent
            summary["total_shard_calls"] = broker.total_shard_calls
            hedged_repeats.append(summary)
            broker.close()

    baseline_p99 = statistics.median([r["latency"]["p99_us"] for r in baseline_repeats])
    hedged_p99 = statistics.median([r["latency"]["p99_us"] for r in hedged_repeats])
    total_backup = sum(r["backup_calls_sent"] for r in hedged_repeats)
    total_calls = sum(r["total_shard_calls"] for r in hedged_repeats)
    extra_load_pct = total_backup / total_calls if total_calls else 0.0
    print(f"baseline p99={baseline_p99/1000:.2f}ms   hedged p99={hedged_p99/1000:.2f}ms   "
          f"extra load={extra_load_pct:.1%}")

    output = {
        "server": "cascade-grpc-hedged-broker",
        "n": args.n,
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "qps": args.qps,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "zipf_s": args.zipf_s,
        "client_workers": args.client_workers,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "cache_capacity": args.cache_capacity,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "shard_p95_us_warmup": shard_p95s,
        "hedge_delay_us": hedge_delay_us,
        "baseline_repeats": baseline_repeats,
        "hedged_repeats": hedged_repeats,
        "baseline_p99_us_median": baseline_p99,
        "hedged_p99_us_median": hedged_p99,
        "extra_load_fraction": extra_load_pct,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-hedging.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
