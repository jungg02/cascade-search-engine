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
from harness.histogram import LatencyRecorder
from harness.loadgen import ZipfQuerySampler, run_open_loop
from harness.runmeta import run_metadata
from server.client import SearchClient
from server.process import ServerProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(client: SearchClient, qps: float, duration_s: float, workers: int,
            hits: int, queries: list[str], zipf_s: float, seed: int) -> dict:
    # queue_wait_us comes straight off the response (SearchServiceImpl sets
    # it from the server's own BoundedQueue) — this is what actually
    # distinguishes "the server's bounded queue is backing up" from
    # "run_open_loop's queue_delay is backing up" (that one measures time
    # waiting for a free *client*-side ThreadPoolExecutor slot, not
    # anything server-side). cache_hit is tracked the same way
    # cache_sensitivity.py already does it, so this run reports its own
    # actual hit rate instead of borrowing one from a different sweep.
    server_queue_wait = LatencyRecorder()
    cache_hits = 0
    cache_total = 0

    # Warm-up: pay whatever fixed per-process cost the first ~75 requests
    # incur (most likely LRU fill fraction — the cache starting empty, not
    # mmap page faults; see below) before the timed window, not during it.
    # Confirmed against real 10s-duration sweep data (not assumed): only the
    # lowest-QPS point tested (20 QPS, ~190-210 samples per repeat) shows a
    # monotone repeat-to-repeat decay in p50/p99/service time (one run: p50
    # 7.97ms -> 3.71ms -> 3.49ms, service 5.90ms -> 1.50ms -> 1.12ms), while
    # 100 QPS and 200 QPS (5-10x more samples per repeat, same fixed cost)
    # don't show it. That's the signature of a fixed per-process cost getting
    # diluted by an ever-larger fraction of the run as offered rate rises,
    # not a real capacity effect.
    #
    # Drawn from a real ZipfQuerySampler, not queries[:75]: `queries` is
    # sorted by qid (an arbitrary order), and ZipfQuerySampler weights
    # strictly by list index over whatever list it's given — so a raw
    # queries[:75] slice is exactly the 75 highest-weighted draws under the
    # real run's own sampler too, since it's the same list in the same
    # order. That's ~52% of all traffic mass by Zipfian weight
    # (H_75/H_6980), which deterministically preloaded exactly the queries
    # most likely to be resampled and inflated the very number (cache hit
    # rate) Important #3 already required be measured honestly — a review
    # finding on the first version of this warm-up. Sampling instead makes
    # the warm-up statistically indistinguishable from "75 real requests
    # that happened to arrive early," which no longer privileges one
    # specific ordering (and resolves the mmap-vs-LRU-fill question above:
    # a uniform-content warm-up wouldn't touch this at all, but a resampled
    # one does, so LRU fill fraction — not mmap paging — was always the
    # more likely mechanism). seed + 1000, not seed, so the warm-up draws
    # are decorrelated from the timed run's own sampler rather than
    # reproducing its exact draw sequence.
    warmup_sampler = ZipfQuerySampler(queries, s=zipf_s, seed=seed + 1000)
    for _ in range(75):
        try:
            client.dispatch(warmup_sampler.sample(), k=hits)
        except Exception:
            pass

    def dispatch(query: str) -> None:
        nonlocal cache_hits, cache_total
        response = client.dispatch(query, k=hits)
        server_queue_wait.record(response.queue_wait_us)
        cache_total += 1
        if response.cache_hit:
            cache_hits += 1

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    # RESOURCE_EXHAUSTED under load is an *expected* outcome at high offered
    # QPS, not a broken run: run_open_loop's own dispatch wrapper already
    # catches any exception and records it as an error rather than raising,
    # so a high error count at high QPS is exactly the knee showing up.
    summary = result.summary()
    summary["server_queue_wait_us"] = server_queue_wait.summary()
    summary["cache_hit_rate"] = cache_hits / cache_total if cache_total else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--duration", type=float, default=10.0)
    # 32, not 8: at 8, run_open_loop's own client-side ThreadPoolExecutor
    # (at most 8 requests in flight) becomes the bottleneck well before the
    # server's queue_depth=64 could ever fill, so a sweep run at
    # client-workers=8 can't tell "server capacity-limited" from "client
    # dispatch-thread-limited" apart. 32 gives the client enough concurrency
    # to actually pressure the server's queue at the QPS range this sweep
    # tests; going much higher (e.g. 96+) risks a different artifact — CPU
    # contention between client dispatch threads and the server's own
    # worker threads, since both run on one 8-core machine here.
    parser.add_argument("--client-workers", type=int, default=32)
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

    points = []
    for qps in args.qps:
        with ServerProcess(
            INDEX_DIR, port=args.port, workers=args.server_workers,
            queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
            algorithm=args.algorithm,
        ) as server:
            client = SearchClient(server.address)
            repeats = [
                measure(client, qps, args.duration, args.client_workers, args.hits,
                        queries, args.zipf_s, seed)
                for seed in range(args.repeats)
            ]
            client.close()
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
