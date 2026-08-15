"""Renders bench/phase2a.md from the throughput-knee and cache-sensitivity
results — the two of the plan's four Phase 2 experiments a single server
instance can produce (sharding and hedging are sub-project B).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def find_knee(points: list[dict], duration_s: float) -> tuple[float | None, float | None]:
    """(knee_qps, sustained_qps): the offered rate where achieved throughput
    falls short *and* the run's wall time outlives the arrival window —
    Phase 0's validated criterion (bench/baselines.md), duplicated rather
    than imported from baselines/report.py to avoid touching that
    already-shipped Phase 0 report generator for an unrelated phase.
    """
    knee = None
    sustained = None
    for point in points:
        reps = [r for r in point["repeats"] if r["latency"].get("count", 0) > 0]
        if not reps:
            continue
        achieved = statistics.median([r["achieved_qps"] for r in reps])
        wall = statistics.median([r["wall_s"] for r in reps])
        if achieved < 0.95 * point["offered_qps"] and wall > 1.1 * duration_s:
            knee = point["offered_qps"]
            break
        sustained = point["offered_qps"]
    return knee, sustained


def render_markdown(knee_data: dict, cache_data: dict) -> str:
    lines = [
        "# Phase 2, sub-project A — single-node gRPC server",
        "",
        "Exit artifact for Phase 2 sub-project A (see "
        "`docs/superpowers/specs/2026-08-15-phase2-server-design.md`). "
        "`bench/results/server-throughput-knee.json` and "
        "`bench/results/server-cache-sensitivity.json` are committed; the "
        "built index is not (see README's Phase 1 setup). Sharding, the "
        "broker, and hedged requests are sub-project B.",
        "",
        "## Server configuration",
        "",
        f"{knee_data['server_workers']} worker threads, queue depth "
        f"{knee_data['queue_depth']}, {knee_data['cache_capacity']}-entry LRU "
        f"cache, `{knee_data['algorithm']}` (Phase 1 measured WAND faster than "
        "BlockMax-WAND in wall-clock on this corpus at k=10 despite scoring "
        "more postings — see `bench/phase1.md` — so it's the server default), "
        f"top-{knee_data['hits']}.",
        "",
        "## Throughput knee",
        "",
        f"Open-loop, Poisson arrivals, Zipfian query popularity (s="
        f"{knee_data['zipf_s']}) over {knee_data['num_distinct_queries']:,} dev "
        f"queries, {knee_data['client_workers']} client dispatch threads. Each "
        f"point is the median of {knee_data['repeats']} runs of "
        f"{knee_data['duration_s']:.0f}s.",
        "",
        "| Offered QPS | Achieved QPS | p50 | p95 | p99 | p99.9 | errors (median) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in knee_data["points"]:
        reps = [r for r in point["repeats"] if r["latency"].get("count", 0) > 0]
        if not reps:
            lines.append(f"| {point['offered_qps']:.0f} | no samples recorded |")
            continue
        mid = sorted(reps, key=lambda r: r["latency"]["p99_us"])[len(reps) // 2]
        lat = mid["latency"]
        errors = statistics.median([r["errors"] for r in reps])
        lines.append(
            f"| {point['offered_qps']:.0f} | {mid['achieved_qps']:.1f} | "
            f"{lat['p50_us']/1000:.2f}ms | {lat['p95_us']/1000:.2f}ms | "
            f"{lat['p99_us']/1000:.2f}ms | {lat['p999_us']/1000:.2f}ms | {errors:.0f} |"
        )

    knee, sustained = find_knee(knee_data["points"], knee_data["duration_s"])
    if sustained is not None:
        sustained_row = next(p for p in knee_data["points"] if p["offered_qps"] == sustained)
        sustained_p99 = statistics.median(
            [r["latency"]["p99_us"] for r in sustained_row["repeats"]]
        )
        cache_note = ""
        if cache_data["points"]:
            rates = [p["hit_rate"] for p in cache_data["points"]]
            cache_note = (
                f", {statistics.median(rates):.0%} median cache hit rate "
                "across the cache-sensitivity sweep"
            )
        lines += [
            "",
            f"**Capacity: sustains {sustained:.0f} QPS at p99 < "
            f"{sustained_p99/1000:.0f}ms** with {knee_data['server_workers']} "
            f"worker threads, queue depth {knee_data['queue_depth']}{cache_note}.",
        ]
        if knee is not None:
            lines.append(
                f"At {knee:.0f} QPS the server stops keeping up: achieved throughput "
                "falls below offered and the queue outlives the arrival window."
            )
    else:
        lines += [
            "",
            "No offered QPS in this sweep stayed under the knee — widen `--qps` "
            "with lower values and re-run.",
        ]

    lines += [
        "",
        "## Cache sensitivity",
        "",
        f"Fixed QPS ({cache_data['qps']}, chosen below the throughput knee above "
        "so this reflects cache behavior rather than queueing) across a sweep "
        "of the Zipfian skew. Cache-hit and cache-miss latency are reported "
        "separately — a blended number would hide which one actually matters.",
        "",
        "| Zipf s | hit rate | hit p50 | hit p99 | miss p50 | miss p99 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in cache_data["points"]:
        hit = point["hit_latency"]
        miss = point["miss_latency"]
        hit_p50 = f"{hit['p50_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        hit_p99 = f"{hit['p99_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        miss_p50 = f"{miss['p50_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        miss_p99 = f"{miss['p99_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        lines.append(
            f"| {point['zipf_s']:.1f} | {point['hit_rate']:.1%} | {hit_p50} | "
            f"{hit_p99} | {miss_p50} | {miss_p99} |"
        )

    lines += [
        "",
        "## Known limitations",
        "",
        "- Single server process, single machine. Sharding and its own",
        "  tail-latency amplification are sub-project B's subject, not this",
        "  document's.",
        "- The gRPC layer's own thread pool is bounded via `ResourceQuota`",
        "  (workers + queue depth) so it can't grow unbounded, but the fixed",
        "  worker pool and bounded queue are the actual concurrency control —",
        "  see the design spec for why.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    knee_path = RESULTS_DIR / "server-throughput-knee.json"
    cache_path = RESULTS_DIR / "server-cache-sensitivity.json"
    if not knee_path.exists():
        raise SystemExit(f"no {knee_path}; run `python -m server.throughput_knee` first")
    if not cache_path.exists():
        raise SystemExit(f"no {cache_path}; run `python -m server.cache_sensitivity` first")

    knee_data = json.loads(knee_path.read_text())
    cache_data = json.loads(cache_path.read_text())

    (REPO_ROOT / "bench" / "phase2a.md").write_text(render_markdown(knee_data, cache_data))
    print("wrote bench/phase2a.md")


if __name__ == "__main__":
    main()
