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
            lines.append(
                f"| {point['offered_qps']:.0f} | no samples recorded | - | - | - | - | - |"
            )
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
        sustained_reps = [
            r for r in sustained_row["repeats"] if r["latency"].get("count", 0) > 0
        ]
        sustained_p99 = statistics.median([r["latency"]["p99_us"] for r in sustained_reps])
        # This run's own hit rate, not the cache-sensitivity sweep's median:
        # that sweep runs at a different (fixed) QPS and its own zipf_s
        # sweep, so borrowing its number here would describe a different
        # run's cache behavior as if it were this one's.
        cache_note = ""
        hit_rates = [
            r["cache_hit_rate"] for r in sustained_reps if "cache_hit_rate" in r
        ]
        if hit_rates:
            cache_note = (
                f", {statistics.median(hit_rates):.0%} cache hit rate at this "
                "point (this sweep's own traffic, s="
                f"{knee_data['zipf_s']}, not borrowed from the cache-sensitivity sweep below)"
            )
        # "At least", not "sustains exactly": find_knee only proves the
        # server kept up *at* this offered rate, not that this is where it
        # would actually fall over — that would require testing the QPS
        # values in between this point and the one where the criterion
        # trips (see below), which this sweep's fixed grid doesn't do. Check
        # whether the server's own queue was even under pressure here: if
        # server_queue_wait_us stayed near zero relative to service time,
        # this point demonstrably wasn't near the server's own ceiling
        # either — it's a measured floor, not a located edge.
        sustained_qwait_reps = [
            r for r in sustained_reps if r.get("server_queue_wait_us", {}).get("count", 0) > 0
        ]
        floor_note = ""
        if sustained_qwait_reps:
            sq_p50 = statistics.median(
                [r["server_queue_wait_us"]["p50_us"] for r in sustained_qwait_reps]
            )
            ssvc_p50 = statistics.median([r["service"]["p50_us"] for r in sustained_qwait_reps])
            if sq_p50 < 0.5 * ssvc_p50:
                floor_note = (
                    " The server's own queue wait stayed near zero at this "
                    f"point ({sq_p50/1000:.2f}ms vs. {ssvc_p50/1000:.1f}ms service "
                    "time), so this is a measured floor on server capacity, not "
                    "a located ceiling — the sweep's fixed QPS grid didn't test "
                    "closely enough above it to say where the server's own "
                    "limit actually sits."
                )
            else:
                floor_note = (
                    f" The server's own queue wait ({sq_p50/1000:.2f}ms) was "
                    f"already a meaningful fraction of service time "
                    f"({ssvc_p50/1000:.1f}ms) at this point, consistent with "
                    "approaching the server's own capacity rather than just a "
                    "harness artifact."
                )
        lines += [
            "",
            f"**Capacity: sustains at least {sustained:.0f} QPS at p99 < "
            f"{sustained_p99/1000:.0f}ms** with {knee_data['server_workers']} "
            f"worker threads, queue depth {knee_data['queue_depth']}{cache_note}."
            f"{floor_note}",
            '"Sustains N QPS" means N is the highest offered rate tested '
            "*before* the point where `find_knee`'s criterion trips (achieved "
            "throughput falls below offered and wall time outlives the "
            "arrival window) — not the highest point under some fixed "
            "latency bound. A lower-QPS row can still show a higher p99 than "
            "the sustained row on ordinary run-to-run variance; that's not a "
            "contradiction, just noise at a point below the knee.",
        ]
        if knee is not None:
            knee_row = next(p for p in knee_data["points"] if p["offered_qps"] == knee)
            knee_reps = [
                r for r in knee_row["repeats"]
                if r["latency"].get("count", 0) > 0
                and r.get("server_queue_wait_us", {}).get("count", 0) > 0
            ]
            mechanism = (
                "achieved throughput falls below offered and the run's wall "
                "time outlives the arrival window"
            )
            if knee_reps:
                # Three numbers, three different possible culprits: server-side
                # queue wait (SearchServiceImpl's own BoundedQueue — the thing
                # this whole experiment is nominally about), client-side queue
                # wait (run_open_loop's queue_delay: time waiting for a free
                # client-side dispatch thread, nothing to do with the server),
                # and service time itself (the server genuinely getting slower
                # per query, e.g. CPU contention between client and server
                # threads on one machine). Attribute to whichever actually
                # dominates instead of assuming it's the server's queue.
                qwait_p50 = statistics.median(
                    [r["server_queue_wait_us"]["p50_us"] for r in knee_reps]
                )
                cli_qd_p50 = statistics.median([r["queue_delay"]["p50_us"] for r in knee_reps])
                svc_p50 = statistics.median([r["service"]["p50_us"] for r in knee_reps])
                # The 5x multiplier below is a rough dominance test (is the
                # client-side number *clearly* bigger, not just nominally
                # bigger), not a calibrated threshold — nothing here has ever
                # required distinguishing a close call.
                if qwait_p50 >= svc_p50:
                    mechanism += (
                        f" — the server's own `BoundedQueue` is where the time "
                        f"goes (median server-side queue wait {qwait_p50/1000:.1f}ms "
                        f"vs. {svc_p50/1000:.1f}ms actually processing the query), "
                        "so this is the bounded queue filling, not a client-side "
                        "artifact"
                    )
                elif cli_qd_p50 > svc_p50 and cli_qd_p50 > qwait_p50 * 5:
                    mechanism += (
                        f" — but almost all of that growth is on the client "
                        f"side: median time waiting for a free client dispatch "
                        f"thread ({cli_qd_p50/1000:.1f}ms, out of "
                        f"{knee_data['client_workers']} client workers) dwarfs "
                        f"both server-side queue wait ({qwait_p50/1000:.2f}ms) "
                        f"and service time ({svc_p50/1000:.1f}ms). This sweep's "
                        "load-generator harness is what saturates at this "
                        "point, not necessarily the server — its own true "
                        f"ceiling may sit above {knee:.0f} QPS; see Known "
                        "limitations"
                    )
                else:
                    mechanism += (
                        f" — median server-side queue wait "
                        f"({qwait_p50/1000:.1f}ms) stays well under service time "
                        f"({svc_p50/1000:.1f}ms) even here, so the server's fixed "
                        "worker pool is throughput-saturated rather than its "
                        "queue backing up; the growth in end-to-end latency at "
                        "this point is scheduling delay, not the `BoundedQueue` "
                        "filling"
                    )
            lines.append(f"At {knee:.0f} QPS the server stops keeping up: {mechanism}.")
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
        f"Fixed QPS ({cache_data['qps']}, well below the sustained-QPS region "
        "found above so this reflects cache behavior rather than queueing) "
        "across a sweep of the Zipfian skew. Cache-hit and cache-miss latency "
        "are reported separately — a blended number would hide which one "
        "actually matters.",
        "",
        "| Zipf s | hit rate | hits (n) | misses (n) | hit p50 | hit p99 | miss p50 | miss p99 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in cache_data["points"]:
        hit = point["hit_latency"]
        miss = point["miss_latency"]
        hit_p50 = f"{hit['p50_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        hit_p99 = f"{hit['p99_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        miss_p50 = f"{miss['p50_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        miss_p99 = f"{miss['p99_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        lines.append(
            f"| {point['zipf_s']:.1f} | {point['hit_rate']:.1%} | "
            f"{point['hit_count']} | {point['misses']} | {hit_p50} | "
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
        "  (workers + queue depth + a small fixed headroom, currently 8, so",
        "  that BoundedQueue's own \"queue full\" rejection is reachable via",
        "  real network traffic instead of always being preempted by gRPC's",
        "  coarser admission control — see main.cc's ResourceQuota comment)",
        "  so it can't grow unbounded, but the fixed worker pool and bounded",
        "  queue are the actual concurrency control — see the design spec for",
        "  why.",
        f"- The load generator's own client-side dispatch pool "
        f"({knee_data['client_workers']} threads, one blocking `dispatch()` "
        "call per thread) is itself a finite-capacity queueing system. It's",
        "  sized well above the server's configured concurrency so it doesn't",
        "  become the bottleneck across most of this sweep, but at the very",
        "  top of the QPS range tested here it can still saturate first —",
        "  `server_queue_wait_us` (server-side) staying low while",
        "  `queue_delay` (client-side) explodes is the signature, and the",
        "  throughput-knee section above calls this out explicitly when it's",
        "  what the data shows. Going higher than the current client-workers",
        "  value risks a different artifact: CPU contention with the",
        "  server's own worker threads, since client and server share one",
        "  machine here. A production load test would run the client on",
        "  separate hardware from the server.",
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
