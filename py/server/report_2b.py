"""Renders bench/phase2b.md from the tail-latency and hedging results --
sub-project B's two remaining Phase 2 experiments (sharded broker,
hedged requests). Kept separate from server.report (sub-project A's
renderer) rather than merged into it, matching this project's established
practice of not touching an already-shipped phase's report generator for a
different phase's data.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"


def render_tail_latency_plot(tail_data: dict, output_path: Path) -> None:
    ns = [p["n"] for p in tail_data["points"]]
    p50s = [
        statistics.median([r["latency"]["p50_us"] for r in p["repeats"]]) / 1000
        for p in tail_data["points"]
    ]
    p99s = [p["p99_us_across_repeats"]["median"] / 1000 for p in tail_data["points"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ns, p99s, marker="o", label="p99")
    ax.plot(ns, p50s, marker="o", label="p50")
    ax.set_xlabel("shard count (N)")
    ax.set_ylabel("client-observed latency (ms)")
    ax.set_xticks(ns)
    ax.set_title("Tail-latency amplification vs. shard count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_tail_latency_table(tail_data: dict) -> str:
    lines = [
        "| N | p50 | p95 | p99 | p99.9 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in tail_data["points"]:
        reps = point["repeats"]
        mid = sorted(reps, key=lambda r: r["latency"]["p99_us"])[len(reps) // 2]
        lat = mid["latency"]
        lines.append(
            f"| {point['n']} | {lat['p50_us']/1000:.2f}ms | {lat['p95_us']/1000:.2f}ms | "
            f"{lat['p99_us']/1000:.2f}ms | {lat['p999_us']/1000:.2f}ms |"
        )
    return "\n".join(lines)


def render_hedging_table(hedging_data: dict) -> str:
    baseline_p99 = hedging_data["baseline_p99_us_median"] / 1000
    hedged_p99 = hedging_data["hedged_p99_us_median"] / 1000
    delta_pct = (hedged_p99 - baseline_p99) / baseline_p99 * 100 if baseline_p99 else 0.0
    extra_load = hedging_data["extra_load_fraction"]
    return "\n".join([
        "| discipline | p99 | delta vs. baseline | extra load |",
        "|---|---:|---:|---:|",
        f"| baseline (no hedging) | {baseline_p99:.2f}ms | - | - |",
        f"| hedged | {hedged_p99:.2f}ms | {delta_pct:+.1f}% | {extra_load:.1%} |",
    ])


def render_markdown(tail_data: dict, hedging_data: dict) -> str:
    warmup_ms = [round(p / 1000, 2) for p in hedging_data["shard_p95_us_warmup"]]
    lines = [
        "# Phase 2, sub-project B — sharded broker, tail latency, hedging",
        "",
        "Exit artifact for Phase 2 sub-project B (see "
        "`docs/superpowers/specs/2026-09-10-phase2b-broker-design.md`). "
        "Combined with `bench/phase2a.md`, this completes Phase 2's four-plot "
        "exit bar.",
        "",
        "## Docid-semantics limitation",
        "",
        "Each shard's internal docid is local to that shard and BM25 IDF is "
        "computed from that shard's own document frequencies, not the "
        "corpus-global df. The broker's merged top-k is a real computation "
        "over real shard responses, but this document makes no NDCG/quality "
        "claim for the sharded configuration — see the design spec §3.",
        "",
        "## Tail-latency amplification vs. shard count",
        "",
        "![Tail latency vs N](plots/phase2b-tail-latency.png)",
        "",
        f"Open-loop, Poisson arrivals, Zipfian query popularity (s="
        f"{tail_data['zipf_s']}) over {tail_data['num_distinct_queries']:,} dev "
        f"queries, fixed QPS={tail_data['qps']}, "
        f"{tail_data['client_workers']} client dispatch threads. Each row is "
        f"the median of {tail_data['repeats']} runs of "
        f"{tail_data['duration_s']:.0f}s. N=1 is measured through the same "
        f"`Broker` path as every other row (a single-shard broker, not "
        "sub-project A's direct-client number), so every point pays the "
        "same broker overhead.",
        "",
        render_tail_latency_table(tail_data),
        "",
        "## Hedged requests",
        "",
        f"At N={hedging_data['n']} shards, fixed QPS={hedging_data['qps']}. "
        f"Hedge delay ({hedging_data['hedge_delay_us']/1000:.2f}ms) is the max "
        f"across shards' own measured p95 service latency from an un-hedged "
        f"warm-up run ({warmup_ms} ms per shard).",
        "",
        render_hedging_table(hedging_data),
        "",
        "## Configuration",
        "",
        f"{tail_data['server_workers']} worker threads per shard, queue depth "
        f"{tail_data['queue_depth']}, {tail_data['cache_capacity']}-entry LRU "
        f"cache per shard, `{tail_data['algorithm']}`, top-{tail_data['hits']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    tail_path = RESULTS_DIR / "server-tail-latency.json"
    hedging_path = RESULTS_DIR / "server-hedging.json"
    if not tail_path.exists():
        raise SystemExit(f"no {tail_path}; run `python -m server.tail_latency` first")
    if not hedging_path.exists():
        raise SystemExit(f"no {hedging_path}; run `python -m server.hedging` first")

    tail_data = json.loads(tail_path.read_text())
    hedging_data = json.loads(hedging_path.read_text())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    render_tail_latency_plot(tail_data, PLOTS_DIR / "phase2b-tail-latency.png")

    (REPO_ROOT / "bench" / "phase2b.md").write_text(render_markdown(tail_data, hedging_data))
    print("wrote bench/phase2b.md and bench/plots/phase2b-tail-latency.png")


if __name__ == "__main__":
    main()
