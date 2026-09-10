"""Renders bench/phase6.md from nrt-lag.json (6a + 6c) and
nrt-segment-sweep.json (6b). Every analysis sentence below is computed
from the committed result JSON -- nothing here is a fixed string
describing a specific run's numbers, matching py/server/report_2b.py's
established practice.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"
LAG_PATH = RESULTS_DIR / "nrt-lag.json"
SWEEP_PATH = RESULTS_DIR / "nrt-segment-sweep.json"
REPORT_PATH = REPO_ROOT / "bench" / "phase6.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_freshness(points: list[dict]) -> Path:
    thresholds = [p["flush_threshold_docs"] for p in points]
    lag_ms = [p["lag_p99_us"] / 1000 for p in points]
    query_ms = [p["query_p99_us"] / 1000 for p in points]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(lag_ms, query_ms, marker="o")
    for t, x, y in zip(thresholds, lag_ms, query_ms):
        ax.annotate(f"t={t}", (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("index-to-searchable lag, p99 (ms)")
    ax.set_ylabel("query latency, p99 (ms)")
    ax.set_title("Phase 6: freshness/latency tradeoff")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phase6-freshness.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_segment_count(points: list[dict]) -> Path:
    ns = [p["n"] for p in points]
    p50_ms = [p["p50_us"] / 1000 for p in points]
    p99_ms = [p["p99_us"] / 1000 for p in points]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ns, p50_ms, marker="o", label="p50")
    ax.plot(ns, p99_ms, marker="o", label="p99")
    ax.set_xlabel("live segment count (N), fixed 200,000-doc corpus")
    ax.set_ylabel("query latency (ms)")
    ax.set_title("Phase 6: query latency vs. segment count")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "phase6-segment-count.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _freshness_table(points: list[dict]) -> str:
    lines = ["| flush threshold (docs) | lag p99 (ms) | query p99 (ms) | final segments | collisions skipped |",
             "|---|---|---|---|---|"]
    for p in points:
        lines.append(
            f"| {p['flush_threshold_docs']} | {p['lag_p99_us']/1000:.2f} | "
            f"{p['query_p99_us']/1000:.2f} | {p['final_segment_count']} | {p['collisions_skipped']} |"
        )
    return "\n".join(lines)


def _sweep_table(points: list[dict]) -> str:
    lines = ["| N | p50 (ms) | p99 (ms) |", "|---|---|---|"]
    for p in points:
        lines.append(f"| {p['n']} | {p['p50_us']/1000:.2f} | {p['p99_us']/1000:.2f} |")
    return "\n".join(lines)


def _merge_evidence(points: list[dict]) -> str:
    for p in points:
        if p.get("merge_before_after"):
            before = p["merge_before_after"]["before"]
            after = p["merge_before_after"]["after"]
            return (
                f"At `flush_threshold_docs={p['flush_threshold_docs']}`, a merge fired "
                f"during the run: segment count went from {before} to {after}, "
                f"consistent with the configured merge policy actually bounding growth."
            )
    return "No merge event was captured in this run's trace (re-run with a smaller threshold if this section is empty)."


def render() -> None:
    lag = _load(LAG_PATH)
    sweep = _load(SWEEP_PATH)

    freshness_plot = _plot_freshness(lag["points"])
    segment_plot = _plot_segment_count(sweep["points"])

    report = f"""# Phase 6: Near-Real-Time Indexing

## Freshness/latency tradeoff

{_freshness_table(lag["points"])}

![freshness/latency tradeoff]({freshness_plot.relative_to(REPORT_PATH.parent)})

Base corpus: {lag['base_doc_count']:,} docs (already indexed). Write stream: \
{lag['stream_doc_count']:,} docs, added one at a time. `merge_factor={lag['merge_factor']}`, \
`max_segments={lag['max_segments']}`.

## Segment-count effect (fixed corpus size)

{_sweep_table(sweep["points"])}

![query latency vs. segment count]({segment_plot.relative_to(REPORT_PATH.parent)})

Fixed corpus: {sweep['fixed_corpus_docs']:,} docs, partitioned into N segments for each point \
-- the same corpus-shrinkage-per-segment effect `bench/phase2b.md` disclosed for its shards is \
still present here. What isolates the fan-out/merge cost specifically is that all N segments \
live in one process with no network hop and no per-shard worker pool, so the only things that \
scale with N are Python-level fan-out overhead and the merge/sort/tombstone-filter step, not \
shard process scheduling or broker-pool sizing. This is a different, valid question from \
Phase 2B's "does more shards help tail latency," not a claim to have fixed Phase 2B's confound.

## Merge-in-action evidence

{_merge_evidence(lag["points"])}

## Known limitations

- **Merge is re-tokenization, not a postings-level merge.** Every merge rebuilds from source \
TSV text via the unmodified `build_index` binary; it is not a sorted-run merge over already-built \
postings (which would require new C++ in `cpp/index/builder.cc`, out of scope here). Merge cost \
scales with merged segment size the same way a fresh build does.
- **In-process latency is not served latency.** There is no gRPC, no worker pool, no request \
queueing in this layer -- these numbers are not comparable to `bench/phase2a.md`'s or \
`bench/phase2b.md`'s served p99s.
- **No NDCG/quality claim for the multi-segment configuration.** Per-segment BM25 statistics \
mean the multi-segment configuration's relevance is not evaluated here, the same framing \
`bench/phase2b.md` established for shards.
- **Flush is synchronous and blocks new writes for its duration.** This design does not overlap \
flush with the next write batch, unlike Lucene's concurrent in-memory segment writer.
- **The tombstone over-fetch bound (`k + len(tombstones)`) is a fixed heuristic**, not a tuned \
bound against how many tombstones plausibly cluster near the top-k of a single segment.
"""
    REPORT_PATH.write_text(report)
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {freshness_plot.relative_to(REPO_ROOT)}")
    print(f"wrote {segment_plot.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    render()
