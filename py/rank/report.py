"""Renders bench/phase4.md: the batch-window tradeoff plot, the
prerank-consistency number, the precision table, and the queue-discipline
comparison.
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
BENCH_DIR = REPO_ROOT / "bench"

# Phase 1's own first-stage baseline over the same 97 dl19+dl20 queries this
# phase's prerank-consistency section evaluates against -- pooled from
# bench/phase1.md's per-set NDCG@10 (dl19 0.5052 / 43 queries, dl20 0.4785 /
# 54 queries): (43*0.5052 + 54*0.4785) / 97. Phase 1 is historical and fixed,
# not re-derived from its rerun, so this is a constant rather than something
# read from bench/phase1.md at render time.
PHASE1_BASELINE_NDCG_10 = 0.4903


def render_batching_plot(batching: dict, output_path: Path) -> None:
    points = batching["points"]
    fig, ax = plt.subplots(figsize=(8, 6))
    throughput = [p["throughput_qps"] for p in points]
    p99_ms = [p["latency_us"]["p99_us"] / 1000 for p in points]
    labels = [f"b={p['max_batch_size']},w={p['max_wait_ms']}ms" for p in points]
    ax.scatter(throughput, p99_ms, s=80, alpha=0.7, color="tab:blue")
    for x, y, label in zip(throughput, p99_ms, labels):
        ax.annotate(label, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("throughput (qps)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Dynamic batching: throughput vs. p99 latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_batching_table(batching: dict) -> str:
    lines = [
        "| max_batch_size | max_wait_ms | throughput (qps) | p50 (ms) | p99 (ms) |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(batching["points"], key=lambda p: -p["throughput_qps"]):
        lines.append(
            f"| {p['max_batch_size']} | {p['max_wait_ms']} | {p['throughput_qps']:.1f} | "
            f"{p['latency_us']['p50_us']/1000:.2f} | {p['latency_us']['p99_us']/1000:.2f} |"
        )
    return "\n".join(lines)


def render_precision_table(precision: dict) -> str:
    # p999 and max, not just p99: the real fp16 run's p999/max were ~10x its
    # own p99 (244ms/252ms against a ~25ms p99), a tail entirely invisible in
    # a p99-only table. .get(..., 0) rather than direct indexing so a
    # rank-precision.json written before these keys existed still renders.
    lines = [
        "| precision | throughput (qps) | p99 (ms) | p999 (ms) | max (ms) | cascade NDCG@10 | delta vs. fp32 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in precision["results"].items():
        lines.append(
            f"| {name} | {row['throughput_qps']:.1f} | {row['latency_us']['p99_us']/1000:.2f} | "
            f"{row['latency_us'].get('p999_us', 0)/1000:.2f} | {row['latency_us'].get('max_us', 0)/1000:.2f} | "
            f"{row['cascade_ndcg_10']:.4f} | {row['ndcg_10_delta_vs_fp32']:+.4f} |"
        )
    return "\n".join(lines)


def render_queue_table(queue: dict) -> str:
    # achieved_qps next to the arrival rate is the whole point of an open-loop
    # generator: the gap between offered and achieved is the backlog, and
    # queue_delay p99 says how much of the client-visible latency it cost.
    lines = [
        "| discipline | arrival rate (qps) | achieved (qps) | p99 (ms) | queue delay p99 (ms) | shed count |",
        "|---|---|---|---|---|---|",
    ]
    for row in queue["runs"]:
        lines.append(
            f"| {row['discipline']} | {row['arrival_rate_qps']:.1f} | "
            f"{row.get('achieved_qps', 0):.1f} | "
            f"{row['latency_us'].get('p99_us', 0)/1000:.2f} | "
            f"{row.get('queue_delay_us', {}).get('p99_us', 0)/1000:.2f} | {row['shed_count']} |"
        )
    return "\n".join(lines)


def render_provider_warning(prerank: dict, batching: dict, precision: dict, queue: dict) -> str:
    """Every rank-*.json result now records active_providers -- the ONNX
    Runtime execution providers that actually initialized, which can
    silently differ from what the driver requested (a CUDA/cuDNN version
    mismatch logs a warning, not an error, and ONNX Runtime falls back to
    CPU). Surfaced prominently, near the top of the report, rather than left
    to a reader who happens to open the raw JSON -- a CPU-fallback run's
    numbers are real, but they measure the wrong hardware for this phase's
    stated question.
    """
    checks = {
        "prerank/consistency": prerank.get("active_providers", []),
        "batching sweep": batching.get("active_providers", []),
        **{
            f"precision ({name})": row.get("active_providers", [])
            for name, row in precision.get("results", {}).items()
        },
        "queue discipline": queue.get("active_providers", []),
    }
    # No `providers and ...` guard: a missing/empty active_providers field is
    # not evidence the run used CUDA, it is evidence the run didn't record
    # what it used -- which is exactly the silent-fallback case this check
    # exists to catch, so it must fire rather than pass.
    fell_back = {
        name: providers
        for name, providers in checks.items()
        if "CUDAExecutionProvider" not in providers
    }
    if not fell_back:
        return "All sub-experiments ran on `CUDAExecutionProvider` as requested."
    lines = [
        "**⚠️ CUDA execution provider unavailable for at least one "
        "sub-experiment -- the numbers below reflect CPU, not GPU, for:**",
        "",
    ]
    for name, providers in fell_back.items():
        lines.append(f"- {name}: ran on `{', '.join(providers)}`")
    return "\n".join(lines)


def render_markdown(prerank: dict, batching: dict, precision: dict, queue: dict) -> str:
    provenance = prerank["provenance"]
    return f"""# Phase 4: Ranking Cascade and Heterogeneous Serving

## Execution provider check

{render_provider_warning(prerank, batching, precision, queue)}

## Candidate pool (limitation, stated up front)

The pre-rank/rank cascade runs over Phase 1's existing lexical WAND
candidates (depth 1000, full 8.8M-passage corpus) — not fused with Phase 3's
dense recall channel. Dense score is used only as a pre-rank *feature*, not
as a separate recall channel.

## Prerank-consistency

Of the true top-{prerank['top_k_survivors']} under the full cross-encoder
ranker, **{prerank['mean_prerank_consistency']:.1%}** survive pre-ranking
(mean over {prerank['num_queries']} dl19+dl20 queries). Full-cascade result:
NDCG@10 = {prerank['cascade_ndcg_10']:.4f}, recall@100 = {prerank['cascade_recall_100']:.4f}.
Phase 1's own first-stage (lexical-only) baseline over the same 97
queries: NDCG@10 = {PHASE1_BASELINE_NDCG_10:.4f}.

## Dynamic batching: throughput vs. p99 latency

![Batching plot](plots/phase4-batching.png)

{render_batching_table(batching)}

## Precision comparison (at the best-throughput batching setting)

{render_precision_table(precision)}

## Queue discipline: load-shedding vs. unbounded

{render_queue_table(queue)}

## Configuration

- model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- git SHA: `{provenance['git_sha']}`
- GPU: {provenance.get('kaggle_hardware', {}).get('gpu_name', 'unknown')}
- timestamp: {provenance['timestamp_utc']}
"""


def main() -> None:
    prerank = json.loads((RESULTS_DIR / "rank-prerank.json").read_text())
    batching = json.loads((RESULTS_DIR / "rank-batching.json").read_text())
    precision = json.loads((RESULTS_DIR / "rank-precision.json").read_text())
    queue = json.loads((RESULTS_DIR / "rank-queue.json").read_text())

    render_batching_plot(batching, PLOTS_DIR / "phase4-batching.png")
    markdown = render_markdown(prerank, batching, precision, queue)
    (BENCH_DIR / "phase4.md").write_text(markdown)
    print(f"wrote {(BENCH_DIR / 'phase4.md').relative_to(REPO_ROOT)}")
    print(f"wrote {(PLOTS_DIR / 'phase4-batching.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
