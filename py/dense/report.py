"""Renders bench/phase3.md: the Pareto plot (recall@100 vs p99 latency,
memory as point size) and the 4-row hybrid-fusion table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dense.subset import SEED as SUBSET_SEED

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"
BENCH_DIR = REPO_ROOT / "bench"


def render_pareto_plot(sweep: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for structure, color, label in (("hnsw", "tab:blue", "HNSW"), ("ivfpq", "tab:orange", "IVF-PQ")):
        points = [p for p in sweep["points"] if p["structure"] == structure]
        recalls = [p["recall_at_100"] for p in points]
        p99_ms = [p["latency_us"]["p99_us"] / 1000 for p in points]
        sizes = [max(20.0, p["memory_gb"] * 400) for p in points]
        ax.scatter(recalls, p99_ms, s=sizes, alpha=0.6, color=color, label=label)
    ax.set_xlabel("recall@100 (vs. exact brute-force)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title(f"HNSW vs IVF-PQ: recall/latency/memory ({sweep['subset_size']:,}-passage subset)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_sweep_table(sweep: dict) -> str:
    lines = [
        "| structure | config | recall@100 | p50 (ms) | p99 (ms) | memory (GB) |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(sweep["points"], key=lambda p: (-p["recall_at_100"])):
        if p["structure"] == "hnsw":
            config = f"M={p['M']} efSearch={p['ef_search']}"
        else:
            config = f"nlist={p['nlist']} m={p['m']} nprobe={p['nprobe']}"
        lines.append(
            f"| {p['structure']} | {config} | {p['recall_at_100']:.4f} | "
            f"{p['latency_us']['p50_us']/1000:.2f} | {p['latency_us']['p99_us']/1000:.2f} | "
            f"{p['memory_gb']:.3f} |"
        )
    return "\n".join(lines)


def render_fusion_table(fusion: dict) -> str:
    lines = [
        "| channel | NDCG@10 | recall@1000 |",
        "|---|---|---|",
    ]
    order = ["lexical_only", "dense_only", "rrf", "score_fusion"]
    for name in order:
        row = fusion["results"][name]
        lines.append(f"| {name} | {row['ndcg_10']:.4f} | {row['recall_1000']:.4f} |")
    return "\n".join(lines)


def render_markdown(sweep: dict, fusion: dict) -> str:
    provenance = sweep["provenance"]
    return f"""# Phase 3: Dense Recall and the ANN Pareto Frontier

## Corpus subset (limitation, stated up front)

This machine has 8GB RAM and no GPU. The full 8.8M-passage corpus would need
~13.5GB just for float32 embeddings before any index overhead, so this phase
subsets to **{sweep['subset_size']:,} passages**: every document judged in the
dl19+dl20 qrels, plus a seeded random fill. The Pareto-frontier finding (which
ANN structure wins at which recall target) does not depend on corpus size;
only the absolute latency/memory numbers would shift at 8.8M.

Encoder: `BAAI/bge-small-en-v1.5`. Evaluated over {sweep['num_dev_queries']}
dev queries (recall@100 / latency) and dl19+dl20 (fusion table).

## Pareto frontier: recall@100 vs. p99 latency

![Pareto plot](plots/phase3-pareto.png)

{render_sweep_table(sweep)}

## Hybrid fusion (dl19+dl20)

{render_fusion_table(fusion)}

## Configuration

- subset seed: {SUBSET_SEED}
- git SHA: `{provenance['git_sha']}`
- hardware: {provenance['hardware']['cpu']}, {int(int(provenance['hardware']['memory_bytes']) / 1e9)}GB RAM
- timestamp: {provenance['timestamp_utc']}
"""


def main() -> None:
    sweep = json.loads((RESULTS_DIR / "dense-ann-sweep.json").read_text())
    fusion = json.loads((RESULTS_DIR / "dense-fusion.json").read_text())

    render_pareto_plot(sweep, PLOTS_DIR / "phase3-pareto.png")
    markdown = render_markdown(sweep, fusion)
    (BENCH_DIR / "phase3.md").write_text(markdown)
    print(f"wrote {(BENCH_DIR / 'phase3.md').relative_to(REPO_ROOT)}")
    print(f"wrote {(PLOTS_DIR / 'phase3-pareto.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
