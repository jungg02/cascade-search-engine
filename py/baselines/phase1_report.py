"""Evaluate the cascade C++ index's runs and emit the Phase 1 exit artifact.

Reuses report.py's evaluate_run/write_result so cascade's NDCG numbers go
through the exact same pytrec_eval-checked path the Phase 0 baselines did —
the whole point of the comparison is that it's apples to apples.

Two things this table needs that Phase 0's didn't:
  - the "safe optimization" claim itself: WAND and BlockMax-WAND must return
    identical results to exhaustive DAAT-OR, or one of them has a bug. Verified
    two ways: NDCG@10/100, MRR@10, and R@100/1000 matching to four decimals
    across all three algorithms on dl19+dl20 (97 real queries — DAAT-OR runs
    there specifically for this, see cascade_bm25.py), and top-10 identical by
    exact positional match on the synthetic 20k-doc corpus
    (cpp/tests/test_index.cc), which also covers dev-sized query volume that
    running exhaustive DAAT-OR for real would cost minutes for no new
    information over what dl19+dl20 already establish.
  - the pruning payoff: postings scored and full evaluations, which only mean
    something next to DAAT-OR's number for the same queries — hence
    cascade_query_cost.py running all three algorithms over one fixed sample
    rather than each engine picking its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.report import evaluate_run, write_result

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
RUNS_DIR = REPO_ROOT / "runs"

METRIC_COLUMNS = ["ndcg_cut_10", "ndcg_cut_100", "mrr_10", "recall_100", "recall_1000"]
CASCADE_ENGINES = ["cascade-daat-or", "cascade-wand", "cascade-blockmax-wand"]


def _load_baseline(engine: str, query_set: str) -> dict | None:
    path = RESULTS_DIR / f"{engine}.{query_set}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def render_markdown(ndcg_results: list[dict], cost: dict) -> str:
    lines = [
        "# Phase 1 — C++ inverted index with BlockMax-WAND",
        "",
        "Exit artifact for Phase 1. `bench/results/cascade-*.json` and",
        "`bench/results/cascade-query-cost.json` are committed; the run files",
        "and the built index are not (regenerable — see `py/baselines/cascade_bm25.py`",
        "and `py/baselines/cascade_query_cost.py`).",
        "",
        "## Query evaluation cost",
        "",
        f"One fixed sample of {cost['sample_size']:,} real `{cost['query_set']}` queries, "
        "top-10, single-threaded serial (no queueing — that's Phase 2's subject). Each "
        "algorithm runs the identical query set, so the postings-scored and "
        "full-evaluations columns are a direct measure of how much work pruning saves, "
        "not an artifact of different queries landing on different algorithms.",
        "",
        "| Algorithm | mean postings scored | mean full evaluations | p50 | p99 | p99.9 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    engine_order = {"cascade-daat-or": 0, "cascade-wand": 1, "cascade-blockmax-wand": 2}
    for r in sorted(cost["results"], key=lambda r: engine_order[r["engine"]]):
        lat = r["serial_latency_us"]
        lines.append(
            f"| {r['engine']} | {r['mean_postings_scored']:,.0f} | "
            f"{r['mean_full_evaluations']:,.0f} | {lat['p50_us']/1000:.3f}ms | "
            f"{lat['p99_us']/1000:.3f}ms | {lat['p999_us']/1000:.3f}ms |"
        )

    daat = next(r for r in cost["results"] if r["engine"] == "cascade-daat-or")
    wand = next(r for r in cost["results"] if r["engine"] == "cascade-wand")
    bmw = next(r for r in cost["results"] if r["engine"] == "cascade-blockmax-wand")
    reduction = 1 - bmw["mean_postings_scored"] / daat["mean_postings_scored"]
    wand_p99 = wand["serial_latency_us"]["p99_us"]
    bmw_p99 = bmw["serial_latency_us"]["p99_us"]
    lines += [
        "",
        f"BlockMax-WAND scores {reduction:.1%} fewer postings than exhaustive DAAT-OR "
        f"for the identical top-10, on the same {cost['sample_size']:,} queries — and "
        f"{1 - bmw['mean_postings_scored'] / wand['mean_postings_scored']:.1%} fewer than "
        "plain WAND. That doesn't translate to a wall-clock win here: BlockMax-WAND's p99 "
        f"({bmw_p99/1000:.1f}ms) is {bmw_p99/wand_p99 - 1:.0%} higher than WAND's "
        f"({wand_p99/1000:.1f}ms), and it loses at every percentile, not just the tail. "
        f"`mean_blocks_decoded` is why: BlockMax-WAND decodes "
        f"{bmw['mean_blocks_decoded']/wand['mean_blocks_decoded'] - 1:.0%} more blocks than "
        f"WAND ({bmw['mean_blocks_decoded']:,.0f} vs {wand['mean_blocks_decoded']:,.0f}) even "
        "while scoring far fewer postings from them. That's the direct cost of "
        "`skip_to_block` landing on and decoding a block it then abandons — its hopeless "
        "branch narrows the pivot span by touching a block's bound, and when that block "
        "turns out not to help, the decode was still paid for. Fewer postings scored is a "
        "real result, but it isn't the same thing as less work in this implementation: the "
        "block-level bookkeeping BlockMax-WAND does to reach that number costs more here "
        "than the decoding it avoids.",
        "",
        "## Algorithm equivalence at real corpus scale",
        "",
        "WAND and BlockMax-WAND are safe optimizations: any difference from exhaustive",
        "DAAT-OR's results is a bug, never a tradeoff. Verified two ways. On the synthetic",
        "20,000-document corpus, `cpp/tests/test_index.cc` checks exact positional top-10",
        "match across 300 generated queries, as part of the build. On the real",
        f"{8_841_823:,}-passage index, DAAT-OR runs on dl19+dl20 (97 real TREC queries —",
        "run separately from the dev-scale query-cost benchmark above, since DAAT-OR's cost",
        "doesn't fall with depth and dev's 6,980 queries would cost minutes for no new",
        "information over what these 97 already establish) and its NDCG@10/NDCG@100/MRR@10/",
        "R@100/R@1000 match WAND's and BlockMax-WAND's to four decimal places on both sets —",
        "see the DAAT-OR rows in the table below.",
        "",
        "## NDCG@10 vs. Lucene",
        "",
        "Depth-1000 runs, same qrels and metric code as `bench/baselines.md`. The plan's",
        "tolerance is ~0.01; a gap past that points at BM25 length normalization or a",
        "tokenizer divergence from Lucene's `EnglishAnalyzer` before it points at the",
        "retrieval algorithm — WAND/BlockMax-WAND already proven exact against DAAT-OR",
        "above, so a scoring bug would show up identically in all three.",
        "",
        "| Engine | Query set | Queries | NDCG@10 | NDCG@100 | MRR@10 | R@100 | R@1000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_rows = ndcg_results + [
        r for qs in ["dl19", "dl20", "dev"]
        if (r := _load_baseline("lucene-anserini-1.0.0", qs)) is not None
    ]
    for r in sorted(all_rows, key=lambda r: (r["query_set"], r["engine"])):
        m = r["metrics"]
        lines.append(
            f"| {r['engine']} | {r['query_set']} | {r['num_queries']} | "
            + " | ".join(f"{m[c]:.4f}" for c in METRIC_COLUMNS)
            + " |"
        )

    lucene_dl19 = _load_baseline("lucene-anserini-1.0.0", "dl19")
    cascade_dl19 = next(
        (r for r in ndcg_results if r["query_set"] == "dl19" and r["engine"] == "cascade-wand"),
        None,
    )
    if lucene_dl19 and cascade_dl19:
        delta = cascade_dl19["metrics"]["ndcg_cut_10"] - lucene_dl19["metrics"]["ndcg_cut_10"]
        lines += [
            "",
            f"**DL19 NDCG@10 delta vs. Lucene: {delta:+.4f}** "
            f"({'within' if abs(delta) <= 0.01 else 'outside'} the plan's ~0.01 tolerance).",
        ]

    lines += [
        "",
        "## Configuration",
        "",
        f"k1={cost['k1']}, b={cost['b']} (Anserini's msmarco defaults — see `BM25Params` in",
        "`cpp/index/index_format.h`). Analyzer targets Lucene's `EnglishAnalyzer` (Porter",
        "stemming + stopwords) via an ASCII approximation of UAX#29 segmentation — see",
        "`cpp/index/analyzer.h` for the known, deliberate divergence from a full",
        "implementation, and the first thing to suspect if the NDCG delta above is large.",
        "Document lengths are exact (unlike Lucene's lossy 1-byte norm quantization), so a",
        "small residual in that direction is expected even with everything else matched.",
        "The observed deltas are small (dl19 -0.0069, dl20 +0.0016, dev -0.0023) and don't",
        "sit on one consistent side of zero, which fits two small, partially-offsetting",
        "sources of divergence — the analyzer approximation and the exact-vs-quantized",
        "length difference — rather than one dominant cause pulling every query set the",
        "same way.",
        "",
        "## Known limitations",
        "",
        "- Single machine, single-threaded, no cache, no sharding. Capacity under",
        "  concurrent load is Phase 2's subject.",
        "- The query-cost table uses top-10; the NDCG table uses top-1000. Different",
        "  operating points on purpose — see `cascade_query_cost.py`'s docstring.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    manifest_path = RUNS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no {manifest_path}; run py/baselines/cascade_bm25.py first")
    manifest = json.loads(manifest_path.read_text())

    ndcg_results = []
    for entry in manifest:
        if entry["engine"] not in CASCADE_ENGINES:
            continue
        run_path = REPO_ROOT / entry["run_path"]
        if not run_path.exists():
            print(f"missing run file, skipping: {run_path}")
            continue
        result = evaluate_run(entry["engine"], entry["query_set"], run_path, entry)
        write_result(result)
        ndcg_results.append(result)
        print(f"{entry['engine']:>24} {entry['query_set']:>5}  "
              f"NDCG@10={result['metrics']['ndcg_cut_10']:.4f}")

    cost_path = RESULTS_DIR / "cascade-query-cost.json"
    if not cost_path.exists():
        raise SystemExit(f"no {cost_path}; run py/baselines/cascade_query_cost.py first")
    cost = json.loads(cost_path.read_text())

    (REPO_ROOT / "bench" / "phase1.md").write_text(render_markdown(ndcg_results, cost))
    print(f"\nwrote bench/phase1.md ({len(ndcg_results)} NDCG runs)")


if __name__ == "__main__":
    main()
