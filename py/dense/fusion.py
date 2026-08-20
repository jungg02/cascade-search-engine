"""Hybrid fusion: RRF and normalized score fusion over the lexical (WAND)
and dense (exact brute-force) channels, compared against each channel
alone on dl19+dl20 NDCG@10 / recall@1000.

Both fusion functions operate on a single query's channels at a time --
a list of (docid -> score) dicts, one per channel -- and are pure, so they
are unit tested directly rather than only through an end-to-end run.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.datasets import load_qrels
from harness.metrics import evaluate
from harness.runfile import read_run
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MANIFEST_PATH = REPO_ROOT / "runs" / "manifest.json"

RRF_K = 60
QUERY_SETS = ("dl19", "dl20")


def reciprocal_rank_fusion(
    channels: list[dict[str, float]], k: int = RRF_K
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for channel in channels:
        ranked = sorted(channel.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (docid, _) in enumerate(ranked, start=1):
            fused[docid] = fused.get(docid, 0.0) + 1.0 / (k + rank)
    return fused


def _min_max_normalize(channel: dict[str, float]) -> dict[str, float]:
    if not channel:
        return {}
    lo, hi = min(channel.values()), max(channel.values())
    if lo == hi:
        return {docid: 0.0 for docid in channel}
    return {docid: (score - lo) / (hi - lo) for docid, score in channel.items()}


def normalized_score_fusion(channels: list[dict[str, float]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for channel in channels:
        for docid, score in _min_max_normalize(channel).items():
            fused[docid] = fused.get(docid, 0.0) + score
    return fused


def load_lexical_run(query_set: str) -> dict[str, dict[str, float]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for entry in manifest:
        if entry["engine"] == "cascade-wand" and entry["query_set"] == query_set:
            return read_run(REPO_ROOT / entry["run_path"])
    raise RuntimeError(f"no cascade-wand manifest entry for query_set={query_set!r}")


def load_dense_run(query_set: str) -> dict[str, dict[str, float]]:
    ground_truth = json.loads(
        (RESULTS_DIR / "dense-ground-truth-dl19-dl20.json").read_text()
    )["results"]
    qrels = load_qrels(query_set)
    return {
        qid: {str(docid): score for docid, score in ground_truth[qid]}
        for qid in qrels
        if qid in ground_truth
    }


def fuse_run(
    lexical: dict[str, dict[str, float]],
    dense: dict[str, dict[str, float]],
    method: str,
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    all_qids = set(lexical) | set(dense)
    for qid in all_qids:
        channels = [lexical.get(qid, {}), dense.get(qid, {})]
        if method == "rrf":
            fused[qid] = reciprocal_rank_fusion(channels)
        elif method == "score":
            fused[qid] = normalized_score_fusion(channels)
        else:
            raise ValueError(f"unknown method {method!r}")
    return fused


def main() -> None:
    all_qrels: dict[str, dict[str, int]] = {}
    lexical_run: dict[str, dict[str, float]] = {}
    dense_run: dict[str, dict[str, float]] = {}
    for query_set in QUERY_SETS:
        all_qrels.update(load_qrels(query_set))
        lexical_run.update(load_lexical_run(query_set))
        dense_run.update(load_dense_run(query_set))

    rows = {
        "lexical_only": lexical_run,
        "dense_only": dense_run,
        "rrf": fuse_run(lexical_run, dense_run, "rrf"),
        "score_fusion": fuse_run(lexical_run, dense_run, "score"),
    }

    results = {}
    for name, run in rows.items():
        eval_result = evaluate(all_qrels, run, ndcg_k=(10,), recall_k=(1000,))
        results[name] = {
            "ndcg_10": eval_result.mean["ndcg_cut_10"],
            "recall_1000": eval_result.mean["recall_1000"],
            "num_queries": eval_result.num_queries,
        }
        print(
            f"{name:>14}: ndcg@10={results[name]['ndcg_10']:.4f} "
            f"recall@1000={results[name]['recall_1000']:.4f}"
        )

    output = {
        "query_sets": list(QUERY_SETS),
        "rrf_k": RRF_K,
        "results": results,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "dense-fusion.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
