"""Exact (brute-force) nearest-neighbor search over the 1M-passage subset,
via faiss.IndexFlatIP. Two outputs, two purposes:

  - dev query set (6,980 queries): the recall@100 reference and latency
    source for the ANN sweep (dense/ann_sweep.py). Top-100 only --
    recall@100 needs no more, and dev's query count makes top-1000 an
    unreasonably large commit (~140MB vs ~14MB at top-100).
  - dl19+dl20 query set (97 queries): the exact dense channel for hybrid
    fusion (dense/fusion.py), which needs recall@1000. Using the
    *exact* ranking here (not an ANN approximation) isolates "does fusion
    help" from "how good is the ANN approximation" -- a separate question
    the Pareto sweep already answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUBSET_SIZE = 1_000_000


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def exact_search(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    docids: np.ndarray,
    k: int,
) -> dict[str, list[list]]:
    scores, positions = index.search(query_embeddings, k)
    results: dict[str, list[list]] = {}
    for row, qid in enumerate(query_ids):
        results[qid] = [
            [int(docids[positions[row, col]]), round(float(scores[row, col]), 6)]
            for col in range(positions.shape[1])
        ]
    return results


def run_query_set(
    index: faiss.IndexFlatIP, docids: np.ndarray, query_set: str, k: int
) -> dict[str, list[list]]:
    query_embeddings = np.load(DATA_DIR / f"dense-query-embeddings-{query_set}.npy")
    query_ids = json.loads((DATA_DIR / f"dense-query-ids-{query_set}.json").read_text())
    return exact_search(index, query_embeddings, query_ids, docids, k)


def main() -> None:
    embeddings = np.load(DATA_DIR / "dense-embeddings.npy")
    docids = np.load(DATA_DIR / "dense-docids.npy")
    assert embeddings.shape[0] == SUBSET_SIZE
    assert docids.shape[0] == embeddings.shape[0], (
        "dense-embeddings.npy and dense-docids.npy are out of sync -- encode.py "
        "writes them in two separate steps, so a crash between those writes "
        "would otherwise go undetected here"
    )
    index = build_flat_index(embeddings)

    dev_results = run_query_set(index, docids, "dev", k=100)
    dl19_results = run_query_set(index, docids, "dl19", k=1000)
    dl20_results = run_query_set(index, docids, "dl20", k=1000)
    dl_results = {**dl19_results, **dl20_results}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev_output = {
        "query_set": "dev",
        "k": 100,
        "model": MODEL_NAME,
        "subset_size": SUBSET_SIZE,
        "results": dev_results,
        "provenance": run_metadata(),
    }
    (RESULTS_DIR / "dense-ground-truth-dev.json").write_text(
        json.dumps(dev_output, separators=(",", ":"))
    )

    dl_output = {
        "query_set": "dl19+dl20",
        "k": 1000,
        "model": MODEL_NAME,
        "subset_size": SUBSET_SIZE,
        "results": dl_results,
        "provenance": run_metadata(),
    }
    (RESULTS_DIR / "dense-ground-truth-dl19-dl20.json").write_text(
        json.dumps(dl_output, separators=(",", ":"))
    )

    print(f"dev: {len(dev_results)} queries, top-100")
    print(f"dl19+dl20: {len(dl_results)} queries, top-1000")


if __name__ == "__main__":
    main()
