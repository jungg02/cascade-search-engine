"""HNSW and IVF-PQ recall/latency/memory sweep against the exact dev
ground truth -- the phase's money chart.

Latency is measured single-threaded, one search() call per dev query
(6,980 samples), matching this project's no-mean, percentile-only rule
(harness.histogram.LatencyRecorder). faiss and hnswlib are both told to
use one thread per call so their own internal multithreading doesn't
blur a single-query measurement.

Memory footprint, as implemented below (current_rss_gb, a getrusage RSS
delta around each build call), describes this script's *original*
methodology only. It has a known same-process high-water-mark flaw: running
all seven builds (3 HNSW + 4 IVF-PQ) sequentially in one process means only
the first heavy allocation produces a meaningful delta -- every later
build's "before" baseline already sits at or above its own peak, so its
delta reads near-zero (this is exactly what happened: 30/35 points read
memory_gb=0.0 or non-monotonic on the real run). The `memory_gb` values
actually committed in bench/results/dense-ann-sweep.json were NOT produced
by rerunning this script -- they came from a separate one-off remeasurement
that switched to serialized-index-size (hnswlib save_index / faiss
serialize_index byte length), which is deterministic and immune to this
bug. See that JSON's own top-level `memory_measurement` field for the real
method and provenance. Platform note (still true of the RSS path below):
ru_maxrss is *bytes* on macOS (this project's only target platform),
*kilobytes* on Linux -- verified empirically at plan-writing time. Divide
by 1e9 for GB; do not port this constant elsewhere without re-checking.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import faiss
import hnswlib
import numpy as np

from harness.histogram import LatencyRecorder
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

EMBEDDING_DIM = 384
HNSW_M_VALUES = (16, 32, 64)
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH_VALUES = (32, 64, 128, 256, 512)
IVFPQ_NLIST_VALUES = (1024, 4096)
IVFPQ_M_VALUES = (32, 64)
IVFPQ_NBITS = 8
IVFPQ_NPROBE_VALUES = (1, 8, 16, 32, 64)
RECALL_K = 100

faiss.omp_set_num_threads(1)


def current_rss_gb() -> float:
    """macOS: ru_maxrss is bytes. See module docstring.

    Known bug (see module docstring): same-process high-water-mark RSS makes
    every build after the first read a near-zero delta. Rerunning this script
    will NOT reproduce the memory_gb values currently committed in
    bench/results/dense-ann-sweep.json -- those came from a separate
    serialized-index-size remeasurement (see that JSON's memory_measurement
    field), not from this function.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def recall_at_k(candidate_ids: list[int], exact_ids: list[int], k: int) -> float:
    """Fraction of the exact top-k that appear anywhere in candidate_ids."""
    exact_top_k = set(exact_ids[:k])
    if not exact_top_k:
        return 0.0
    found = exact_top_k & set(candidate_ids)
    return len(found) / len(exact_top_k)


def measure_latency_hnsw(index: hnswlib.Index, queries: np.ndarray, k: int) -> LatencyRecorder:
    recorder = LatencyRecorder()
    for row in range(queries.shape[0]):
        query = queries[row : row + 1]
        started = time.perf_counter_ns()
        index.knn_query(query, k=k, num_threads=1)
        recorder.record((time.perf_counter_ns() - started) / 1000)
    return recorder


def measure_latency_faiss(index, queries: np.ndarray, k: int) -> LatencyRecorder:
    recorder = LatencyRecorder()
    for row in range(queries.shape[0]):
        query = queries[row : row + 1]
        started = time.perf_counter_ns()
        index.search(query, k)
        recorder.record((time.perf_counter_ns() - started) / 1000)
    return recorder


def sweep_hnsw(
    embeddings: np.ndarray,
    docids: np.ndarray,
    dev_queries: np.ndarray,
    dev_qids: list[str],
    ground_truth: dict[str, list[list]],
) -> list[dict]:
    points = []
    n = embeddings.shape[0]
    max_ef_search = max(HNSW_EF_SEARCH_VALUES)
    for m in HNSW_M_VALUES:
        rss_before = current_rss_gb()
        index = hnswlib.Index(space="ip", dim=EMBEDDING_DIM)
        index.init_index(max_elements=n, M=m, ef_construction=HNSW_EF_CONSTRUCTION)
        index.add_items(embeddings, docids)
        memory_gb = current_rss_gb() - rss_before

        recall_at_max_ef_search = None
        for ef_search in HNSW_EF_SEARCH_VALUES:
            index.set_ef(ef_search)
            labels, _ = index.knn_query(dev_queries, k=RECALL_K, num_threads=1)
            recalls = [
                recall_at_k(
                    [int(x) for x in labels[row]], [d for d, _ in ground_truth[qid]], RECALL_K
                )
                for row, qid in enumerate(dev_qids)
            ]
            mean_recall = sum(recalls) / len(recalls)
            latency = measure_latency_hnsw(index, dev_queries, RECALL_K)

            point = {
                "structure": "hnsw",
                "M": m,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "ef_search": ef_search,
                "recall_at_100": mean_recall,
                "latency_us": latency.summary(),
                "memory_gb": memory_gb,
            }
            points.append(point)
            print(
                f"hnsw M={m} efSearch={ef_search}: recall@100={mean_recall:.4f} "
                f"p99={latency.percentile(99)/1000:.2f}ms mem={memory_gb:.3f}GB"
            )
            assert 0.0 <= mean_recall <= 1.0
            if ef_search == max_ef_search:
                recall_at_max_ef_search = mean_recall

        assert recall_at_max_ef_search > 0.9, (
            f"hnsw M={m} recall@100 at max efSearch ({max_ef_search}) was only "
            f"{recall_at_max_ef_search:.4f} -- expected close to 1.0"
        )
    return points


def sweep_ivfpq(
    embeddings: np.ndarray,
    docids: np.ndarray,
    dev_queries: np.ndarray,
    dev_qids: list[str],
    ground_truth: dict[str, list[list]],
) -> list[dict]:
    points = []
    max_nprobe = max(IVFPQ_NPROBE_VALUES)
    for nlist in IVFPQ_NLIST_VALUES:
        for m in IVFPQ_M_VALUES:
            rss_before = current_rss_gb()
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFPQ(
                quantizer, EMBEDDING_DIM, nlist, m, IVFPQ_NBITS, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.add(embeddings)
            memory_gb = current_rss_gb() - rss_before

            recall_at_max_nprobe = None
            for nprobe in IVFPQ_NPROBE_VALUES:
                index.nprobe = nprobe
                _, positions = index.search(dev_queries, RECALL_K)
                recalls = []
                for row, qid in enumerate(dev_qids):
                    candidate_ids = [int(docids[p]) for p in positions[row] if p != -1]
                    exact_ids = [d for d, _ in ground_truth[qid]]
                    recalls.append(recall_at_k(candidate_ids, exact_ids, RECALL_K))
                mean_recall = sum(recalls) / len(recalls)
                latency = measure_latency_faiss(index, dev_queries, RECALL_K)

                point = {
                    "structure": "ivfpq",
                    "nlist": nlist,
                    "m": m,
                    "nbits": IVFPQ_NBITS,
                    "nprobe": nprobe,
                    "recall_at_100": mean_recall,
                    "latency_us": latency.summary(),
                    "memory_gb": memory_gb,
                }
                points.append(point)
                print(
                    f"ivfpq nlist={nlist} m={m} nprobe={nprobe}: recall@100={mean_recall:.4f} "
                    f"p99={latency.percentile(99)/1000:.2f}ms mem={memory_gb:.3f}GB"
                )
                assert 0.0 <= mean_recall <= 1.0
                if nprobe == max_nprobe:
                    recall_at_max_nprobe = mean_recall

            if recall_at_max_nprobe <= 0.5:
                print(
                    f"WARNING: ivfpq nlist={nlist} m={m} recall@100 at max nprobe "
                    f"({max_nprobe}) was only {recall_at_max_nprobe:.4f} -- cluster "
                    "coverage likely the binding constraint, not a bug"
                )
    return points


def main() -> None:
    embeddings = np.load(DATA_DIR / "dense-embeddings.npy")
    docids = np.load(DATA_DIR / "dense-docids.npy")
    dev_queries = np.load(DATA_DIR / "dense-query-embeddings-dev.npy")
    dev_qids = json.loads((DATA_DIR / "dense-query-ids-dev.json").read_text())
    ground_truth = json.loads(
        (RESULTS_DIR / "dense-ground-truth-dev.json").read_text()
    )["results"]

    hnsw_points = sweep_hnsw(embeddings, docids, dev_queries, dev_qids, ground_truth)
    ivfpq_points = sweep_ivfpq(embeddings, docids, dev_queries, dev_qids, ground_truth)

    output = {
        "subset_size": embeddings.shape[0],
        "num_dev_queries": len(dev_qids),
        "points": hnsw_points + ivfpq_points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "dense-ann-sweep.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(output['points'])} points)")


if __name__ == "__main__":
    main()
