"""Kaggle-run orchestration for Phase 4: prerank + prerank-consistency,
dynamic batching sweep, precision comparison, queue discipline. Writes each
sub-experiment's bench/results/rank-*.json as it completes (see this phase's
design spec's "Kaggle provenance gap" -- a session interruption shouldn't
lose earlier sub-experiments' results).

This session (the one running this plan) cannot execute this file --
it is written and reviewed here, then run by hand on Kaggle. See
kaggle/README.md for the upload/run steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from harness.datasets import load_qrels, load_queries
from harness.metrics import evaluate
from harness.runfile import read_run
from harness.runmeta import run_metadata
from rank.crossencoder_harness import (
    CrossEncoderSession,
    convert_to_fp16,
    export_onnx,
    quantize_int8,
    run_batching_sweep,
    run_queue_discipline,
)
from rank.encode_candidates import DEV_NEGATIVE_POOL_DEPTH, truncate_to_top_k
from rank.prerank_consistency import prerank_consistency
from rank.prerank_features import fit_scaler
from rank.prerank_mlp import (
    build_mlp,
    build_training_examples,
    load_dense_scores_and_lengths,
    survivors_for_query,
    train_mlp,
)

# Filled in by hand before uploading -- `git rev-parse HEAD` on this repo,
# immediately before uploading this file. Kaggle has no git repo to read it
# from; see this phase's design spec's "Kaggle provenance gap."
SOURCE_GIT_SHA = "4a98fb3ef9c46bf9466ebc4379330f135bf002ab"

RESULTS_DIR = Path("bench/results")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_K = 100  # matches rank.prerank_mlp.TOP_K_SURVIVORS

BATCHING_GRID = [
    (max_batch_size, max_wait_ms)
    for max_batch_size in (1, 8, 32)
    for max_wait_ms in (0, 5, 20)
]
QUEUE_DISCIPLINES = ("shed", "unbounded")
QUEUE_OVERLOAD_MULTIPLIERS = (1.5, 2.0, 3.0)


def kaggle_hardware_info() -> dict:
    """torch.cuda fields to supplement run_metadata()'s CPU-oriented,
    sysctl-based hardware block, which reads 'unknown' on Kaggle's Linux
    host (no sysctl there)."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def provenance() -> dict:
    meta = run_metadata()
    meta["git_sha"] = SOURCE_GIT_SHA  # overrides run_metadata()'s "unknown"
    meta["kaggle_hardware"] = kaggle_hardware_info()
    return meta


def write_result(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"rank-{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def load_candidate_texts() -> dict[str, str]:
    return json.loads(Path("data/rank-candidate-texts.json").read_text())


def run_prerank_and_consistency() -> dict:
    # Truncated to match exactly what Task 2 encoded features for -- dev's
    # union of untruncated depth-1000 candidates is 3.77M unique docs, far
    # more than the 1-positive-plus-4-negatives-per-query training loop ever
    # consumes. See this phase's negative-sampling Global Constraint.
    dev_run = truncate_to_top_k(read_run("runs/cascade-wand.dev.txt"), DEV_NEGATIVE_POOL_DEPTH)
    dev_qrels = load_qrels("dev")
    dense_scores, doc_lengths = load_dense_scores_and_lengths(Path("data/rank-dense-scores.jsonl"))
    candidate_texts = load_candidate_texts()

    features, labels = build_training_examples(dev_run, dev_qrels, dense_scores, doc_lengths)
    print(f"training on {len(features)} examples from {len(dev_run)} dev queries")
    scaler = fit_scaler(features)
    model = build_mlp()
    train_mlp(model, features, labels, scaler, epochs=50)

    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    export_onnx(MODEL_NAME, fp32_path)
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])

    consistency_scores = []
    fused_qrels = {}
    fused_run = {}
    for query_set in ("dl19", "dl20"):
        run = read_run(f"runs/cascade-wand.{query_set}.txt")
        qrels = load_qrels(query_set)
        queries = load_queries(query_set)
        fused_qrels.update(qrels)
        for qid, candidates in run.items():
            docids = list(candidates)
            # true top-k under the FULL ranker: score every candidate, not
            # just the pre-rank survivors -- this is the one place the
            # expensive model runs over the entire candidate pool.
            pairs = [(queries[qid], candidate_texts[docid]) for docid in docids]
            all_scores = session.score(pairs)
            ranked = sorted(zip(docids, all_scores), key=lambda kv: kv[1], reverse=True)
            true_top_k = [docid for docid, _ in ranked[:CROSS_ENCODER_K]]

            survivors = survivors_for_query(
                model, scaler, candidates, dense_scores, doc_lengths, qid, top_k=CROSS_ENCODER_K
            )
            consistency_scores.append(prerank_consistency(true_top_k, survivors))

            fused_run[qid] = {
                docid: score
                for docid, score in zip(docids, all_scores)
                if docid in survivors
            }

    mean_consistency = sum(consistency_scores) / len(consistency_scores)
    # recall_k must match CROSS_ENCODER_K, not the WAND run's original depth
    # 1000 -- fused_run only ever holds each query's pre-rank survivors (at
    # most CROSS_ENCODER_K docs), so recall@1000 over it would silently
    # report a recall@CROSS_ENCODER_K number under a misleading deeper-sounding
    # name. harness.metrics.EvalResult.shallow_metrics exists to catch exactly
    # this (a cutoff deeper than the run); asked for the correct depth here so
    # there's nothing for it to flag, and asserted below as a second guard.
    eval_result = evaluate(
        fused_qrels, fused_run, ndcg_k=(10,), recall_k=(CROSS_ENCODER_K,), rel_threshold=2
    )
    assert not eval_result.shallow_metrics, eval_result.shallow_metrics

    write_result(
        "prerank",
        {
            "mean_prerank_consistency": mean_consistency,
            "num_queries": len(consistency_scores),
            "cascade_ndcg_10": eval_result.mean["ndcg_cut_10"],
            "cascade_recall_100": eval_result.mean[f"recall_{CROSS_ENCODER_K}"],
            "top_k_survivors": CROSS_ENCODER_K,
            "run_depth": eval_result.run_depth,
            # Requested CUDA -- may have silently fallen back to CPU (a
            # CUDA/cuDNN version mismatch logs a warning, not an error).
            # Recorded so a CPU-fallback run is never read as a GPU number.
            "active_providers": session.active_providers,
            "provenance": provenance(),
        },
    )

    # Returned (not just written) so run_precision() can re-score exactly
    # these survivors at fp16/INT8 without repeating the MLP pass -- the
    # pre-rank survivor set doesn't depend on the cross-encoder's precision,
    # only the cross-encoder's *score* of each survivor does.
    return {
        "fused_qrels": fused_qrels,
        "survivors_by_qid": {
            qid: set(candidates) for qid, candidates in fused_run.items()
        },
        "candidate_texts": candidate_texts,
        "cascade_ndcg_10_fp32": eval_result.mean["ndcg_cut_10"],
    }


def run_batching() -> list[dict]:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    # Created once (not a factory that builds a fresh session per call) so
    # active_providers below reflects the actual session run_batching_sweep
    # used across the whole grid -- run_batching_sweep only calls its
    # session_factory once internally anyway, so this changes nothing about
    # the sweep itself, only makes the session inspectable afterward.
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])
    request_pairs = [("what does this query mean", f"candidate passage {i}") for i in range(256)]

    points = run_batching_sweep(lambda: session, BATCHING_GRID, request_pairs, duration_s=10.0)
    write_result(
        "batching",
        {
            "points": points,
            "grid": BATCHING_GRID,
            "active_providers": session.active_providers,
            "provenance": provenance(),
        },
    )
    return points


def score_survivors_with_session(
    session: CrossEncoderSession,
    survivors_by_qid: dict[str, set[str]],
    candidate_texts: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Re-score exactly the pre-rank survivors (not the full candidate pool)
    with one precision's session -- this is the run fed to evaluate() for
    that precision's cascade NDCG@10."""
    queries = {**load_queries("dl19"), **load_queries("dl20")}
    run: dict[str, dict[str, float]] = {}
    for qid, docids in survivors_by_qid.items():
        docids = list(docids)
        pairs = [(queries[qid], candidate_texts[docid]) for docid in docids]
        scores = session.score(pairs)
        run[qid] = dict(zip(docids, scores))
    return run


def run_precision(
    best_setting: tuple[int, float],
    fused_qrels: dict[str, dict[str, int]],
    survivors_by_qid: dict[str, set[str]],
    candidate_texts: dict[str, str],
    cascade_ndcg_10_fp32: float,
) -> None:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    fp16_path = onnx_dir / "model_fp16.onnx"
    int8_path = onnx_dir / "model_int8.onnx"
    convert_to_fp16(fp32_path, fp16_path)
    quantize_int8(fp32_path, int8_path)

    request_pairs = [("what does this query mean", f"candidate passage {i}") for i in range(256)]
    max_batch_size, max_wait_ms = best_setting
    results = {}
    for precision, path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        session = CrossEncoderSession(path, MODEL_NAME, providers=["CUDAExecutionProvider"])
        session_factory = lambda s=session: s
        points = run_batching_sweep(session_factory, [(max_batch_size, max_wait_ms)], request_pairs, duration_s=10.0)

        precision_run = score_survivors_with_session(session, survivors_by_qid, candidate_texts)
        # recall_k matches CROSS_ENCODER_K for the same reason as
        # run_prerank_and_consistency() -- precision_run also only ever
        # holds the pre-rank survivors, not the full depth-1000 pool.
        eval_result = evaluate(
            fused_qrels, precision_run, ndcg_k=(10,), recall_k=(CROSS_ENCODER_K,), rel_threshold=2
        )
        assert not eval_result.shallow_metrics, eval_result.shallow_metrics
        ndcg_10 = eval_result.mean["ndcg_cut_10"]

        results[precision] = {
            **points[0],
            "cascade_ndcg_10": ndcg_10,
            "ndcg_10_delta_vs_fp32": ndcg_10 - cascade_ndcg_10_fp32,
            "active_providers": session.active_providers,
        }

    write_result(
        "precision",
        {"setting": {"max_batch_size": max_batch_size, "max_wait_ms": max_wait_ms}, "results": results, "provenance": provenance()},
    )


def run_queue_disciplines(best_setting: tuple[int, float], baseline_throughput_qps: float) -> None:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    # Created once, reused across all 6 discipline/multiplier runs -- same
    # reasoning as run_batching() above.
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])
    request_pairs = [("what does this query mean", f"candidate passage {i}") for i in range(256)]

    runs = []
    for discipline in QUEUE_DISCIPLINES:
        for multiplier in QUEUE_OVERLOAD_MULTIPLIERS:
            result = run_queue_discipline(
                lambda: session,
                discipline=discipline,
                arrival_rate_qps=baseline_throughput_qps * multiplier,
                batcher_config=best_setting,
                request_pairs=request_pairs,
                duration_s=10.0,
            )
            runs.append(result)
    write_result(
        "queue",
        {"runs": runs, "active_providers": session.active_providers, "provenance": provenance()},
    )


def main() -> None:
    assert SOURCE_GIT_SHA != "REPLACE_ME_BEFORE_UPLOADING", (
        "fill in SOURCE_GIT_SHA from a local `git rev-parse HEAD` before running on Kaggle"
    )
    prerank_state = run_prerank_and_consistency()
    batching_points = run_batching()
    best_point = max(batching_points, key=lambda p: p["throughput_qps"])
    best_setting = (best_point["max_batch_size"], best_point["max_wait_ms"])
    run_precision(
        best_setting,
        fused_qrels=prerank_state["fused_qrels"],
        survivors_by_qid=prerank_state["survivors_by_qid"],
        candidate_texts=prerank_state["candidate_texts"],
        cascade_ndcg_10_fp32=prerank_state["cascade_ndcg_10_fp32"],
    )
    run_queue_disciplines(best_setting, baseline_throughput_qps=best_point["throughput_qps"])


if __name__ == "__main__":
    main()
