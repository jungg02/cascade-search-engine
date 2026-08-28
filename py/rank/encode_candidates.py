"""Dense-score and doc-length features for Phase 4's pre-rank MLP.

Not a re-run of dense.encode.py -- that encoded Phase 3's 1M-passage
*subset*; this phase's WAND candidates are a different, much smaller
document set. dl19/dl20 stay at their full depth-1000 (every candidate there
is genuinely scored, for prerank-consistency's true-top-k). dev is truncated
to each query's top DEV_NEGATIVE_POOL_DEPTH WAND-scored candidates before
encoding -- dev's *un*truncated union across 6,980 queries is 3.77M unique
docs (measured; an 8+ hour encoding job), when MLP training only ever
samples 1 positive + 4 negatives per query from it. Encodes just that
truncated/full-depth set, plus the dev/dl19/dl20 queries, and writes a flat
scalar feature file -- Kaggle only needs the (qid, docid) -> dense_score
scalar as an MLP feature, not the raw embeddings, so nothing here ships a
1.5GB-scale artifact the way Phase 3's encode.py did.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.datasets import iter_docs, load_queries
from harness.runfile import read_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

QUERY_SETS = ("dev", "dl19", "dl20")
DEV_NEGATIVE_POOL_DEPTH = 50


def unique_candidate_docids(runs: list[dict[str, dict[str, float]]]) -> set[str]:
    """Every docid that appears anywhere across a list of loaded run dicts."""
    docids: set[str] = set()
    for run in runs:
        for candidates in run.values():
            docids.update(candidates)
    return docids


def truncate_to_top_k(run: dict[str, dict[str, float]], k: int) -> dict[str, dict[str, float]]:
    """Keep only each query's k highest-scoring candidates. Used to bound
    dev's contribution to the encoding workload -- Task 6/Task 8 apply this
    identically to dev_run before sampling training negatives, so the
    encoded feature set and the training-sampling pool always agree."""
    return {
        qid: dict(sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:k])
        for qid, candidates in run.items()
    }


def load_wand_runs() -> dict[str, dict[str, dict[str, float]]]:
    return {
        query_set: read_run(RUNS_DIR / f"cascade-wand.{query_set}.txt")
        for query_set in QUERY_SETS
    }


def read_run_truncated(path: str | Path, k: int) -> dict[str, dict[str, float]]:
    """Streaming equivalent of `harness.runfile.read_run(path)` followed by
    `truncate_to_top_k(run, k)`, but never holds more than k entries per
    query in memory -- relies on each query's block being pre-sorted
    descending by score, which is the format `harness.runfile.write_run`
    always produces and what every WAND run in this project is written by
    (verified directly against cascade-wand.dev.txt: no query's block ever
    has a later line score higher than an earlier one).

    Exists because dev's untruncated file is 6,974,879 lines / 374MB --
    materializing that whole nested dict via read_run() just to immediately
    discard all but the top k per query (at most 349,000 of 3.77M entries)
    is real, avoidable memory pressure on a resource-constrained host. Found
    during this phase's real Kaggle run: the driver's dev_run load was the
    single largest avoidable allocation contributing to an out-of-memory
    kernel restart.
    """
    run: dict[str, dict[str, float]] = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            qid, _, docid, _, score, *_ = fields
            bucket = run.setdefault(qid, {})
            if len(bucket) >= k:
                continue
            bucket[docid] = float(score)
    return run


def collect_candidate_texts(candidate_docids: set[str]) -> dict[str, str]:
    """Single streaming pass over the corpus, matching dense/subset.py's pattern
    for pulling a bounded docid set out of the full 8.8M-passage stream."""
    texts: dict[str, str] = {}
    for docid, text in iter_docs():
        if docid in candidate_docids:
            texts[docid] = text
            if len(texts) == len(candidate_docids):
                break
    missing = candidate_docids - texts.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} candidate docids never seen while streaming the "
            f"corpus (e.g. {sorted(missing)[:5]})"
        )
    return texts


def main() -> None:
    # Deliberately local, not module-level: Task 8's Kaggle driver imports
    # this module only for DEV_NEGATIVE_POOL_DEPTH/truncate_to_top_k (pure,
    # no ML dependency), and Kaggle never uploads py/dense/ -- a module-level
    # `from dense.encode import ...` would crash that import with
    # ModuleNotFoundError before the driver ever reached the names it
    # actually wants. Only main() (the local, non-Kaggle encoding driver)
    # needs the real encoder.
    from sentence_transformers import SentenceTransformer

    from dense.encode import MODEL_NAME, apply_query_prefix, select_device

    wand_runs = load_wand_runs()
    wand_runs["dev"] = truncate_to_top_k(wand_runs["dev"], DEV_NEGATIVE_POOL_DEPTH)
    candidate_docids = unique_candidate_docids(list(wand_runs.values()))
    print(
        f"unique candidate docids across dev (top {DEV_NEGATIVE_POOL_DEPTH}/query)"
        f"+dl19+dl20: {len(candidate_docids)}"
    )

    device = select_device()
    print(f"device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("encoding candidate passages...")
    candidate_texts = collect_candidate_texts(candidate_docids)
    ordered_docids = sorted(candidate_texts)
    doc_embeddings = model.encode(
        [candidate_texts[d] for d in ordered_docids],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    doc_embedding_by_id = dict(zip(ordered_docids, doc_embeddings))
    doc_length_by_id = {d: len(candidate_texts[d]) for d in ordered_docids}

    # Written alongside the scalar features so Task 8's Kaggle driver can
    # build real (query_text, passage_text) pairs for the cross-encoder --
    # query text comes from harness.datasets.load_queries directly (no local
    # artifact needed for that half), but passage text needs this file since
    # Kaggle never sees the 8.8M-passage corpus itself, only this bounded
    # candidate-scoped lookup.
    candidate_texts_path = DATA_DIR / "rank-candidate-texts.json"
    candidate_texts_path.write_text(json.dumps(candidate_texts))
    print(f"wrote {candidate_texts_path.relative_to(REPO_ROOT)} ({len(candidate_texts)} docs)")

    output_path = DATA_DIR / "rank-dense-scores.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(output_path, "w") as out:
        for query_set in QUERY_SETS:
            queries = load_queries(query_set)
            run = wand_runs[query_set]
            qids = sorted(run)
            print(f"encoding {len(qids)} {query_set} queries...")
            query_texts = [apply_query_prefix(queries[qid]) for qid in qids]
            query_embeddings = model.encode(
                query_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            for qid, q_embedding in zip(qids, query_embeddings):
                for docid in run[qid]:
                    dense_score = float(np.dot(q_embedding, doc_embedding_by_id[docid]))
                    row = {
                        "qid": qid,
                        "docid": docid,
                        "dense_score": round(dense_score, 6),
                        "doc_length": doc_length_by_id[docid],
                    }
                    out.write(json.dumps(row) + "\n")
                    rows_written += 1

    print(f"wrote {output_path.relative_to(REPO_ROOT)} ({rows_written} rows)")


if __name__ == "__main__":
    main()
