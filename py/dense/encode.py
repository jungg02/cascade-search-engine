"""Encode the 1M-passage subset and the dev/dl19/dl20 query sets with
bge-small-en-v1.5.

BGE is asymmetric: passages are encoded plain, queries get an instruction
prefix. Getting this backwards silently degrades retrieval quality without
raising an error, so apply_query_prefix is its own small, tested function
rather than an inline string concatenation buried in the encoding loop.

This is the one long-running step in this phase -- everything downstream
reads its .npy output and never re-encodes. Passages are encoded in
CHUNK_SIZE-row slices written straight into a pre-sized memmap: a single
SentenceTransformer.encode() call over all 1M texts holds the full output
array (and a duplicate from .astype()) in memory at once, which pushed this
host into heavy swapping on first run (8GB RAM, shared with the rest of the
desktop). Chunking bounds peak RSS to one chunk. The memmap is built at a
`.tmp` path and only renamed into place after the last chunk, so a crash
mid-run leaves no file at the real path rather than a full-size npy that
*looks* complete but has trailing zero rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from harness.datasets import load_queries

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384
# 128 caused an earlier run to die without a Python traceback (MPS OOM under
# tokenizer padding on this host tends to fail silently rather than raise).
BATCH_SIZE = 32
CHUNK_SIZE = 50_000
QUERY_SETS = ("dev", "dl19", "dl20")

# select_device() detects whatever this *host* has at call time -- fine for
# actually running encode.py, wrong for a report reading it later, since the
# report needs to say what device produced the *committed* embeddings, not
# what's available on whatever machine happens to render the report. This is
# a recorded literal, not a re-detection: this repo's only real encoding run
# (see the SDD ledger's Task 3 notes) used MPS.
MEASURED_DEVICE = "mps"


def apply_query_prefix(text: str) -> str:
    return QUERY_PREFIX + text


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_subset_texts(jsonl_path: Path) -> tuple[list[str], list[int]]:
    docids: list[int] = []
    texts: list[str] = []
    with open(jsonl_path) as handle:
        for line in handle:
            row = json.loads(line)
            docids.append(int(row["docid"]))
            texts.append(row["text"])
    return texts, docids


def encode_passages_to_file(model: SentenceTransformer, texts: list[str], out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    memmap = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(len(texts), EMBEDDING_DIM)
    )
    try:
        for start in range(0, len(texts), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(texts))
            memmap[start:end] = model.encode(
                texts[start:end],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            print(f"  encoded {end}/{len(texts)} passages", flush=True)
            if model.device.type == "mps":
                torch.mps.empty_cache()
        memmap.flush()
    finally:
        del memmap
    tmp_path.replace(out_path)


def encode_queries(model: SentenceTransformer, query_set: str) -> tuple[np.ndarray, list[str]]:
    queries = load_queries(query_set)
    qids = sorted(queries)
    texts = [apply_query_prefix(queries[qid]) for qid in qids]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return embeddings, qids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-jsonl", default=str(DATA_DIR / "dense-subset.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="encode only the first N passages (smoke-testing)"
    )
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts, docids = load_subset_texts(Path(args.subset_jsonl))
    if args.limit is not None:
        texts, docids = texts[: args.limit], docids[: args.limit]
    print(f"encoding {len(texts)} passages...")
    t0 = time.time()
    out_path = DATA_DIR / "dense-embeddings.npy"
    encode_passages_to_file(model, texts, out_path)
    print(f"passages encoded in {time.time() - t0:.1f}s")

    np.save(DATA_DIR / "dense-docids.npy", np.array(docids, dtype=np.int64))
    print(f"wrote {out_path.name} ({len(texts)}, {EMBEDDING_DIM}) and dense-docids.npy")

    for query_set in QUERY_SETS:
        q_embeddings, qids = encode_queries(model, query_set)
        np.save(DATA_DIR / f"dense-query-embeddings-{query_set}.npy", q_embeddings)
        (DATA_DIR / f"dense-query-ids-{query_set}.json").write_text(json.dumps(qids))
        print(f"{query_set}: encoded {len(qids)} queries, shape {q_embeddings.shape}")


if __name__ == "__main__":
    main()
