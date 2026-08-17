"""Build the 1M-passage subset for dense recall: every doc judged in
dl19+dl20 qrels, plus a random fill to reach exactly 1,000,000 docs.

Random subsetting alone risks dropping qrels-judged documents -- a document
missing from the subset can never be retrieved, which would make every
downstream recall/NDCG number against it meaningless. So the subset is
always the qrels union first, then filled to size with a seeded random
sample of the remaining corpus.

MS MARCO passage docids are dense sequential integers ("0".."8841822",
matching row order in the corpus) -- verified against harness.datasets at
plan-writing time. This lets subset selection work directly on an integer
id range instead of needing a true streaming reservoir sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harness.datasets import doc_count, iter_docs, load_qrels

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

SUBSET_SIZE = 1_000_000
SEED = 0


def qrels_docid_union() -> set[int]:
    """Every docid judged (at any relevance level) in dl19 or dl20 qrels."""
    ids: set[int] = set()
    for query_set in ("dl19", "dl20"):
        qrels = load_qrels(query_set)
        for judgments in qrels.values():
            ids.update(int(docid) for docid in judgments)
    return ids


def select_subset_ids(
    qrels_ids: set[int],
    total_docs: int,
    seed: int = SEED,
    subset_size: int = SUBSET_SIZE,
) -> set[int]:
    """The qrels union plus a seeded random fill, as a set of integer docids."""
    if len(qrels_ids) > subset_size:
        raise ValueError(
            f"qrels union ({len(qrels_ids)}) exceeds subset_size ({subset_size})"
        )
    rng = np.random.default_rng(seed)
    all_ids = np.arange(total_docs)
    candidate_pool = np.setdiff1d(
        all_ids, np.array(sorted(qrels_ids), dtype=np.int64), assume_unique=True
    )
    fill_needed = subset_size - len(qrels_ids)
    fill = rng.choice(candidate_pool, size=fill_needed, replace=False)
    return qrels_ids | {int(x) for x in fill.tolist()}


def write_subset(subset_ids: set[int], jsonl_path: Path, docids_path: Path) -> None:
    """Single streaming pass over the corpus, writing matched docs in corpus order."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(jsonl_path, "w") as jsonl_out, open(docids_path, "w") as ids_out:
        for docid, text in iter_docs():
            if int(docid) in subset_ids:
                jsonl_out.write(json.dumps({"docid": docid, "text": text}) + "\n")
                ids_out.write(docid + "\n")
                written += 1
    if written != len(subset_ids):
        raise RuntimeError(
            f"wrote {written} docs but subset_ids had {len(subset_ids)} -- "
            "a selected docid was never seen while streaming the corpus"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()

    qrels_ids = qrels_docid_union()
    print(f"qrels union: {len(qrels_ids)} docids")
    total_docs = doc_count()
    subset_ids = select_subset_ids(
        qrels_ids, total_docs, seed=args.seed, subset_size=args.subset_size
    )
    print(f"selected {len(subset_ids)} docids (seed={args.seed})")

    jsonl_path = DATA_DIR / "dense-subset.jsonl"
    docids_path = DATA_DIR / "dense-subset-docids.txt"
    write_subset(subset_ids, jsonl_path, docids_path)
    print(
        f"wrote {jsonl_path.relative_to(REPO_ROOT)} and "
        f"{docids_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
