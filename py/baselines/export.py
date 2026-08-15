"""Export the passage corpus and query sets to the formats the baselines consume.

Every engine in this project reads the same text from the same source
(`ir_datasets`), so a scoring difference between engines is a scoring difference
and not a preprocessing difference. Phase 1's C++ index will read this same JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.datasets import DATA_DIR, QUERY_SETS, iter_docs, load_queries

JSONL_PATH = DATA_DIR / "msmarco-passage-jsonl" / "docs.jsonl"
TSV_PATH = DATA_DIR / "msmarco-passage.tsv"


def export_docs(path: Path = JSONL_PATH, limit: int | None = None) -> int:
    """Write Anserini JsonCollection format: one {"id", "contents"} object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as out:
        for doc_id, text in iter_docs(limit=limit):
            out.write(json.dumps({"id": doc_id, "contents": text}) + "\n")
            count += 1
            if count % 1_000_000 == 0:
                print(f"  {count:,} docs", flush=True)
    return count


def export_tsv(path: Path = TSV_PATH, limit: int | None = None) -> int:
    """Write `docid \\t text` for the C++ index builder.

    A second format rather than a JSON parser in C++, because MS MARCO passages
    contain no tabs or newlines (the source collection is itself a TSV) so this
    is lossless. The assertion below is what keeps that claim true — if it ever
    fires, the C++ index and the Lucene index are reading different text and
    every cross-engine comparison downstream is invalid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as out:
        for doc_id, text in iter_docs(limit=limit):
            if "\t" in text or "\n" in text:
                raise ValueError(f"doc {doc_id} contains a tab or newline; TSV is lossy")
            out.write(f"{doc_id}\t{text}\n")
            count += 1
            if count % 1_000_000 == 0:
                print(f"  {count:,} docs", flush=True)
    return count


def export_queries(directory: Path = DATA_DIR / "queries") -> dict[str, int]:
    """Write Anserini's TSV query format: `qid\\tquery`."""
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in QUERY_SETS:
        queries = load_queries(name)
        with open(directory / f"{name}.tsv", "w") as out:
            for qid, text in queries.items():
                out.write(f"{qid}\t{text}\n")
        counts[name] = len(queries)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="docs to export (default: all)")
    parser.add_argument("--queries-only", action="store_true")
    parser.add_argument("--tsv-only", action="store_true", help="skip the Lucene JSONL")
    args = parser.parse_args()

    counts = export_queries()
    for name, n in counts.items():
        print(f"{name}: {n} queries")

    if args.queries_only:
        return
    if not args.tsv_only:
        n = export_docs(limit=args.limit)
        print(f"wrote {n:,} docs to {JSONL_PATH}")
    n = export_tsv(limit=args.limit)
    print(f"wrote {n:,} docs to {TSV_PATH}")


if __name__ == "__main__":
    main()
