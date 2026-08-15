"""Lucene BM25 baseline via the Anserini fatjar.

Anserini rather than PyLucene because the fatjar is a single download with no
build step, and its published msmarco-passage numbers give us an external check
on the whole pipeline — qrels loading, run formatting, and metric code together.
Agreeing with pytrec_eval only proves the metric code is right.

Config that must travel with every number (recorded into the run's JSON):
  - analyzer: Anserini's default EnglishAnalyzer, i.e. Porter stemming +
    stopword removal. Not Lucene's StandardAnalyzer. Phase 1's C++ tokenizer has
    to target whichever of these we actually ran.
  - BM25 k1/b: Anserini's msmarco defaults are 0.9/0.4; its tuned setting for
    this corpus is 0.82/0.68. Lucene's own default is 1.2/0.75. They disagree,
    so an untagged cross-engine NDCG comparison means nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from baselines import append_manifest
from harness.datasets import DATA_DIR, QUERY_SETS

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / ".tools"
JAVA = TOOLS / "jdk-21.0.12+8" / "Contents" / "Home" / "bin" / "java"
FATJAR = TOOLS / "anserini-fatjar.jar"
INDEX_DIR = REPO_ROOT / "indexes" / "lucene-msmarco-passage"
JSONL_DIR = DATA_DIR / "msmarco-passage-jsonl"
RUNS_DIR = REPO_ROOT / "runs"

# 4g leaves room for the OS and Python on an 8GB machine; 4 threads rather than
# all 8 for the same reason.
HEAP = "-Xmx4g"
INDEX_THREADS = 4
ANALYZER = "EnglishAnalyzer (Porter stemming + stopwords), Anserini default"


def build_index(force: bool = False) -> dict:
    """Index the JSONL corpus. Positions/docvectors/raw are skipped: BM25 needs
    none of them, and each would cost several GB."""
    if INDEX_DIR.exists() and not force:
        print(f"index already present at {INDEX_DIR}")
        return {"skipped": True}

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(JAVA), HEAP, "-cp", str(FATJAR),
        "io.anserini.index.IndexCollection",
        "-collection", "JsonCollection",
        "-input", str(JSONL_DIR),
        "-index", str(INDEX_DIR),
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", str(INDEX_THREADS),
    ]
    started = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - started
    return {
        "skipped": False,
        "build_seconds": elapsed,
        "index_bytes": sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file()),
        "threads": INDEX_THREADS,
        "heap": HEAP,
    }


def search(query_set: str, k1: float = 0.9, b: float = 0.4, hits: int = 1000) -> dict:
    """Run one query set and return the run path plus the config that produced it."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"lucene.{query_set}.k1_{k1}.b_{b}.txt"
    cmd = [
        str(JAVA), HEAP, "-cp", str(FATJAR),
        "io.anserini.search.SearchCollection",
        "-index", str(INDEX_DIR),
        "-topics", str(DATA_DIR / "queries" / f"{query_set}.tsv"),
        "-topicReader", "TsvString",
        "-output", str(run_path),
        "-bm25", "-bm25.k1", str(k1), "-bm25.b", str(b),
        "-hits", str(hits),
        "-threads", "4",
    ]
    started = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - started

    return {
        "engine": "lucene-anserini-1.0.0",
        "query_set": query_set,
        "run_path": str(run_path.relative_to(REPO_ROOT)),
        "k1": k1,
        "b": b,
        "hits": hits,
        "analyzer": ANALYZER,
        # Left at Anserini's default (false), which is what published numbers use.
        # It means Lucene's lossy 1-byte norm quantization for document length.
        # Phase 1's C++ index will use exact lengths, so this is the first thing
        # to suspect if our NDCG@10 lands within ~0.01 of Lucene's but not on it.
        "bm25_accurate_doc_lengths": False,
        "wall_seconds": elapsed,
        "note": "wall time includes JVM startup; see the loadgen runs for latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()

    index_stats = build_index(force=args.force_index)
    print(json.dumps(index_stats, indent=2))

    for query_set in QUERY_SETS:
        entry = search(query_set, k1=args.k1, b=args.b)
        append_manifest(entry)
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
