"""Tantivy BM25 baseline.

Tantivy is the secondary baseline. Two differences from Anserini that have to be
stated wherever the numbers appear, because they are not tuning choices — they
are not adjustable:
  - BM25 k1/b are fixed at Lucene's classic 1.2/0.75, not Anserini's 0.9/0.4.
  - the `en_stem` tokenizer stems but does *not* remove stopwords, where
    Anserini's EnglishAnalyzer does both.

So a Tantivy/Anserini NDCG gap is a preprocessing and parameter gap, not
evidence that one engine retrieves better. The point of including it is a second
independent implementation of BM25 to sanity-check the first.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import tantivy

from baselines import append_manifest
from harness.datasets import QUERY_SETS, iter_docs, load_queries
from harness.histogram import LatencyRecorder
from harness.runfile import write_run

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "indexes" / "tantivy-msmarco-passage"
RUNS_DIR = REPO_ROOT / "runs"

K1, B = 1.2, 0.75
TOKENIZER = "en_stem"
# The query parser treats +, -, ", :, (, ) and friends as syntax. MS MARCO
# queries are natural language and contain them, so strip to a bag of words
# rather than letting a stray colon turn into a field lookup.
_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def sanitize(query: str) -> str:
    return _NON_WORD.sub(" ", query.lower()).strip()


def _schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("contents", stored=False, tokenizer_name=TOKENIZER)
    return builder.build()


def build_index(force: bool = False, limit: int | None = None) -> dict:
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()) and not force:
        print(f"index already present at {INDEX_DIR}")
        return {"skipped": True}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(_schema(), path=str(INDEX_DIR))
    writer = index.writer(heap_size=1_000_000_000, num_threads=4)

    started = time.perf_counter()
    count = 0
    for doc_id, text in iter_docs(limit=limit):
        writer.add_document(tantivy.Document(doc_id=doc_id, contents=text))
        count += 1
        if count % 1_000_000 == 0:
            print(f"  {count:,} docs", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    elapsed = time.perf_counter() - started

    return {
        "skipped": False,
        "docs": count,
        "build_seconds": elapsed,
        "index_bytes": sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file()),
    }


def search(query_set: str, hits: int = 1000) -> dict:
    index = tantivy.Index.open(str(INDEX_DIR))
    index.reload()
    searcher = index.searcher()
    queries = load_queries(query_set)

    latency = LatencyRecorder()
    run: dict[str, dict[str, float]] = {}
    for qid, text in queries.items():
        cleaned = sanitize(text)
        if not cleaned:
            run[qid] = {}
            continue
        started = time.perf_counter_ns()
        parsed = index.parse_query(cleaned, ["contents"])
        result = searcher.search(parsed, hits)
        latency.record((time.perf_counter_ns() - started) / 1000)
        run[qid] = {
            searcher.doc(address)["doc_id"][0]: float(score)
            for score, address in result.hits
        }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"tantivy.{query_set}.txt"
    write_run(run, run_path, tag="tantivy", depth=hits)

    return {
        # tantivy.__version__ is a prose string ("tantivy v0.26.0, index_format v7"),
        # and this value becomes a filename. Keep the full string as metadata.
        "engine": "tantivy-0.26.0",
        "engine_version_string": getattr(tantivy, "__version__", "unknown"),
        "query_set": query_set,
        "run_path": str(run_path.relative_to(REPO_ROOT)),
        "k1": K1,
        "b": B,
        "hits": hits,
        "analyzer": f"{TOKENIZER} (stemming, no stopword removal)",
        # Single-threaded, serial, no load applied. This is a per-query cost, not
        # a capacity measurement; Phase 2 does capacity properly under load.
        "serial_latency_us": latency.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(json.dumps(build_index(force=args.force_index, limit=args.limit), indent=2))
    for query_set in QUERY_SETS:
        entry = search(query_set)
        append_manifest(entry)
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
