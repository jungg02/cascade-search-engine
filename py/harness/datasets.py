"""Corpus and judgment loading, normalized to `(doc_id, text)` and `(qid, text)`.

`IR_DATASETS_HOME` is pinned to `data/ir_datasets` inside the repo rather than
left at its `~/.ir_datasets` default, so the multi-GB cache is visible, gitignored,
and recorded in the run config.

Dataset choices worth knowing:
  - `dev/small` (6,980 queries) is the split every published MS MARCO MRR@10 uses.
    The full dev split is a different, larger set and gives a different number.
  - The `/judged` variants of DL19/DL20 restrict to queries that actually have
    judgments (43 and 54), which is what the TREC-DL results tables report.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
os.environ.setdefault("IR_DATASETS_HOME", str(DATA_DIR / "ir_datasets"))

import ir_datasets  # noqa: E402  (must follow IR_DATASETS_HOME)

# TREC-DL evaluates NDCG on the graded qrels but binarizes at rel >= 2 for the
# binary measures (MRR, recall, MAP). MS MARCO dev qrels are already binary.
QUERY_SETS = {
    "dl19": ("msmarco-passage/trec-dl-2019/judged", 2),
    "dl20": ("msmarco-passage/trec-dl-2020/judged", 2),
    "dev": ("msmarco-passage/dev/small", 1),
}
CORPUS = "msmarco-passage"


def rel_threshold(query_set: str) -> int:
    """Relevance level at which a judgment counts as positive for MRR/recall."""
    return QUERY_SETS[query_set][1]


def load_queries(query_set: str) -> dict[str, str]:
    dataset = ir_datasets.load(QUERY_SETS[query_set][0])
    return {q.query_id: q.text for q in dataset.queries_iter()}


def load_qrels(query_set: str) -> dict[str, dict[str, int]]:
    dataset = ir_datasets.load(QUERY_SETS[query_set][0])
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels


def iter_docs(limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Stream the passage corpus. 8.8M passages — never materialize this."""
    dataset = ir_datasets.load(CORPUS)
    for i, doc in enumerate(dataset.docs_iter()):
        if limit is not None and i >= limit:
            return
        yield doc.doc_id, doc.text


def doc_count() -> int:
    return ir_datasets.load(CORPUS).docs_count()
