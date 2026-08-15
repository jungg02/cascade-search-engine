"""Retrieval metrics, implemented to match trec_eval exactly.

Conventions taken from trec_eval (and therefore pytrec_eval), each verified
against pytrec_eval in py/tests/test_metrics.py:
  - gain is *linear* in the judgment (gain = rel), not 2^rel - 1. Much of the IR
    literature uses the exponential gain; trec_eval does not, and Anserini's
    published NDCG@10 numbers come from trec_eval. Matching trec_eval is what
    makes our numbers comparable to theirs.
  - discount is 1 / log2(rank + 1), ranks starting at 1
  - IDCG is the ideal ordering of every judged document for that query, cut at k
  - documents in the run with no judgment are treated as rel = 0
  - ties in the run are broken by docid in *descending* lexicographic order
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


def _ranked_docids(doc_scores: dict[str, float]) -> list[str]:
    """Run order under trec_eval's tie-breaking: score desc, then docid desc."""
    return [
        docid
        for docid, _ in sorted(doc_scores.items(), key=lambda kv: (-kv[1], _NegStr(kv[0])))
    ]


class _NegStr:
    """Sort key that reverses string ordering, to mirror trec_eval's tie-break."""

    __slots__ = ("s",)

    def __init__(self, s: str) -> None:
        self.s = s

    def __lt__(self, other: "_NegStr") -> bool:
        return self.s > other.s


def ndcg_at_k(qrels: Qrels, run: Run, k: int) -> dict[str, float]:
    """NDCG@k per query, over every query present in qrels."""
    scores: dict[str, float] = {}
    for qid, judgments in qrels.items():
        ideal = sorted((rel for rel in judgments.values() if rel > 0), reverse=True)[:k]
        idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
        if idcg == 0:
            scores[qid] = 0.0
            continue

        dcg = 0.0
        for rank, docid in enumerate(_ranked_docids(run.get(qid, {}))[:k], start=1):
            rel = judgments.get(docid, 0)
            if rel > 0:
                dcg += rel / math.log2(rank + 1)
        scores[qid] = dcg / idcg
    return scores


def mrr_at_k(qrels: Qrels, run: Run, k: int, rel_threshold: int = 1) -> dict[str, float]:
    """Reciprocal rank of the first relevant document within the top k, per query."""
    scores: dict[str, float] = {}
    for qid, judgments in qrels.items():
        scores[qid] = 0.0
        for rank, docid in enumerate(_ranked_docids(run.get(qid, {}))[:k], start=1):
            if judgments.get(docid, 0) >= rel_threshold:
                scores[qid] = 1.0 / rank
                break
    return scores


def recall_at_k(qrels: Qrels, run: Run, k: int, rel_threshold: int = 1) -> dict[str, float]:
    """Fraction of a query's relevant documents that appear in the top k."""
    scores: dict[str, float] = {}
    for qid, judgments in qrels.items():
        relevant = {d for d, rel in judgments.items() if rel >= rel_threshold}
        if not relevant:
            scores[qid] = 0.0
            continue
        retrieved = _ranked_docids(run.get(qid, {}))[:k]
        scores[qid] = sum(1 for d in retrieved if d in relevant) / len(relevant)
    return scores


@dataclass
class EvalResult:
    """Per-query and averaged metrics, plus the context needed to trust them.

    `run_depth` and `shallow_metrics` exist because a cutoff deeper than the run
    silently reports a shallower metric under a deeper name.
    """

    mean: dict[str, float]
    per_query: dict[str, dict[str, float]]
    run_depth: int
    num_queries: int
    rel_threshold: int
    shallow_metrics: list[str] = field(default_factory=list)


def evaluate(
    qrels: Qrels,
    run: Run,
    *,
    ndcg_k: tuple[int, ...] = (10, 100),
    mrr_k: tuple[int, ...] = (10,),
    recall_k: tuple[int, ...] = (100, 1000),
    rel_threshold: int = 1,
) -> EvalResult:
    """Evaluate a run over every query in qrels.

    Queries in qrels but absent from the run score 0 rather than being dropped,
    matching `trec_eval -c`. `rel_threshold` applies to MRR and recall only; NDCG
    always uses the graded judgments, which is the TREC-DL convention.
    """
    per_query: dict[str, dict[str, float]] = {}
    for k in ndcg_k:
        per_query[f"ndcg_cut_{k}"] = ndcg_at_k(qrels, run, k)
    for k in mrr_k:
        per_query[f"mrr_{k}"] = mrr_at_k(qrels, run, k, rel_threshold)
    for k in recall_k:
        per_query[f"recall_{k}"] = recall_at_k(qrels, run, k, rel_threshold)

    run_depth = max((len(docs) for docs in run.values()), default=0)
    shallow = [
        name
        for name, k in [(f"ndcg_cut_{k}", k) for k in ndcg_k]
        + [(f"mrr_{k}", k) for k in mrr_k]
        + [(f"recall_{k}", k) for k in recall_k]
        if k > run_depth
    ]

    return EvalResult(
        mean={
            name: (sum(scores.values()) / len(scores) if scores else 0.0)
            for name, scores in per_query.items()
        },
        per_query=per_query,
        run_depth=run_depth,
        num_queries=len(qrels),
        rel_threshold=rel_threshold,
        shallow_metrics=shallow,
    )
