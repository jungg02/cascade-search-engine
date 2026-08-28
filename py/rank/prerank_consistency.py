"""Prerank-consistency: of the true top-k under the full ranker, how many
survive pre-ranking? A low number means the cascade discards good documents
before the expensive model ever sees them -- same shape as
dense/ann_sweep.py's recall_at_k, one level up the cascade.
"""

from __future__ import annotations


def prerank_consistency(true_top_k: list[str], survivors: set[str]) -> float:
    if not true_top_k:
        return 0.0
    found = sum(1 for docid in true_top_k if docid in survivors)
    return found / len(true_top_k)
