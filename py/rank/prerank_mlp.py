"""Pre-rank MLP: a small 3-input model (BM25 score, dense score, doc length)
trained on dev's sparse binary qrels (1 positive + 4 sampled negatives per
query, sampled from that query's top-50 WAND-scored candidates -- see
rank.encode_candidates.truncate_to_top_k and this phase's negative-sampling
Global Constraint -- with a judged-relevant doc), evaluated on dl19+dl20 --
reduces each query's up-to-1000 WAND candidates to the top ~100 by MLP score.

Training a 3-input MLP has no GPU dependency -- this runs identically on CPU
locally (for tests and smoke checks) and on Kaggle (at full dev/dl19/dl20
scale), unlike the cross-encoder harness which genuinely needs a GPU to be
fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rank.prerank_features import FeatureScaler, FeatureVector, build_feature_vector

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

NUM_NEGATIVES = 4
SEED = 0
TOP_K_SURVIVORS = 100


def build_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))


def load_dense_scores_and_lengths(
    path: Path,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    dense_scores: dict[tuple[str, str], float] = {}
    doc_lengths: dict[tuple[str, str], int] = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["qid"], row["docid"])
            dense_scores[key] = row["dense_score"]
            doc_lengths[key] = row["doc_length"]
    return dense_scores, doc_lengths


def build_training_examples(
    dev_run: dict[str, dict[str, float]],
    dev_qrels: dict[str, dict[str, int]],
    dense_scores: dict[tuple[str, str], float],
    doc_lengths: dict[tuple[str, str], int],
    seed: int = SEED,
    num_negatives: int = NUM_NEGATIVES,
) -> tuple[list[FeatureVector], list[float]]:
    rng = np.random.default_rng(seed)
    features: list[FeatureVector] = []
    labels: list[float] = []
    for qid, candidates in dev_run.items():
        # Defensive: restrict to docids we actually have features for. Callers
        # are expected to pass an already-truncated dev_run (see
        # rank.encode_candidates.truncate_to_top_k) so this is normally a
        # no-op, but it protects against a silent KeyError below if a caller
        # ever passes an untruncated run that outruns what was encoded.
        available = [d for d in candidates if (qid, d) in dense_scores]
        judgments = dev_qrels.get(qid, {})
        positives = [docid for docid in available if judgments.get(docid, 0) > 0]
        if not positives:
            continue
        positive_docid = positives[0]
        negative_pool = [d for d in available if d != positive_docid]
        sample_size = min(num_negatives, len(negative_pool))
        negative_docids = rng.choice(negative_pool, size=sample_size, replace=False)

        for docid, label in [(positive_docid, 1.0)] + [(d, 0.0) for d in negative_docids]:
            key = (qid, docid)
            features.append(
                build_feature_vector(
                    bm25_score=candidates[docid],
                    dense_score=dense_scores[key],
                    doc_length=doc_lengths[key],
                )
            )
            labels.append(label)
    return features, labels


def train_mlp(
    model: nn.Module,
    features: list[FeatureVector],
    labels: list[float],
    scaler: FeatureScaler,
    epochs: int = 50,
    lr: float = 0.01,
) -> None:
    scaled = torch.tensor([scaler.transform(f) for f in features], dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(scaled)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()


def score_candidates(
    model: nn.Module, scaler: FeatureScaler, query_features: dict[str, FeatureVector]
) -> dict[str, float]:
    model.eval()
    docids = list(query_features)
    scaled = torch.tensor([scaler.transform(query_features[d]) for d in docids], dtype=torch.float32)
    with torch.no_grad():
        logits = model(scaled).squeeze(1)
    return dict(zip(docids, (float(x) for x in logits)))


def survivors_for_query(
    model: nn.Module,
    scaler: FeatureScaler,
    candidates: dict[str, float],
    dense_scores: dict[tuple[str, str], float],
    doc_lengths: dict[tuple[str, str], int],
    qid: str,
    top_k: int = TOP_K_SURVIVORS,
) -> set[str]:
    query_features = {
        docid: build_feature_vector(
            bm25_score=bm25_score,
            dense_score=dense_scores[(qid, docid)],
            doc_length=doc_lengths[(qid, docid)],
        )
        for docid, bm25_score in candidates.items()
    }
    scores = score_candidates(model, scaler, query_features)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {docid for docid, _ in ranked[:top_k]}
