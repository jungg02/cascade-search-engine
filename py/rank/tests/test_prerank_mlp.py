"""build_training_examples is pure given already-loaded run/qrels/feature
dicts. train_mlp/score_candidates are exercised on a tiny synthetic dataset
here (no GPU needed -- a 3-input MLP trains in milliseconds on CPU), which is
the same code path the Kaggle run uses at full scale."""

from __future__ import annotations

import torch

from rank.prerank_features import fit_scaler
from rank.prerank_mlp import (
    build_mlp,
    build_training_examples,
    score_candidates,
    train_mlp,
)


def test_build_training_examples_positive_plus_negatives():
    dev_run = {"q1": {"pos": 5.0, "neg1": 4.0, "neg2": 3.0, "neg3": 2.0, "neg4": 1.0, "neg5": 0.5}}
    dev_qrels = {"q1": {"pos": 1}}
    dense_scores = {("q1", d): 0.1 for d in dev_run["q1"]}
    doc_lengths = {("q1", d): 100 for d in dev_run["q1"]}
    features, labels = build_training_examples(
        dev_run, dev_qrels, dense_scores, doc_lengths, seed=0, num_negatives=4
    )
    assert len(features) == 5  # 1 positive + 4 negatives
    assert sum(labels) == 1.0  # exactly one positive label
    assert len(labels) == 5


def test_build_training_examples_skips_queries_with_no_qrels_positive():
    dev_run = {"q1": {"d1": 1.0}}
    dev_qrels = {}  # q1 has no judged-relevant doc at all
    dense_scores = {("q1", "d1"): 0.1}
    doc_lengths = {("q1", "d1"): 100}
    features, labels = build_training_examples(
        dev_run, dev_qrels, dense_scores, doc_lengths, seed=0, num_negatives=4
    )
    assert features == []
    assert labels == []


def test_train_and_score_tiny_synthetic_dataset():
    # A trivially separable dataset: feature value > 0 means relevant.
    features = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)]
    labels = [1.0, 1.0, 0.0, 0.0]
    scaler = fit_scaler(features)
    torch.manual_seed(0)  # before build_mlp(): seeds weight init, not just training
    model = build_mlp()
    train_mlp(model, features, labels, scaler, epochs=200)

    query_features = {"pos_doc": (1.0, 1.0, 1.0), "neg_doc": (-1.0, -1.0, -1.0)}
    scores = score_candidates(model, scaler, query_features)
    assert scores["pos_doc"] > scores["neg_doc"]
