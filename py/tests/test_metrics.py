"""The eval harness is only credible if it agrees with trec_eval.

Phase 1 compares our C++ index's NDCG@10 against Lucene's to within ~0.01, so a
systematic bias in the metric code would silently invalidate that check. These
tests pin our implementation to pytrec_eval (the trec_eval C code) to 4 decimals.
"""

import math
import random

import pytest
import pytrec_eval

from harness.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def _random_qrels_and_run(seed: int, *, run_depth: int = 50):
    """Judged and unjudged docs, graded relevance, score ties — the messy cases."""
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(12):
        qid = f"q{q}"
        qrels[qid] = {f"d{d}": rng.choice([0, 0, 1, 2, 3]) for d in rng.sample(range(200), 30)}
        # Coarse scores so ties are common; tie-breaking has to match trec_eval.
        run[qid] = {
            f"d{d}": float(rng.randint(0, 8))
            for d in rng.sample(range(200), run_depth)
        }
    return qrels, run


def _pytrec_eval(qrels, run, measures, relevance_level=1):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures, relevance_level=relevance_level)
    return evaluator.evaluate(run)


def test_ndcg_at_10_matches_hand_computed_value():
    # Two relevant docs retrieved at ranks 1 and 3, one relevant doc missed.
    # trec_eval gain is linear in rel; discount is 1/log2(rank + 1).
    qrels = {"q1": {"d1": 3, "d2": 1, "d3": 2}}
    run = {"q1": {"d1": 5.0, "dX": 4.0, "d2": 3.0}}

    dcg = 3 / math.log2(2) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)

    assert ndcg_at_k(qrels, run, 10)["q1"] == pytest.approx(dcg / idcg, abs=1e-9)


def test_ndcg_uses_linear_gain_not_exponential_gain():
    # The distinguishing case: a rel=1 doc ranked above a rel=3 doc. Exponential
    # gain (2^rel - 1) gives 0.7098 here; trec_eval gives 0.7967.
    qrels = {"q1": {"lo": 1, "hi": 3}}
    run = {"q1": {"lo": 2.0, "hi": 1.0}}

    assert ndcg_at_k(qrels, run, 2)["q1"] == pytest.approx(0.796708, abs=1e-6)


def test_ndcg_is_zero_when_query_has_no_relevant_documents():
    qrels = {"q1": {"d1": 0}}
    run = {"q1": {"d1": 5.0}}

    assert ndcg_at_k(qrels, run, 10)["q1"] == 0.0


@pytest.mark.parametrize("k", [10, 100])
@pytest.mark.parametrize("seed", range(5))
def test_ndcg_agrees_with_pytrec_eval(seed, k):
    qrels, run = _random_qrels_and_run(seed)
    expected = _pytrec_eval(qrels, run, {f"ndcg_cut.{k}"})

    ours = ndcg_at_k(qrels, run, k)

    for qid in qrels:
        assert ours[qid] == pytest.approx(expected[qid][f"ndcg_cut_{k}"], abs=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_mrr_at_10_agrees_with_pytrec_eval_on_a_truncated_run(seed):
    # trec_eval has no recip_rank cutoff, so MRR@10 is recip_rank over a run
    # truncated to depth 10 — the same trick the MS MARCO eval script uses.
    qrels, run = _random_qrels_and_run(seed)
    # reverse=True on (score, docid) gives score desc with ties broken by docid
    # desc — the order trec_eval itself uses, so the truncation is the same top 10.
    truncated = {
        qid: dict(sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
        for qid, docs in run.items()
    }
    expected = _pytrec_eval(qrels, truncated, {"recip_rank"})

    ours = mrr_at_k(qrels, run, 10)

    for qid in qrels:
        assert ours[qid] == pytest.approx(expected[qid]["recip_rank"], abs=1e-4)


@pytest.mark.parametrize("relevance_level", [1, 2])
@pytest.mark.parametrize("seed", range(5))
def test_recall_agrees_with_pytrec_eval_at_each_relevance_level(seed, relevance_level):
    # TREC-DL convention: NDCG uses graded qrels, but recall and MRR count
    # rel >= 2 as relevant. Getting this wrong detaches us from published numbers.
    qrels, run = _random_qrels_and_run(seed)
    expected = _pytrec_eval(qrels, run, {"recall.100"}, relevance_level=relevance_level)

    ours = recall_at_k(qrels, run, 100, rel_threshold=relevance_level)

    for qid in qrels:
        assert ours[qid] == pytest.approx(expected[qid]["recall_100"], abs=1e-4)


def test_recall_at_1000_reports_the_run_depth_it_actually_saw():
    # recall@1000 silently degenerates to recall@100 if the run is only 100 deep.
    # The result has to carry the depth so the report can't overstate it.
    qrels, run = _random_qrels_and_run(0, run_depth=100)

    result = evaluate(qrels, run, recall_k=(1000,))

    assert result.run_depth == 100


def test_evaluate_reports_the_mean_over_queries_present_in_qrels():
    qrels, run = _random_qrels_and_run(1)

    result = evaluate(qrels, run, ndcg_k=(10,), mrr_k=(), recall_k=())

    per_query = ndcg_at_k(qrels, run, 10)
    assert result.mean["ndcg_cut_10"] == pytest.approx(
        sum(per_query.values()) / len(per_query), abs=1e-9
    )


def test_evaluate_scores_zero_for_a_judged_query_missing_from_the_run():
    # trec_eval -c semantics: a query the engine failed to answer counts as 0,
    # it does not vanish from the average.
    qrels = {"q1": {"d1": 3}, "q2": {"d9": 3}}
    run = {"q1": {"d1": 1.0}}

    result = evaluate(qrels, run, ndcg_k=(10,), mrr_k=(), recall_k=())

    assert result.per_query["ndcg_cut_10"]["q2"] == 0.0
    assert result.mean["ndcg_cut_10"] == pytest.approx(0.5, abs=1e-9)
