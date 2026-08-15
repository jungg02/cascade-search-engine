"""The report generator does its own averaging over pytrec_eval's per-query
output. That averaging has to use the same denominator as harness.metrics, or
the self-check in baselines.md would report a difference that is an artifact of
the check rather than of the metrics."""

import random

from baselines.report import _pytrec_eval_check, render_markdown
from harness.metrics import evaluate


def _qrels_and_run(seed: int):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(8):
        qid = f"q{q}"
        qrels[qid] = {f"d{d}": rng.choice([0, 1, 2, 3]) for d in rng.sample(range(300), 40)}
        run[qid] = {f"d{d}": float(rng.randint(0, 500)) for d in rng.sample(range(300), 120)}
    return qrels, run


def test_pytrec_eval_cross_check_agrees_with_our_metrics_on_a_realistic_run():
    qrels, run = _qrels_and_run(5)

    ours = evaluate(qrels, run, ndcg_k=(10, 100), mrr_k=(10,), recall_k=(100, 1000),
                    rel_threshold=2)
    reference = _pytrec_eval_check(qrels, run, threshold=2)

    for metric, value in reference.items():
        assert abs(ours.mean[metric] - value) < 1e-4


def test_cross_check_uses_the_qrels_count_as_denominator_for_unanswered_queries():
    qrels, run = _qrels_and_run(6)
    run.pop("q0")  # engine returned nothing for this query

    ours = evaluate(qrels, run, ndcg_k=(10,), mrr_k=(), recall_k=(), rel_threshold=1)
    reference = _pytrec_eval_check(qrels, run, threshold=1)

    assert abs(ours.mean["ndcg_cut_10"] - reference["ndcg_cut_10"]) < 1e-4


def test_markdown_reports_the_configuration_alongside_every_metric():
    result = {
        "engine": "lucene", "query_set": "dl19", "num_queries": 43, "run_depth": 1000,
        "metrics": {"ndcg_cut_10": 0.5, "ndcg_cut_100": 0.4, "mrr_10": 0.9,
                    "recall_100": 0.6, "recall_1000": 0.7},
        "config": {"k1": 0.9, "b": 0.4, "analyzer": "EnglishAnalyzer"},
        "pytrec_eval_agreement": {"max_abs_difference": 1e-9},
    }

    markdown = render_markdown([result])

    assert "0.5000" in markdown
    assert "EnglishAnalyzer" in markdown
    assert "0.9" in markdown and "0.4" in markdown
