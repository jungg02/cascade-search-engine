# Phase 0 — BM25 baselines

Exit artifact for Phase 0. Every number here is reproducible from the
committed JSON in `bench/results/`; the run files themselves are gitignored
(they are large and regenerable).

## Metrics

| Engine | Query set | Queries | NDCG@10 | NDCG@100 | MRR@10 | R@100 | R@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| lucene-anserini-1.0.0 | dev | 6980 | 0.2301 | 0.2892 | 0.1855 | 0.6622 | 0.8575 |
| tantivy-0.26.0 | dev | 6980 | 0.2281 | 0.2880 | 0.1819 | 0.6698 | 0.8592 |
| lucene-anserini-1.0.0 | dl19 | 43 | 0.5121 | 0.5072 | 0.7138 | 0.4988 | 0.7539 |
| tantivy-0.26.0 | dl19 | 43 | 0.4812 | 0.4905 | 0.6612 | 0.4898 | 0.7594 |
| lucene-anserini-1.0.0 | dl20 | 54 | 0.4769 | 0.4910 | 0.6653 | 0.5623 | 0.7865 |
| tantivy-0.26.0 | dl20 | 54 | 0.4751 | 0.4776 | 0.6197 | 0.5559 | 0.7929 |

## Configuration

BM25 parameters and analyzers differ between engines and are not
interchangeable. A cross-engine NDCG gap is largely a preprocessing gap.

| Engine | Query set | k1 | b | Analyzer | Depth |
|---|---|---:|---:|---|---:|
| lucene-anserini-1.0.0 | dev | 0.9 | 0.4 | EnglishAnalyzer (Porter stemming + stopwords), Anserini default | 1000 |
| tantivy-0.26.0 | dev | 1.2 | 0.75 | en_stem (stemming, no stopword removal) | 1000 |
| lucene-anserini-1.0.0 | dl19 | 0.9 | 0.4 | EnglishAnalyzer (Porter stemming + stopwords), Anserini default | 1000 |
| tantivy-0.26.0 | dl19 | 1.2 | 0.75 | en_stem (stemming, no stopword removal) | 1000 |
| lucene-anserini-1.0.0 | dl20 | 0.9 | 0.4 | EnglishAnalyzer (Porter stemming + stopwords), Anserini default | 1000 |
| tantivy-0.26.0 | dl20 | 1.2 | 0.75 | en_stem (stemming, no stopword removal) | 1000 |

## Latency (Tantivy, open-loop)

Poisson arrivals at a fixed rate, Zipfian query popularity (s=1.0) over 6,980 dev queries, 4 workers, top-10. Latency is measured from the *scheduled* arrival, so queueing is included rather than hidden.

Each point is the median of 3 runs of 10s; the p99 spread across repeats is shown because a laptop under thermal throttle does not produce a repeatable tail.

| Offered QPS | Achieved QPS | p50 | p95 | p99 | p99.9 | p99 spread |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 9.8 | 38.86ms | 118.32ms | 138.58ms | 138.58ms | 128.72–143.89ms |
| 25 | 23.5 | 46.76ms | 97.99ms | 120.10ms | 149.93ms | 117.63–269.35ms |
| 40 | 40.8 | 40.81ms | 101.10ms | 115.76ms | 144.82ms | 113.23–210.64ms |
| 55 | 52.6 | 44.84ms | 107.39ms | 128.93ms | 164.43ms | 122.14–201.67ms |
| 70 | 68.2 | 82.79ms | 237.80ms | 332.46ms | 347.85ms | 271.05–402.50ms |
| 85 | 64.6 | 1305.47ms | 2916.39ms | 3026.90ms | 3087.05ms | 1661.91–3454.55ms |

Lucene per-query latency is deliberately absent: driving Anserini one query
at a time means a JVM subprocess per request, which would measure JVM startup.
It is deferred to Phase 2, where the index sits behind a persistent server.

**Capacity: sustains 70 QPS at p99 < 332ms on 4 workers, top-10, no cache and no sharding.** At 85 QPS the server stops keeping up: achieved throughput falls below offered, the queue outlives the arrival window, and p50 crosses from tens of milliseconds into seconds.

A closed-loop generator could not have produced that knee. It only sends the next request after the previous one returns, so it throttles itself to the service rate and reports a flat, healthy-looking latency at every offered rate — the queue never forms, so it never shows up.

## Metric validation

Maximum absolute difference between `harness.metrics` and `pytrec_eval` across every run above: **1.11e-16**.

Relevance thresholds follow the TREC-DL convention: NDCG uses the graded
judgments, while MRR and recall count `rel >= 2` as relevant on DL19/DL20.
MS MARCO dev qrels are binary, so the threshold there is 1.

### External check

Agreeing with `pytrec_eval` only proves the metric arithmetic. The check that
covers qrels loading, topic formatting, and run parsing is comparing our Lucene
numbers against Anserini's own published regressions for the same engine and the
same parameters:

| Query set | Metric | Anserini published | Ours | Δ |
|---|---|---:|---:|---:|
| DL19 | nDCG@10 | 0.5058 | 0.5121 | +0.0063 |
| DL20 | nDCG@10 | 0.4796 | 0.4769 | −0.0027 |
| dev  | RR@10   | 0.1840 | 0.1855 | +0.0015 |
| dev  | R@1000  | 0.8526 | 0.8575 | +0.0049 |

Source: `docs/regressions/regressions-{dl19,dl20}-passage.md` and
`regressions-msmarco-v1-passage.md` at tag `anserini-1.0.0`, BM25 (default) column.

The residual is small and one-directional-ish rather than random, which is what a
corpus difference looks like: `ir_datasets` applies encoding fixes to MS MARCO
passages that Anserini's own conversion script does not. Both engines here read
the `ir_datasets` text, so the comparison *within* this table is clean; the
comparison to Anserini's published numbers carries that one known difference.

## Known limitations

- Lucene latency is not measured here (see above); only Tantivy is under load.
- The Tantivy latency sweep uses top-10, while the run files that produced the
  metrics above use top-1000. They are different operating points on purpose:
  depth 1000 through the Python binding is dominated by stored-field fetches
  rather than by the engine.
- Single machine, 4 workers, no sharding and no cache. Capacity and tail
  behaviour at realistic fan-out are Phase 2's subject, not this document's.
