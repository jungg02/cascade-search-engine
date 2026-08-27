# Phase 4: Ranking Cascade and Heterogeneous Serving

> **⚠️ STALE — every number below is known wrong, pending a re-run.** This
> report was rendered before a fix wave corrected two defects in how it was
> measured: (1) the cross-encoder was exported without `token_type_ids`,
> silently zeroing its query/passage segment signal, so every NDCG@10 and
> prerank-consistency number below reflects a cascade *worse than the BM25
> first-stage baseline it's meant to improve on* — not this phase's real
> result; (2) the batching/queue-discipline harness was effectively
> closed-loop, so `shed_count: 0` across every row below is an artifact of a
> load generator that never reached the offered rate, not a genuine "no
> shedding needed" finding. Both are fixed in code; this file will be
> regenerated (`PYTHONPATH=py uv run python -m rank.report`) once the
> corrected pipeline has been re-run on Kaggle and fresh `bench/results/
> rank-*.json` files are downloaded.

## Execution provider check

All sub-experiments ran on `CUDAExecutionProvider` as requested.

## Candidate pool (limitation, stated up front)

The pre-rank/rank cascade runs over Phase 1's existing lexical WAND
candidates (depth 1000, full 8.8M-passage corpus) — not fused with Phase 3's
dense recall channel. Dense score is used only as a pre-rank *feature*, not
as a separate recall channel.

## Prerank-consistency

Of the true top-100 under the full cross-encoder
ranker, **18.5%** survive pre-ranking
(mean over 97 dl19+dl20 queries). Full-cascade result:
NDCG@10 = 0.3593, recall@100 = 0.6400.

## Dynamic batching: throughput vs. p99 latency

![Batching plot](plots/phase4-batching.png)

| max_batch_size | max_wait_ms | throughput (qps) | p50 (ms) | p99 (ms) |
|---|---|---|---|---|
| 32 | 20 | 664.0 | 17.25 | 27.33 |
| 8 | 20 | 601.3 | 9.44 | 13.61 |
| 32 | 5 | 537.0 | 6.99 | 9.77 |
| 8 | 5 | 528.3 | 7.09 | 9.81 |
| 1 | 5 | 316.2 | 3.09 | 3.76 |
| 1 | 0 | 313.8 | 3.11 | 3.94 |
| 1 | 20 | 312.4 | 3.11 | 4.31 |
| 8 | 0 | 312.4 | 3.13 | 3.92 |
| 32 | 0 | 307.4 | 3.17 | 3.95 |

## Precision comparison (at the best-throughput batching setting)

| precision | throughput (qps) | p99 (ms) | cascade NDCG@10 | delta vs. fp32 |
|---|---|---|---|---|
| fp32 | 665.3 | 27.27 | 0.3594 | +0.0000 |
| fp16 | 703.2 | 25.21 | 0.3593 | +0.0000 |
| int8 | 350.3 | 62.48 | 0.3617 | +0.0024 |

## Queue discipline: load-shedding vs. unbounded

| discipline | arrival rate (qps) | p99 (ms) | shed count |
|---|---|---|---|
| shed | 996.0 | 26.35 | 0 |
| shed | 1328.0 | 26.72 | 0 |
| shed | 1992.0 | 26.61 | 0 |
| unbounded | 996.0 | 27.08 | 0 |
| unbounded | 1328.0 | 27.24 | 0 |
| unbounded | 1992.0 | 27.14 | 0 |

## Configuration

- model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- git SHA: `cde154ec38ae4b94ac4b78eb98aec5e5173246ae`
- GPU: Tesla T4
- timestamp: 2026-08-27T10:56:38.598880+00:00
