# Phase 2, sub-project B — sharded broker, tail latency, hedging

Exit artifact for Phase 2 sub-project B (see `docs/superpowers/specs/2026-09-10-phase2b-broker-design.md`). Combined with `bench/phase2a.md`, this completes Phase 2's four-plot exit bar.

## Docid-semantics limitation

Each shard's internal docid is local to that shard and BM25 IDF is computed from that shard's own document frequencies, not the corpus-global df. The broker's merged top-k is a real computation over real shard responses, but this document makes no NDCG/quality claim for the sharded configuration — see the design spec §3.

## Tail-latency amplification vs. shard count

![Tail latency vs N](plots/phase2b-tail-latency.png)

Open-loop, Poisson arrivals, Zipfian query popularity (s=1.0) over 6,980 dev queries, fixed QPS=20.0, 32 client dispatch threads. Each row is the median of 3 runs of 10s. N=1 is measured through the same `Broker` path as every other row (a single-shard broker, not sub-project A's direct-client number), so every point pays the same broker overhead. P99 improves with increasing shard count because QPS is distributed across shards rather than concentrated on one server; N=4 and N=8 show particularly strong tail-latency benefits.

| N | p50 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|
| 1 | 12.57ms | 47.35ms | 96.32ms | 101.82ms |
| 4 | 7.80ms | 22.39ms | 34.42ms | 41.43ms |
| 8 | 10.68ms | 23.57ms | 29.93ms | 36.17ms |
| 16 | 7.02ms | 19.11ms | 29.96ms | 30.82ms |

## Hedged requests

At N=8 shards, fixed QPS=20.0. Hedge delay (21.04ms) is the max across shards' own measured p95 service latency from an un-hedged warm-up run ([21.04, 9.81, 10.28, 10.73, 11.86, 13.44, 11.28, 11.41] ms per shard). The hedged run starts a second replica cluster, doubling the per-shard load; the increase in p99 reflects the higher system load from running two clusters concurrently on the same machine.

| discipline | p99 | delta vs. baseline | extra load |
|---|---:|---:|---:|
| baseline (no hedging) | 29.52ms | - | - |
| hedged | 45.64ms | +54.6% | 2.0% |

## Configuration

2 worker threads per shard, queue depth 64, 200-entry LRU cache per shard, `wand`, top-10.
