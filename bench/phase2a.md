# Phase 2, sub-project A — single-node gRPC server

Exit artifact for Phase 2 sub-project A (see `docs/superpowers/specs/2026-08-15-phase2-server-design.md`). `bench/results/server-throughput-knee.json` and `bench/results/server-cache-sensitivity.json` are committed; the built index is not (see README's Phase 1 setup). Sharding, the broker, and hedged requests are sub-project B.

## Server configuration

4 worker threads, queue depth 64, 200-entry LRU cache, `wand` (Phase 1 measured WAND faster than BlockMax-WAND in wall-clock on this corpus at k=10 despite scoring more postings — see `bench/phase1.md` — so it's the server default), top-10.

## Throughput knee

Open-loop, Poisson arrivals, Zipfian query popularity (s=1.0) over 6,980 dev queries, 8 client dispatch threads. Each point is the median of 3 runs of 10s.

| Offered QPS | Achieved QPS | p50 | p95 | p99 | p99.9 | errors (median) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 21.2 | 4.38ms | 40.64ms | 72.93ms | 107.42ms | 0 |
| 100 | 96.7 | 2.32ms | 21.15ms | 39.80ms | 87.24ms | 0 |
| 200 | 198.6 | 2.01ms | 14.15ms | 27.80ms | 62.43ms | 0 |
| 400 | 401.3 | 5.73ms | 83.46ms | 130.11ms | 167.60ms | 0 |
| 800 | 796.2 | 14.09ms | 64.74ms | 86.83ms | 108.74ms | 0 |
| 1200 | 917.9 | 3257.95ms | 3989.27ms | 4063.62ms | 4088.17ms | 0 |

**Capacity: sustains 800 QPS at p99 < 87ms** with 4 worker threads, queue depth 64, 52% median cache hit rate across the cache-sensitivity sweep.
At 1200 QPS the server stops keeping up: achieved throughput falls below offered and the queue outlives the arrival window.

## Cache sensitivity

Fixed QPS (20.0, chosen below the throughput knee above so this reflects cache behavior rather than queueing) across a sweep of the Zipfian skew. Cache-hit and cache-miss latency are reported separately — a blended number would hide which one actually matters.

| Zipf s | hit rate | hit p50 | hit p99 | miss p50 | miss p99 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 3.7% | 0.37ms | 0.47ms | 10.35ms | 123.41ms |
| 1.0 | 30.0% | 0.43ms | 4.02ms | 9.04ms | 82.40ms |
| 1.5 | 74.2% | 0.86ms | 5.00ms | 10.29ms | 46.85ms |
| 2.0 | 88.4% | 0.81ms | 2.54ms | 13.94ms | 39.11ms |

## Known limitations

- Single server process, single machine. Sharding and its own
  tail-latency amplification are sub-project B's subject, not this
  document's.
- The gRPC layer's own thread pool is bounded via `ResourceQuota`
  (workers + queue depth) so it can't grow unbounded, but the fixed
  worker pool and bounded queue are the actual concurrency control —
  see the design spec for why.
