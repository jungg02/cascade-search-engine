# Phase 2, sub-project A — single-node gRPC server

Exit artifact for Phase 2 sub-project A (see `docs/superpowers/specs/2026-08-15-phase2-server-design.md`). `bench/results/server-throughput-knee.json` and `bench/results/server-cache-sensitivity.json` are committed; the built index is not (see README's Phase 1 setup). Sharding, the broker, and hedged requests are sub-project B.

## Server configuration

4 worker threads, queue depth 64, 200-entry LRU cache, `wand` (Phase 1 measured WAND faster than BlockMax-WAND in wall-clock on this corpus at k=10 despite scoring more postings — see `bench/phase1.md` — so it's the server default), top-10.

## Throughput knee

Open-loop, Poisson arrivals, Zipfian query popularity (s=1.0) over 6,980 dev queries, 32 client dispatch threads. Each point is the median of 3 runs of 10s.

| Offered QPS | Achieved QPS | p50 | p95 | p99 | p99.9 | errors (median) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 19.7 | 3.57ms | 29.65ms | 51.53ms | 70.22ms | 0 |
| 100 | 99.3 | 1.88ms | 16.07ms | 34.22ms | 69.79ms | 0 |
| 200 | 198.6 | 2.05ms | 14.48ms | 28.07ms | 59.32ms | 0 |
| 400 | 398.5 | 3.17ms | 16.47ms | 29.42ms | 67.22ms | 0 |
| 800 | 798.2 | 12.49ms | 45.28ms | 113.63ms | 146.68ms | 0 |
| 1200 | 1001.9 | 1574.54ms | 2492.29ms | 2537.65ms | 2571.22ms | 0 |

**Capacity: sustains at least 800 QPS at p99 < 114ms** with 4 worker threads, queue depth 64, 50% cache hit rate at this point (this sweep's own traffic, s=1.0, not borrowed from the cache-sensitivity sweep below). The server's own queue wait stayed near zero at this point (0.03ms vs. 8.8ms service time), so this is a measured floor on server capacity, not a located ceiling — the sweep's fixed QPS grid didn't test closely enough above it to say where the server's own limit actually sits.
"Sustains N QPS" means N is the highest offered rate tested *before* the point where `find_knee`'s criterion trips (achieved throughput falls below offered and wall time outlives the arrival window) — not the highest point under some fixed latency bound. A lower-QPS row can still show a higher p99 than the sustained row on ordinary run-to-run variance; that's not a contradiction, just noise at a point below the knee.
At 1200 QPS the server stops keeping up: achieved throughput falls below offered and the run's wall time outlives the arrival window — but almost all of that growth is on the client side: median time waiting for a free client dispatch thread (1547.1ms, out of 32 client workers) dwarfs both server-side queue wait (5.87ms) and service time (26.7ms). This sweep's load-generator harness is what saturates at this point, not necessarily the server — its own true ceiling may sit above 1200 QPS; see Known limitations.

## Cache sensitivity

Fixed QPS (20.0, well below the sustained-QPS region found above so this reflects cache behavior rather than queueing) across a sweep of the Zipfian skew. Cache-hit and cache-miss latency are reported separately — a blended number would hide which one actually matters.

| Zipf s | hit rate | hits (n) | misses (n) | hit p50 | hit p99 | miss p50 | miss p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 3.7% | 7 | 183 | 0.46ms | 0.91ms | 9.09ms | 102.84ms |
| 1.0 | 30.0% | 57 | 133 | 0.46ms | 1.41ms | 9.71ms | 77.44ms |
| 1.5 | 74.2% | 141 | 49 | 0.61ms | 4.40ms | 10.10ms | 49.26ms |
| 2.0 | 88.4% | 168 | 22 | 1.01ms | 7.81ms | 12.99ms | 41.73ms |

## Known limitations

- Single server process, single machine. Sharding and its own
  tail-latency amplification are sub-project B's subject, not this
  document's.
- The gRPC layer's own thread pool is bounded via `ResourceQuota`
  (workers + queue depth + a small fixed headroom, currently 8, so
  that BoundedQueue's own "queue full" rejection is reachable via
  real network traffic instead of always being preempted by gRPC's
  coarser admission control — see main.cc's ResourceQuota comment)
  so it can't grow unbounded, but the fixed worker pool and bounded
  queue are the actual concurrency control — see the design spec for
  why.
- The load generator's own client-side dispatch pool (32 threads, one blocking `dispatch()` call per thread) is itself a finite-capacity queueing system. It's
  sized well above the server's configured concurrency so it doesn't
  become the bottleneck across most of this sweep, but at the very
  top of the QPS range tested here it can still saturate first —
  `server_queue_wait_us` (server-side) staying low while
  `queue_delay` (client-side) explodes is the signature, and the
  throughput-knee section above calls this out explicitly when it's
  what the data shows. Going higher than the current client-workers
  value risks a different artifact: CPU contention with the
  server's own worker threads, since client and server share one
  machine here. A production load test would run the client on
  separate hardware from the server.
