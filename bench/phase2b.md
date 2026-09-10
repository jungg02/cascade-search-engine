# Phase 2, sub-project B — sharded broker, tail latency, hedging

Exit artifact for Phase 2 sub-project B (see `docs/superpowers/specs/2026-09-10-phase2b-broker-design.md`). Combined with `bench/phase2a.md`, this completes Phase 2's four-plot exit bar.

## Docid-semantics limitation

Each shard's internal docid is local to that shard and BM25 IDF is computed from that shard's own document frequencies, not the corpus-global df. The broker's merged top-k is a real computation over real shard responses, but this document makes no NDCG/quality claim for the sharded configuration — see the design spec §3.

## Tail-latency amplification vs. shard count

![Client-observed latency vs shard count](plots/phase2b-tail-latency.png)

Open-loop, Poisson arrivals, Zipfian query popularity (s=1.0) over 6,980 dev queries, fixed QPS=20.0, 32 client dispatch threads. Each row is the median of 3 runs of 10s. N=1 is measured through the same `Broker` path as every other row (a single-shard broker, not sub-project A's direct-client number), so every point pays the same broker overhead.

The `Broker.dispatch` method (`py/server/broker.py`) fans every query out to every shard concurrently and waits for all N to respond, so offered load per shard does NOT drop as N grows — each shard receives ~20 QPS regardless of shard count. The measured p99 improvement with increasing N reflects a different mechanism: the 8,841,823-passage corpus is split N ways, so each shard's local corpus shrinks to 1/N size. Smaller corpus means each shard's per-query WAND search does less work, dominating the classic "tail at scale" wait-for-slowest-of-N amplification effect that Dean & Barroso describe (design spec §1). At this corpus size, corpus-shrinkage wins over amplification; cleanly isolating amplification the way the design spec intended would require either a much larger corpus (where per-shard costs saturate and stop shrinking) or fixed-size replicas of the full index rather than document-partitioned shards. Additionally, at only 2 worker threads per shard and 20 QPS offered load, N=1's single full-corpus shard may experience queueing constraints that the cheaper N≥4 shards avoid, compounding the corpus-size effect. The result is architecturally sound and well-measured, but it demonstrates corpus-partitioning benefit under this specific load, not the tail-amplification phenomenon the experiment design intended to isolate.

| N | p50 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|
| 1 | 9.63ms | 47.35ms | 96.32ms | 101.82ms |
| 4 | 8.90ms | 25.55ms | 34.42ms | 52.79ms |
| 8 | 10.68ms | 23.57ms | 29.93ms | 37.02ms |
| 16 | 7.78ms | 20.96ms | 29.96ms | 34.34ms |

## Hedged requests

At N=8 shards, fixed QPS=20.0. Hedge delay (21.04ms) is the max across shards' own measured p95 service latency from an un-hedged warm-up run ([21.04, 9.81, 10.28, 10.73, 11.86, 13.44, 11.28, 11.41] ms per shard).

| discipline | p99 | delta vs. baseline | extra load (same repeat as p99) |
|---|---:|---:|---:|
| baseline (no hedging) | 29.52ms | - | - |
| hedged | 45.64ms | +54.6% | 2.7% |

The extra-load figure in the table is the backup-call ratio for the specific repeat whose p99 is shown, not a blend across repeats. Pooled across all three hedged repeats, 96 of 4776 shard calls sent a backup (2.0%) — lower than what the reported-p99 repeat itself experienced (2.7%), because one of the three hedged repeats sent zero backup calls at all (0 of 1568 shard calls; see the explanation below).

The p99 regression is not explained by the mere existence of 16 co-resident processes (8 primaries + 8 replicas): `queue_delay` p99 (client-side wait for a free dispatch thread) stays essentially flat between the three baseline repeats (3.6-8.8ms) and the three hedged repeats (4.8-5.7ms) — both ranges stay under 8.8ms, far smaller than the 16.1ms p99 gap between baseline and hedged — so the added latency lives in `service` time inside `dispatch()` (the shard round-trip itself), not client-side queueing. Direct evidence: one of the three hedged repeats sent zero backup calls (0 of 1568 shard calls) despite running with the identical 16 co-resident processes as the other two hedged repeats, and its service p99 (23.11ms) was faster than 2 of the 3 baseline repeats' own service p99s (19.81ms, 26.49ms, 37.53ms) — not systematically worse across the board, which is what pure process-residency contention would predict. The repeats that did trigger backups paid for it (50 backups out of 1520 shard calls, service p99 47.37ms; 46 backups out of 1688 shard calls, service p99 43.47ms) — this tracks whether hedging actually fired, not whether the replica processes merely existed. The more likely mechanism: the small fraction of requests that trigger a backup call are doing real extra work — a full WAND search on a replica process — on an 8-core machine already running 8 primary and 8 replica `server_bin` processes plus a 32-thread Python load generator. A second contributing factor is cache asymmetry: each `server_bin` process keeps its own independent LRU cache (`cpp/server/lru_cache.h`), and since replicas only ever see the traffic that actually gets hedged, their caches stay far colder than the primaries' (which see 100% of traffic) — bench/phase2a.md's own cache-sensitivity sweep at this same zipf_s shows roughly a 21x p50 latency gap between cache hits and misses, so a backup call is effectively racing a likely-warm primary from a likely-cold replica.

The design spec anticipated extending the capacity statement to "hedging cuts p99 by Z% at a W% extra-request cost" (design spec §8); the measured result at this corpus/machine scale is the opposite — p99 got 54.6% worse under hedging, for the reasons above (CPU contention from real backup work plus cold-replica-cache asymmetry), not because the hedging mechanism itself is broken.

## Configuration

2 worker threads per shard (reduced from sub-project A's default of 4 — on this 8GB test machine, the process-per-shard model caused system thrashing once 8-16 concurrent shard processes were running; applied uniformly across every N in this run, not per-point, to keep all rows comparable despite the constraint), queue depth 64, 200-entry LRU cache per shard, `wand`, top-10.

Ports: the tail-latency experiment uses base port 50300 (`server.tail_latency --base-port`, one port per shard, consecutive from there); the hedging experiment uses base port 50400 for primaries and 50500 for replicas (`server.hedging --base-port`/`--replica-base-port`) — script defaults, not overridden for the committed runs.

Corpus: 8,841,823 passages (counted directly from data/msmarco-passage.tsv). Approximate per-shard document count at each N tested: N=1: ~8,841,823 docs/shard, N=4: ~2,210,455 docs/shard, N=8: ~1,105,227 docs/shard, N=16: ~552,613 docs/shard.

## Known limitations

- **Dispersion isn't shown**: the tail-latency table above shows only the median-repeat's numbers, with no spread. At N=1, the three repeats' p99s ranged 58.3-212.1ms (median 96.3ms); at N=4, 30.8-55.7ms (median 34.4ms). Checking every adjacent-N pair's p99 range, N=4 vs. N=8, N=8 vs. N=16 overlap: run-to-run variance is real and, for those pairs, comparable in size to the N-to-N difference the plot shows, so the median-only table row alone cannot rule out those adjacent N's being statistically indistinguishable at this repeat count.
- **N=8 vs. N=16 plateau**: their p99s are within 0.03ms of each other (29.93ms vs. 29.96ms), essentially flat rather than continuing to improve. This is plausibly where the wait-for-slowest-of-N amplification effect the design spec discusses (§1) starts to reassert itself against the shrinking-per-shard-corpus effect described above, rather than corpus-shrinkage continuing to dominate indefinitely as N grows further.
- **Broker thread-pool sizing confound**: `Broker`'s internal `ThreadPoolExecutor` is sized to exactly N workers (one per shard), but both experiment drivers run 32 concurrent client threads issuing requests through the same broker instance — meaning up to 32 concurrent queries compete for only N pool threads. This under-provisioning is worst at N=1 (only 1 pool thread) and improves as N grows, meaning part of the tail-latency improvement with N is a broker-pool-capacity artifact, not purely the corpus-shrinkage effect described above. Disclosed here rather than fixed — a pool-sizing change would require re-running every experiment to know its effect, which is out of scope for this fix wave.
- **p99.9 resolution limit**: with ~190-211 samples per repeat and nearest-rank percentiles, p99.9 collapses to the single maximum sample in 18 of 18 repeats checked across both experiments — p99.9 numbers in this document should be read as "worst observed," not as a stable percentile estimate.
- **Hedge-delay measurement condition mismatch**: `hedging.py`'s warm-up measures each shard's own p95 one shard at a time (via a direct `SearchClient`, with the other shards idle), but the actual hedged run loads all shards simultaneously. The measured hedge delay is likely optimistic (too low) relative to real per-shard latency under full concurrent load, since the warm-up doesn't include the same contention the real run has.
