# Phase 6: Near-Real-Time Indexing

## Freshness/latency tradeoff

| flush threshold (docs) | lag p99 (ms) | query p99 (ms) | final segments | collisions skipped |
|---|---|---|---|---|
| 50 | 55.59 | 6.81 | 7 | 4118 |
| 200 | 109.24 | 27.46 | 6 | 4115 |
| 1000 | 209.05 | 5.74 | 2 | 4106 |

![freshness/latency tradeoff](plots/phase6-freshness.png)

Base corpus: 50,000 docs (already indexed). Write stream: 5,000 docs, added one at a time. `merge_factor=4`, `max_segments=8`.

## Segment-count effect (fixed corpus size)

| N | p50 (ms) | p99 (ms) |
|---|---|---|
| 1 | 3.55 | 8.07 |
| 2 | 3.11 | 8.76 |
| 4 | 2.90 | 8.48 |
| 8 | 3.40 | 37.12 |

![query latency vs. segment count](plots/phase6-segment-count.png)

Fixed corpus: 200,000 docs, partitioned into N segments for each point -- the same corpus-shrinkage-per-segment effect `bench/phase2b.md` disclosed for its shards is still present here. What isolates the fan-out/merge cost specifically is that all N segments live in one process with no network hop and no per-shard worker pool, so the only things that scale with N are Python-level fan-out overhead and the merge/sort/tombstone-filter step, not shard process scheduling or broker-pool sizing. This is a different, valid question from Phase 2B's "does more shards help tail latency," not a claim to have fixed Phase 2B's confound.

## Merge-in-action evidence

At `flush_threshold_docs=50`, a merge fired during the run: segment count went from 8 to 6, consistent with the configured merge policy actually bounding growth.

## Known limitations

- **Merge is re-tokenization, not a postings-level merge.** Every merge rebuilds from source TSV text via the unmodified `build_index` binary; it is not a sorted-run merge over already-built postings (which would require new C++ in `cpp/index/builder.cc`, out of scope here). Merge cost scales with merged segment size the same way a fresh build does.
- **In-process latency is not served latency.** There is no gRPC, no worker pool, no request queueing in this layer -- these numbers are not comparable to `bench/phase2a.md`'s or `bench/phase2b.md`'s served p99s.
- **No NDCG/quality claim for the multi-segment configuration.** Per-segment BM25 statistics mean the multi-segment configuration's relevance is not evaluated here, the same framing `bench/phase2b.md` established for shards.
- **Flush is synchronous and blocks new writes for its duration.** This design does not overlap flush with the next write batch, unlike Lucene's concurrent in-memory segment writer.
- **The tombstone over-fetch bound (`k + len(tombstones)`) is a fixed heuristic**, not a tuned bound against how many tombstones plausibly cluster near the top-k of a single segment.
