# Phase 1 — C++ inverted index with BlockMax-WAND

Exit artifact for Phase 1. `bench/results/cascade-*.json` and
`bench/results/cascade-query-cost.json` are committed; the run files
and the built index are not (regenerable — see `py/baselines/cascade_bm25.py`
and `py/baselines/cascade_query_cost.py`).

## Query evaluation cost

One fixed sample of 6,980 real `dev` queries, top-10, single-threaded serial (no queueing — that's Phase 2's subject). Each algorithm runs the identical query set, so the postings-scored and full-evaluations columns are a direct measure of how much work pruning saves, not an artifact of different queries landing on different algorithms.

| Algorithm | mean postings scored | mean full evaluations | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|
| cascade-daat-or | 1,054,955 | 953,085 | 9.323ms | 69.183ms | 107.121ms |
| cascade-wand | 15,579 | 11,741 | 2.865ms | 29.032ms | 54.208ms |
| cascade-blockmax-wand | 3,229 | 1,598 | 4.026ms | 41.332ms | 73.382ms |

BlockMax-WAND scores 99.7% fewer postings than exhaustive DAAT-OR for the identical top-10, on the same 6,980 queries — and 79.3% fewer than plain WAND. That doesn't translate to a wall-clock win here: BlockMax-WAND's p99 (41.3ms) is 42% higher than WAND's (29.0ms), and it loses at every percentile, not just the tail. `mean_blocks_decoded` is why: BlockMax-WAND decodes 49% more blocks than WAND (7,213 vs 4,826) even while scoring far fewer postings from them. That's the direct cost of `skip_to_block` landing on and decoding a block it then abandons — its hopeless branch narrows the pivot span by touching a block's bound, and when that block turns out not to help, the decode was still paid for. Fewer postings scored is a real result, but it isn't the same thing as less work in this implementation: the block-level bookkeeping BlockMax-WAND does to reach that number costs more here than the decoding it avoids.

## Algorithm equivalence at real corpus scale

WAND and BlockMax-WAND are safe optimizations: any difference from exhaustive
DAAT-OR's results is a bug, never a tradeoff. Verified two ways. On the synthetic
20,000-document corpus, `cpp/tests/test_index.cc` checks exact positional top-10
match across 300 generated queries, as part of the build. On the real
8,841,823-passage index, DAAT-OR runs on dl19+dl20 (97 real TREC queries —
run separately from the dev-scale query-cost benchmark above, since DAAT-OR's cost
doesn't fall with depth and dev's 6,980 queries would cost minutes for no new
information over what these 97 already establish) and its NDCG@10/NDCG@100/MRR@10/
R@100/R@1000 match WAND's and BlockMax-WAND's to four decimal places on both sets —
see the DAAT-OR rows in the table below.

## NDCG@10 vs. Lucene

Depth-1000 runs, same qrels and metric code as `bench/baselines.md`. The plan's
tolerance is ~0.01; a gap past that points at BM25 length normalization or a
tokenizer divergence from Lucene's `EnglishAnalyzer` before it points at the
retrieval algorithm — WAND/BlockMax-WAND already proven exact against DAAT-OR
above, so a scoring bug would show up identically in all three.

| Engine | Query set | Queries | NDCG@10 | NDCG@100 | MRR@10 | R@100 | R@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| cascade-blockmax-wand | dev | 6980 | 0.2278 | 0.2868 | 0.1833 | 0.6590 | 0.8530 |
| cascade-wand | dev | 6980 | 0.2278 | 0.2868 | 0.1833 | 0.6590 | 0.8530 |
| lucene-anserini-1.0.0 | dev | 6980 | 0.2301 | 0.2892 | 0.1855 | 0.6622 | 0.8575 |
| cascade-blockmax-wand | dl19 | 43 | 0.5052 | 0.5015 | 0.6911 | 0.4926 | 0.7502 |
| cascade-daat-or | dl19 | 43 | 0.5052 | 0.5015 | 0.6911 | 0.4926 | 0.7502 |
| cascade-wand | dl19 | 43 | 0.5052 | 0.5015 | 0.6911 | 0.4926 | 0.7502 |
| lucene-anserini-1.0.0 | dl19 | 43 | 0.5121 | 0.5072 | 0.7138 | 0.4988 | 0.7539 |
| cascade-blockmax-wand | dl20 | 54 | 0.4785 | 0.4900 | 0.6545 | 0.5643 | 0.7886 |
| cascade-daat-or | dl20 | 54 | 0.4785 | 0.4900 | 0.6545 | 0.5643 | 0.7886 |
| cascade-wand | dl20 | 54 | 0.4785 | 0.4900 | 0.6545 | 0.5643 | 0.7886 |
| lucene-anserini-1.0.0 | dl20 | 54 | 0.4769 | 0.4910 | 0.6653 | 0.5623 | 0.7865 |

**DL19 NDCG@10 delta vs. Lucene: -0.0069** (within the plan's ~0.01 tolerance).

## Configuration

k1=0.9, b=0.4 (Anserini's msmarco defaults — see `BM25Params` in
`cpp/index/index_format.h`). Analyzer targets Lucene's `EnglishAnalyzer` (Porter
stemming + stopwords) via an ASCII approximation of UAX#29 segmentation — see
`cpp/index/analyzer.h` for the known, deliberate divergence from a full
implementation, and the first thing to suspect if the NDCG delta above is large.
Document lengths are exact (unlike Lucene's lossy 1-byte norm quantization), so a
small residual in that direction is expected even with everything else matched.
The observed deltas are small (dl19 -0.0069, dl20 +0.0016, dev -0.0023) and don't
sit on one consistent side of zero, which fits two small, partially-offsetting
sources of divergence — the analyzer approximation and the exact-vs-quantized
length difference — rather than one dominant cause pulling every query set the
same way.

## Known limitations

- Single machine, single-threaded, no cache, no sharding. Capacity under
  concurrent load is Phase 2's subject.
- The query-cost table uses top-10; the NDCG table uses top-1000. Different
  operating points on purpose — see `cascade_query_cost.py`'s docstring.
