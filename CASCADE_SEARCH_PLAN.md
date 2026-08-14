# Cascade — a multi-stage search engine with an enforced latency budget

**Target role:** Software Engineer Intern, TikTok Search Architecture
**Owner:** Hon Jung
**Format:** phased build; each phase ends in a benchmark artifact, not a demo

---

## 0. Positioning

### What this project is
A search engine built as a **cascade of stages under a fixed end-to-end latency budget**, where every stage is measured, ablated, and defended with numbers.

### What this project is not
- Not a RAG app. No LLM generation anywhere in the serving path.
- Not a chatbot or a UI project. The deliverable is a benchmark report.
- Not a model-training project. Models are off-the-shelf or distilled; the contribution is systems.

### The one-sentence pitch you should be able to give
> "I built a sharded multi-stage search engine: a C++ inverted index with BlockMax-WAND that matches Lucene's NDCG@10 on TREC-DL 2019, a dense channel with a measured recall/latency Pareto frontier, and a GPU cross-encoder reranker with dynamic batching — all under a 100ms p99 budget, characterized with a Zipfian load generator."

Every clause in that sentence is a number you can produce on demand. That is the entire point.

### Why this shape, for this JD
| JD phrase | Phase that answers it |
|---|---|
| "retrieval, ranking, and recall systems" | 1, 3, 4 |
| "heterogeneous computing architectures" | 4 (CPU index / GPU ranker split, batching, quantization) |
| "high-concurrency, low-latency" | 2 (sharding, fan-out, tail latency, hedging) |
| "large-scale system design" | 2, 6 |
| "C++/Python/Java" | 1 and 2 are C++; everything else Python |

---

## 1. Corpora

### Primary — quality harness
**MS MARCO passage v1** (8.8M passages) with **TREC Deep Learning 2019 + 2020** qrels (43 + 54 judged queries, graded relevance).

Why: real human relevance judgments. This is what makes your NDCG claims falsifiable. Access via `ir_datasets` (`msmarco-passage/trec-dl-2019/judged`), which also gives you the dev set (~6.9k queries, sparse qrels) for MRR@10.

### Secondary — domain corpus
Pick one, in order of preference:
1. **HowTo100M metadata** — ASR transcripts as the dominant text field. Mirrors TikTok's situation exactly: no authored text, everything derived.
2. **YouTube-8M** — precomputed frame-level visual features + vocabulary labels. Gives you a visual channel without needing to encode video yourself.
3. **MSR-VTT** (10k videos, 200k captions) — small, but fine for wiring up multimodal plumbing before scaling.

> **Availability risk:** WebVid-2M/10M was withdrawn. Verify each dataset is still distributable before you build against it. Have MSR-VTT as the always-available fallback.

### Memory budget — check this before you start
| Artifact | Full 8.8M | Notes |
|---|---|---|
| Inverted index (compressed postings) | ~1.0–1.5 GB | fine on a laptop |
| Doc store (text, for reranking) | ~3 GB | memory-map it, don't load |
| Embeddings, fp32, 384-dim | ~13.5 GB | **too big for a laptop** |
| Embeddings, PQ-48 | ~0.4 GB | viable |
| HNSW graph, M=32 | ~2–3 GB | on top of vectors |

**Decision:** run the lexical channel on the full 8.8M. For the dense channel, either subset to ~2M passages or go PQ from the start. State which you did in the report — an interviewer will ask.

### Hardware
Phases 0–3 run on a laptop. Phase 4 needs a GPU: Colab/Kaggle T4 is enough, or a spot instance for a few hours. Don't buy anything.

---

## 2. Architecture

```
                      ┌──────────────────────────┐
   query ────────────▶│  Query understanding svc │  (Python; cache-fronted)
                      └────────────┬─────────────┘
                                   │ structured query
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
            ┌────────────┐  ┌────────────┐  ┌────────────┐
            │ Lexical    │  │ Dense      │  │ Entity /   │   RECALL
            │ (C++ BMW)  │  │ (HNSW/PQ)  │  │ vertical   │   CPU
            │ N shards   │  │            │  │            │
            └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
                  └───────────────┼───────────────┘
                                  ▼
                        ┌──────────────────┐
                        │ Merge + dedup    │
                        │ RRF / score fuse │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ Pre-rank (MLP)   │   CPU
                        └────────┬─────────┘
                                 ▼  batch queue
                        ┌──────────────────┐
                        │ Rank             │   GPU
                        │ cross-encoder    │   dynamic batching
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ Blend + MMR      │
                        └──────────────────┘
```

The queue between pre-rank and rank is the heterogeneous-compute boundary. It is the most interesting thing in the diagram and the thing you should be able to talk about for ten minutes.

### Latency budget (p99, end to end 100ms)

| Stage | Budget | Where it goes |
|---|---:|---|
| Query understanding (cache miss) | 8ms | fan-out: max(intent, entity, encoder) |
| Lexical recall | 25ms | scatter-gather over N shards |
| Dense recall | 20ms | runs concurrently with lexical |
| Merge + dedup | 3ms | |
| Pre-rank | 8ms | 10k → 1k |
| Rank (GPU) | 30ms | includes batch wait |
| Blend + MMR | 4ms | |
| Slack | 12ms | RPC, serialization, GC |

Recall channels run **concurrently**, so their budgets overlap — the recall stage costs `max(25, 20)`, not `45`. Write the budget as a critical-path calculation, not a sum. Interviewers check this.

---

## 3. Phases

Each phase has an **exit artifact**. If you cannot produce the artifact, the phase is not done, regardless of how much code exists.

---

### Phase 0 — Harness (est. 6–8h) — **do not skip**

Build the measuring instruments before the thing being measured.

**Components**
- `ir_datasets` loader → normalized `(doc_id, text)` and `(query_id, query, qrels)`
- Eval script: NDCG@10, NDCG@100, MRR@10, recall@100, recall@1000. Validate your implementation against `pytrec_eval` — they must agree to 4 decimal places.
- Load generator: Zipfian query sampler (s ≈ 1.0), configurable QPS, open-loop (fixed arrival rate, not closed-loop request-response — closed-loop hides queueing).
- Latency recording: HDR histogram. Report p50/p95/p99/p99.9. **Never report a mean.**
- Baselines: run Lucene (via PyLucene or Anserini) and Tantivy for BM25. Record their NDCG@10 and their latency.

**Exit artifact:** `baselines.md` with a table of Lucene/Tantivy NDCG@10 and MRR@10 on TREC-DL 2019, 2020, and MS MARCO dev.

**Why open-loop matters:** a closed-loop generator can't produce a queue, so it will never show you the latency knee. This single choice separates a real benchmark from a fake one.

---

### Phase 1 — C++ inverted index with BlockMax-WAND (est. 15–20h) — **highest signal**

This is the component that makes the project credible for this specific JD.

**Index build**
- Tokenizer: shared C++ implementation, Python binding via pybind11. One code path for indexing and querying — this is your training/serving-skew defense.
- Term dictionary: sorted terms + binary search, or an FST if you're feeling ambitious.
- Postings: docid gap-encoded, StreamVByte or PForDelta compressed, stored in fixed-size blocks (64 or 128 docs).
- Per-block metadata: `max_docid`, `max_impact` (the max BM25 term contribution in that block). The max-impact is what makes BlockMax possible.
- Skip list over blocks.

**On-disk layout**
```
index.terms   [term_count][term_len|term|df|postings_offset]*
index.post    [block: max_docid|max_impact|len|docid_gaps|freqs]*
index.docs    [doc_count][doc_len]*          # for BM25 normalization
index.store   [doc_id → text offset]         # mmapped, for reranking
```

**Query evaluation** — implement in this order and benchmark each against the last:
1. Exhaustive DAAT-OR (baseline, correct by construction)
2. WAND (pivot selection on term upper bounds)
3. BlockMax-WAND (pivot selection on block max-impacts)

**Exit artifact:** a table with three rows —

| Algorithm | NDCG@10 | postings scored | full evaluations | p99 latency |
|---|---|---|---|---|

BMW must produce **identical top-10** to exhaustive OR (it's a safe optimization — if results differ you have a bug). The value is in columns 3–5.

**Also required:** your NDCG@10 vs. Lucene's from Phase 0. If you're more than ~0.01 off, you have a scoring bug — most likely BM25 length normalization or your tokenizer diverging from Lucene's `StandardAnalyzer`. Chase it down; the debugging story is itself good interview material.

---

### Phase 2 — Serving, sharding, tail latency (est. 12–15h)

**Components**
- gRPC service around the C++ index (C++ server, or pybind11 + Python server — C++ is the better signal).
- Fixed-size thread pool; explicit queue with bounded depth and load shedding (return `RESOURCE_EXHAUSTED` past the bound rather than queueing unboundedly).
- Query result cache: LRU keyed on the *normalized* query string. Report hit rate.
- Document-partitioned sharding: split the index into N shards (N = 4, 8, 16), each its own process. A broker scatters and gathers.

**The experiments that matter**

1. **Tail latency amplification.** Measure single-shard p99. Then measure N-shard fan-out p99. The fan-out p99 is materially worse than the single-shard p99, because the request waits on the slowest of N. Plot p99 vs N. This is the Dean & Barroso result and reproducing it yourself is worth more than citing it.
2. **Hedged requests.** Send a backup request to a replica once the first exceeds p95. Show the p99 improvement and the extra load cost (should be ~5% extra requests). Report both.
3. **Throughput knee.** QPS on x-axis, p99 on y-axis. Find the point where latency goes vertical. That number is your capacity.
4. **Cache sensitivity.** Sweep the Zipf parameter and show hit rate → p99 relationship. Report cache-hit and cache-miss latency *separately*; the blended number only tells you your hit rate.

**Exit artifact:** four plots + a capacity statement: "sustains X QPS at p99 < 25ms with N=8 shards and a 43% cache hit rate."

> **Stop here if time runs out.** Phases 0–2 alone are a strong, complete, defensible project. Everything after this is upside.

---

### Phase 3 — Dense recall and the ANN Pareto frontier (est. 10–12h)

**Encoder:** start with an off-the-shelf bi-encoder (`bge-small-en`, `all-MiniLM-L6-v2`). Do not train from scratch. If you want a training story later, distill it or fine-tune with in-batch negatives plus hard negatives mined from your own BM25 output — mining your negatives with your own Phase 1 index is a nice touch.

**ANN indexes:** HNSW (`hnswlib`) and IVF-PQ (`faiss`).

**The money chart:** sweep parameters and plot **recall@100 (vs. exact brute-force) against p99 latency**, with a third series for memory footprint.
- HNSW: sweep `M ∈ {16, 32, 64}`, `efSearch ∈ {32 … 512}`
- IVF-PQ: sweep `nlist`, `nprobe`, PQ subquantizer count

Every point on that plot is a real engineering choice. Being able to say "at 95% recall HNSW is 3× faster but 6× the memory of IVF-PQ, so at our corpus size the choice is X" is exactly the conversation this role runs on.

**Hybrid fusion:** compare reciprocal rank fusion against normalized score fusion, and both against each channel alone.

**Exit artifact:** the Pareto plot, plus a fusion table (lexical / dense / RRF / score-fusion × NDCG@10, recall@1000).

---

### Phase 4 — Ranking cascade and heterogeneous serving (est. 12–15h)

**Pre-rank.** A cheap scorer taking 10k → 1k. Options: a small MLP over lightweight features (BM25 score, dense score, doc length, channel-of-origin), or a bi-encoder dot product you already have.

Measure **prerank–rank consistency**: of the true top-100 under the full ranker, how many survive pre-ranking? This metric is standard in industry and almost never appears in student projects. A low number means your cascade is throwing away good documents before the good model ever sees them.

**Rank.** A cross-encoder (`ms-marco-MiniLM-L-6-v2` or similar), served on GPU.

**The heterogeneous compute work — the core of this phase**
- **Dynamic batching.** Accumulate requests into a batch until either `max_batch_size` or `max_wait_ms` is hit. Sweep both. Plot throughput and p99 against batch window. There is a clean tradeoff curve here: larger windows buy throughput and cost latency.
- **Precision.** fp32 → fp16 → INT8 (ONNX Runtime, or TensorRT if you want the harder path). For each: latency, throughput, and **NDCG@10 delta**. Quantization that costs quality is a different decision from quantization that's free.
- **Queue discipline.** What happens when GPU capacity is exceeded? Show load shedding vs. unbounded queueing, and what each does to p99.
- **CPU/GPU split characterization.** Where does time actually go — index traversal, feature assembly, H2D transfer, kernel execution, D2H? Profile it. "The transfer dominated until I batched" is a real finding.

**Exit artifact:** batch-window tradeoff plot, precision table (latency × throughput × NDCG delta), prerank-consistency number.

---

### Phase 5 — Video, multi-field, multi-vertical (est. 10–12h)

Now the domain corpus enters.

- **Multi-field indexing with BM25F.** Fields: ASR transcript, caption, hashtags, title, creator name. Per-field length normalization and per-field weights. Ablate the weights — show which derived signal carries the retrieval load. (Prediction: ASR dominates, and that finding is itself the interesting result.)
- **Visual channel.** CLIP text→frame embeddings, or YouTube-8M precomputed features. A third recall channel.
- **Query understanding service**, per the design discussed separately:
  - segmentation ablation (whitespace vs. real segmenter → recall@1000 delta)
  - cache-hit vs cache-miss latency split
  - serial vs. parallel fan-out p99 comparison
- **Vertical blending.** Intent distribution over {video, creator, sound, hashtag}, blended results.
- **Diversity rerank.** MMR over the final list; report a diversity metric alongside NDCG and show the tradeoff.

**Exit artifact:** field-weight ablation table, channel contribution table (which channel supplied the eventual top-10), query-understanding ablations.

---

### Phase 6 (stretch) — Near-real-time indexing (est. 10h)

Almost no portfolio project does this, and it is directly relevant to TikTok, where content freshness is the product.

- Segment-based indexing: writes land in an in-memory segment, flush on size/time threshold, tiered merge policy in the background.
- Searches fan out across all live segments and merge.
- **Measure:** index-to-searchable lag (p99), and query latency as a function of unmerged segment count. The tradeoff — more segments means fresher but slower — is the whole design tension in Lucene, Elasticsearch, and every real-time search system.
- Deletes via a tombstone bitset, applied at query time.

**Exit artifact:** freshness/latency tradeoff plot.

---

## 4. Repo structure

```
cascade/
  cpp/
    index/          builder, postings codec, term dict
    query/          daat.cc  wand.cc  blockmax_wand.cc
    server/         grpc service, thread pool, cache
    bindings/       pybind11
    tests/          correctness: BMW top-k == exhaustive top-k
  py/
    harness/        eval.py  loadgen.py  histogram.py
    dense/          encode.py  ann_sweep.py
    rank/           prerank.py  crossencoder_server.py  batcher.py
    qu/             normalize.py  intent.py  entity.py
  proto/            search.proto
  bench/
    results/        raw json per run — commit these
    plots/
    REPORT.md       the actual deliverable
  README.md
```

`bench/REPORT.md` is the thing you link on your resume. Structure it as: latency budget → per-stage results → ablations → capacity statement → known limitations.

**Commit your raw benchmark JSON.** Reproducibility is the difference between a benchmark and a claim.

---

## 5. Instrumentation spec

Emit per request: `stage_latency_us` for every stage, `cache_hit`, `shard_count`, `postings_scored`, `full_evaluations`, `candidates_in/out` per stage, `batch_size` at the ranker, `shed` flag.

Every benchmark run records: git SHA, corpus size, shard count, hardware, and the full config. A number without its config is not a result.

---

## 6. Known pitfalls

| Pitfall | Mitigation |
|---|---|
| Closed-loop load generation hides queueing | open-loop with fixed arrival rate |
| Reporting mean latency | report p50/p95/p99/p99.9, never a mean |
| Blended cache latency | always split hit vs. miss |
| BMW returning different results from OR | it's a bug, not a tradeoff — safe pruning is exact |
| Tokenizer differs offline vs. online | one shared C++ implementation, both paths bind to it |
| Dense index OOM at 8.8M | subset to 2M or use PQ; state which |
| Measuring latency on a laptop under thermal throttle | pin frequency, run the load test 3×, report variance |
| Scope creep into a UI | there is no UI in this plan |

---

## 7. Resume mapping

Write bullets only for phases you've actually finished, with real numbers.

**After Phase 2**
> Built a sharded search engine in C++ with a BlockMax-WAND inverted index over 8.8M documents, matching Lucene's NDCG@10 on TREC-DL 2019 to within [X] while scoring [Y]% fewer postings; characterized tail-latency amplification across [N] shards and cut p99 by [Z]% with hedged requests.

**After Phase 4**
> Extended to a four-stage retrieval cascade (lexical + dense recall → pre-rank → GPU cross-encoder) under a 100ms p99 budget; mapped the HNSW recall/latency/memory Pareto frontier and tuned dynamic batching to sustain [X] QPS at [Y]ms p99, with INT8 quantization costing [Z] NDCG@10.

Leave the placeholders as placeholders until you have the measurement. Fabricated metrics are the single fastest way to lose a systems interview, because the follow-up question is always "how did you measure that."

---

## 8. Interview questions this project earns you the right to answer

- Why does fan-out to more shards hurt p99 even when each shard is faster?
- Why is WAND safe but top-k pruning by score threshold sometimes not?
- When would you choose IVF-PQ over HNSW?
- What breaks first when QPS doubles?
- How do you decide the pre-ranking cutoff?
- What's the cost of dynamic batching and how do you pick the window?
- Your cache hit rate drops 10 points overnight — what happened?
- How do you keep a search index fresh without tanking query latency?

If you can answer these from your own measurements rather than from a blog post, you are ahead of most candidates for this role.

---

## 9. Sequencing against your calendar

Assuming ~8h/week alongside the GovTech internship:

| Weeks | Phase | Cumulative state |
|---|---|---|
| 1 | 0 | harness + baselines |
| 2–4 | 1 | **first resume-worthy milestone** |
| 5–6 | 2 | **minimum defensible version complete** |
| 7–8 | 3 | Pareto frontier |
| 9–10 | 4 | full cascade + GPU |
| 11–12 | 5 | video/multimodal |
| 13+ | 6 | stretch |

If applications force a cut, cut from the back. Phases 0–2 finished and measured beats all six phases half-built.
