# Phase 3: dense recall and the ANN Pareto frontier

Status: approved for implementation planning
Scope: all of Phase 3 as one sub-project (see CASCADE_SEARCH_PLAN.md §Phase 3) —
encoding, brute-force ground truth, HNSW sweep, IVF-PQ sweep, and hybrid fusion,
all in one pass, since they share one embedding artifact and one exit deliverable.

## 1. Purpose

Phase 1/2 built and served a lexical (BM25/WAND) index. Phase 3 asks a different
question: what does a dense (embedding-based) recall channel buy, and at what
cost in latency and memory relative to exact search? Two ANN structures
(HNSW, IVF-PQ) trade recall for latency and memory differently; mapping that
tradeoff (the "Pareto frontier") is the core deliverable. A second question —
does combining lexical and dense recall beat either alone? — is answered by
the hybrid-fusion table.

## 2. Constraint: this machine has 8GB RAM, no GPU

The full corpus is 8.8M passages. Dense embeddings for all of them (384-dim
float32) would be ~13.5GB before any ANN index overhead — doesn't fit. Per
the plan's own pitfall table ("Dense index OOM at 8.8M → subset to 2M or use
PQ; state which"), this sub-project subsets to **1,000,000 passages** and
states that clearly as a documented limitation in the exit artifact. The
Pareto-frontier finding (which ANN structure wins at which recall target) does
not depend on corpus size; only the absolute latency/memory numbers would
shift at 8.8M, and the report says so.

## 3. Component: corpus subset

`py/dense/subset.py`. Random subsetting risks silently dropping qrels-judged
documents for the dl19/dl20 eval queries — if a relevant document isn't in the
subset, it can never be retrieved, and recall/NDCG against it becomes
meaningless. So the subset is constructed as:

```
subset = (all docids judged in dl19 ∪ dl20 qrels)
       ∪ (random sample, seeded, filling up to exactly 1,000,000)
```

This guarantees every relevance judgment used anywhere downstream is actually
retrievable from the subset. Output: `data/dense-subset-docids.txt` (one
docid per line) plus `data/dense-subset.jsonl` (docid + passage text, the
input to encoding). Seed and both qrels sets' union size are recorded in the
output's own header line for reproducibility.

## 4. Component: encoding

`py/dense/encode.py`. Model: `bge-small-en-v1.5` via `sentence-transformers`,
384-dim output, L2-normalized (inner product = cosine similarity downstream).
BGE is asymmetric:

- Passages: encoded as-is.
- Queries: encoded with the instruction prefix
  `"Represent this sentence for searching relevant passages: "` prepended.

Device selection: try MPS, fall back to CPU, log whichever was used — no
requirement to force one or the other, but the exit artifact must state which
one actually ran, since it affects the latency numbers' portability.

Output: `data/dense-embeddings.npy` (1,000,000 × 384 float32, ~1.5GB) and a
parallel `data/dense-docids.npy` (same row order, so row *i* of the embedding
matrix is docid `dense-docids.npy[i]`). Written once; every downstream step
(ground truth, HNSW build, IVF-PQ build) reads these files and never touches
the encoder again — re-encoding is the expensive step (minutes to an hour on
this CPU), so nothing after this point re-runs it.

Query embeddings for the 97 dl19+dl20 eval queries are encoded once into
`data/dense-query-embeddings.npy` alongside a `data/dense-query-ids.json`
ordering file, reused by ground truth, both ANN sweeps, and fusion.

## 5. Component: exact ground truth

`py/dense/ground_truth.py`. `faiss.IndexFlatIP` over the full 1M embedding
matrix. For each of the 97 eval queries, exact top-1000 (docid, score) pairs.
This serves two purposes:

- **Recall@100 reference** for both ANN sweeps (§6): an ANN result set's
  recall@100 is "how many of the exact top-100 appear in the ANN top-100."
- **The dense channel for fusion** (§7): using the *exact* dense ranking
  rather than an ANN approximation there isolates "does fusion help" from
  "how good is the ANN approximation," which is a separate question already
  covered by the Pareto plot.

Output: `bench/results/dense-ground-truth.json` — per query, top-1000
(docid, score), plus config (subset size, seed, model name, device).

## 6. Component: ANN sweeps (the money chart)

`py/dense/ann_sweep.py`.

**HNSW (`hnswlib`).** Build three indexes, one per `M ∈ {16, 32, 64}` (`M` is
build-time; each value needs a full rebuild). For each built index, sweep
`efSearch ∈ {32, 64, 128, 256, 512}` at query time (no rebuild) — 15 points.

**IVF-PQ (`faiss`).** Grid: `nlist ∈ {1024, 4096}` × PQ subquantizer count
`m ∈ {32, 64}` (both divide the 384-dim embedding evenly) at train/build time
— 4 trained indexes. `nprobe ∈ {1, 8, 16, 32, 64}` at search time — 20 points.

**Per point, three measurements:**

- **recall@100** vs. the exact ground truth (§5).
- **Latency** (p50/p95/p99, never mean — same rule as everywhere else in this
  repo). No server or queue is involved here — these are in-process library
  calls, not a system under concurrent load — so open-loop generation doesn't
  apply. Instead: single-threaded, repeat the 97 eval queries with replacement
  until there are enough samples for a stable p99 (same resampling approach
  Phase 1's `bench_query` used), timing each individual `search()` call via
  `harness.histogram.LatencyRecorder`.
- **Memory footprint**: measured RSS delta (`resource.getrusage` before/after
  the index build call) — a portable, dependency-free proxy for index size
  that works uniformly across `hnswlib` and `faiss`.

Output: `bench/results/dense-ann-sweep.json` — all 35 points (15 HNSW + 20
IVF-PQ), each carrying its exact build/search config, recall@100, p50/p95/p99,
memory delta, git SHA, hardware, corpus subset size.

**Exit chart:** recall@100 (x) vs. p99 latency (y) scatter/line, HNSW and
IVF-PQ as two series, memory footprint encoded as point size or a third
subplot. Rendered with `matplotlib` (first plotting dependency in this repo —
everything before Phase 3 was tables).

## 7. Component: hybrid fusion

`py/dense/fusion.py`.

**Channels:**
- **Lexical**: Phase 1's existing WAND run for dl19+dl20, already on disk —
  read via `harness.runfile.read_run` against the `run_path` recorded for
  engine `cascade-wand` in `runs/manifest.json` (depth 1000). No re-query of
  the C++ index.
- **Dense**: the exact brute-force top-1000 from §5.

**Fusion methods** (compared against each channel alone — 4 rows total):

- **Lexical only**, **Dense only** — baselines.
- **RRF** (reciprocal rank fusion): `score(d) = Σ_channels 1/(60 + rank_in_channel(d))`,
  k=60 (the standard default). A document absent from a channel contributes 0
  from that channel. Rank-based, so no cross-channel score normalization is
  needed, and it's insensitive to each channel's raw score scale.
- **Normalized score fusion**: min-max normalize each channel's scores to
  [0, 1] independently per query, then sum. A document absent from a channel
  contributes 0 from that channel.

**Metrics:** NDCG@10 and recall@1000 for all 4 rows, computed with the
existing `harness.metrics.evaluate` — the same function Phase 1 validated
against `pytrec_eval` to 4 decimals, on the same dl19+dl20 query/qrels pair.

Output: `bench/results/dense-fusion.json` (per-row metrics, full config) and
the 4-row table in the exit artifact.

## 8. File layout

```
py/dense/
  __init__.py
  subset.py            builds the 1M-doc subset (qrels ∪ random fill)
  encode.py             bge-small-en-v1.5 encoding -> embeddings + docids
  ground_truth.py       exact IndexFlatIP top-1000 per eval query
  ann_sweep.py           HNSW + IVF-PQ builds/sweeps -> dense-ann-sweep.json
  fusion.py               RRF / score-fusion / lexical / dense-only -> dense-fusion.json
  report.py               renders bench/phase3.md (Pareto plot + fusion table)
  tests/
    __init__.py
    test_subset.py         qrels docids always present in the subset
    test_fusion.py          RRF math, score normalization, absent-document handling
```

New dependencies (`pyproject.toml`, a new `dense` extra):
`sentence-transformers`, `torch` (CPU wheel), `faiss-cpu`, `hnswlib`,
`matplotlib`.

## 9. Testing

- `test_subset.py`: every docid in the dl19+dl20 qrels union appears in the
  constructed subset; subset size is exactly 1,000,000; re-running with the
  same seed produces an identical subset (determinism).
- `test_fusion.py`: RRF score formula on a small hand-constructed
  two-channel example with known expected ranking; score-fusion's min-max
  normalization on a small example with known expected values; a document
  present in only one channel is scored correctly (not dropped, not treated
  as a tie with unseen documents).
- No correctness re-testing of WAND itself, and no re-validation of
  `ndcg_at_k`/`recall_at_k` against `pytrec_eval` — both already proven in
  Phase 1; this phase only adds a new candidate-generation path in front of
  the same, already-validated metrics functions.
- ANN sweep and encoding steps are exercised by running them, not by a unit
  test — like Phase 1's `cascade_bm25`/`cascade_query_cost`, these are
  benchmark drivers whose correctness is checked by their output (recall@100
  sanity: HNSW/IVF-PQ recall must never exceed 1.0 and must reach ~1.0 at
  maximal `efSearch`/`nprobe`, which is itself asserted in `ann_sweep.py`'s
  own run, not a separate pytest).

## 10. Exit artifact

`bench/phase3.md` (mirrors `bench/phase1.md`/`bench/phase2a.md`'s structure):
subset methodology and size stated up front as a limitation, the Pareto plot
(recall@100 vs p99, memory as third dimension), the 4-row fusion table
(NDCG@10, recall@1000), and full run configuration (model, device, subset
size/seed, git SHA, hardware) per the project's "a number without its config
is not a result" rule. `bench/results/dense-ground-truth.json`,
`bench/results/dense-ann-sweep.json`, `bench/results/dense-fusion.json` are
committed raw JSON, same as every prior phase.

## 11. Out of scope (this sub-project)

- Full 8.8M-corpus dense indexing — explicitly deferred per §2; the pitfall
  table's own mitigation is what this sub-project follows.
- Training or fine-tuning an encoder — the plan says start with an
  off-the-shelf bi-encoder; distillation/fine-tuning is a stated "later" item,
  not part of this phase's exit bar.
- Weighted score fusion (a weight sweep between lexical and dense) — the
  plan's exit artifact asks for RRF vs. normalized score fusion vs. each
  channel alone, not a weight sweep; a natural follow-up but not required here.
- On-disk / memory-mapped ANN indexes — the 1M-passage subset (§2) removes
  the need; revisit only if a later phase needs the full corpus size.
- Any change to the Phase 1 C++ index or Phase 2 server — this phase reads
  Phase 1's existing WAND run file and otherwise operates independently in
  Python.
