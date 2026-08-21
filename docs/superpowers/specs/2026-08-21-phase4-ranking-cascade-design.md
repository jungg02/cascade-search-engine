# Phase 4: ranking cascade and heterogeneous serving

Status: approved for implementation planning
Scope: pre-rank (MLP) + rank (cross-encoder) cascade, prerank-consistency,
dynamic batching, fp32/fp16/INT8 precision comparison, and queue discipline
(load-shedding vs. unbounded), all in one pass. CPU/GPU transfer/kernel
profiling is explicitly deferred (see §10) — trimmed from the plan's full
scope to fit the session constraint in §2.

## 1. Purpose

Phases 1–3 built recall (lexical WAND, dense ANN). Phase 4 asks what happens
*after* recall: given up to 1000 lexical candidates per query, how do you cut
that down to a manageable set cheaply (pre-rank), re-score the survivors with
a much more expensive model (rank), and serve that expensive model under load
without either wasting GPU capacity or blowing the latency budget? The
pre-rank stage's real risk is silent: if it discards documents the full
ranker would have ranked well, the cascade is lossy in a way no single-stage
metric shows — that's what **prerank-consistency** measures. The rank stage's
real risk is a heterogeneous-compute one: a GPU model can be made faster by
batching, cheaper by quantizing, or protected by shedding load — each is a
real, measurable trade, not a free win, and this phase's job is to make each
trade's cost explicit.

## 2. Constraint: no local GPU, one Kaggle session

This machine has no CUDA GPU (MPS only, and MPS doesn't support the
ONNX Runtime INT8 path this phase needs). All GPU-bound work runs on a
**Kaggle Notebook** (free T4), which the user drives directly — this session
cannot execute code there. Results come back as `bench/results/*.json`,
downloaded from Kaggle and dropped into the repo, exactly as every prior
phase's results arrived, just via a manual step instead of a local run.

This constrains the design in two concrete ways:

- **Everything must plausibly fit in one Kaggle session.** Kaggle GPU
  sessions have a hard continuous-runtime ceiling (order of hours) and a
  weekly quota; there is no persistent environment between sessions the way
  a long-lived local shell has. So every experiment grid in this spec is
  sized to keep total GPU wall-clock time to roughly an hour, leaving real
  headroom for the debugging that WILL happen (ONNX export and INT8
  calibration are the likeliest friction points, per experience with every
  prior phase's first real-data run surfacing a bug pytest couldn't catch).
- **Debugging happens over a relay, not directly.** The user runs a cell,
  something breaks, they report back what happened, this session fixes the
  code, they re-run. This is slower than this project's established pattern
  of directly re-running a failed step — so the design favors fewer,
  coarser experiment points over exhaustive grids, and favors the Kaggle
  driver writing each sub-experiment's output as it completes (see §9)
  rather than one script that only saves at the very end.

## 3. Candidate pool: Phase 1's WAND run, unmodified

The cascade's input is **Phase 1's existing lexical WAND run**
(`runs/cascade-wand.{dev,dl19,dl20}.txt`, depth 1000, already on disk, no
re-query of the C++ index) — not a fresh recall pass, and not fused with
Phase 3's dense channel. This keeps Phase 4 focused on its actual stated
scope (pre-rank/rank/batching/quantization), and reuses real, already-scored
candidates over the full 8.8M-passage corpus rather than Phase 3's 1M-passage
subset.

One consequence: the plan's suggested pre-rank feature "channel-of-origin"
does not apply — with a single candidate channel it is a constant,
contributing nothing to the MLP. Dropped.

## 4. Component: candidate encoding (local)

`py/rank/encode_candidates.py`. Not a full re-run of Phase 3's `encode.py` —
that encoded the 1M-passage *subset*; this phase's WAND candidates are a
different, much smaller document set (the union of every doc that appears in
any of the three WAND runs' depth-1000 lists, likely tens of thousands of
unique docs, not 1M). This step:

- Encodes the **dev, dl19, dl20 queries** (same instruction-prefixed
  `bge-small-en-v1.5` convention as Phase 3 — reuses `dense.encode`'s
  `apply_query_prefix`/`select_device` rather than duplicating them).
- Encodes every **unique candidate docid** across the three WAND run files
  (a single streaming pass over `harness.datasets.iter_docs()`, collecting
  text for docids in the candidate set — the same access pattern Phase 3's
  `subset.py` already established for pulling a bounded docid set out of a
  full corpus stream).
- Writes a flat scalar feature file, not raw embeddings: `(qid, docid,
  dense_score, doc_length)` for every (query, candidate) pair across all
  three query sets — `doc_length` (token/char count of the candidate's
  text) is cheap to compute in the same corpus pass that already reads each
  candidate's text for encoding, so it's carried alongside `dense_score`
  rather than requiring a second pass. Kaggle only needs these two scalars
  as MLP features, not the raw embeddings — shipping scores instead of
  vectors keeps the Kaggle-side upload small and avoids re-deriving
  embeddings there.

Output: `data/rank-dense-scores.jsonl` (gitignored, matching `data/`'s
existing convention) — one line per (qid, docid, dense_score, doc_length).

## 5. Component: pre-rank (MLP, Kaggle)

`py/rank/prerank_features.py` (pure, local + Kaggle-shared) builds the
3-feature vector per (query, candidate) pair:

- **BM25 score** — read directly from the WAND run file (`harness.runfile.
  read_run`), no re-computation.
- **Dense score** — looked up from §4's output file.
- **Doc length** — read from §4's output file (`data/rank-dense-scores.jsonl`),
  which already carries it alongside `dense_score`.

`py/rank/prerank_mlp.py` (Kaggle-run driver, no unit test — a benchmark
driver like every GPU/real-data script in this repo, correctness checked by
running and inspecting output, not pytest):

- **Train** a small MLP (2–3 layers, the plan's own suggestion) on the
  **dev** query set: pointwise binary classification, label from dev's
  sparse qrels (relevant/not, per MS MARCO dev's convention — each query has
  ~1 labeled relevant doc; negatives are sampled from the same query's
  non-relevant WAND candidates). Dev has 6,980 queries — ample training
  signal for a model this small.
- **Evaluate** on **dl19+dl20** (held out, never seen in training, real
  graded relevance judgments) — reduces each query's up-to-1000 candidates
  to the top ~100 by MLP score.

## 6. Component: prerank-consistency (Kaggle)

The core pre-rank metric, per the plan: of the *true* top-100 under the full
ranker, how many survive pre-ranking? "True top-100" requires scoring **all**
dl19+dl20 candidates (up to 1000 each, ~97k pairs total) with the
cross-encoder once, at fp32 — this is the one point where the expensive
ranker runs over the *entire* candidate pool rather than just the pre-rank
survivors, specifically to establish the ground truth this metric compares
against.

`py/rank/prerank_consistency.py` (pure, unit-tested — same shape as Phase
3's `recall_at_k`): given the true top-100 docids and the set of docids that
survived pre-ranking, returns the overlap fraction. A low number means the
cascade is throwing away good documents before the expensive model ever sees
them — the number itself is the finding, whatever it turns out to be.

## 7. Component: dynamic batching + precision + queue discipline (Kaggle)

`py/rank/batcher.py` — pure dynamic-batching queue logic, unit-tested with a
mock clock and a mock scorer (no GPU needed for this part): accumulate
incoming requests until either `max_batch_size` or `max_wait_ms` is reached,
then flush as one batch. This is the one piece of Phase 4 that is genuinely
pure and local-testable, matching this project's "test what's pure, run what
needs real data" split.

`py/rank/crossencoder_harness.py` (Kaggle-run driver) — an **in-process
async harness**, not a real client/server: a single Python process where an
open-loop request generator feeds `batcher.py`'s queue, which forwards
batches to the cross-encoder (`ms-marco-MiniLM-L-6-v2` via ONNX Runtime's
CUDA execution provider). No real network hop — batching/queueing/shedding
behavior is fully captured without needing a client/server split, and this
runs entirely inside one Kaggle notebook. Latency recorded via the existing
`harness.histogram.LatencyRecorder` (p50/p95/p99, never mean — same rule as
every prior phase).

**Dynamic batching sweep** — bounded, not exhaustive: `max_batch_size ∈ {1,
8, 32}` × `max_wait_ms ∈ {0, 5, 20}` (9 points), run once at fp32. Per point:
throughput and p99 latency.

**Precision comparison** — fp32 (baseline), fp16, INT8, all via ONNX Runtime
(not TensorRT — the plan's own "harder path" is explicitly declined here for
Kaggle-setup simplicity; the export/quantize path is simpler and still
produces real per-precision numbers). Rather than re-running the full
9-point batching sweep at all three precisions (3× the cost), the 1–2
best-looking settings from the fp32 sweep are re-run at fp16 and INT8. Per
precision, per selected setting: latency, throughput, and **NDCG@10 delta**
against the fp32 cascade result from §6 — "quantization that costs quality
is a different decision from quantization that's free," so this delta is
measured, never assumed zero.

**Queue discipline** — a handful of runs (2 disciplines × 2–3 sustained
overload levels), not a sweep: load-shedding (reject when the queue is full)
vs. unbounded queueing, observing what each does to p99 and to the shed
rate. Not swept finely — the qualitative "shedding bounds p99, unbounded
queueing doesn't" finding is the point, not a dense curve.

## 8. Out of scope (deferred from the plan's full scope)

- **CPU/GPU transfer/kernel profiling** — the plan's stage-by-stage
  breakdown (index traversal, feature assembly, H2D, kernel, D2H) is
  deferred per §2's session-time constraint; the hardest of the plan's six
  pieces to instrument well, and the one most likely to blow the one-session
  budget on its own. Revisit if time allows after the other five land.
- **TensorRT** — ONNX Runtime is used instead throughout §7; TensorRT is the
  plan's own explicitly-optional "harder path."
- **Fusing Phase 3's dense recall into the candidate pool** — per §3, the
  candidate pool is lexical-only; a natural follow-up, not required here.
- **Training/fine-tuning the cross-encoder** — off-the-shelf
  `ms-marco-MiniLM-L-6-v2`, matching the plan's "don't train from scratch"
  rule (same rule Phase 3 followed for the bi-encoder).
- **Re-querying the C++ WAND index** — Phase 1's existing run files are read
  as-is; no change to the C++ index or Phase 2's server.

## 9. File layout

```
py/rank/
  __init__.py
  encode_candidates.py    local: dense scores + doc length for WAND candidates
  prerank_features.py     pure: (BM25, dense, doclen) feature vector
  prerank_mlp.py           Kaggle: train on dev, eval on dl19+dl20
  prerank_consistency.py  pure: true-top-100-survival overlap
  batcher.py               pure: dynamic-batching queue logic
  crossencoder_harness.py Kaggle: async batching + ONNX precision + queue-discipline harness
  report.py                local: renders bench/phase4.md
  tests/
    __init__.py
    test_prerank_features.py
    test_prerank_consistency.py
    test_batcher.py
kaggle/
  phase4_notebook.ipynb    thin driver: installs deps, imports py/rank,
                           runs prerank_mlp + crossencoder_harness, writes
                           bench/results/*.json as each piece completes
```

New dependencies (`pyproject.toml`, a new `rank` extra, Kaggle-side only
where GPU-specific): `torch`, `onnx`, `onnxruntime-gpu` (Kaggle),
`onnxruntime` (CPU, for local test fixtures if any pure test needs it),
`transformers` (cross-encoder model loading).

Each of the Kaggle driver's sub-experiments (prerank+consistency, batching
sweep, precision comparison, queue discipline) writes its own
`bench/results/*.json` as it completes, not one file at the very end — per
§2's session-interruption concern.

## 10. Testing

- `test_prerank_features.py`: feature vector matches hand-computed values
  for a small fixture (known BM25/dense scores, known doc length).
- `test_prerank_consistency.py`: overlap fraction on a small hand-constructed
  true-top-k vs. survivor-set example, including the edge case where the
  survivor set is a strict superset (consistency = 1.0) and strict subset
  (consistency < 1.0).
- `test_batcher.py`: with a mock clock, confirms a batch flushes at exactly
  `max_batch_size` requests OR at `max_wait_ms` elapsed, whichever comes
  first; confirms an empty queue never flushes early.
- MLP training/eval, cross-encoder scoring, ONNX export/quantization, and
  the full batching/precision/queue-discipline harness are Kaggle-only,
  exercised by running — like every GPU-bound or real-data driver in this
  repo (Phase 3's `encode.py`/`ann_sweep.py`), correctness is checked by
  their output, not by pytest.

## 11. Exit artifact

`bench/phase4.md` (mirrors `bench/phase1.md`/`phase2a.md`/`phase3.md`'s
structure): candidate-pool methodology stated up front (lexical-only, depth
1000, no dense fusion — a documented scope limitation, same convention as
Phase 3's subset-size limitation), the prerank-consistency number, the
batch-window tradeoff plot, the 3-way precision table (latency × throughput
× NDCG@10 delta), the queue-discipline comparison (shed vs. unbounded p99),
and full run configuration (model, device, git SHA, hardware) per the
project's "a number without its config is not a result" rule.
`bench/results/rank-prerank.json`, `bench/results/rank-batching.json`,
`bench/results/rank-precision.json`, `bench/results/rank-queue.json` are
committed raw JSON, same as every prior phase.
