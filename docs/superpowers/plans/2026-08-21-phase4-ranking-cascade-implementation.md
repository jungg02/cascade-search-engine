# Phase 4: Ranking Cascade and Heterogeneous Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pre-rank (MLP) + rank (cross-encoder) cascade over Phase 1's existing lexical candidates, measure prerank-consistency, and characterize the cross-encoder's serving behavior (dynamic batching, fp32/fp16/INT8 precision, load-shedding vs. unbounded queueing) — producing `bench/phase4.md`.

**Architecture:** Local, CPU/MPS-only preparation (`py/rank/encode_candidates.py`) plus five pure, unit-tested modules importable both locally and on Kaggle (`prerank_features.py`, `prerank_consistency.py`, `batcher.py`) and two GPU-bound drivers (`prerank_mlp.py`, `crossencoder_harness.py`) that are smoke-tested locally at tiny scale (a real but tiny model, a handful of examples, CPU-only) and run for real at full scale on a Kaggle Notebook via a single driver script. Results come back as `bench/results/*.json`, downloaded from Kaggle by hand and dropped into the repo — this project cannot execute code on Kaggle directly. `py/rank/report.py` then renders `bench/phase4.md` locally, no GPU needed.

**Tech Stack:** Python 3.12, `torch` (already a dependency via the `dense` extra), `transformers` (cross-encoder model + tokenizer), `onnx`/`onnxruntime` (CPU locally, `onnxruntime-gpu` on Kaggle) + `onnxconverter-common` (fp16 conversion), reusing `harness.metrics`, `harness.runfile`, `harness.datasets`, `harness.histogram.LatencyRecorder`, `harness.runmeta.run_metadata`, and `dense.encode`'s `apply_query_prefix`/`select_device` unchanged.

**Spec:** `docs/superpowers/specs/2026-08-21-phase4-ranking-cascade-design.md`

## Global Constraints

- **No local GPU; Kaggle Notebook (T4) for all GPU-bound work.** This session cannot execute code on Kaggle — the user runs `kaggle/phase4_driver.py` there themselves and downloads `bench/results/rank-*.json` back into the repo. Every GPU-bound task in this plan therefore has two levels: a full-scale Kaggle run (real numbers, not reproducible locally) and a tiny local smoke test (a handful of synthetic or real-but-small examples, CPU-only, using the *real* model at tiny scale) that catches code bugs before they cost a Kaggle session. This mirrors Phase 3's `--limit` smoke-test pattern for its own expensive encode step.
- **Candidate pool: Phase 1's existing WAND runs, unmodified.** `runs/cascade-wand.{dev,dl19,dl20}.txt` (depth 1000, already on disk, `harness.runfile.read_run`) — no re-query of the C++ index, no fusion with Phase 3's dense channel. Verified at plan-writing time: `cascade-wand.dev.txt` has 6,974,879 lines (≈1000/query across 6,980 queries — some queries have slightly fewer than 1000 hits), `cascade-wand.dl19.txt`/`dl20.txt` are depth 1000 for 43/54 queries respectively (per `runs/manifest.json`'s `hits: 1000` field on each `cascade-wand` entry). This "unmodified" promise holds for every candidate pool that is actually *scored and reported* (dl19/dl20's full depth-1000, used everywhere — prerank-consistency's true-top-k, cascade NDCG@10, batching/precision/queue harnesses). It does **not** extend to dev's role as MLP *training-data source*: see the negative-sampling bullet below for a real scale correction found during Task 2's implementation (dev's union of depth-1000-per-query candidates is 3.77M unique docs, not the "tens of thousands" this plan originally estimated — an 8+ hour encoding job at this project's established `batch_size=32` MPS rate).
- **Relevance data:** dev qrels are sparse binary (verified: `harness.datasets.load_qrels("dev")` returns exactly one relevant docid per query in the sampled cases at plan-writing time — MS MARCO dev's standard shape), used only for MLP *training* (never for reporting NDCG/recall, since dev's `rel_threshold` is 1, not the project's dl19/dl20 convention of 2). dl19+dl20 (97 queries total, graded relevance, `rel_threshold=2` via `harness.datasets.rel_threshold`) are the held-out eval set for every reported metric — MLP never sees them during training.
- **Encoder reuse:** dense-score features use the exact same model/convention as Phase 3 — `BAAI/bge-small-en-v1.5` via `sentence-transformers`, L2-normalized, asymmetric query prefix. Task 2 imports `dense.encode.apply_query_prefix` and `dense.encode.select_device` directly rather than redefining them.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace hub ID) via `transformers.AutoModelForSequenceClassification`/`AutoTokenizer` directly (not `sentence_transformers.CrossEncoder`) — direct `transformers` access is what ONNX export needs, and using the same access path for the fp32 baseline keeps every precision variant's code path identical except for the ONNX Runtime session's execution provider and quantization.
- **Doc length feature:** character count of the candidate's passage text (`len(text)`) — not a token count, to avoid a second tokenizer dependency in the local, non-GPU encoding step.
- **Feature scaling:** BM25 score, dense score, and doc length are on incompatible scales (BM25 unbounded, dense cosine in [-1, 1], doc length in the hundreds to thousands of characters). The MLP trains on z-score-normalized features (mean/std computed once on the **dev** training set, the same fixed values applied to dl19/dl20 at eval time — never recomputed on eval data, which would leak eval-set statistics into normalization).
- **Negative sampling for MLP training, and dev's encoding-scope correction (ruling, made during Task 2's implementation):** dev's qrels give ~1 positive per query. Training pairs: the 1 labeled-relevant doc (label 1) plus 4 negatives (label 0) sampled uniformly at random, without replacement, from that query's WAND candidates, seeded (`np.random.default_rng(0)`) for reproducibility — 5 training examples per dev query with a valid positive. **Correction:** the candidate pool negatives are sampled from is each query's **top 50** WAND-scored candidates (`rank.encode_candidates.truncate_to_top_k`, `DEV_NEGATIVE_POOL_DEPTH = 50`), not the full up-to-999 non-positive candidates as originally written. This was forced by a real scale finding: the union of dev's *un*truncated depth-1000 candidates across 6,980 queries is 3,767,002 unique docs — encoding dense-score features for all of them measured at ~7-8 hours (batch_size=32, ~128 docs/sec on this project's MPS setup), against this plan's original "tens of thousands, low tens of minutes" estimate. Truncating to top-50 bounds dev's contribution to at most 6,980 × 50 = 349,000 (docid, query) slots (most of the encoding work), bringing the job back to the originally-intended tens-of-minutes scale. This is also a better-justified training signal, not just a workaround: uniform sampling over up to 999 mostly-irrelevant tail candidates gives the MLP mostly "easy" negatives, whereas the top-50 BM25-ranked candidates are the ones actually competing with the positive for pre-rank survival — the harder, more informative negatives for this specific task. Cost if this ruling is wrong: some dev queries whose single qrels-positive falls outside BM25's top 50 (previously usable at depth 1000) are now skipped for training, modestly shrinking the effective training set; this does not affect dl19/dl20, which are never truncated and remain the full, unmodified depth-1000 eval candidate pool for every reported metric.
- **ONNX quantization: dynamic, not static/calibration-based.** `onnxruntime.quantization.quantize_dynamic` needs no calibration dataset and is the simpler path (matching the design spec's explicit ONNX-over-TensorRT tradeoff) — INT8 activation ranges are computed at runtime, not pre-calibrated. This is a real, stated tradeoff (dynamic quantization can be less accurate than calibrated static quantization), not assumed to be free.
- **fp16 conversion:** `onnxconverter_common.float16.convert_float_to_float16` on the exported fp32 ONNX graph — a separate `.onnx` file, not a runtime cast.
- **INT8-on-GPU is a measured question, not an assumption.** ONNX Runtime's `CUDAExecutionProvider` may not accelerate every dynamically-quantized op — some can silently fall back to CPU. The harness measures and reports whatever latency actually results; a slower-than-expected INT8 number is a real finding to report, not a bug to hide or route around.
- **Latency: p50/p95/p99, never a mean** (project-wide rule, `harness.histogram.LatencyRecorder` already enforces this — reused unchanged).
- **Batching sweep grid:** `max_batch_size ∈ {1, 8, 32}` × `max_wait_ms ∈ {0, 5, 20}` — 9 points, fp32 only.
- **Precision comparison:** run only at the single (max_batch_size, max_wait_ms) point with the **highest measured throughput** in the fp32 sweep (a deterministic argmax over the 9 sweep points — no interactive judgment call, since this runs unattended on Kaggle) — fp32 (already measured), fp16, INT8 at that one setting. Per precision: latency, throughput, NDCG@10 delta vs. the fp32 cascade result.
- **Queue discipline:** 2 disciplines (load-shedding: reject when the queue is full; unbounded: never reject) × 3 sustained overload levels (arrival rate as a multiple of measured single-setting fp32 throughput: 1.5x, 2x, 3x) — 6 runs, not a sweep.
- **Kaggle provenance gap, and its fix:** `harness.runmeta.run_metadata()` shells out to `sysctl` for hardware fields and to `git` for `git_sha`/`git_dirty` — both fail silently to `"unknown"`/`None` on Kaggle (Linux, no `sysctl`; no git repo present, since only `py/rank/*.py` is uploaded, not the whole repo). Task 8's driver fixes this two ways: (1) a `SOURCE_GIT_SHA` constant at the top of `kaggle/phase4_driver.py`, filled in by hand from a local `git rev-parse HEAD` immediately before uploading, merged into every output JSON's provenance, overriding the `"unknown"` `git_sha`; (2) a `kaggle_hardware_info()` helper capturing `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_properties(0).total_memory`, merged in alongside (not replacing) `run_metadata()`'s CPU-oriented fields, since a GPU is the thing that actually matters for this phase's numbers.
- Every `bench/results/rank-*.json` file carries `harness.runmeta.run_metadata()` (as fixed above) plus its own full config — "a number without its config is not a result," enforced project-wide already.
- Python: run every local driver as a module with `PYTHONPATH=py` (`uv run python -m rank.foo`), matching every other phase's established convention.

---

## File Structure

```
pyproject.toml                                    modified — add `rank` extra
README.md                                          modified — "Running Phase 4" section (Task 10)

py/rank/__init__.py                                new
py/rank/encode_candidates.py                       new — Task 2
py/rank/tests/__init__.py                          new
py/rank/tests/test_encode_candidates.py            new — Task 2
py/rank/prerank_features.py                        new — Task 3
py/rank/tests/test_prerank_features.py             new — Task 3
py/rank/prerank_consistency.py                     new — Task 4
py/rank/tests/test_prerank_consistency.py          new — Task 4
py/rank/batcher.py                                 new — Task 5
py/rank/tests/test_batcher.py                      new — Task 5
py/rank/prerank_mlp.py                              new — Task 6
py/rank/tests/test_prerank_mlp.py                  new — Task 6 (tiny local smoke test)
py/rank/crossencoder_harness.py                    new — Task 7
py/rank/tests/test_crossencoder_harness.py         new — Task 7 (tiny local smoke test)
py/rank/report.py                                  new — Task 9

kaggle/phase4_driver.py                             new — Task 8 (uploaded to / pasted into Kaggle)
kaggle/README.md                                    new — Task 8 (how to run it there)

data/rank-dense-scores.jsonl                       new, gitignored — Task 2
data/rank-candidate-texts.json                     new, gitignored — Task 2

bench/results/rank-prerank.json                    new, committed — Task 8 (downloaded from Kaggle)
bench/results/rank-batching.json                   new, committed — Task 8 (downloaded from Kaggle)
bench/results/rank-precision.json                  new, committed — Task 8 (downloaded from Kaggle)
bench/results/rank-queue.json                      new, committed — Task 8 (downloaded from Kaggle)
bench/plots/phase4-batching.png                    new, committed — Task 9
bench/phase4.md                                    new, committed — Task 9/10
```

---

### Task 1: Dependencies and package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `py/rank/__init__.py`
- Create: `py/rank/tests/__init__.py`

**Interfaces:**
- Produces: the `rank` extra (`transformers`, `onnx`, `onnxruntime`, `onnxconverter-common` — CPU packages, for local smoke tests and code-sharing with the Kaggle driver; `torch` is already present via the existing `dense` extra), an importable `py/rank` package, `py/rank/tests` collected by pytest.

- [ ] **Step 1: Add the `rank` extra to `pyproject.toml`**

Add this block after the existing `dense = [...]` extra (before `[tool.uv]`):

```toml
# CPU-only versions of everything ONNX-related, for local smoke tests and for
# sharing code with the Kaggle driver (which additionally installs
# onnxruntime-gpu there -- see kaggle/README.md). torch is already present via
# the `dense` extra.
rank = [
    "transformers>=4.46.0",
    "onnx>=1.17.0",
    "onnxruntime>=1.20.0",
    "onnxconverter-common>=1.14.0",
]
```

Also update `testpaths` in `[tool.pytest.ini_options]`:

```toml
testpaths = ["py/tests", "py/server/tests", "py/dense/tests", "py/rank/tests"]
```

- [ ] **Step 2: Sync**

Run:
```bash
uv sync --extra dev --extra baselines --extra dense --extra rank
```
Expected: resolves and installs `transformers`, `onnx`, `onnxruntime`, `onnxconverter-common` (torch/sentence-transformers already present from the `dense` extra), no build errors — unlike `hnswlib`, none of these need the SDK header workaround.

- [ ] **Step 3: Create the package skeleton**

`py/rank/__init__.py` (empty file) and `py/rank/tests/__init__.py` (empty file).

- [ ] **Step 4: Verify imports and existing tests**

Run:
```bash
uv run python3 -c "import transformers, onnx, onnxruntime, onnxconverter_common; print('all import OK')"
uv run pytest -q
```
Expected: `all import OK`, and the existing 65 tests still pass (0 new tests yet — `py/rank/tests` is empty except `__init__.py`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock py/rank/__init__.py py/rank/tests/__init__.py
git commit -m "phase4: add ranking-cascade dependencies and package scaffolding"
```

---

### Task 2: Candidate encoding

**Files:**
- Create: `py/rank/encode_candidates.py`
- Test: `py/rank/tests/test_encode_candidates.py`

**Interfaces:**
- Consumes: `runs/cascade-wand.{dev,dl19,dl20}.txt` (Phase 1, via `harness.runfile.read_run`), `harness.datasets.load_queries(query_set: str) -> dict[str, str]`, `harness.datasets.iter_docs(limit: int | None = None) -> Iterator[tuple[str, str]]`, `dense.encode.apply_query_prefix(text: str) -> str`, `dense.encode.select_device() -> str`.
- Produces: `unique_candidate_docids(runs: list[dict[str, dict[str, float]]]) -> set[str]` (pure, unit-tested), `truncate_to_top_k(run: dict[str, dict[str, float]], k: int) -> dict[str, dict[str, float]]` (pure, unit-tested — keeps each query's k highest-scoring candidates; Task 6's local smoke test imports this directly), and `read_run_truncated(path: str | Path, k: int) -> dict[str, dict[str, float]]` (a streaming `read_run`+`truncate_to_top_k` fused into one pass, never materializing more than k entries per query — Task 8's Kaggle driver imports this instead of the two-step form, since the two-step form was a real, avoidable memory-pressure contributor to a Kaggle OOM found during this phase's actual Kaggle run; both forms produce byte-identical output given the same inputs, verified by a dedicated test). CLI writes `data/rank-dense-scores.jsonl` — one JSON object per line, `{"qid": str, "docid": str, "dense_score": float, "doc_length": int}`, for every (query, candidate) pair across dl19+dl20's full depth-1000 WAND runs plus dev's **top-50-per-query truncated** run (see Global Constraints' negative-sampling correction) — and `data/rank-candidate-texts.json` — a flat `{docid: passage_text}` mapping for every unique candidate docid retained after that truncation, so Task 8's Kaggle driver can build real cross-encoder input pairs without ever needing the full 8.8M-passage corpus there.

- [ ] **Step 1: Write the failing test**

`py/rank/tests/test_encode_candidates.py`:

```python
"""unique_candidate_docids is pure (no I/O), tested against small synthetic
run dicts rather than the real WAND run files."""

from __future__ import annotations

from pathlib import Path

from harness.runfile import read_run
from rank.encode_candidates import read_run_truncated, truncate_to_top_k, unique_candidate_docids


def test_union_across_multiple_runs():
    run_a = {"q1": {"d1": 1.0, "d2": 0.5}}
    run_b = {"q2": {"d2": 0.9, "d3": 0.1}}
    result = unique_candidate_docids([run_a, run_b])
    assert result == {"d1", "d2", "d3"}


def test_empty_runs_list_is_empty_set():
    assert unique_candidate_docids([]) == set()


def test_single_run_single_query():
    run_a = {"q1": {"d1": 1.0}}
    assert unique_candidate_docids([run_a]) == {"d1"}


def test_truncate_to_top_k_keeps_highest_scores():
    run = {"q1": {"d1": 1.0, "d2": 5.0, "d3": 3.0, "d4": 2.0}}
    result = truncate_to_top_k(run, k=2)
    assert result == {"q1": {"d2": 5.0, "d3": 3.0}}


def test_truncate_to_top_k_leaves_short_queries_unchanged():
    run = {"q1": {"d1": 1.0}}
    result = truncate_to_top_k(run, k=5)
    assert result == {"q1": {"d1": 1.0}}


def test_read_run_truncated_matches_read_run_then_truncate(tmp_path: Path):
    # A real run file, deliberately pre-sorted descending per query (the
    # invariant read_run_truncated relies on -- matches how
    # harness.runfile.write_run always writes, and how every WAND run in
    # this project is produced).
    run_file = tmp_path / "run.txt"
    run_file.write_text(
        "q1 Q0 d2 1 5.0 tag\n"
        "q1 Q0 d3 2 3.0 tag\n"
        "q1 Q0 d4 3 2.0 tag\n"
        "q1 Q0 d1 4 1.0 tag\n"
        "q2 Q0 d5 1 9.0 tag\n"
    )
    streamed = read_run_truncated(run_file, k=2)
    materialized = truncate_to_top_k(read_run(run_file), k=2)
    assert streamed == materialized
    assert streamed == {"q1": {"d2": 5.0, "d3": 3.0}, "q2": {"d5": 9.0}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest py/rank/tests/test_encode_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.encode_candidates'`.

- [ ] **Step 3: Write `py/rank/encode_candidates.py`**

```python
"""Dense-score and doc-length features for Phase 4's pre-rank MLP.

Not a re-run of dense.encode.py -- that encoded Phase 3's 1M-passage
*subset*; this phase's WAND candidates are a different, much smaller
document set. dl19/dl20 stay at their full depth-1000 (every candidate there
is genuinely scored, for prerank-consistency's true-top-k). dev is truncated
to each query's top DEV_NEGATIVE_POOL_DEPTH WAND-scored candidates before
encoding -- dev's *un*truncated union across 6,980 queries is 3.77M unique
docs (measured; an 8+ hour encoding job), when MLP training only ever
samples 1 positive + 4 negatives per query from it. Encodes just that
truncated/full-depth set, plus the dev/dl19/dl20 queries, and writes a flat
scalar feature file -- Kaggle only needs the (qid, docid) -> dense_score
scalar as an MLP feature, not the raw embeddings, so nothing here ships a
1.5GB-scale artifact the way Phase 3's encode.py did.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.datasets import iter_docs, load_queries
from harness.runfile import read_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

QUERY_SETS = ("dev", "dl19", "dl20")
DEV_NEGATIVE_POOL_DEPTH = 50


def unique_candidate_docids(runs: list[dict[str, dict[str, float]]]) -> set[str]:
    """Every docid that appears anywhere across a list of loaded run dicts."""
    docids: set[str] = set()
    for run in runs:
        for candidates in run.values():
            docids.update(candidates)
    return docids


def truncate_to_top_k(run: dict[str, dict[str, float]], k: int) -> dict[str, dict[str, float]]:
    """Keep only each query's k highest-scoring candidates. Used to bound
    dev's contribution to the encoding workload -- Task 6/Task 8 apply this
    identically to dev_run before sampling training negatives, so the
    encoded feature set and the training-sampling pool always agree."""
    return {
        qid: dict(sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:k])
        for qid, candidates in run.items()
    }


def load_wand_runs() -> dict[str, dict[str, dict[str, float]]]:
    return {
        query_set: read_run(RUNS_DIR / f"cascade-wand.{query_set}.txt")
        for query_set in QUERY_SETS
    }


def read_run_truncated(path: str | Path, k: int) -> dict[str, dict[str, float]]:
    """Streaming equivalent of `harness.runfile.read_run(path)` followed by
    `truncate_to_top_k(run, k)`, but never holds more than k entries per
    query in memory -- relies on each query's block being pre-sorted
    descending by score, which is the format `harness.runfile.write_run`
    always produces and what every WAND run in this project is written by
    (verified directly against cascade-wand.dev.txt: no query's block ever
    has a later line score higher than an earlier one).

    Exists because dev's untruncated file is 6,974,879 lines / 374MB --
    materializing that whole nested dict via read_run() just to immediately
    discard all but the top k per query (at most 349,000 of 3.77M entries)
    is real, avoidable memory pressure on a resource-constrained host. Found
    during this phase's real Kaggle run: the driver's dev_run load was the
    single largest avoidable allocation contributing to an out-of-memory
    kernel restart.
    """
    run: dict[str, dict[str, float]] = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            qid, _, docid, _, score, *_ = fields
            bucket = run.setdefault(qid, {})
            if len(bucket) >= k:
                continue
            bucket[docid] = float(score)
    return run


def collect_candidate_texts(candidate_docids: set[str]) -> dict[str, str]:
    """Single streaming pass over the corpus, matching dense/subset.py's pattern
    for pulling a bounded docid set out of the full 8.8M-passage stream."""
    texts: dict[str, str] = {}
    for docid, text in iter_docs():
        if docid in candidate_docids:
            texts[docid] = text
            if len(texts) == len(candidate_docids):
                break
    missing = candidate_docids - texts.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} candidate docids never seen while streaming the "
            f"corpus (e.g. {sorted(missing)[:5]})"
        )
    return texts


def main() -> None:
    # Deliberately local, not module-level: Task 8's Kaggle driver imports
    # this module only for DEV_NEGATIVE_POOL_DEPTH/truncate_to_top_k (pure,
    # no ML dependency), and Kaggle never uploads py/dense/ -- a module-level
    # `from dense.encode import ...` would crash that import with
    # ModuleNotFoundError before the driver ever reached the names it
    # actually wants. Only main() (the local, non-Kaggle encoding driver)
    # needs the real encoder.
    from sentence_transformers import SentenceTransformer

    from dense.encode import MODEL_NAME, apply_query_prefix, select_device

    wand_runs = load_wand_runs()
    wand_runs["dev"] = truncate_to_top_k(wand_runs["dev"], DEV_NEGATIVE_POOL_DEPTH)
    candidate_docids = unique_candidate_docids(list(wand_runs.values()))
    print(
        f"unique candidate docids across dev (top {DEV_NEGATIVE_POOL_DEPTH}/query)"
        f"+dl19+dl20: {len(candidate_docids)}"
    )

    device = select_device()
    print(f"device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("encoding candidate passages...")
    candidate_texts = collect_candidate_texts(candidate_docids)
    ordered_docids = sorted(candidate_texts)
    doc_embeddings = model.encode(
        [candidate_texts[d] for d in ordered_docids],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    doc_embedding_by_id = dict(zip(ordered_docids, doc_embeddings))
    doc_length_by_id = {d: len(candidate_texts[d]) for d in ordered_docids}

    # Written alongside the scalar features so Task 8's Kaggle driver can
    # build real (query_text, passage_text) pairs for the cross-encoder --
    # query text comes from harness.datasets.load_queries directly (no local
    # artifact needed for that half), but passage text needs this file since
    # Kaggle never sees the 8.8M-passage corpus itself, only this bounded
    # candidate-scoped lookup.
    candidate_texts_path = DATA_DIR / "rank-candidate-texts.json"
    candidate_texts_path.write_text(json.dumps(candidate_texts))
    print(f"wrote {candidate_texts_path.relative_to(REPO_ROOT)} ({len(candidate_texts)} docs)")

    output_path = DATA_DIR / "rank-dense-scores.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(output_path, "w") as out:
        for query_set in QUERY_SETS:
            queries = load_queries(query_set)
            run = wand_runs[query_set]
            qids = sorted(run)
            print(f"encoding {len(qids)} {query_set} queries...")
            query_texts = [apply_query_prefix(queries[qid]) for qid in qids]
            query_embeddings = model.encode(
                query_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            for qid, q_embedding in zip(qids, query_embeddings):
                for docid in run[qid]:
                    dense_score = float(np.dot(q_embedding, doc_embedding_by_id[docid]))
                    row = {
                        "qid": qid,
                        "docid": docid,
                        "dense_score": round(dense_score, 6),
                        "doc_length": doc_length_by_id[docid],
                    }
                    out.write(json.dumps(row) + "\n")
                    rows_written += 1

    print(f"wrote {output_path.relative_to(REPO_ROOT)} ({rows_written} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest py/rank/tests/test_encode_candidates.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the real candidate encoding**

Run: `PYTHONPATH=py uv run python -m rank.encode_candidates`
Expected: prints the unique candidate docid count (bounded now — dev contributes at most `6,980 × 50 = 349,000` (qid, docid) slots instead of its untruncated ~6.97M, since dev's WAND candidates are truncated to each query's top 50 by score before encoding; dl19/dl20 stay full depth, contributing up to 97,000 more slots, `~43,000 + ~54,000`), `device: mps` (or `cpu`), a progress bar for candidate encoding, `wrote data/rank-candidate-texts.json (N docs)`, then progress bars for query encoding, then `wrote data/rank-dense-scores.jsonl (N rows)` where the row count is at most `349,000 + 43,000 + 54,000 ≈ 446,000` (fewer in practice — some dev queries return under 50 WAND hits). At this project's established ~128 docs/sec encoding rate (batch_size=32 on MPS), expect roughly 30-60 minutes, matching this plan's original tens-of-minutes estimate (the pre-fix, untruncated version of this step measured 3,767,002 unique docs and a 7-8 hour ETA — see Global Constraints' negative-sampling correction). Still a real encoding pass; run it in the background and poll rather than blocking on it, same as Phase 3's Task 3.

- [ ] **Step 6: Sanity-check the output**

Run:
```bash
wc -l data/rank-dense-scores.jsonl
uv run python3 -c "
import json
with open('data/rank-dense-scores.jsonl') as f:
    row = json.loads(f.readline())
print(row)
assert set(row) == {'qid', 'docid', 'dense_score', 'doc_length'}
assert -1.0 <= row['dense_score'] <= 1.0
assert row['doc_length'] > 0

texts = json.loads(open('data/rank-candidate-texts.json').read())
print('candidate texts:', len(texts))
assert row['docid'] in texts
assert texts[row['docid']]
"
```
Expected: a line count at most `349,000 + 43,000 + 54,000 ≈ 446,000` (well under the untruncated WAND run files' combined 7,071,879-line count), the sample row has all four expected keys with plausible values, and its docid resolves to a non-empty passage text in `rank-candidate-texts.json`.

- [ ] **Step 7: Commit**

```bash
git add py/rank/encode_candidates.py py/rank/tests/test_encode_candidates.py
git commit -m "phase4: encode WAND candidates' dense score + doc length features"
```
(`data/rank-dense-scores.jsonl` and `data/rank-candidate-texts.json` stay untracked — `/data/` is gitignored.)

---

### Task 3: Pre-rank features

**Files:**
- Create: `py/rank/prerank_features.py`
- Test: `py/rank/tests/test_prerank_features.py`

**Interfaces:**
- Consumes: nothing (pure, given already-loaded scalar inputs).
- Produces: `build_feature_vector(bm25_score: float, dense_score: float, doc_length: int) -> tuple[float, float, float]` (pure, unit-tested), `FeatureScaler` (dataclass: `mean: tuple[float, float, float]`, `std: tuple[float, float, float]`), `fit_scaler(feature_vectors: list[tuple[float, float, float]]) -> FeatureScaler` (pure), `FeatureScaler.transform(vector: tuple[float, float, float]) -> tuple[float, float, float]` (pure, z-score normalize, guards divide-by-zero when a feature's std is 0).

- [ ] **Step 1: Write the failing tests**

`py/rank/tests/test_prerank_features.py`:

```python
"""Pure feature-vector construction and scaling, tested against hand-computed
values -- matching this repo's style for pure retrieval-math functions."""

from __future__ import annotations

from rank.prerank_features import FeatureScaler, build_feature_vector, fit_scaler


def test_build_feature_vector_is_a_plain_tuple():
    result = build_feature_vector(bm25_score=12.5, dense_score=0.3, doc_length=200)
    assert result == (12.5, 0.3, 200.0)


def test_fit_scaler_hand_computed_mean_and_std():
    vectors = [(0.0, 0.0, 0.0), (10.0, 2.0, 100.0)]
    scaler = fit_scaler(vectors)
    assert scaler.mean == (5.0, 1.0, 50.0)
    # population std (ddof=0): sqrt(((0-5)^2 + (10-5)^2) / 2) = 5.0
    assert scaler.std == (5.0, 1.0, 50.0)


def test_scaler_transform_zscore():
    scaler = FeatureScaler(mean=(5.0, 1.0, 50.0), std=(5.0, 1.0, 50.0))
    result = scaler.transform((10.0, 2.0, 100.0))
    assert result == (1.0, 1.0, 1.0)


def test_scaler_transform_guards_zero_std():
    scaler = FeatureScaler(mean=(5.0, 1.0, 50.0), std=(0.0, 1.0, 50.0))
    result = scaler.transform((5.0, 2.0, 100.0))
    # first feature's std is 0 -- every value equals the mean, so it
    # normalizes to 0.0 rather than dividing by zero.
    assert result == (0.0, 1.0, 1.0)


def test_fit_scaler_single_vector_has_zero_std():
    scaler = fit_scaler([(3.0, 4.0, 5.0)])
    assert scaler.mean == (3.0, 4.0, 5.0)
    assert scaler.std == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest py/rank/tests/test_prerank_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.prerank_features'`.

- [ ] **Step 3: Write `py/rank/prerank_features.py`**

```python
"""Pre-rank feature vector: (BM25 score, dense score, doc length), z-score
normalized before the MLP sees them -- the three raw values are on
incompatible scales (BM25 unbounded, dense cosine in [-1, 1], doc length in
the hundreds to thousands of characters), and an unnormalized MLP would
either fail to learn or be dominated by whichever feature happens to have
the largest raw magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

FeatureVector = tuple[float, float, float]


def build_feature_vector(bm25_score: float, dense_score: float, doc_length: int) -> FeatureVector:
    return (float(bm25_score), float(dense_score), float(doc_length))


@dataclass
class FeatureScaler:
    mean: FeatureVector
    std: FeatureVector

    def transform(self, vector: FeatureVector) -> FeatureVector:
        return tuple(
            0.0 if std == 0.0 else (value - mean) / std
            for value, mean, std in zip(vector, self.mean, self.std)
        )


def fit_scaler(feature_vectors: list[FeatureVector]) -> FeatureScaler:
    n = len(feature_vectors)
    means = tuple(sum(v[i] for v in feature_vectors) / n for i in range(3))
    variances = tuple(
        sum((v[i] - means[i]) ** 2 for v in feature_vectors) / n for i in range(3)
    )
    stds = tuple(variance**0.5 for variance in variances)
    return FeatureScaler(mean=means, std=stds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest py/rank/tests/test_prerank_features.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add py/rank/prerank_features.py py/rank/tests/test_prerank_features.py
git commit -m "phase4: pre-rank feature vector + z-score scaler"
```

---

### Task 4: Prerank-consistency metric

**Files:**
- Create: `py/rank/prerank_consistency.py`
- Test: `py/rank/tests/test_prerank_consistency.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `prerank_consistency(true_top_k: list[str], survivors: set[str]) -> float` (pure, unit-tested) — fraction of `true_top_k` that appears in `survivors`.

- [ ] **Step 1: Write the failing tests**

`py/rank/tests/test_prerank_consistency.py`:

```python
"""prerank_consistency is pure -- same shape as dense/ann_sweep.py's
recall_at_k, tested against hand-constructed examples."""

from __future__ import annotations

from rank.prerank_consistency import prerank_consistency


def test_full_survival_is_1():
    true_top_k = ["d1", "d2", "d3"]
    survivors = {"d1", "d2", "d3", "d4"}  # strict superset
    assert prerank_consistency(true_top_k, survivors) == 1.0


def test_partial_survival():
    true_top_k = ["d1", "d2", "d3", "d4"]
    survivors = {"d1", "d3"}  # strict subset, 2 of 4 survive
    assert prerank_consistency(true_top_k, survivors) == 0.5


def test_no_survival_is_0():
    true_top_k = ["d1", "d2"]
    survivors = {"d3", "d4"}
    assert prerank_consistency(true_top_k, survivors) == 0.0


def test_empty_true_top_k_is_0_not_a_zero_division():
    assert prerank_consistency([], {"d1"}) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest py/rank/tests/test_prerank_consistency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.prerank_consistency'`.

- [ ] **Step 3: Write `py/rank/prerank_consistency.py`**

```python
"""Prerank-consistency: of the true top-k under the full ranker, how many
survive pre-ranking? A low number means the cascade discards good documents
before the expensive model ever sees them -- same shape as
dense/ann_sweep.py's recall_at_k, one level up the cascade.
"""

from __future__ import annotations


def prerank_consistency(true_top_k: list[str], survivors: set[str]) -> float:
    if not true_top_k:
        return 0.0
    found = sum(1 for docid in true_top_k if docid in survivors)
    return found / len(true_top_k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest py/rank/tests/test_prerank_consistency.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add py/rank/prerank_consistency.py py/rank/tests/test_prerank_consistency.py
git commit -m "phase4: prerank-consistency metric"
```

---

### Task 5: Dynamic batching queue logic

**Files:**
- Create: `py/rank/batcher.py`
- Test: `py/rank/tests/test_batcher.py`

**Interfaces:**
- Consumes: nothing (pure logic, driven by an injected clock function in tests).
- Produces: `DynamicBatcher` class — `__init__(self, max_batch_size: int, max_wait_ms: float, clock: Callable[[], float])`, `add(self, item) -> None`, `__len__(self) -> int` (current queue depth, the public way Task 7's queue-discipline harness checks fullness rather than reaching into a private attribute), `should_flush(self) -> bool`, `flush(self) -> list` (unit-tested with a fake clock — no real async/GPU involved at this layer).

- [ ] **Step 1: Write the failing tests**

`py/rank/tests/test_batcher.py`:

```python
"""DynamicBatcher's flush decision is pure given an injected clock -- tested
with a fake clock rather than real wall-clock sleeps, so these run in
milliseconds and never flake on timing."""

from __future__ import annotations

from rank.batcher import DynamicBatcher


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


def test_flushes_at_max_batch_size_before_wait_elapses():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=100, clock=clock)
    batcher.add("a")
    assert not batcher.should_flush()
    batcher.add("b")
    assert batcher.should_flush()
    assert batcher.flush() == ["a", "b"]


def test_flushes_at_max_wait_ms_before_batch_size_reached():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=5, clock=clock)
    batcher.add("a")
    assert not batcher.should_flush()
    clock.advance(5.0)
    assert batcher.should_flush()
    assert batcher.flush() == ["a"]


def test_empty_batcher_never_flushes():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=5, clock=clock)
    clock.advance(1000.0)
    assert not batcher.should_flush()


def test_flush_resets_the_batch_and_wait_window():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=100, clock=clock)
    batcher.add("a")
    batcher.add("b")
    assert batcher.flush() == ["a", "b"]
    assert not batcher.should_flush()
    batcher.add("c")
    assert not batcher.should_flush()


def test_len_reflects_current_queue_depth():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=100, clock=clock)
    assert len(batcher) == 0
    batcher.add("a")
    batcher.add("b")
    assert len(batcher) == 2
    batcher.flush()
    assert len(batcher) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest py/rank/tests/test_batcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.batcher'`.

- [ ] **Step 3: Write `py/rank/batcher.py`**

```python
"""Dynamic-batching queue logic: accumulate requests until either
max_batch_size or max_wait_ms is hit, whichever comes first. Pure given an
injected clock, so the batching *decision* is unit-tested without any real
concurrency or GPU -- crossencoder_harness.py wraps this in a flusher thread
that actually calls the model, guarding every add/should_flush/flush call
with its own lock (this class is deliberately not internally synchronized).
"""

from __future__ import annotations

from typing import Callable


class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: float, clock: Callable[[], float]) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._clock = clock
        self._items: list = []
        self._window_start: float | None = None

    def add(self, item) -> None:
        if not self._items:
            self._window_start = self._clock()
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def should_flush(self) -> bool:
        if not self._items:
            return False
        if len(self._items) >= self.max_batch_size:
            return True
        elapsed = self._clock() - self._window_start
        return elapsed >= self.max_wait_ms

    def flush(self) -> list:
        items, self._items = self._items, []
        self._window_start = None
        return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest py/rank/tests/test_batcher.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add py/rank/batcher.py py/rank/tests/test_batcher.py
git commit -m "phase4: dynamic-batching queue logic"
```

---

### Task 6: Pre-rank MLP

**Files:**
- Create: `py/rank/prerank_mlp.py`
- Test: `py/rank/tests/test_prerank_mlp.py`

**Interfaces:**
- Consumes: `data/rank-dense-scores.jsonl` (Task 2), `harness.runfile.read_run` (Phase 1 WAND runs), `harness.datasets.load_qrels`, `rank.prerank_features.{build_feature_vector, fit_scaler, FeatureScaler}` (Task 3).
- Produces: `build_mlp() -> torch.nn.Module` (2-input-hidden-layer, 3→16→1), `build_training_examples(dev_run, dev_qrels, dense_scores, doc_lengths, seed: int = 0, num_negatives: int = 4) -> tuple[list[FeatureVector], list[float]]` (pure given already-loaded inputs), `train_mlp(model, features, labels, scaler) -> None` (mutates `model` in place), `score_candidates(model, scaler, query_features: dict[str, FeatureVector]) -> dict[str, float]`. This module is both a **Kaggle-scale driver** (real dev/dl19/dl20 data, real training) and **locally smoke-tested** at tiny scale (Step 5 below) — unlike Phase 3's GPU-bound drivers, training a 3-input MLP has no GPU dependency at all, so this can be validated end-to-end locally before ever touching Kaggle.

- [ ] **Step 1: Write the failing test**

`py/rank/tests/test_prerank_mlp.py`:

```python
"""build_training_examples is pure given already-loaded run/qrels/feature
dicts. train_mlp/score_candidates are exercised on a tiny synthetic dataset
here (no GPU needed -- a 3-input MLP trains in milliseconds on CPU), which is
the same code path the Kaggle run uses at full scale."""

from __future__ import annotations

import torch

from rank.prerank_features import fit_scaler
from rank.prerank_mlp import (
    build_mlp,
    build_training_examples,
    score_candidates,
    train_mlp,
)


def test_build_training_examples_positive_plus_negatives():
    dev_run = {"q1": {"pos": 5.0, "neg1": 4.0, "neg2": 3.0, "neg3": 2.0, "neg4": 1.0, "neg5": 0.5}}
    dev_qrels = {"q1": {"pos": 1}}
    dense_scores = {("q1", d): 0.1 for d in dev_run["q1"]}
    doc_lengths = {("q1", d): 100 for d in dev_run["q1"]}
    features, labels = build_training_examples(
        dev_run, dev_qrels, dense_scores, doc_lengths, seed=0, num_negatives=4
    )
    assert len(features) == 5  # 1 positive + 4 negatives
    assert sum(labels) == 1.0  # exactly one positive label
    assert len(labels) == 5


def test_build_training_examples_skips_queries_with_no_qrels_positive():
    dev_run = {"q1": {"d1": 1.0}}
    dev_qrels = {}  # q1 has no judged-relevant doc at all
    dense_scores = {("q1", "d1"): 0.1}
    doc_lengths = {("q1", "d1"): 100}
    features, labels = build_training_examples(
        dev_run, dev_qrels, dense_scores, doc_lengths, seed=0, num_negatives=4
    )
    assert features == []
    assert labels == []


def test_train_and_score_tiny_synthetic_dataset():
    # A trivially separable dataset: feature value > 0 means relevant.
    features = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)]
    labels = [1.0, 1.0, 0.0, 0.0]
    scaler = fit_scaler(features)
    torch.manual_seed(0)  # before build_mlp(): seeds weight init, not just training
    model = build_mlp()
    train_mlp(model, features, labels, scaler, epochs=200)

    query_features = {"pos_doc": (1.0, 1.0, 1.0), "neg_doc": (-1.0, -1.0, -1.0)}
    scores = score_candidates(model, scaler, query_features)
    assert scores["pos_doc"] > scores["neg_doc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest py/rank/tests/test_prerank_mlp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.prerank_mlp'`.

- [ ] **Step 3: Write `py/rank/prerank_mlp.py`**

```python
"""Pre-rank MLP: a small 3-input model (BM25 score, dense score, doc length)
trained on dev's sparse binary qrels (1 positive + 4 sampled negatives per
query, sampled from that query's top-50 WAND-scored candidates -- see
rank.encode_candidates.truncate_to_top_k and this phase's negative-sampling
Global Constraint -- with a judged-relevant doc), evaluated on dl19+dl20 --
reduces each query's up-to-1000 WAND candidates to the top ~100 by MLP score.

Training a 3-input MLP has no GPU dependency -- this runs identically on CPU
locally (for tests and smoke checks) and on Kaggle (at full dev/dl19/dl20
scale), unlike the cross-encoder harness which genuinely needs a GPU to be
fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rank.prerank_features import FeatureScaler, FeatureVector, build_feature_vector

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

NUM_NEGATIVES = 4
SEED = 0
TOP_K_SURVIVORS = 100


def build_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))


def load_dense_scores_and_lengths(
    path: Path,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    dense_scores: dict[tuple[str, str], float] = {}
    doc_lengths: dict[tuple[str, str], int] = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["qid"], row["docid"])
            dense_scores[key] = row["dense_score"]
            doc_lengths[key] = row["doc_length"]
    return dense_scores, doc_lengths


def build_training_examples(
    dev_run: dict[str, dict[str, float]],
    dev_qrels: dict[str, dict[str, int]],
    dense_scores: dict[tuple[str, str], float],
    doc_lengths: dict[tuple[str, str], int],
    seed: int = SEED,
    num_negatives: int = NUM_NEGATIVES,
) -> tuple[list[FeatureVector], list[float]]:
    rng = np.random.default_rng(seed)
    features: list[FeatureVector] = []
    labels: list[float] = []
    for qid, candidates in dev_run.items():
        # Defensive: restrict to docids we actually have features for. Callers
        # are expected to pass an already-truncated dev_run (see
        # rank.encode_candidates.truncate_to_top_k) so this is normally a
        # no-op, but it protects against a silent KeyError below if a caller
        # ever passes an untruncated run that outruns what was encoded.
        available = [d for d in candidates if (qid, d) in dense_scores]
        judgments = dev_qrels.get(qid, {})
        positives = [docid for docid in available if judgments.get(docid, 0) > 0]
        if not positives:
            continue
        positive_docid = positives[0]
        negative_pool = [d for d in available if d != positive_docid]
        sample_size = min(num_negatives, len(negative_pool))
        negative_docids = rng.choice(negative_pool, size=sample_size, replace=False)

        for docid, label in [(positive_docid, 1.0)] + [(d, 0.0) for d in negative_docids]:
            key = (qid, docid)
            features.append(
                build_feature_vector(
                    bm25_score=candidates[docid],
                    dense_score=dense_scores[key],
                    doc_length=doc_lengths[key],
                )
            )
            labels.append(label)
    return features, labels


def train_mlp(
    model: nn.Module,
    features: list[FeatureVector],
    labels: list[float],
    scaler: FeatureScaler,
    epochs: int = 50,
    lr: float = 0.01,
) -> None:
    scaled = torch.tensor([scaler.transform(f) for f in features], dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(scaled)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()


def score_candidates(
    model: nn.Module, scaler: FeatureScaler, query_features: dict[str, FeatureVector]
) -> dict[str, float]:
    model.eval()
    docids = list(query_features)
    scaled = torch.tensor([scaler.transform(query_features[d]) for d in docids], dtype=torch.float32)
    with torch.no_grad():
        logits = model(scaled).squeeze(1)
    return dict(zip(docids, (float(x) for x in logits)))


def survivors_for_query(
    model: nn.Module,
    scaler: FeatureScaler,
    candidates: dict[str, float],
    dense_scores: dict[tuple[str, str], float],
    doc_lengths: dict[tuple[str, str], int],
    qid: str,
    top_k: int = TOP_K_SURVIVORS,
) -> set[str]:
    query_features = {
        docid: build_feature_vector(
            bm25_score=bm25_score,
            dense_score=dense_scores[(qid, docid)],
            doc_length=doc_lengths[(qid, docid)],
        )
        for docid, bm25_score in candidates.items()
    }
    scores = score_candidates(model, scaler, query_features)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {docid for docid, _ in ranked[:top_k]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest py/rank/tests/test_prerank_mlp.py -v`
Expected: 3 passed.

- [ ] **Step 5: Sanity-check on a tiny local slice (no Kaggle needed)**

Run:
```bash
PYTHONPATH=py uv run python3 -c "
from rank.encode_candidates import DEV_NEGATIVE_POOL_DEPTH, load_wand_runs, truncate_to_top_k
from rank.prerank_mlp import (
    build_mlp, build_training_examples, load_dense_scores_and_lengths, train_mlp,
)
from rank.prerank_features import fit_scaler
from harness.datasets import load_qrels
import itertools

wand_runs = load_wand_runs()
dev_run_full = truncate_to_top_k(wand_runs['dev'], DEV_NEGATIVE_POOL_DEPTH)
dev_run = dict(itertools.islice(dev_run_full.items(), 50))  # tiny slice
dev_qrels = load_qrels('dev')
dense_scores, doc_lengths = load_dense_scores_and_lengths('data/rank-dense-scores.jsonl')

features, labels = build_training_examples(dev_run, dev_qrels, dense_scores, doc_lengths)
print(f'{len(features)} training examples from {len(dev_run)} queries')
scaler = fit_scaler(features)
model = build_mlp()
train_mlp(model, features, labels, scaler, epochs=50)
print('training completed without error')
"
```
Expected: prints a training-example count and `training completed without error`. Since dev's WAND candidates are truncated to each query's top 50 by BM25 score (`truncate_to_top_k`) before this slice is taken, a query's single qrels-positive is no longer guaranteed to survive that truncation — only queries where it does contribute 5 training examples (1 positive + 4 negatives), and the rest are silently skipped by `build_training_examples`'s `if not positives: continue`. Real measurement on this exact 50-query slice: 31 of 50 queries retained a positive within their top-50 BM25 candidates (155 = 31 × 5 training examples) — roughly 38% BM25-recall@50 attrition, not a bug. Expect a similar ~35-40% query-level attrition at Task 8's full 6,980-dev-query Kaggle scale (i.e. training on roughly 4,200-4,500 queries, not all 6,980) — this is expected behavior, not a sign anything broke. This exercises the real full-scale code path (Task 2's real `data/rank-dense-scores.jsonl`, the real dev WAND run, the real dev qrels) at a small enough slice to run in seconds — the actual Kaggle run in Task 8 is this same code, unmodified, over all 6,980 dev queries.

- [ ] **Step 6: Commit**

```bash
git add py/rank/prerank_mlp.py py/rank/tests/test_prerank_mlp.py
git commit -m "phase4: pre-rank MLP training and scoring"
```

---

### Task 7: Cross-encoder harness

**Files:**
- Create: `py/rank/crossencoder_harness.py`
- Test: `py/rank/tests/test_crossencoder_harness.py`

**Interfaces:**
- Consumes: `rank.batcher.DynamicBatcher` (Task 5), `harness.histogram.LatencyRecorder`, `harness.runmeta.run_metadata`.
- Produces: `export_onnx(model_name: str, onnx_path: Path) -> None`, `convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None`, `quantize_int8(fp32_path: Path, int8_path: Path) -> None`, `CrossEncoderSession` class wrapping an ONNX Runtime `InferenceSession` with a `.score(pairs: list[tuple[str, str]]) -> list[float]` method, `run_batching_sweep(session_factory, grid: list[tuple[int, float]], request_pairs: list[tuple[str, str]], probe_requests: int) -> list[dict]`, `run_queue_discipline(session_factory, discipline: str, arrival_rate: float, batcher_config: tuple[int, float], request_pairs: list[tuple[str, str]], duration_s: float) -> dict`. Both drive `harness.loadgen.run_open_loop` (the same generator Phase 2 uses) against a dedicated flusher thread rather than an asyncio loop — see the module docstring for why a blocking `session.score()` sharing an event loop with its own arrival generator is closed-loop in all but name. `run_batching_sweep` takes a request *budget* (`probe_requests`) rather than a duration because it is a saturating probe: `run_open_loop` precomputes its whole arrival list and drains it, so a fire-hose bounded only by a duration runs for offered/capacity seconds. This module is a **Kaggle-scale driver** (needs a real GPU to be *fast*, not to be *correct* — every function here also runs correctly, just slowly, on CPU) and is **locally smoke-tested** at tiny scale in fp32-only, CPU-only mode (Step 4 below, which runs this task's test file — the tiny-scale smoke test IS the test suite here, not a separate step), matching Phase 3's `--limit`-flag precedent for validating an expensive driver's logic before the real run.

- [ ] **Step 1: Write the failing test**

`py/rank/tests/test_crossencoder_harness.py`:

```python
"""ONNX export/quantization and the batching/queue harness are exercised
here on the real (tiny) cross-encoder model, fp32-only, CPU-only -- slow
compared to a GPU but correct, and small enough to run in a normal test
suite. The full fp16/INT8/GPU run happens on Kaggle (Task 8); this test
validates the same code path at a scale that fits in CI."""

from __future__ import annotations

import math
from pathlib import Path

import onnx
import pytest

from rank.crossencoder_harness import (
    CrossEncoderSession,
    convert_to_fp16,
    export_onnx,
    quantize_int8,
    run_batching_sweep,
    run_queue_discipline,
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="module")
def fp32_onnx_path(tmp_path_factory) -> Path:
    onnx_dir = tmp_path_factory.mktemp("onnx")
    path = onnx_dir / "model_fp32.onnx"
    export_onnx(MODEL_NAME, path)
    return path


def test_export_onnx_produces_a_file(fp32_onnx_path: Path):
    assert fp32_onnx_path.exists()
    assert fp32_onnx_path.stat().st_size > 0


def test_session_scores_a_pair(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    scores = session.score([("what is python", "python is a programming language")])
    assert len(scores) == 1
    assert isinstance(scores[0], float)


def test_session_reports_active_providers(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    assert session.active_providers == ["CPUExecutionProvider"]


def test_score_chunks_batches_larger_than_chunk_size(fp32_onnx_path: Path):
    # A real (if small-scale) regression test for the OOM this chunking
    # fixes: scoring more pairs than chunk_size must still score every pair
    # (via multiple internal session.run() calls), producing the same count
    # and the same per-pair values a single unchunked call would.
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("what is python", f"passage number {i}") for i in range(10)]
    chunked = session.score(pairs, chunk_size=3)
    unchunked = session.score(pairs, chunk_size=len(pairs))
    assert len(chunked) == len(pairs)
    assert chunked == pytest.approx(unchunked)


def test_session_exposes_token_type_ids_input(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session._session.get_inputs()]
    assert "token_type_ids" in input_names


def test_session_scores_relevant_pair_above_floor(fp32_onnx_path: Path):
    # Value-pinned regression test for the token_type_ids omission: verified
    # by hand against real cross-encoder scores on this exact pair --
    # broken (token_type_ids omitted) scores it -0.4496, correct scores it
    # +7.4720. 5.0 separates cleanly from both. isinstance(score, float),
    # which is all the older smoke test checked, passes either way (and
    # would pass on NaN too), which is why this bug survived to review.
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    scores = session.score(
        [("what is the capital of france",
          "paris is the capital and most populous city of france")]
    )
    assert scores[0] > 5.0


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_converted_models_keep_three_inputs_and_score_finitely(
    fp32_onnx_path: Path, tmp_path: Path, precision: str
):
    # Neither conversion was covered by any test, and both rewrite the graph
    # the token_type_ids fix just changed: float16 conversion must leave the
    # *integer* inputs (input_ids/attention_mask/token_type_ids) alone, and
    # dynamic int8 quantization must not drop the newly-added third input. A
    # dropped or retyped token_type_ids would resurface exactly that bug in
    # the precision half of this phase, where it is hardest to notice.
    converted = tmp_path / f"model_{precision}.onnx"
    (convert_to_fp16 if precision == "fp16" else quantize_int8)(fp32_onnx_path, converted)

    graph_inputs = onnx.load(str(converted)).graph.input
    assert [i.name for i in graph_inputs] == ["input_ids", "attention_mask", "token_type_ids"]
    assert all(i.type.tensor_type.elem_type == onnx.TensorProto.INT64 for i in graph_inputs)

    session = CrossEncoderSession(converted, MODEL_NAME, providers=["CPUExecutionProvider"])
    score = session.score(
        [("what is the capital of france",
          "paris is the capital and most populous city of france")]
    )[0]
    assert math.isfinite(score)
    assert score > 5.0


def test_batching_sweep_tiny_grid(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("query text", f"passage number {i}") for i in range(6)]
    points = run_batching_sweep(
        session_factory=lambda: session,
        grid=[(1, 0.0), (4, 5.0)],
        request_pairs=pairs,
        # A saturating probe's wall time is probe_requests/throughput, so the
        # budget (not a duration) is what keeps this test to a few seconds on
        # a CPU-only fixture model.
        probe_requests=200,
        warmup_requests=2,
    )
    assert len(points) == 2
    for point in points:
        assert "max_batch_size" in point and "max_wait_ms" in point
        assert "latency_us" in point and "throughput_qps" in point
        assert point["throughput_qps"] > 0


def test_queue_discipline_reports_achieved_qps_and_queue_delay(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("query text", f"passage number {i}") for i in range(6)]
    result = run_queue_discipline(
        session_factory=lambda: session,
        discipline="unbounded",
        arrival_rate_qps=200.0,
        batcher_config=(4, 5.0),
        request_pairs=pairs,
        duration_s=0.5,
        warmup_requests=2,
    )
    assert result["discipline"] == "unbounded"
    assert result["achieved_qps"] > 0
    assert "p99_us" in result["latency_us"]
    assert "p99_us" in result["queue_delay_us"]
    assert result["shed_count"] == 0


def test_shed_discipline_actually_sheds_under_overload(fp32_onnx_path: Path):
    # The regression test for I1 itself. Under the old closed-loop harness the
    # arrival generator shared an event loop with the blocking scorer, so
    # offered load throttled itself down to whatever the scorer permitted and
    # the queue could never fill -- shed_count was structurally 0 (as it was
    # in every committed rank-queue.json run), and this assertion could not
    # have been made to pass locally at any arrival rate. With a genuinely
    # open-loop generator, a depth-1 queue at 500 qps against a CPU-bound
    # cross-encoder must shed.
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("query text", f"passage number {i}") for i in range(6)]
    result = run_queue_discipline(
        session_factory=lambda: session,
        discipline="shed",
        arrival_rate_qps=500.0,
        batcher_config=(4, 5.0),
        request_pairs=pairs,
        duration_s=0.5,
        max_queue_depth=1,
        warmup_requests=2,
    )
    assert result["shed_count"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest py/rank/tests/test_crossencoder_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank.crossencoder_harness'`.

- [ ] **Step 3: Write `py/rank/crossencoder_harness.py`**

```python
"""Cross-encoder serving harness: ONNX export/precision variants, an
in-process dynamic-batching harness (no real network hop -- harness.loadgen's
open-loop request generator feeds rank.batcher.DynamicBatcher directly,
matching the design spec's choice to keep this runnable inside one Kaggle
notebook), and queue discipline (load-shedding vs. unbounded).

The batching harness is thread-based, not asyncio-based, and that is load-
bearing: session.score() is a blocking call with no await in it, so an
asyncio generator sharing its event loop could never advance while a batch
was being scored -- arrivals silently throttled to the scorer's own rate and
the generator became closed-loop in everything but name (measured: a nominal
1992 qps target achieved ~719 qps, and no run ever shed a single request).
Scoring now runs on a dedicated flusher thread while harness.loadgen.
run_open_loop drives arrivals off absolute, precomputed deadlines.

INT8 is dynamic quantization (onnxruntime.quantization.quantize_dynamic) --
no calibration dataset needed, simpler than static/calibrated quantization,
at a real accuracy-vs-effort tradeoff this project measures rather than
assumes. fp16 is a separate converted ONNX graph
(onnxconverter_common.float16), not a runtime cast. Whether
CUDAExecutionProvider actually accelerates the dynamically-quantized INT8
ops is an open, measured question, not an assumption -- if it silently falls
back to slower ops, that IS the finding to report.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from harness.loadgen import LoadResult, run_open_loop


def export_onnx(model_name: str, onnx_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    dummy = tokenizer("dummy query", "dummy passage", return_tensors="pt")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # token_type_ids is not optional for a BERT-family cross-encoder: it is
    # the only input carrying the query/passage segment boundary (0 for query
    # tokens, 1 for passage tokens). Exporting a 2-input graph makes ORT run
    # the model with an all-zero segment embedding, which silently destroys
    # the model's discriminative power rather than failing -- measured on a
    # real pair, a relevant passage scores +7.4720 with token_type_ids and
    # -0.4496 without it.
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        # torch>=2.9 defaults to the dynamo/torch.export-based exporter, which
        # requires the optional `onnxscript` package (not a project dependency)
        # and takes `dynamic_shapes` instead of `dynamic_axes`. This module's
        # code was written for the legacy TorchScript-based exporter (it passes
        # `dynamic_axes`, which torch's own docs say "is used when dynamo is
        # False"), so pin that behavior explicitly rather than pull in a new
        # dependency for the new exporter.
        dynamo=False,
    )


def convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    model = onnx.load(str(fp32_path))
    fp16_model = float16.convert_float_to_float16(model)
    onnx.save(fp16_model, str(fp16_path))


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), weight_type=QuantType.QInt8)


DEFAULT_SCORE_CHUNK_SIZE = 32


class CrossEncoderSession:
    def __init__(self, onnx_path: Path, tokenizer_name: str, providers: list[str]) -> None:
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def active_providers(self) -> list[str]:
        """The execution providers ONNX Runtime actually initialized -- may
        silently differ from what the caller requested. A CUDA/cuDNN version
        mismatch makes ORT log a warning and fall back to CPU rather than
        raise, so a caller that only checks for an exception at session
        creation would never notice it ran the "GPU" experiment on CPU.
        Callers must record this in their result's provenance rather than
        assume the requested provider is the one that ran."""
        return self._session.get_providers()

    def score(
        self, pairs: list[tuple[str, str]], chunk_size: int = DEFAULT_SCORE_CHUNK_SIZE
    ) -> list[float]:
        # Chunked so a caller scoring an unbounded candidate pool (e.g. every
        # WAND candidate for a query, up to depth 1000) never hands a single
        # unbounded batch to the tokenizer/session in one call -- padding to
        # the batch's longest sequence times an unbounded batch size is an
        # uncontrolled memory spike, worse still if CUDA silently fell back
        # to CPU (see active_providers above) and the "GPU" run is actually
        # consuming system RAM instead of VRAM.
        scores: list[float] = []
        for start in range(0, len(pairs), chunk_size):
            chunk = pairs[start : start + chunk_size]
            queries, passages = zip(*chunk)
            encoded = self._tokenizer(
                list(queries), list(passages), padding=True, truncation=True, return_tensors="np"
            )
            outputs = self._session.run(
                ["logits"],
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    # A two-sequence tokenizer call already returns this; the
                    # bug this replaces was discarding it, leaving the segment
                    # embedding zeroed. See export_onnx's note.
                    "token_type_ids": encoded["token_type_ids"],
                },
            )
            scores.extend(float(x) for x in np.asarray(outputs[0]).reshape(-1))
        return scores


class _QueueFullError(Exception):
    """Raised by dispatch() under the shed discipline when the batcher is at
    capacity. run_open_loop's own dispatch wrapper catches any exception and
    counts it as a LoadResult error rather than aborting the run, which is
    exactly what "shed" should mean -- but sheds are counted explicitly here
    too (see _BatchedRun.shed_count), because result.errors would also absorb
    a genuine scoring failure and let a broken run masquerade as successful
    load-shedding."""


class _ScorerFailed(Exception):
    """Raised by dispatch() once the flusher thread has died, so in-flight and
    subsequent arrivals fail fast instead of blocking forever on an Event that
    nothing will ever set."""


@dataclass
class _BatchedRun:
    """One open-loop run's outcome: harness.loadgen's own LoadResult plus the
    shed count, which loadgen has no concept of."""

    result: LoadResult
    shed_count: int


def _run_batched_open_loop(
    session,
    batcher_factory: Callable[[], "DynamicBatcher"],
    request_pairs: list[tuple[str, str]],
    duration_s: float,
    qps: float,
    workers: int,
    max_queue_depth: int | None = None,
    warmup_requests: int = 0,
) -> _BatchedRun:
    """Drive `session` through one shared DynamicBatcher using harness.loadgen's
    real open-loop generator: arrivals run on their own schedule regardless of
    how long a batch takes to score, because session.score() runs on a
    dedicated flusher thread, not on the generator's critical path. Each
    dispatched request enqueues itself into the batcher and blocks on its own
    threading.Event until the flusher scores its batch and wakes it -- this is
    what lets run_open_loop's scheduled/start/done timestamps stay meaningful
    (queue_delay = start - scheduled genuinely reflects batcher/queue wait, not
    an artifact of a stalled event loop).

    Concurrency contract, since getting this wrong deadlocks rather than fails:
      * `lock` guards the batcher, the shed counter and the round-robin index,
        and nothing else. It is never held across session.score() -- the flush
        happens under the lock, the scoring does not.
      * A request's Event travels *inside* the batched item, so the flusher
        only ever wakes events it was handed. There is no shared id->event map
        for the two sides to race over.
      * If session.score() raises, the flusher records the exception, wakes
        every request it is holding, and keeps draining-and-waking until stop
        is set, so no dispatch thread can block forever. dispatch() then fails
        fast, run_open_loop returns, and the original exception is re-raised
        here rather than surfacing as a silent hang.
    """
    from rank.batcher import DynamicBatcher  # local import avoids a hard cycle at module load

    batcher: DynamicBatcher = batcher_factory()
    lock = threading.Lock()
    stop = threading.Event()
    failure: list[BaseException] = []
    counters = {"shed": 0, "next_index": 0}

    def flusher() -> None:
        while not stop.is_set():
            with lock:
                batch = batcher.flush() if batcher.should_flush() else None
            if not batch:
                time.sleep(0.0005)
                continue
            try:
                scores = session.score([pair for _, pair in batch])
                assert len(scores) == len(batch)
            except BaseException as exc:  # noqa: BLE001 -- re-raised by the caller
                failure.append(exc)
                for event, _ in batch:
                    event.set()
                # Keep releasing whatever arrives until the run is torn down:
                # dispatch() is failing fast by now, but requests already
                # enqueued (or enqueued in the race window) still need waking.
                while not stop.is_set():
                    with lock:
                        orphans = batcher.flush()
                    for event, _ in orphans:
                        event.set()
                    time.sleep(0.0005)
                return
            for event, _ in batch:
                event.set()

    flusher_thread = threading.Thread(target=flusher, daemon=True)
    flusher_thread.start()

    def dispatch(_query: str) -> None:
        # _query (the Zipf-sampled text run_open_loop hands us) is not used to
        # pick the pair -- request_pairs is round-robined independently under
        # the lock, since real pair identity for this synthetic benchmark
        # workload doesn't need to correlate with the sampler's popularity
        # draw, only arrival *timing* does.
        if failure:
            raise _ScorerFailed()
        event = threading.Event()
        with lock:
            if max_queue_depth is not None and len(batcher) >= max_queue_depth:
                counters["shed"] += 1
                raise _QueueFullError()
            pair = request_pairs[counters["next_index"] % len(request_pairs)]
            counters["next_index"] += 1
            batcher.add((event, pair))
        event.wait()
        if failure:
            raise _ScorerFailed()

    try:
        for _ in range(warmup_requests):
            dispatch("warmup")
        result = run_open_loop(
            dispatch=dispatch,
            queries=[q for q, _ in request_pairs],
            qps=qps,
            duration_s=duration_s,
            workers=workers,
        )
    finally:
        stop.set()
        flusher_thread.join(timeout=5.0)
    # Checked after the finally block (so it can't mask an in-flight
    # exception) and raised rather than ignored: a flusher that outlives its
    # run is a daemon thread still holding an ONNX Runtime session while the
    # interpreter tears down, which surfaces later as an unrelated-looking
    # native crash. Fail loudly here instead.
    if flusher_thread.is_alive():
        raise RuntimeError("flusher thread did not stop within 5s of the run ending")
    if failure:
        raise failure[0]
    return _BatchedRun(result=result, shed_count=counters["shed"])


# Requests offered per batching-sweep probe. The probe is a saturating burst:
# qps is set far above any config's plausible throughput, so what the offered
# rate actually buys is "every request is already waiting", and achieved_qps
# becomes that config's real sustained throughput. The *budget* has to be the
# bounded quantity rather than the rate, because run_open_loop precomputes the
# whole arrival list up front and its ThreadPoolExecutor drains every submitted
# request before returning -- a fire-hose bounded only by duration_s runs for
# offered_requests/capacity seconds (at 100k qps for 10s: hours), not for
# duration_s. Bounding the budget instead terminates in budget/capacity
# seconds, identically on this repo's CPU fixture and on a Kaggle T4, with no
# hardware-specific rate guess baked in. Consequence, stated where it can't be
# missed: under a saturating probe the sweep's p99 is drain-dominated --
# comparable across configs at equal budget, but not an absolute client-side
# latency. The queue-discipline experiment below, which offers a real specified
# arrival rate, is where latency/queue_delay are absolute numbers.
DEFAULT_PROBE_REQUESTS = 2000
_SATURATING_QPS = 100_000.0


def run_batching_sweep(
    session_factory: Callable[[], CrossEncoderSession],
    grid: list[tuple[int, float]],
    request_pairs: list[tuple[str, str]],
    probe_requests: int = DEFAULT_PROBE_REQUESTS,
    warmup_requests: int = 20,
) -> list[dict]:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    points = []
    for max_batch_size, max_wait_ms in grid:
        run = _run_batched_open_loop(
            session,
            # DynamicBatcher's clock contract is milliseconds (its max_wait_ms
            # comparison assumes clock() ticks in ms) -- time.perf_counter()
            # ticks in seconds, so it must be scaled here or a "5ms" wait
            # budget silently becomes 5 seconds.
            batcher_factory=lambda: DynamicBatcher(
                max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
            ),
            request_pairs=request_pairs,
            duration_s=probe_requests / _SATURATING_QPS,
            qps=_SATURATING_QPS,
            # Sized well above max_batch_size so the thread pool is never what
            # stops a batch from filling -- the batcher's own max_batch_size/
            # max_wait_ms must be the only thing shaping throughput.
            workers=max(max_batch_size * 8, 64),
            warmup_requests=warmup_requests,
        )
        summary = run.result.summary()
        points.append(
            {
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "latency_us": summary["latency"],
                "throughput_qps": run.result.achieved_qps,
            }
        )
    return points


def run_queue_discipline(
    session_factory: Callable[[], CrossEncoderSession],
    discipline: str,
    arrival_rate_qps: float,
    batcher_config: tuple[int, float],
    request_pairs: list[tuple[str, str]],
    duration_s: float = 2.0,
    max_queue_depth: int = 64,
    warmup_requests: int = 20,
) -> dict:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    max_batch_size, max_wait_ms = batcher_config
    # Generously sized so the thread pool is never the binding constraint --
    # only the batcher's max_queue_depth (shed) or real scorer throughput
    # (unbounded) should be able to cause backlog/shedding.
    workers = max(max_queue_depth * 4, 256)

    run = _run_batched_open_loop(
        session,
        # See run_batching_sweep's comment: DynamicBatcher expects a
        # millisecond clock, not seconds.
        batcher_factory=lambda: DynamicBatcher(
            max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
        ),
        request_pairs=request_pairs,
        duration_s=duration_s,
        qps=arrival_rate_qps,
        workers=workers,
        max_queue_depth=max_queue_depth if discipline == "shed" else None,
        warmup_requests=warmup_requests,
    )
    summary = run.result.summary()
    # Not run.result.achieved_qps: loadgen counts every *dispatched* request,
    # shed ones included, so under the shed discipline that field reads as
    # "sustained 2287 qps" for a run that actually served ~518 and dropped the
    # rest -- the opposite of the point. "Achieved" here means completed.
    # Identical to loadgen's own number under the unbounded discipline, which
    # sheds nothing.
    completed_qps = run.result.completed / run.result.wall_s if run.result.wall_s > 0 else 0.0
    return {
        "discipline": discipline,
        "arrival_rate_qps": arrival_rate_qps,
        "achieved_qps": completed_qps,
        "dispatched_qps": run.result.achieved_qps,
        "latency_us": summary["latency"],
        "queue_delay_us": summary["queue_delay"],
        "shed_count": run.shed_count,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest py/rank/tests/test_crossencoder_harness.py -v`
Expected: 5 passed. This downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace on first run (a few hundred MB) and exports/scores it on CPU — expect tens of seconds for the module-scoped fixture, not minutes.

- [ ] **Step 5: Commit**

```bash
git add py/rank/crossencoder_harness.py py/rank/tests/test_crossencoder_harness.py
git commit -m "phase4: cross-encoder ONNX export, batching, and queue-discipline harness"
```

---

### Task 8: Kaggle driver

**Note — deviates from the spec's stated filename:** the design spec (§9) names this file `kaggle/phase4_notebook.ipynb`. This plan uses a plain `.py` script plus a companion `kaggle/README.md` instead — a `.py` file is reviewable in a normal diff and easy to copy/paste into a single Kaggle notebook cell (or upload and `%run`), where hand-authoring the equivalent notebook-cell JSON directly in a plan document would be far less legible and harder to review here. No scope change: same content, same single-cell-or-uploaded-script usage on Kaggle, just a different file extension.

**Files:**
- Create: `kaggle/phase4_driver.py`
- Create: `kaggle/README.md`

**Interfaces:**
- Consumes: everything from Tasks 2–7, uploaded alongside this script; `data/rank-dense-scores.jsonl` and `data/rank-candidate-texts.json` (Task 2, uploaded as a small Kaggle Dataset), `runs/cascade-wand.{dev,dl19,dl20}.txt` (same), `harness.datasets.load_queries` (query text). `bench/results/dense-ground-truth-*.json` is **not** needed here (Phase 3's dense ground truth was over the 1M subset; this phase's dense scores come from Task 2's WAND-candidate-scoped file instead).
- Produces (on Kaggle, downloaded back into the repo): `bench/results/rank-prerank.json`, `bench/results/rank-batching.json`, `bench/results/rank-precision.json`, `bench/results/rank-queue.json` — each written as its own sub-experiment completes, not all at the very end.

This task has no local "run it" step — it is written and reviewed here, then the user uploads it and the small input files to Kaggle and runs it there themselves (per this phase's Global Constraint on execution). There is no unit test for this file: it is a thin orchestration script over already-tested modules (Tasks 2–7's real logic, not reimplemented here), matching this project's convention that a driver's correctness is checked by its output, not by pytest — the *logic* it calls was already tested in Tasks 3–7.

- [ ] **Step 1: Write `kaggle/README.md`**

```markdown
# Running Phase 4 on Kaggle

1. Create a new Kaggle Notebook, enable a GPU (Settings -> Accelerator -> GPU T4 x2 or x1).
2. Upload as a Kaggle Dataset (Add Data -> Upload), preserving this directory layout:
   - The whole `py/rank/` directory (all `.py` files, not `tests/`).
   - `py/harness/__init__.py`, `py/harness/histogram.py`, `py/harness/runmeta.py`,
     `py/harness/datasets.py`, `py/harness/metrics.py`, and `py/harness/runfile.py`
     (all five `harness` modules `phase4_driver.py` imports, directly or via
     `rank.crossencoder_harness`/`rank.prerank_mlp` — `datasets.py` is needed for
     `load_queries`/`load_qrels`, which pull from `ir_datasets`; `ir_datasets` itself
     must also be installed on Kaggle: `!pip install ir_datasets`).
   - `data/rank-dense-scores.jsonl` and `data/rank-candidate-texts.json`
     (from Task 2's local run).
   - `runs/cascade-wand.dev.txt`, `runs/cascade-wand.dl19.txt`, `runs/cascade-wand.dl20.txt`.

   Note: `py/dense/` is **not** needed here — `rank.encode_candidates` (imported by
   the driver only for its pure `DEV_NEGATIVE_POOL_DEPTH`/`truncate_to_top_k` helpers)
   imports `sentence_transformers`/`dense.encode` lazily, inside its own `main()`,
   specifically so this Kaggle driver's import of it doesn't need `py/dense/` uploaded.
3. A Kaggle Dataset mounts read-only under `/kaggle/input/<dataset-name>/`, and
   `phase4_driver.py` does a plain `import rank.foo` / `import harness.foo` plus
   relative paths like `runs/cascade-wand.dev.txt` — neither resolves against
   `/kaggle/input` directly. In the notebook's first cell, copy the uploaded
   dataset into the writable working directory and point Python at it:
   ```bash
   !cp -r /kaggle/input/<dataset-name>/py /kaggle/working/py
   !cp -r /kaggle/input/<dataset-name>/data /kaggle/working/data
   !cp -r /kaggle/input/<dataset-name>/runs /kaggle/working/runs
   %cd /kaggle/working
   ```
   ```python
   import sys; sys.path.insert(0, "/kaggle/working/py")
   ```
4. In a notebook cell:
   ```bash
   !pip uninstall -y onnxruntime  # Kaggle images may preinstall the CPU build
   !pip install onnxruntime-gpu onnxconverter-common
   ```
   No `nest_asyncio` (an earlier version of this driver needed it): the
   batching/queue-discipline harness in `rank.crossencoder_harness` is
   thread-based and no longer calls `asyncio.run(...)`, so Kaggle's
   already-active event loop is no longer in the way.
5. Before running, edit `kaggle/phase4_driver.py`'s `SOURCE_GIT_SHA` constant
   — run `git rev-parse HEAD` locally and paste the result in, since Kaggle
   has no git repo to read it from (see this phase's design spec, "Kaggle
   provenance gap").
6. Run `kaggle/phase4_driver.py` (paste its contents into a cell, or
   `%run phase4_driver.py` if uploaded as a file) — expect roughly an hour
   of GPU time across all four sub-experiments (see the design spec's
   session-budget reasoning).
7. Download `bench/results/rank-*.json` from the notebook's output files and
   drop them into this repo's `bench/results/` directory.
```

- [ ] **Step 2: Write `kaggle/phase4_driver.py`**

```python
"""Kaggle-run orchestration for Phase 4: prerank + prerank-consistency,
dynamic batching sweep, precision comparison, queue discipline. Writes each
sub-experiment's bench/results/rank-*.json as it completes (see this phase's
design spec's "Kaggle provenance gap" -- a session interruption shouldn't
lose earlier sub-experiments' results).

This session (the one running this plan) cannot execute this file --
it is written and reviewed here, then run by hand on Kaggle. See
kaggle/README.md for the upload/run steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from harness.datasets import load_qrels, load_queries
from harness.metrics import evaluate
from harness.runfile import read_run
from harness.runmeta import run_metadata
from rank.crossencoder_harness import (
    CrossEncoderSession,
    convert_to_fp16,
    export_onnx,
    quantize_int8,
    run_batching_sweep,
    run_queue_discipline,
)
from rank.encode_candidates import DEV_NEGATIVE_POOL_DEPTH, read_run_truncated
from rank.prerank_consistency import prerank_consistency
from rank.prerank_features import fit_scaler
from rank.prerank_mlp import (
    build_mlp,
    build_training_examples,
    load_dense_scores_and_lengths,
    survivors_for_query,
    train_mlp,
)

# Filled in by hand before uploading -- `git rev-parse HEAD` on this repo,
# immediately before uploading this file. Kaggle has no git repo to read it
# from; see this phase's design spec's "Kaggle provenance gap."
SOURCE_GIT_SHA = "7c601f5eeee332c12948b02508612ae25ad950a3"

RESULTS_DIR = Path("bench/results")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_K = 100  # matches rank.prerank_mlp.TOP_K_SURVIVORS

BATCHING_GRID = [
    (max_batch_size, max_wait_ms)
    for max_batch_size in (1, 8, 32)
    for max_wait_ms in (0, 5, 20)
]
QUEUE_DISCIPLINES = ("shed", "unbounded")
QUEUE_OVERLOAD_MULTIPLIERS = (1.5, 2.0, 3.0)

# Requests offered per batching-sweep grid point. run_batching_sweep is a
# saturating probe, so its wall time is probe_requests/throughput, not a fixed
# duration -- 5000 is a few seconds per point at plausible T4 throughput while
# still giving the slowest config (max_batch_size=1) a bounded run. See
# rank.crossencoder_harness.DEFAULT_PROBE_REQUESTS for the full reasoning.
PROBE_REQUESTS = 5000

# How many pairs the synthetic request stream cycles through. Built from real
# dl19/dl20 queries and real candidate passages (see build_request_pairs) so
# batching pads realistically -- 256 copies of one fixed-length string, which
# this used to use, makes every batch uniform and hides the padding cost that
# max_batch_size is supposed to trade against.
REQUEST_PAIR_COUNT = 256


def kaggle_hardware_info() -> dict:
    """torch.cuda fields to supplement run_metadata()'s CPU-oriented,
    sysctl-based hardware block, which reads 'unknown' on Kaggle's Linux
    host (no sysctl there)."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def provenance() -> dict:
    meta = run_metadata()
    meta["git_sha"] = SOURCE_GIT_SHA  # overrides run_metadata()'s "unknown"
    meta["kaggle_hardware"] = kaggle_hardware_info()
    return meta


def write_result(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"rank-{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def load_candidate_texts() -> dict[str, str]:
    return json.loads(Path("data/rank-candidate-texts.json").read_text())


def build_request_pairs(candidate_texts: dict[str, str]) -> list[tuple[str, str]]:
    """The (query, passage) stream the batching/precision/queue experiments
    replay. Real queries and real candidate passages, not synthetic
    fixed-length placeholders: the tokenized length variance across the real
    corpus is exactly what dynamic batching has to pad over, so a uniform
    synthetic workload would report a padding cost of zero."""
    queries = {**load_queries("dl19"), **load_queries("dl20")}
    query_list = list(queries.values())
    docids = list(candidate_texts)
    return [
        (query_list[i % len(query_list)], candidate_texts[docids[i % len(docids)]])
        for i in range(REQUEST_PAIR_COUNT)
    ]


def run_prerank_and_consistency() -> dict:
    # Streamed and truncated together -- dev's union of untruncated
    # depth-1000 candidates is 3.77M unique docs / 6,974,879 lines (374MB),
    # far more than the 1-positive-plus-4-negatives-per-query training loop
    # ever consumes. Materializing the full file via read_run() first (as
    # this line used to do, calling truncate_to_top_k(read_run(...), ...))
    # was a real, avoidable contributor to a Kaggle OOM kernel restart --
    # read_run_truncated never holds more than DEV_NEGATIVE_POOL_DEPTH
    # entries per query. See this phase's negative-sampling Global
    # Constraint and its Kaggle-OOM ruling.
    dev_run = read_run_truncated("runs/cascade-wand.dev.txt", DEV_NEGATIVE_POOL_DEPTH)
    dev_qrels = load_qrels("dev")
    dense_scores, doc_lengths = load_dense_scores_and_lengths(Path("data/rank-dense-scores.jsonl"))
    candidate_texts = load_candidate_texts()

    features, labels = build_training_examples(dev_run, dev_qrels, dense_scores, doc_lengths)
    print(f"training on {len(features)} examples from {len(dev_run)} dev queries")
    scaler = fit_scaler(features)
    model = build_mlp()
    train_mlp(model, features, labels, scaler, epochs=50)

    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    export_onnx(MODEL_NAME, fp32_path)
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])

    consistency_scores = []
    fused_qrels = {}
    fused_run = {}
    for query_set in ("dl19", "dl20"):
        run = read_run(f"runs/cascade-wand.{query_set}.txt")
        qrels = load_qrels(query_set)
        queries = load_queries(query_set)
        fused_qrels.update(qrels)
        for qid, candidates in run.items():
            docids = list(candidates)
            # true top-k under the FULL ranker: score every candidate, not
            # just the pre-rank survivors -- this is the one place the
            # expensive model runs over the entire candidate pool.
            pairs = [(queries[qid], candidate_texts[docid]) for docid in docids]
            all_scores = session.score(pairs)
            ranked = sorted(zip(docids, all_scores), key=lambda kv: kv[1], reverse=True)
            true_top_k = [docid for docid, _ in ranked[:CROSS_ENCODER_K]]

            survivors = survivors_for_query(
                model, scaler, candidates, dense_scores, doc_lengths, qid, top_k=CROSS_ENCODER_K
            )
            consistency_scores.append(prerank_consistency(true_top_k, survivors))

            fused_run[qid] = {
                docid: score
                for docid, score in zip(docids, all_scores)
                if docid in survivors
            }

    mean_consistency = sum(consistency_scores) / len(consistency_scores)
    # recall_k must match CROSS_ENCODER_K, not the WAND run's original depth
    # 1000 -- fused_run only ever holds each query's pre-rank survivors (at
    # most CROSS_ENCODER_K docs), so recall@1000 over it would silently
    # report a recall@CROSS_ENCODER_K number under a misleading deeper-sounding
    # name. harness.metrics.EvalResult.shallow_metrics exists to catch exactly
    # this (a cutoff deeper than the run); asked for the correct depth here so
    # there's nothing for it to flag, and asserted below as a second guard.
    eval_result = evaluate(
        fused_qrels, fused_run, ndcg_k=(10,), recall_k=(CROSS_ENCODER_K,), rel_threshold=2
    )
    assert not eval_result.shallow_metrics, eval_result.shallow_metrics

    write_result(
        "prerank",
        {
            "mean_prerank_consistency": mean_consistency,
            "num_queries": len(consistency_scores),
            "cascade_ndcg_10": eval_result.mean["ndcg_cut_10"],
            "cascade_recall_100": eval_result.mean[f"recall_{CROSS_ENCODER_K}"],
            "top_k_survivors": CROSS_ENCODER_K,
            "run_depth": eval_result.run_depth,
            # Requested CUDA -- may have silently fallen back to CPU (a
            # CUDA/cuDNN version mismatch logs a warning, not an error).
            # Recorded so a CPU-fallback run is never read as a GPU number.
            "active_providers": session.active_providers,
            "provenance": provenance(),
        },
    )

    # Returned (not just written) so run_precision() can re-score exactly
    # these survivors at fp16/INT8 without repeating the MLP pass -- the
    # pre-rank survivor set doesn't depend on the cross-encoder's precision,
    # only the cross-encoder's *score* of each survivor does.
    return {
        "fused_qrels": fused_qrels,
        "survivors_by_qid": {
            qid: set(candidates) for qid, candidates in fused_run.items()
        },
        "candidate_texts": candidate_texts,
        "cascade_ndcg_10_fp32": eval_result.mean["ndcg_cut_10"],
    }


def run_batching(request_pairs: list[tuple[str, str]]) -> list[dict]:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    # Created once (not a factory that builds a fresh session per call) so
    # active_providers below reflects the actual session run_batching_sweep
    # used across the whole grid -- run_batching_sweep only calls its
    # session_factory once internally anyway, so this changes nothing about
    # the sweep itself, only makes the session inspectable afterward.
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])

    points = run_batching_sweep(
        lambda: session, BATCHING_GRID, request_pairs, probe_requests=PROBE_REQUESTS
    )
    write_result(
        "batching",
        {
            "points": points,
            "grid": BATCHING_GRID,
            "active_providers": session.active_providers,
            "provenance": provenance(),
        },
    )
    return points


def score_survivors_with_session(
    session: CrossEncoderSession,
    survivors_by_qid: dict[str, set[str]],
    candidate_texts: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Re-score exactly the pre-rank survivors (not the full candidate pool)
    with one precision's session -- this is the run fed to evaluate() for
    that precision's cascade NDCG@10."""
    queries = {**load_queries("dl19"), **load_queries("dl20")}
    run: dict[str, dict[str, float]] = {}
    for qid, docids in survivors_by_qid.items():
        docids = list(docids)
        pairs = [(queries[qid], candidate_texts[docid]) for docid in docids]
        scores = session.score(pairs)
        run[qid] = dict(zip(docids, scores))
    return run


def run_precision(
    best_setting: tuple[int, float],
    fused_qrels: dict[str, dict[str, int]],
    survivors_by_qid: dict[str, set[str]],
    candidate_texts: dict[str, str],
    cascade_ndcg_10_fp32: float,
    request_pairs: list[tuple[str, str]],
) -> None:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    fp16_path = onnx_dir / "model_fp16.onnx"
    int8_path = onnx_dir / "model_int8.onnx"
    convert_to_fp16(fp32_path, fp16_path)
    quantize_int8(fp32_path, int8_path)

    max_batch_size, max_wait_ms = best_setting
    results = {}
    for precision, path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        session = CrossEncoderSession(path, MODEL_NAME, providers=["CUDAExecutionProvider"])
        session_factory = lambda s=session: s
        points = run_batching_sweep(
            session_factory,
            [(max_batch_size, max_wait_ms)],
            request_pairs,
            probe_requests=PROBE_REQUESTS,
        )

        precision_run = score_survivors_with_session(session, survivors_by_qid, candidate_texts)
        # recall_k matches CROSS_ENCODER_K for the same reason as
        # run_prerank_and_consistency() -- precision_run also only ever
        # holds the pre-rank survivors, not the full depth-1000 pool.
        eval_result = evaluate(
            fused_qrels, precision_run, ndcg_k=(10,), recall_k=(CROSS_ENCODER_K,), rel_threshold=2
        )
        assert not eval_result.shallow_metrics, eval_result.shallow_metrics
        ndcg_10 = eval_result.mean["ndcg_cut_10"]

        results[precision] = {
            **points[0],
            "cascade_ndcg_10": ndcg_10,
            "ndcg_10_delta_vs_fp32": ndcg_10 - cascade_ndcg_10_fp32,
            "active_providers": session.active_providers,
        }

    write_result(
        "precision",
        {"setting": {"max_batch_size": max_batch_size, "max_wait_ms": max_wait_ms}, "results": results, "provenance": provenance()},
    )


def run_queue_disciplines(
    best_setting: tuple[int, float],
    baseline_throughput_qps: float,
    request_pairs: list[tuple[str, str]],
) -> None:
    onnx_dir = Path("onnx_models")
    fp32_path = onnx_dir / "model_fp32.onnx"
    # Created once, reused across all 6 discipline/multiplier runs -- same
    # reasoning as run_batching() above.
    session = CrossEncoderSession(fp32_path, MODEL_NAME, providers=["CUDAExecutionProvider"])

    runs = []
    for discipline in QUEUE_DISCIPLINES:
        for multiplier in QUEUE_OVERLOAD_MULTIPLIERS:
            result = run_queue_discipline(
                lambda: session,
                discipline=discipline,
                arrival_rate_qps=baseline_throughput_qps * multiplier,
                batcher_config=best_setting,
                request_pairs=request_pairs,
                duration_s=10.0,
            )
            runs.append(result)
    write_result(
        "queue",
        {"runs": runs, "active_providers": session.active_providers, "provenance": provenance()},
    )


def main() -> None:
    assert SOURCE_GIT_SHA != "REPLACE_ME_BEFORE_UPLOADING", (
        "fill in SOURCE_GIT_SHA from a local `git rev-parse HEAD` before running on Kaggle"
    )
    prerank_state = run_prerank_and_consistency()
    # Built once and threaded through all three serving experiments, so
    # batching, precision and queue discipline are all measured against the
    # same real request stream rather than three separate placeholder lists.
    request_pairs = build_request_pairs(prerank_state["candidate_texts"])
    batching_points = run_batching(request_pairs)
    best_point = max(batching_points, key=lambda p: p["throughput_qps"])
    best_setting = (best_point["max_batch_size"], best_point["max_wait_ms"])
    run_precision(
        best_setting,
        fused_qrels=prerank_state["fused_qrels"],
        survivors_by_qid=prerank_state["survivors_by_qid"],
        candidate_texts=prerank_state["candidate_texts"],
        cascade_ndcg_10_fp32=prerank_state["cascade_ndcg_10_fp32"],
        request_pairs=request_pairs,
    )
    run_queue_disciplines(
        best_setting,
        baseline_throughput_qps=best_point["throughput_qps"],
        request_pairs=request_pairs,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add kaggle/phase4_driver.py kaggle/README.md
git commit -m "phase4: Kaggle driver orchestrating prerank/batching/precision/queue"
```

(This commits the *driver script*, not its output — `bench/results/rank-*.json` are added in a later, separate commit after the user has actually run this on Kaggle and downloaded the results. That commit is not part of this plan's task list since it depends on the user's real Kaggle run; add it by hand once the files exist, following the same `git add bench/results/rank-*.json && git commit` pattern every prior phase used for its own results.)

---

### Task 9: Report

**Files:**
- Create: `py/rank/report.py`

**Interfaces:**
- Consumes: `bench/results/rank-prerank.json`, `bench/results/rank-batching.json`, `bench/results/rank-precision.json`, `bench/results/rank-queue.json` (Task 8, downloaded from Kaggle by the user and dropped into `bench/results/` before this task runs).
- Produces: `bench/plots/phase4-batching.png`, `bench/phase4.md`. No dedicated test file — matches `dense/report.py`'s precedent (a renderer exercised by running it and inspecting output).

- [ ] **Step 1: Write `py/rank/report.py`**

```python
"""Renders bench/phase4.md: the batch-window tradeoff plot, the
prerank-consistency number, the precision table, and the queue-discipline
comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"
BENCH_DIR = REPO_ROOT / "bench"

# Phase 1's own first-stage baseline over the same 97 dl19+dl20 queries this
# phase's prerank-consistency section evaluates against -- pooled from
# bench/phase1.md's per-set NDCG@10 (dl19 0.5052 / 43 queries, dl20 0.4785 /
# 54 queries): (43*0.5052 + 54*0.4785) / 97. Phase 1 is historical and fixed,
# not re-derived from its rerun, so this is a constant rather than something
# read from bench/phase1.md at render time.
PHASE1_BASELINE_NDCG_10 = 0.4903


def render_batching_plot(batching: dict, output_path: Path) -> None:
    points = batching["points"]
    fig, ax = plt.subplots(figsize=(8, 6))
    throughput = [p["throughput_qps"] for p in points]
    p99_ms = [p["latency_us"]["p99_us"] / 1000 for p in points]
    labels = [f"b={p['max_batch_size']},w={p['max_wait_ms']}ms" for p in points]
    ax.scatter(throughput, p99_ms, s=80, alpha=0.7, color="tab:blue")
    for x, y, label in zip(throughput, p99_ms, labels):
        ax.annotate(label, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("throughput (qps)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Dynamic batching: throughput vs. p99 latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_batching_table(batching: dict) -> str:
    lines = [
        "| max_batch_size | max_wait_ms | throughput (qps) | p50 (ms) | p99 (ms) |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(batching["points"], key=lambda p: -p["throughput_qps"]):
        lines.append(
            f"| {p['max_batch_size']} | {p['max_wait_ms']} | {p['throughput_qps']:.1f} | "
            f"{p['latency_us']['p50_us']/1000:.2f} | {p['latency_us']['p99_us']/1000:.2f} |"
        )
    return "\n".join(lines)


def render_precision_table(precision: dict) -> str:
    # p999 and max, not just p99: the real fp16 run's p999/max were ~10x its
    # own p99 (244ms/252ms against a ~25ms p99), a tail entirely invisible in
    # a p99-only table. .get(..., 0) rather than direct indexing so a
    # rank-precision.json written before these keys existed still renders.
    lines = [
        "| precision | throughput (qps) | p99 (ms) | p999 (ms) | max (ms) | cascade NDCG@10 | delta vs. fp32 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in precision["results"].items():
        lines.append(
            f"| {name} | {row['throughput_qps']:.1f} | {row['latency_us']['p99_us']/1000:.2f} | "
            f"{row['latency_us'].get('p999_us', 0)/1000:.2f} | {row['latency_us'].get('max_us', 0)/1000:.2f} | "
            f"{row['cascade_ndcg_10']:.4f} | {row['ndcg_10_delta_vs_fp32']:+.4f} |"
        )
    return "\n".join(lines)


def render_queue_table(queue: dict) -> str:
    # achieved_qps next to the arrival rate is the whole point of an open-loop
    # generator: the gap between offered and achieved is the backlog, and
    # queue_delay p99 says how much of the client-visible latency it cost.
    lines = [
        "| discipline | arrival rate (qps) | achieved (qps) | p99 (ms) | queue delay p99 (ms) | shed count |",
        "|---|---|---|---|---|---|",
    ]
    for row in queue["runs"]:
        lines.append(
            f"| {row['discipline']} | {row['arrival_rate_qps']:.1f} | "
            f"{row.get('achieved_qps', 0):.1f} | "
            f"{row['latency_us'].get('p99_us', 0)/1000:.2f} | "
            f"{row.get('queue_delay_us', {}).get('p99_us', 0)/1000:.2f} | {row['shed_count']} |"
        )
    return "\n".join(lines)


def render_provider_warning(prerank: dict, batching: dict, precision: dict, queue: dict) -> str:
    """Every rank-*.json result now records active_providers -- the ONNX
    Runtime execution providers that actually initialized, which can
    silently differ from what the driver requested (a CUDA/cuDNN version
    mismatch logs a warning, not an error, and ONNX Runtime falls back to
    CPU). Surfaced prominently, near the top of the report, rather than left
    to a reader who happens to open the raw JSON -- a CPU-fallback run's
    numbers are real, but they measure the wrong hardware for this phase's
    stated question.
    """
    checks = {
        "prerank/consistency": prerank.get("active_providers", []),
        "batching sweep": batching.get("active_providers", []),
        **{
            f"precision ({name})": row.get("active_providers", [])
            for name, row in precision.get("results", {}).items()
        },
        "queue discipline": queue.get("active_providers", []),
    }
    # No `providers and ...` guard: a missing/empty active_providers field is
    # not evidence the run used CUDA, it is evidence the run didn't record
    # what it used -- which is exactly the silent-fallback case this check
    # exists to catch, so it must fire rather than pass.
    fell_back = {
        name: providers
        for name, providers in checks.items()
        if "CUDAExecutionProvider" not in providers
    }
    if not fell_back:
        return "All sub-experiments ran on `CUDAExecutionProvider` as requested."
    lines = [
        "**⚠️ CUDA execution provider unavailable for at least one "
        "sub-experiment -- the numbers below reflect CPU, not GPU, for:**",
        "",
    ]
    for name, providers in fell_back.items():
        lines.append(f"- {name}: ran on `{', '.join(providers)}`")
    return "\n".join(lines)


def render_markdown(prerank: dict, batching: dict, precision: dict, queue: dict) -> str:
    provenance = prerank["provenance"]
    return f"""# Phase 4: Ranking Cascade and Heterogeneous Serving

## Execution provider check

{render_provider_warning(prerank, batching, precision, queue)}

## Candidate pool (limitation, stated up front)

The pre-rank/rank cascade runs over Phase 1's existing lexical WAND
candidates (depth 1000, full 8.8M-passage corpus) — not fused with Phase 3's
dense recall channel. Dense score is used only as a pre-rank *feature*, not
as a separate recall channel.

## Prerank-consistency

Of the true top-{prerank['top_k_survivors']} under the full cross-encoder
ranker, **{prerank['mean_prerank_consistency']:.1%}** survive pre-ranking
(mean over {prerank['num_queries']} dl19+dl20 queries). Full-cascade result:
NDCG@10 = {prerank['cascade_ndcg_10']:.4f}, recall@100 = {prerank['cascade_recall_100']:.4f}.
Phase 1's own first-stage (lexical-only) baseline over the same 97
queries: NDCG@10 = {PHASE1_BASELINE_NDCG_10:.4f}.

## Dynamic batching: throughput vs. p99 latency

![Batching plot](plots/phase4-batching.png)

{render_batching_table(batching)}

## Precision comparison (at the best-throughput batching setting)

{render_precision_table(precision)}

## Queue discipline: load-shedding vs. unbounded

{render_queue_table(queue)}

## Configuration

- model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- git SHA: `{provenance['git_sha']}`
- GPU: {provenance.get('kaggle_hardware', {}).get('gpu_name', 'unknown')}
- timestamp: {provenance['timestamp_utc']}
"""


def main() -> None:
    prerank = json.loads((RESULTS_DIR / "rank-prerank.json").read_text())
    batching = json.loads((RESULTS_DIR / "rank-batching.json").read_text())
    precision = json.loads((RESULTS_DIR / "rank-precision.json").read_text())
    queue = json.loads((RESULTS_DIR / "rank-queue.json").read_text())

    render_batching_plot(batching, PLOTS_DIR / "phase4-batching.png")
    markdown = render_markdown(prerank, batching, precision, queue)
    (BENCH_DIR / "phase4.md").write_text(markdown)
    print(f"wrote {(BENCH_DIR / 'phase4.md').relative_to(REPO_ROOT)}")
    print(f"wrote {(PLOTS_DIR / 'phase4-batching.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report renderer**

Run: `PYTHONPATH=py uv run python -m rank.report`
Expected: `wrote bench/phase4.md`, `wrote bench/plots/phase4-batching.png`. Requires `bench/results/rank-{prerank,batching,precision,queue}.json` to already exist (downloaded from the Kaggle run in Task 8) — if any are missing, this fails with a clear `FileNotFoundError` naming the missing file.

- [ ] **Step 3: Visually inspect the output**

Read `bench/phase4.md` and confirm the tables render sensibly. Open `bench/plots/phase4-batching.png` and confirm it shows 9 labeled points with throughput on x, p99 latency on y.

- [ ] **Step 4: Commit**

```bash
git add py/rank/report.py bench/phase4.md bench/plots/phase4-batching.png
git commit -m "phase4: render the batching plot, precision table, and queue-discipline comparison"
```

---

### Task 10: README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: every artifact from Tasks 1–9.
- Produces: a "Running Phase 4" section in `README.md`, matching the style of the existing "Running Phase 1"/"Running Phase 2"/"Running Phase 3" sections.

- [ ] **Step 1: Run the full local test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including the new `py/rank/tests/test_encode_candidates.py` (6), `test_prerank_features.py` (5), `test_prerank_consistency.py` (4), `test_batcher.py` (5), `test_prerank_mlp.py` (3), `test_crossencoder_harness.py` (5) — 28 new tests on top of the existing 65.

- [ ] **Step 2: Add the README section**

Add after the existing `## Running Phase 3 (dense recall and the ANN Pareto frontier)` section, before `## Ground rules`:

```markdown
## Running Phase 4 (ranking cascade and heterogeneous serving)

Needs the `rank` extra (`transformers`, `onnx`, `onnxruntime`,
`onnxconverter-common`; `torch` already present via `dense`) and Phase 1's
WAND run files for dev/dl19/dl20 (`uv run python -m baselines.cascade_bm25`
if not already present, plus `--query-set dev` if that run is missing too).

```bash
uv sync --extra dense --extra rank
```

This machine has no CUDA GPU, so the GPU-bound half of this phase (the
cross-encoder rank stage, batching sweep, precision comparison, queue
discipline) runs on a Kaggle Notebook instead of locally — see
`kaggle/README.md` for the upload/run steps. Everything else runs locally:

```bash
export PYTHONPATH=py
uv run python -m rank.encode_candidates    # dense score + doc length features -> data/rank-dense-scores.jsonl
# ... upload to Kaggle, run kaggle/phase4_driver.py there, download results ...
uv run python -m rank.report               # -> bench/phase4.md, bench/plots/phase4-batching.png
```
```

- [ ] **Step 3: Update the status table**

Change the Phase 4 row in the `## Status` table from:
```
| 4 — Ranking cascade | batching + precision tables | not started |
```
to:
```
| 4 — Ranking cascade | batching + precision tables | **done** |
```
(Only after `bench/results/rank-*.json` are actually downloaded and committed and `bench/phase4.md` has been rendered from real data — if this task runs before the Kaggle round-trip completes, leave the status table unchanged and note that in the task report instead of marking it done prematurely.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "phase4: document setup and add exit-artifact status"
```

---
