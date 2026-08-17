# Phase 3: Dense Recall and the ANN Pareto Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode a qrels-preserving 1M-passage subset with `bge-small-en-v1.5`, sweep HNSW and IVF-PQ across a recall/latency/memory grid against an exact brute-force reference, and compare hybrid fusion (RRF, normalized score fusion) against each channel alone — producing `bench/phase3.md`.

**Architecture:** Six sequential Python scripts under `py/dense/`, each reading the previous step's output and writing its own artifact to `data/` (large, gitignored intermediates) or `bench/results/` (small, committed results): `subset.py` -> `encode.py` -> `ground_truth.py` -> `ann_sweep.py` -> `fusion.py` -> `report.py`. No C++ changes, no server — this phase reads Phase 1's existing WAND run file and otherwise operates independently in Python.

**Tech Stack:** Python 3.12, `sentence-transformers`/`torch` (encoding), `faiss-cpu` (exact search + IVF-PQ), `hnswlib` (HNSW), `matplotlib` (Pareto plot — first plotting dependency in this repo), reusing `harness.metrics`, `harness.runfile`, `harness.histogram.LatencyRecorder`, `harness.runmeta.run_metadata` unchanged.

**Spec:** `docs/superpowers/specs/2026-08-16-phase3-dense-recall-design.md`

## Global Constraints

- Corpus subset: exactly **1,000,000** passages = (every docid judged in dl19+dl20 qrels, verified at plan-writing time to be 20,349 docids) union (a seeded random fill of the remaining corpus). MS MARCO passage docids are dense sequential integers `"0"`..`"8841822"` matching row order — verified empirically against `harness.datasets.iter_docs`/`doc_count` at plan-writing time; this is what makes subset selection an integer set-difference instead of a true streaming reservoir sample.
- Encoder: `BAAI/bge-small-en-v1.5` via `sentence-transformers`, 384-dim, L2-normalized (`normalize_embeddings=True`, inner product = cosine similarity downstream). Asymmetric: passages encoded as-is; queries encoded with the prefix `"Represent this sentence for searching relevant passages: "`.
- Device: try `mps`, fall back to `cpu`, log whichever ran. Verified empirically at plan-writing time on this machine: MPS encodes at ~274 passages/sec vs. CPU's ~75/sec (3.7x) — encoding the full 1M-passage subset takes **~60 minutes on MPS** (measured: 8,000 passages in 29.2s). This is the single long-running step in this phase; run it in the background and do not re-run it casually.
- Two query sets, encoded once each and reused throughout: **dev** (6,980 queries — the ANN sweep's recall@100 reference and latency source; no qrels needed since recall@100 there is a dense-vs-exact overlap measure, not an IR-relevance measure) and **dl19+dl20** (97 queries — the fusion table's source, since NDCG@10/recall@1000 need real relevance judgments).
- `hnswlib` ships no macOS wheel (sdist only) and its `setup.py` doesn't point `clang++` at the SDK's libc++ headers — same class of issue as `cpp/Makefile`'s documented `SDK_PATH` workaround. Verified empirically at plan-writing time: building it fails with `fatal error: 'iostream' file not found` unless `CXXFLAGS`/`CPPFLAGS` are set to `-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1` for the `uv sync`/`uv add` invocation. See Task 1 for the exact command.
- Memory footprint measurement: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` before/after each index build. **Platform gotcha, verified empirically at plan-writing time:** this field is in *bytes* on macOS/Darwin but *kilobytes* on Linux — same field, different units. This project only runs on macOS; code divides by 1e9 for GB and comments why, rather than silently being wrong by 1000x if ever run elsewhere.
- Latency: p50/p95/p99 only, **never a mean** (project-wide rule, `harness.histogram.LatencyRecorder` already enforces this). Single-threaded per-call timing for the ANN sweep — `faiss.omp_set_num_threads(1)` and `hnswlib`'s `knn_query(..., num_threads=1)` are both set explicitly so a library's internal multithreading doesn't blur a single-query latency measurement.
- HNSW grid: `M ∈ {16, 32, 64}` (build-time, needs rebuild), `ef_construction = 200` (fixed, not swept — the spec sweeps `M` and `efSearch` only), `efSearch ∈ {32, 64, 128, 256, 512}` (query-time, no rebuild) — 15 points.
- IVF-PQ grid: `nlist ∈ {1024, 4096}` × `m ∈ {32, 64}` (PQ subquantizers; both evenly divide the 384-dim embedding) at train/build time, `nbits = 8` (fixed, standard — verified empirically at plan-writing time it works at real scale), `nprobe ∈ {1, 8, 16, 32, 64}` at search time — 20 points.
- RRF: `score(d) = Σ_channels 1/(60 + rank_in_channel(d))`, k=60 (standard default). A document absent from a channel contributes 0 from it — never dropped, never tied with unseen documents.
- Every `bench/results/*.json` file carries `harness.runmeta.run_metadata()` (git SHA, hardware, timestamp) plus its own full config (seed, subset size, model name, device, grid values) — "a number without its config is not a result," enforced project-wide already.
- Python: run every driver as a module with `PYTHONPATH=py` (`uv run python -m dense.foo`), matching `baselines/*.py`'s and `server/*.py`'s established convention.
- Docids are handled as `int64` throughout (`data/dense-docids.npy`, embedding row order) since they are verified-sequential integers; converted to `str` only at JSON-serialization boundaries, where qrels/run-file docids are strings.

---

## File Structure

```
pyproject.toml                                    modified — add `dense` extra
README.md                                          modified — "Running Phase 3" section (Task 8)

py/dense/__init__.py                               new
py/dense/subset.py                                 new — Task 2
py/dense/tests/__init__.py                         new
py/dense/tests/test_subset.py                      new — Task 2
py/dense/encode.py                                 new — Task 3
py/dense/tests/test_encode.py                      new — Task 3
py/dense/ground_truth.py                           new — Task 4
py/dense/ann_sweep.py                              new — Task 5
py/dense/fusion.py                                 new — Task 6
py/dense/tests/test_fusion.py                       new — Task 6
py/dense/report.py                                 new — Task 7

data/dense-subset.jsonl                            new, gitignored — Task 2
data/dense-subset-docids.txt                       new, gitignored — Task 2
data/dense-embeddings.npy                          new, gitignored — Task 3
data/dense-docids.npy                              new, gitignored — Task 3
data/dense-query-embeddings-{dev,dl19,dl20}.npy    new, gitignored — Task 3
data/dense-query-ids-{dev,dl19,dl20}.json          new, gitignored — Task 3

bench/results/dense-ground-truth-dev.json          new, committed — Task 4
bench/results/dense-ground-truth-dl19-dl20.json    new, committed — Task 4
bench/results/dense-ann-sweep.json                 new, committed — Task 5
bench/results/dense-fusion.json                    new, committed — Task 6
bench/plots/phase3-pareto.png                      new, committed — Task 7
bench/phase3.md                                    new, committed — Task 7/8
```

---

### Task 1: Dependencies and package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `py/dense/__init__.py`
- Create: `py/dense/tests/__init__.py`

**Interfaces:**
- Produces: the `dense` extra (`sentence-transformers`, `torch`, `faiss-cpu`, `hnswlib`, `matplotlib`), an importable `py/dense` package, `py/dense/tests` collected by pytest.

- [ ] **Step 1: Add the `dense` extra to `pyproject.toml`**

Add this block after the existing `dev = [...]` extra (before `[tool.uv]`):

```toml
# hnswlib ships no macOS wheel (sdist only) and its setup.py doesn't point
# clang++ at the SDK's libc++ headers (same class of issue as cpp/Makefile's
# SDK_PATH note) -- `uv sync --extra dense` on a fresh checkout, or after
# `uv cache clean`, needs:
#   SDK_PATH=$(xcrun --show-sdk-path)
#   CXXFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
#   CPPFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
#   uv sync --extra dense
dense = [
    "faiss-cpu>=1.15.0",
    "hnswlib>=0.8.0",
    "matplotlib>=3.11.1",
    "sentence-transformers>=5.7.0",
    "torch>=2.13.0",
]
```

Also update `testpaths` in `[tool.pytest.ini_options]`:

```toml
testpaths = ["py/tests", "py/server/tests", "py/dense/tests"]
```

- [ ] **Step 2: Sync with the SDK header flags**

Run:
```bash
SDK_PATH=$(xcrun --show-sdk-path)
CXXFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
CPPFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
uv sync --extra dev --extra baselines --extra dense
```
Expected: resolves and installs ~40 packages including `faiss-cpu`, `hnswlib`, `torch`, `sentence-transformers`, `matplotlib`, with no build errors. If `hnswlib` fails with `fatal error: 'iostream' file not found`, the env vars above were not picked up — re-check `xcrun --show-sdk-path` returns a real path.

- [ ] **Step 3: Create the package skeleton**

`py/dense/__init__.py` (empty file) and `py/dense/tests/__init__.py` (empty file).

- [ ] **Step 4: Verify imports and existing tests**

Run:
```bash
uv run python3 -c "import hnswlib, faiss, sentence_transformers, matplotlib; print('all import OK')"
uv run pytest -q
```
Expected: `all import OK`, and the existing 53 tests still pass (0 new tests yet — `py/dense/tests` is empty except `__init__.py`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock py/dense/__init__.py py/dense/tests/__init__.py
git commit -m "phase3: add dense-recall dependencies and package scaffolding"
```

---

### Task 2: Corpus subset

**Files:**
- Create: `py/dense/subset.py`
- Test: `py/dense/tests/test_subset.py`

**Interfaces:**
- Consumes: `harness.datasets.doc_count() -> int`, `harness.datasets.iter_docs(limit: int | None = None) -> Iterator[tuple[str, str]]`, `harness.datasets.load_qrels(query_set: str) -> dict[str, dict[str, int]]`.
- Produces: `select_subset_ids(qrels_ids: set[int], total_docs: int, seed: int = 0, subset_size: int = 1_000_000) -> set[int]` (pure, unit-tested), `qrels_docid_union() -> set[int]` (I/O, calls `load_qrels`), `write_subset(subset_ids: set[int], jsonl_path: Path, docids_path: Path) -> None`. CLI writes `data/dense-subset.jsonl` (one `{"docid": ..., "text": ...}` per line) and `data/dense-subset-docids.txt` (one docid per line, corpus order).

- [ ] **Step 1: Write the failing tests**

`py/dense/tests/test_subset.py`:

```python
"""select_subset_ids is pure (no corpus I/O) so these run against small
synthetic inputs, matching this repo's existing test style (py/tests/test_metrics.py
uses randomized synthetic qrels/runs rather than the real corpus)."""

from __future__ import annotations

from dense.subset import select_subset_ids


def test_every_qrels_docid_is_in_the_subset():
    qrels_ids = {5, 10, 999}
    subset = select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=100)
    assert qrels_ids <= subset


def test_subset_size_is_exact():
    qrels_ids = {5, 10, 999}
    subset = select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=100)
    assert len(subset) == 100


def test_same_seed_is_deterministic():
    qrels_ids = {5, 10, 999}
    first = select_subset_ids(qrels_ids, total_docs=5000, seed=42, subset_size=200)
    second = select_subset_ids(qrels_ids, total_docs=5000, seed=42, subset_size=200)
    assert first == second


def test_different_seeds_differ():
    qrels_ids = {5, 10, 999}
    first = select_subset_ids(qrels_ids, total_docs=5000, seed=1, subset_size=200)
    second = select_subset_ids(qrels_ids, total_docs=5000, seed=2, subset_size=200)
    assert first != second


def test_qrels_larger_than_subset_size_raises():
    qrels_ids = set(range(50))
    try:
        select_subset_ids(qrels_ids, total_docs=1000, seed=0, subset_size=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest py/dense/tests/test_subset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dense.subset'` (or `ImportError`).

- [ ] **Step 3: Write `py/dense/subset.py`**

```python
"""Build the 1M-passage subset for dense recall: every doc judged in
dl19+dl20 qrels, plus a random fill to reach exactly 1,000,000 docs.

Random subsetting alone risks dropping qrels-judged documents -- a document
missing from the subset can never be retrieved, which would make every
downstream recall/NDCG number against it meaningless. So the subset is
always the qrels union first, then filled to size with a seeded random
sample of the remaining corpus.

MS MARCO passage docids are dense sequential integers ("0".."8841822",
matching row order in the corpus) -- verified against harness.datasets at
plan-writing time. This lets subset selection work directly on an integer
id range instead of needing a true streaming reservoir sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harness.datasets import doc_count, iter_docs, load_qrels

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

SUBSET_SIZE = 1_000_000
SEED = 0


def qrels_docid_union() -> set[int]:
    """Every docid judged (at any relevance level) in dl19 or dl20 qrels."""
    ids: set[int] = set()
    for query_set in ("dl19", "dl20"):
        qrels = load_qrels(query_set)
        for judgments in qrels.values():
            ids.update(int(docid) for docid in judgments)
    return ids


def select_subset_ids(
    qrels_ids: set[int],
    total_docs: int,
    seed: int = SEED,
    subset_size: int = SUBSET_SIZE,
) -> set[int]:
    """The qrels union plus a seeded random fill, as a set of integer docids."""
    if len(qrels_ids) > subset_size:
        raise ValueError(
            f"qrels union ({len(qrels_ids)}) exceeds subset_size ({subset_size})"
        )
    rng = np.random.default_rng(seed)
    all_ids = np.arange(total_docs)
    candidate_pool = np.setdiff1d(
        all_ids, np.array(sorted(qrels_ids), dtype=np.int64), assume_unique=True
    )
    fill_needed = subset_size - len(qrels_ids)
    fill = rng.choice(candidate_pool, size=fill_needed, replace=False)
    return qrels_ids | {int(x) for x in fill.tolist()}


def write_subset(subset_ids: set[int], jsonl_path: Path, docids_path: Path) -> None:
    """Single streaming pass over the corpus, writing matched docs in corpus order."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(jsonl_path, "w") as jsonl_out, open(docids_path, "w") as ids_out:
        for docid, text in iter_docs():
            if int(docid) in subset_ids:
                jsonl_out.write(json.dumps({"docid": docid, "text": text}) + "\n")
                ids_out.write(docid + "\n")
                written += 1
    if written != len(subset_ids):
        raise RuntimeError(
            f"wrote {written} docs but subset_ids had {len(subset_ids)} -- "
            "a selected docid was never seen while streaming the corpus"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()

    qrels_ids = qrels_docid_union()
    print(f"qrels union: {len(qrels_ids)} docids")
    total_docs = doc_count()
    subset_ids = select_subset_ids(
        qrels_ids, total_docs, seed=args.seed, subset_size=args.subset_size
    )
    print(f"selected {len(subset_ids)} docids (seed={args.seed})")

    jsonl_path = DATA_DIR / "dense-subset.jsonl"
    docids_path = DATA_DIR / "dense-subset-docids.txt"
    write_subset(subset_ids, jsonl_path, docids_path)
    print(
        f"wrote {jsonl_path.relative_to(REPO_ROOT)} and "
        f"{docids_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest py/dense/tests/test_subset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the real subset build**

Run: `PYTHONPATH=py uv run python -m dense.subset`
Expected: prints `qrels union: 20349 docids` (verified at plan-writing time), `selected 1000000 docids (seed=0)`, then the two output paths. Takes well under a minute (full-corpus streaming measured at ~16s at plan-writing time).

- [ ] **Step 6: Sanity-check the output**

Run:
```bash
wc -l data/dense-subset-docids.txt data/dense-subset.jsonl
```
Expected: both report `1000000`.

- [ ] **Step 7: Commit**

```bash
git add py/dense/subset.py py/dense/tests/test_subset.py
git commit -m "phase3: build the qrels-preserving 1M-passage subset"
```
(`data/dense-subset*` stays untracked — `/data/` is gitignored.)

---

### Task 3: Encoding

**Files:**
- Create: `py/dense/encode.py`
- Test: `py/dense/tests/test_encode.py`

**Interfaces:**
- Consumes: `data/dense-subset.jsonl` (Task 2), `harness.datasets.load_queries(query_set: str) -> dict[str, str]`.
- Produces: `apply_query_prefix(text: str) -> str` (pure, unit-tested), `select_device() -> str` (`"mps"` or `"cpu"`). CLI writes `data/dense-embeddings.npy` (float32, shape `(1_000_000, 384)`, row order = `data/dense-docids.npy`), `data/dense-docids.npy` (int64, shape `(1_000_000,)`), and per query set (`dev`, `dl19`, `dl20`): `data/dense-query-embeddings-{set}.npy` (float32, `(N, 384)`) and `data/dense-query-ids-{set}.json` (JSON list of qid strings, row order).

- [ ] **Step 1: Write the failing test**

`py/dense/tests/test_encode.py`:

```python
from dense.encode import QUERY_PREFIX, apply_query_prefix


def test_prefix_is_prepended_exactly_once():
    result = apply_query_prefix("search engine ranking")
    assert result == QUERY_PREFIX + "search engine ranking"


def test_prefix_matches_bge_instruction_convention():
    assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest py/dense/tests/test_encode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dense.encode'`.

- [ ] **Step 3: Write `py/dense/encode.py`**

```python
"""Encode the 1M-passage subset and the dev/dl19/dl20 query sets with
bge-small-en-v1.5.

BGE is asymmetric: passages are encoded plain, queries get an instruction
prefix. Getting this backwards silently degrades retrieval quality without
raising an error, so apply_query_prefix is its own small, tested function
rather than an inline string concatenation buried in the encoding loop.

This is the one long-running step in this phase (~60 minutes for 1M
passages on this machine's MPS backend, measured at plan-writing time) --
everything downstream reads its .npy output and never re-encodes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from harness.datasets import load_queries

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384
BATCH_SIZE = 128
QUERY_SETS = ("dev", "dl19", "dl20")


def apply_query_prefix(text: str) -> str:
    return QUERY_PREFIX + text


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_subset_texts(jsonl_path: Path) -> tuple[list[str], list[int]]:
    docids: list[int] = []
    texts: list[str] = []
    with open(jsonl_path) as handle:
        for line in handle:
            row = json.loads(line)
            docids.append(int(row["docid"]))
            texts.append(row["text"])
    return texts, docids


def encode_passages(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def encode_queries(model: SentenceTransformer, query_set: str) -> tuple[np.ndarray, list[str]]:
    queries = load_queries(query_set)
    qids = sorted(queries)
    texts = [apply_query_prefix(queries[qid]) for qid in qids]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return embeddings, qids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-jsonl", default=str(DATA_DIR / "dense-subset.jsonl"))
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts, docids = load_subset_texts(Path(args.subset_jsonl))
    print(f"encoding {len(texts)} passages...")
    t0 = time.time()
    embeddings = encode_passages(model, texts)
    print(f"passages encoded in {time.time() - t0:.1f}s")
    assert embeddings.shape == (len(texts), EMBEDDING_DIM)

    np.save(DATA_DIR / "dense-embeddings.npy", embeddings)
    np.save(DATA_DIR / "dense-docids.npy", np.array(docids, dtype=np.int64))
    print(f"wrote dense-embeddings.npy {embeddings.shape} and dense-docids.npy")

    for query_set in QUERY_SETS:
        q_embeddings, qids = encode_queries(model, query_set)
        np.save(DATA_DIR / f"dense-query-embeddings-{query_set}.npy", q_embeddings)
        (DATA_DIR / f"dense-query-ids-{query_set}.json").write_text(json.dumps(qids))
        print(f"{query_set}: encoded {len(qids)} queries, shape {q_embeddings.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest py/dense/tests/test_encode.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the real encoding step in the background**

This takes ~60 minutes. Run:
```bash
PYTHONPATH=py uv run python -m dense.encode > /tmp/phase3-encode.log 2>&1 &
```
Poll with `tail -f /tmp/phase3-encode.log` or check the process periodically rather than blocking on it — do not run this in the foreground of an interactive session.

Expected final lines: `device: mps` (or `cpu`, logged either way), `passages encoded in ~3600s`, `wrote dense-embeddings.npy (1000000, 384) and dense-docids.npy`, then three `{set}: encoded N queries, shape (N, 384)` lines for `dev` (6980), `dl19` (43), `dl20` (54).

- [ ] **Step 6: Sanity-check the output**

Run:
```bash
uv run python3 -c "
import numpy as np
emb = np.load('data/dense-embeddings.npy')
docids = np.load('data/dense-docids.npy')
print('embeddings', emb.shape, emb.dtype)
print('docids', docids.shape, docids.dtype)
print('norms sample', np.linalg.norm(emb[:5], axis=1))
"
```
Expected: `embeddings (1000000, 384) float32`, `docids (1000000,) int64`, norms all `~1.0` (L2-normalized).

- [ ] **Step 7: Commit**

```bash
git add py/dense/encode.py py/dense/tests/test_encode.py
git commit -m "phase3: encode the passage subset and eval queries with bge-small-en-v1.5"
```
(`data/dense-embeddings.npy` etc. stay untracked — `/data/` is gitignored.)

---

### Task 4: Exact ground truth

**Files:**
- Create: `py/dense/ground_truth.py`

**Interfaces:**
- Consumes: `data/dense-embeddings.npy`, `data/dense-docids.npy`, `data/dense-query-embeddings-{dev,dl19,dl20}.npy`, `data/dense-query-ids-{dev,dl19,dl20}.json` (Task 3).
- Produces: `exact_search(index: faiss.IndexFlatIP, query_embeddings: np.ndarray, query_ids: list[str], docids: np.ndarray, k: int) -> dict[str, list[list]]` (maps qid -> list of `[docid, score]`, descending score). CLI writes `bench/results/dense-ground-truth-dev.json` (top-100 per query — recall@100 only needs top-100, and dev's 6,980 queries at top-1000 would be a ~140MB commit) and `bench/results/dense-ground-truth-dl19-dl20.json` (top-1000 per query — fusion needs recall@1000).

- [ ] **Step 1: Write `py/dense/ground_truth.py`**

No unit test for this file — it is a benchmark driver over real embeddings, exercised by running it and checking its output, matching this repo's existing convention for `baselines/cascade_bm25.py`/`baselines/cascade_query_cost.py` (driver correctness checked by output, not by pytest).

```python
"""Exact (brute-force) nearest-neighbor search over the 1M-passage subset,
via faiss.IndexFlatIP. Two outputs, two purposes:

  - dev query set (6,980 queries): the recall@100 reference and latency
    source for the ANN sweep (dense_recall/ann_sweep.py). Top-100 only --
    recall@100 needs no more, and dev's query count makes top-1000 an
    unreasonably large commit (~140MB vs ~14MB at top-100).
  - dl19+dl20 query set (97 queries): the exact dense channel for hybrid
    fusion (dense_recall/fusion.py), which needs recall@1000. Using the
    *exact* ranking here (not an ANN approximation) isolates "does fusion
    help" from "how good is the ANN approximation" -- a separate question
    the Pareto sweep already answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUBSET_SIZE = 1_000_000


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def exact_search(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    docids: np.ndarray,
    k: int,
) -> dict[str, list[list]]:
    scores, positions = index.search(query_embeddings, k)
    results: dict[str, list[list]] = {}
    for row, qid in enumerate(query_ids):
        results[qid] = [
            [int(docids[positions[row, col]]), round(float(scores[row, col]), 6)]
            for col in range(positions.shape[1])
        ]
    return results


def run_query_set(
    index: faiss.IndexFlatIP, docids: np.ndarray, query_set: str, k: int
) -> dict[str, list[list]]:
    query_embeddings = np.load(DATA_DIR / f"dense-query-embeddings-{query_set}.npy")
    query_ids = json.loads((DATA_DIR / f"dense-query-ids-{query_set}.json").read_text())
    return exact_search(index, query_embeddings, query_ids, docids, k)


def main() -> None:
    embeddings = np.load(DATA_DIR / "dense-embeddings.npy")
    docids = np.load(DATA_DIR / "dense-docids.npy")
    assert embeddings.shape[0] == SUBSET_SIZE
    index = build_flat_index(embeddings)

    dev_results = run_query_set(index, docids, "dev", k=100)
    dl19_results = run_query_set(index, docids, "dl19", k=1000)
    dl20_results = run_query_set(index, docids, "dl20", k=1000)
    dl_results = {**dl19_results, **dl20_results}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev_output = {
        "query_set": "dev",
        "k": 100,
        "model": MODEL_NAME,
        "subset_size": SUBSET_SIZE,
        "results": dev_results,
        "provenance": run_metadata(),
    }
    (RESULTS_DIR / "dense-ground-truth-dev.json").write_text(
        json.dumps(dev_output, separators=(",", ":"))
    )

    dl_output = {
        "query_set": "dl19+dl20",
        "k": 1000,
        "model": MODEL_NAME,
        "subset_size": SUBSET_SIZE,
        "results": dl_results,
        "provenance": run_metadata(),
    }
    (RESULTS_DIR / "dense-ground-truth-dl19-dl20.json").write_text(
        json.dumps(dl_output, separators=(",", ":"))
    )

    print(f"dev: {len(dev_results)} queries, top-100")
    print(f"dl19+dl20: {len(dl_results)} queries, top-1000")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the real ground-truth build**

Run: `PYTHONPATH=py uv run python -m dense.ground_truth`
Expected: `dev: 6980 queries, top-100`, `dl19+dl20: 97 queries, top-1000`. A single flat-index brute-force search over 1M x 384 for ~7,077 queries total — expect low tens of seconds, not minutes.

- [ ] **Step 3: Sanity-check the output**

Run:
```bash
uv run python3 -c "
import json
dev = json.loads(open('bench/results/dense-ground-truth-dev.json').read())
print('dev queries', len(dev['results']))
qid, hits = next(iter(dev['results'].items()))
print('sample query', qid, 'top-3', hits[:3])
assert all(hits[i][1] >= hits[i+1][1] for i in range(len(hits)-1)), 'not descending'
print('descending: OK')
"
ls -la bench/results/dense-ground-truth-dev.json bench/results/dense-ground-truth-dl19-dl20.json
```
Expected: `dev queries 6980`, scores descending confirmed, dev file roughly 10-20MB, dl19-dl20 file roughly 1-3MB.

- [ ] **Step 4: Commit**

```bash
git add py/dense/ground_truth.py bench/results/dense-ground-truth-dev.json bench/results/dense-ground-truth-dl19-dl20.json
git commit -m "phase3: exact brute-force ground truth for dev and dl19+dl20"
```

---

### Task 5: HNSW and IVF-PQ sweeps

**Files:**
- Create: `py/dense/ann_sweep.py`

**Interfaces:**
- Consumes: `data/dense-embeddings.npy`, `data/dense-docids.npy`, `data/dense-query-embeddings-dev.npy`, `data/dense-query-ids-dev.json` (Task 3), `bench/results/dense-ground-truth-dev.json` (Task 4).
- Produces: `recall_at_k(candidate_ids: list[int], exact_ids: list[int], k: int) -> float` (pure). CLI writes `bench/results/dense-ann-sweep.json`, a list of 35 point dicts (15 HNSW + 20 IVF-PQ), each with `structure`, build config, `nprobe`/`efSearch`, `recall_at_100`, `latency_us` (`LatencyRecorder.summary()`), `memory_gb`.

No unit test for this file (same rationale as Task 4) beyond the in-script sanity assertions the spec calls for: recall must never exceed 1.0, and must reach ~1.0 at the maximal `efSearch`/`nprobe` for each structure.

- [ ] **Step 1: Write `py/dense/ann_sweep.py`**

```python
"""HNSW and IVF-PQ recall/latency/memory sweep against the exact dev
ground truth -- the phase's money chart.

Latency is measured single-threaded, one search() call per dev query
(6,980 samples), matching this project's no-mean, percentile-only rule
(harness.histogram.LatencyRecorder). faiss and hnswlib are both told to
use one thread per call so their own internal multithreading doesn't
blur a single-query measurement.

Memory footprint is a getrusage RSS delta around each build call.
Platform-specific: ru_maxrss is *bytes* on macOS (this project's only
target platform), *kilobytes* on Linux -- verified empirically at
plan-writing time. Divide by 1e9 for GB; do not port this constant
elsewhere without re-checking.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import faiss
import hnswlib
import numpy as np

from harness.histogram import LatencyRecorder
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

EMBEDDING_DIM = 384
HNSW_M_VALUES = (16, 32, 64)
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH_VALUES = (32, 64, 128, 256, 512)
IVFPQ_NLIST_VALUES = (1024, 4096)
IVFPQ_M_VALUES = (32, 64)
IVFPQ_NBITS = 8
IVFPQ_NPROBE_VALUES = (1, 8, 16, 32, 64)
RECALL_K = 100

faiss.omp_set_num_threads(1)


def current_rss_gb() -> float:
    """macOS: ru_maxrss is bytes. See module docstring."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def recall_at_k(candidate_ids: list[int], exact_ids: list[int], k: int) -> float:
    """Fraction of the exact top-k that appear anywhere in candidate_ids."""
    exact_top_k = set(exact_ids[:k])
    if not exact_top_k:
        return 0.0
    found = exact_top_k & set(candidate_ids)
    return len(found) / len(exact_top_k)


def measure_latency_hnsw(index: hnswlib.Index, queries: np.ndarray, k: int) -> LatencyRecorder:
    recorder = LatencyRecorder()
    for row in range(queries.shape[0]):
        query = queries[row : row + 1]
        started = time.perf_counter_ns()
        index.knn_query(query, k=k, num_threads=1)
        recorder.record((time.perf_counter_ns() - started) / 1000)
    return recorder


def measure_latency_faiss(index, queries: np.ndarray, k: int) -> LatencyRecorder:
    recorder = LatencyRecorder()
    for row in range(queries.shape[0]):
        query = queries[row : row + 1]
        started = time.perf_counter_ns()
        index.search(query, k)
        recorder.record((time.perf_counter_ns() - started) / 1000)
    return recorder


def sweep_hnsw(
    embeddings: np.ndarray,
    docids: np.ndarray,
    dev_queries: np.ndarray,
    dev_qids: list[str],
    ground_truth: dict[str, list[list]],
) -> list[dict]:
    points = []
    n = embeddings.shape[0]
    max_ef_search = max(HNSW_EF_SEARCH_VALUES)
    for m in HNSW_M_VALUES:
        rss_before = current_rss_gb()
        index = hnswlib.Index(space="ip", dim=EMBEDDING_DIM)
        index.init_index(max_elements=n, M=m, ef_construction=HNSW_EF_CONSTRUCTION)
        index.add_items(embeddings, docids)
        memory_gb = current_rss_gb() - rss_before

        recall_at_max_ef_search = None
        for ef_search in HNSW_EF_SEARCH_VALUES:
            index.set_ef(ef_search)
            labels, _ = index.knn_query(dev_queries, k=RECALL_K, num_threads=1)
            recalls = [
                recall_at_k(
                    [int(x) for x in labels[row]], [d for d, _ in ground_truth[qid]], RECALL_K
                )
                for row, qid in enumerate(dev_qids)
            ]
            mean_recall = sum(recalls) / len(recalls)
            latency = measure_latency_hnsw(index, dev_queries, RECALL_K)

            point = {
                "structure": "hnsw",
                "M": m,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "ef_search": ef_search,
                "recall_at_100": mean_recall,
                "latency_us": latency.summary(),
                "memory_gb": memory_gb,
            }
            points.append(point)
            print(
                f"hnsw M={m} efSearch={ef_search}: recall@100={mean_recall:.4f} "
                f"p99={latency.percentile(99)/1000:.2f}ms mem={memory_gb:.3f}GB"
            )
            assert 0.0 <= mean_recall <= 1.0
            if ef_search == max_ef_search:
                recall_at_max_ef_search = mean_recall

        assert recall_at_max_ef_search > 0.9, (
            f"hnsw M={m} recall@100 at max efSearch ({max_ef_search}) was only "
            f"{recall_at_max_ef_search:.4f} -- expected close to 1.0"
        )
    return points


def sweep_ivfpq(
    embeddings: np.ndarray,
    docids: np.ndarray,
    dev_queries: np.ndarray,
    dev_qids: list[str],
    ground_truth: dict[str, list[list]],
) -> list[dict]:
    points = []
    max_nprobe = max(IVFPQ_NPROBE_VALUES)
    for nlist in IVFPQ_NLIST_VALUES:
        for m in IVFPQ_M_VALUES:
            rss_before = current_rss_gb()
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFPQ(
                quantizer, EMBEDDING_DIM, nlist, m, IVFPQ_NBITS, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.add(embeddings)
            memory_gb = current_rss_gb() - rss_before

            recall_at_max_nprobe = None
            for nprobe in IVFPQ_NPROBE_VALUES:
                index.nprobe = nprobe
                _, positions = index.search(dev_queries, RECALL_K)
                recalls = []
                for row, qid in enumerate(dev_qids):
                    candidate_ids = [int(docids[p]) for p in positions[row] if p != -1]
                    exact_ids = [d for d, _ in ground_truth[qid]]
                    recalls.append(recall_at_k(candidate_ids, exact_ids, RECALL_K))
                mean_recall = sum(recalls) / len(recalls)
                latency = measure_latency_faiss(index, dev_queries, RECALL_K)

                point = {
                    "structure": "ivfpq",
                    "nlist": nlist,
                    "m": m,
                    "nbits": IVFPQ_NBITS,
                    "nprobe": nprobe,
                    "recall_at_100": mean_recall,
                    "latency_us": latency.summary(),
                    "memory_gb": memory_gb,
                }
                points.append(point)
                print(
                    f"ivfpq nlist={nlist} m={m} nprobe={nprobe}: recall@100={mean_recall:.4f} "
                    f"p99={latency.percentile(99)/1000:.2f}ms mem={memory_gb:.3f}GB"
                )
                assert 0.0 <= mean_recall <= 1.0
                if nprobe == max_nprobe:
                    recall_at_max_nprobe = mean_recall

            assert recall_at_max_nprobe > 0.5, (
                f"ivfpq nlist={nlist} m={m} recall@100 at max nprobe ({max_nprobe}) was "
                f"only {recall_at_max_nprobe:.4f} -- PQ quantization error should still "
                "allow better than this at full nprobe"
            )
    return points


def main() -> None:
    embeddings = np.load(DATA_DIR / "dense-embeddings.npy")
    docids = np.load(DATA_DIR / "dense-docids.npy")
    dev_queries = np.load(DATA_DIR / "dense-query-embeddings-dev.npy")
    dev_qids = json.loads((DATA_DIR / "dense-query-ids-dev.json").read_text())
    ground_truth = json.loads(
        (RESULTS_DIR / "dense-ground-truth-dev.json").read_text()
    )["results"]

    hnsw_points = sweep_hnsw(embeddings, docids, dev_queries, dev_qids, ground_truth)
    ivfpq_points = sweep_ivfpq(embeddings, docids, dev_queries, dev_qids, ground_truth)

    output = {
        "subset_size": embeddings.shape[0],
        "num_dev_queries": len(dev_qids),
        "points": hnsw_points + ivfpq_points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "dense-ann-sweep.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(output['points'])} points)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the real sweep**

Run: `PYTHONPATH=py uv run python -m dense.ann_sweep`
Expected: 15 `hnsw ...` lines followed by 20 `ivfpq ...` lines, each with a plausible `recall@100` (rising toward 1.0 as `efSearch`/`nprobe` increases within each build), then `wrote bench/results/dense-ann-sweep.json (35 points)`. IVF-PQ training prints a `faiss` clustering warning if a `(nlist, m)` combination is under-provisioned — with 1M training vectors against `nlist<=4096`, this should not fire (verified analogous math at plan-writing time on a smaller synthetic example); if it does fire, note it in the task report rather than silently ignoring it.

- [ ] **Step 3: Sanity-check the output**

Run:
```bash
uv run python3 -c "
import json
data = json.loads(open('bench/results/dense-ann-sweep.json').read())
print('points', len(data['points']))
hnsw = [p for p in data['points'] if p['structure'] == 'hnsw']
ivfpq = [p for p in data['points'] if p['structure'] == 'ivfpq']
print('hnsw', len(hnsw), 'ivfpq', len(ivfpq))
assert len(hnsw) == 15 and len(ivfpq) == 20
best_hnsw = max(hnsw, key=lambda p: p['recall_at_100'])
print('best hnsw recall@100', best_hnsw['recall_at_100'], best_hnsw['M'], best_hnsw['ef_search'])
"
```
Expected: `points 35`, `hnsw 15 ivfpq 20`, best HNSW recall@100 close to 1.0.

- [ ] **Step 4: Commit**

```bash
git add py/dense/ann_sweep.py bench/results/dense-ann-sweep.json
git commit -m "phase3: HNSW and IVF-PQ recall/latency/memory sweep"
```

---

### Task 6: Hybrid fusion

**Files:**
- Create: `py/dense/fusion.py`
- Test: `py/dense/tests/test_fusion.py`

**Interfaces:**
- Consumes: `bench/results/dense-ground-truth-dl19-dl20.json` (Task 4), `runs/manifest.json` (existing, engine `cascade-wand`), `harness.runfile.read_run(path) -> dict[str, dict[str, float]]`, `harness.datasets.load_qrels(query_set) -> dict[str, dict[str, int]]`, `harness.metrics.evaluate(qrels, run, **kwargs) -> EvalResult`.
- Produces: `reciprocal_rank_fusion(channels: list[dict[str, float]], k: int = 60) -> dict[str, float]` (pure), `normalized_score_fusion(channels: list[dict[str, float]]) -> dict[str, float]` (pure), both unit-tested. CLI writes `bench/results/dense-fusion.json`.

- [ ] **Step 1: Write the failing tests**

`py/dense/tests/test_fusion.py`:

```python
"""RRF and normalized score fusion are pure functions over per-channel
(docid -> score) dicts for a single query -- tested against small,
hand-computed examples, matching this repo's style for pure retrieval-math
functions (py/tests/test_metrics.py)."""

from __future__ import annotations

from dense.fusion import normalized_score_fusion, reciprocal_rank_fusion


def test_rrf_hand_computed_two_channels():
    # channel A ranks: d1 (1st), d2 (2nd); channel B ranks: d2 (1st), d3 (2nd)
    channel_a = {"d1": 10.0, "d2": 5.0}
    channel_b = {"d2": 0.9, "d3": 0.1}
    fused = reciprocal_rank_fusion([channel_a, channel_b], k=60)
    # d1: rank 1 in A only -> 1/61. d2: rank 2 in A, rank 1 in B -> 1/62 + 1/61.
    # d3: rank 2 in B only -> 1/62.
    assert fused["d1"] == 1 / 61
    assert fused["d2"] == 1 / 62 + 1 / 61
    assert fused["d3"] == 1 / 62
    # d2 should rank highest (present in both channels)
    assert fused["d2"] > fused["d1"] > fused["d3"]


def test_rrf_document_absent_from_a_channel_is_not_dropped():
    channel_a = {"only_in_a": 1.0}
    channel_b: dict[str, float] = {}
    fused = reciprocal_rank_fusion([channel_a, channel_b], k=60)
    assert "only_in_a" in fused
    assert fused["only_in_a"] == 1 / 61


def test_normalized_score_fusion_min_max_per_channel():
    channel_a = {"d1": 10.0, "d2": 0.0}  # normalizes to 1.0, 0.0
    channel_b = {"d1": 2.0, "d2": 4.0}  # normalizes to 0.0, 1.0
    fused = normalized_score_fusion([channel_a, channel_b])
    assert fused["d1"] == 1.0
    assert fused["d2"] == 1.0


def test_normalized_score_fusion_absent_document_contributes_zero():
    channel_a = {"d1": 10.0, "d2": 0.0}
    channel_b = {"d1": 5.0}
    fused = normalized_score_fusion([channel_a, channel_b])
    # d1 is the only doc in channel_b, so it min-max normalizes to 0.0 there
    # (min == max == 5.0 -> defined as 0.0, see implementation).
    assert fused["d1"] == 1.0 + 0.0
    assert fused["d2"] == 0.0 + 0.0


def test_normalized_score_fusion_single_value_channel_does_not_divide_by_zero():
    channel_a = {"only_doc": 7.0}
    fused = normalized_score_fusion([channel_a])
    assert fused["only_doc"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest py/dense/tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dense.fusion'`.

- [ ] **Step 3: Write `py/dense/fusion.py`**

```python
"""Hybrid fusion: RRF and normalized score fusion over the lexical (WAND)
and dense (exact brute-force) channels, compared against each channel
alone on dl19+dl20 NDCG@10 / recall@1000.

Both fusion functions operate on a single query's channels at a time --
a list of (docid -> score) dicts, one per channel -- and are pure, so they
are unit tested directly rather than only through an end-to-end run.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.datasets import load_qrels
from harness.metrics import evaluate
from harness.runfile import read_run
from harness.runmeta import run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MANIFEST_PATH = REPO_ROOT / "runs" / "manifest.json"

RRF_K = 60
QUERY_SETS = ("dl19", "dl20")


def reciprocal_rank_fusion(
    channels: list[dict[str, float]], k: int = RRF_K
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for channel in channels:
        ranked = sorted(channel.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (docid, _) in enumerate(ranked, start=1):
            fused[docid] = fused.get(docid, 0.0) + 1.0 / (k + rank)
    return fused


def _min_max_normalize(channel: dict[str, float]) -> dict[str, float]:
    if not channel:
        return {}
    lo, hi = min(channel.values()), max(channel.values())
    if lo == hi:
        return {docid: 0.0 for docid in channel}
    return {docid: (score - lo) / (hi - lo) for docid, score in channel.items()}


def normalized_score_fusion(channels: list[dict[str, float]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for channel in channels:
        for docid, score in _min_max_normalize(channel).items():
            fused[docid] = fused.get(docid, 0.0) + score
    return fused


def load_lexical_run(query_set: str) -> dict[str, dict[str, float]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for entry in manifest:
        if entry["engine"] == "cascade-wand" and entry["query_set"] == query_set:
            return read_run(REPO_ROOT / entry["run_path"])
    raise RuntimeError(f"no cascade-wand manifest entry for query_set={query_set!r}")


def load_dense_run(query_set: str) -> dict[str, dict[str, float]]:
    ground_truth = json.loads(
        (RESULTS_DIR / "dense-ground-truth-dl19-dl20.json").read_text()
    )["results"]
    qrels = load_qrels(query_set)
    return {
        qid: {str(docid): score for docid, score in ground_truth[qid]}
        for qid in qrels
        if qid in ground_truth
    }


def fuse_run(
    lexical: dict[str, dict[str, float]],
    dense: dict[str, dict[str, float]],
    method: str,
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    all_qids = set(lexical) | set(dense)
    for qid in all_qids:
        channels = [lexical.get(qid, {}), dense.get(qid, {})]
        if method == "rrf":
            fused[qid] = reciprocal_rank_fusion(channels)
        elif method == "score":
            fused[qid] = normalized_score_fusion(channels)
        else:
            raise ValueError(f"unknown method {method!r}")
    return fused


def main() -> None:
    all_qrels: dict[str, dict[str, int]] = {}
    lexical_run: dict[str, dict[str, float]] = {}
    dense_run: dict[str, dict[str, float]] = {}
    for query_set in QUERY_SETS:
        all_qrels.update(load_qrels(query_set))
        lexical_run.update(load_lexical_run(query_set))
        dense_run.update(load_dense_run(query_set))

    rows = {
        "lexical_only": lexical_run,
        "dense_only": dense_run,
        "rrf": fuse_run(lexical_run, dense_run, "rrf"),
        "score_fusion": fuse_run(lexical_run, dense_run, "score"),
    }

    results = {}
    for name, run in rows.items():
        eval_result = evaluate(all_qrels, run, ndcg_k=(10,), recall_k=(1000,))
        results[name] = {
            "ndcg_10": eval_result.mean["ndcg_cut_10"],
            "recall_1000": eval_result.mean["recall_1000"],
            "num_queries": eval_result.num_queries,
        }
        print(
            f"{name:>14}: ndcg@10={results[name]['ndcg_10']:.4f} "
            f"recall@1000={results[name]['recall_1000']:.4f}"
        )

    output = {
        "query_sets": list(QUERY_SETS),
        "rrf_k": RRF_K,
        "results": results,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "dense-fusion.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest py/dense/tests/test_fusion.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the real fusion evaluation**

Run: `PYTHONPATH=py uv run python -m dense.fusion`
Expected: 4 lines (`lexical_only`, `dense_only`, `rrf`, `score_fusion`), each with plausible NDCG@10/recall@1000 values, then `wrote bench/results/dense-fusion.json`. Requires `runs/cascade-wand.dl19.txt` and `runs/cascade-wand.dl20.txt` to already exist (Phase 1 output) — if missing, run `PYTHONPATH=py uv run python -m baselines.cascade_bm25` first.

- [ ] **Step 6: Sanity-check the output**

Run:
```bash
uv run python3 -c "
import json
data = json.loads(open('bench/results/dense-fusion.json').read())
for name, row in data['results'].items():
    print(name, row)
assert data['results']['rrf']['num_queries'] == data['results']['lexical_only']['num_queries']
"
```
Expected: 4 rows printed, all with `num_queries` == 97 (43 + 54).

- [ ] **Step 7: Commit**

```bash
git add py/dense/fusion.py py/dense/tests/test_fusion.py bench/results/dense-fusion.json
git commit -m "phase3: RRF and normalized-score hybrid fusion vs. each channel alone"
```

---

### Task 7: Report

**Files:**
- Create: `py/dense/report.py`

**Interfaces:**
- Consumes: `bench/results/dense-ann-sweep.json` (Task 5), `bench/results/dense-fusion.json` (Task 6).
- Produces: `bench/plots/phase3-pareto.png`, `bench/phase3.md`. No dedicated test file — matches this repo's existing precedent (`baselines/report.py`, `server/report.py` are both renderers exercised by running them and inspecting output, not by pytest).

- [ ] **Step 1: Write `py/dense/report.py`**

```python
"""Renders bench/phase3.md: the Pareto plot (recall@100 vs p99 latency,
memory as point size) and the 4-row hybrid-fusion table.
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

SUBSET_SIZE = 1_000_000


def render_pareto_plot(sweep: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for structure, color, label in (("hnsw", "tab:blue", "HNSW"), ("ivfpq", "tab:orange", "IVF-PQ")):
        points = [p for p in sweep["points"] if p["structure"] == structure]
        recalls = [p["recall_at_100"] for p in points]
        p99_ms = [p["latency_us"]["p99_us"] / 1000 for p in points]
        sizes = [max(20.0, p["memory_gb"] * 400) for p in points]
        ax.scatter(recalls, p99_ms, s=sizes, alpha=0.6, color=color, label=label)
    ax.set_xlabel("recall@100 (vs. exact brute-force)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title(f"HNSW vs IVF-PQ: recall/latency/memory ({SUBSET_SIZE:,}-passage subset)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_sweep_table(sweep: dict) -> str:
    lines = [
        "| structure | config | recall@100 | p50 (ms) | p99 (ms) | memory (GB) |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(sweep["points"], key=lambda p: (-p["recall_at_100"])):
        if p["structure"] == "hnsw":
            config = f"M={p['M']} efSearch={p['ef_search']}"
        else:
            config = f"nlist={p['nlist']} m={p['m']} nprobe={p['nprobe']}"
        lines.append(
            f"| {p['structure']} | {config} | {p['recall_at_100']:.4f} | "
            f"{p['latency_us']['p50_us']/1000:.2f} | {p['latency_us']['p99_us']/1000:.2f} | "
            f"{p['memory_gb']:.3f} |"
        )
    return "\n".join(lines)


def render_fusion_table(fusion: dict) -> str:
    lines = [
        "| channel | NDCG@10 | recall@1000 |",
        "|---|---|---|",
    ]
    order = ["lexical_only", "dense_only", "rrf", "score_fusion"]
    for name in order:
        row = fusion["results"][name]
        lines.append(f"| {name} | {row['ndcg_10']:.4f} | {row['recall_1000']:.4f} |")
    return "\n".join(lines)


def render_markdown(sweep: dict, fusion: dict) -> str:
    provenance = sweep["provenance"]
    return f"""# Phase 3: Dense Recall and the ANN Pareto Frontier

## Corpus subset (limitation, stated up front)

This machine has 8GB RAM and no GPU. The full 8.8M-passage corpus would need
~13.5GB just for float32 embeddings before any index overhead, so this phase
subsets to **{sweep['subset_size']:,} passages**: every document judged in the
dl19+dl20 qrels, plus a seeded random fill. The Pareto-frontier finding (which
ANN structure wins at which recall target) does not depend on corpus size;
only the absolute latency/memory numbers would shift at 8.8M.

Encoder: `BAAI/bge-small-en-v1.5`. Evaluated over {sweep['num_dev_queries']}
dev queries (recall@100 / latency) and dl19+dl20 (fusion table).

## Pareto frontier: recall@100 vs. p99 latency

![Pareto plot](plots/phase3-pareto.png)

{render_sweep_table(sweep)}

## Hybrid fusion (dl19+dl20)

{render_fusion_table(fusion)}

## Configuration

- git SHA: `{provenance['git_sha']}`
- hardware: {provenance['hardware']['cpu']}, {int(int(provenance['hardware']['memory_bytes']) / 1e9)}GB RAM
- timestamp: {provenance['timestamp_utc']}
"""


def main() -> None:
    sweep = json.loads((RESULTS_DIR / "dense-ann-sweep.json").read_text())
    fusion = json.loads((RESULTS_DIR / "dense-fusion.json").read_text())

    render_pareto_plot(sweep, PLOTS_DIR / "phase3-pareto.png")
    markdown = render_markdown(sweep, fusion)
    (BENCH_DIR / "phase3.md").write_text(markdown)
    print(f"wrote {(BENCH_DIR / 'phase3.md').relative_to(REPO_ROOT)}")
    print(f"wrote {(PLOTS_DIR / 'phase3-pareto.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report renderer**

Run: `PYTHONPATH=py uv run python -m dense.report`
Expected: `wrote bench/phase3.md`, `wrote bench/plots/phase3-pareto.png`.

- [ ] **Step 3: Visually inspect the output**

Read `bench/phase3.md` and confirm the tables render sensibly (35 sweep rows sorted by recall@100 descending, 4 fusion rows). Open `bench/plots/phase3-pareto.png` and confirm it shows two colored point clouds (HNSW, IVF-PQ) with recall on x, latency on y, and visibly varying point sizes for memory.

- [ ] **Step 4: Commit**

```bash
git add py/dense/report.py bench/phase3.md bench/plots/phase3-pareto.png
git commit -m "phase3: render the Pareto plot and fusion table into bench/phase3.md"
```

---

### Task 8: End-to-end verification and README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: every artifact from Tasks 1-7.
- Produces: a "Running Phase 3" section in `README.md`, matching the style of the existing "Running Phase 1"/"Running Phase 2" sections.

- [ ] **Step 1: Confirm the committed artifacts are self-consistent**

Tasks 2-7 each already ran their own step against real data and committed
the result — this step does not re-run the ~60-minute encode step again
(it was already validated once in Task 3's Step 5-6; re-running it here
would only re-derive the same intermediates at real cost with no new
information). Instead, confirm the already-committed `bench/results/*.json`
and `bench/phase3.md` are mutually consistent:

```bash
uv run python3 -c "
import json
sweep = json.loads(open('bench/results/dense-ann-sweep.json').read())
fusion = json.loads(open('bench/results/dense-fusion.json').read())
assert len(sweep['points']) == 35, f\"expected 35 points, got {len(sweep['points'])}\"
assert sweep['subset_size'] == 1_000_000
assert set(fusion['results']) == {'lexical_only', 'dense_only', 'rrf', 'score_fusion'}
for name, row in fusion['results'].items():
    assert row['num_queries'] == 97, f\"{name}: expected 97 queries, got {row['num_queries']}\"
print('bench/results/*.json: consistent')
"
grep -c "^|" bench/phase3.md  # sanity: table rows present
ls -la bench/plots/phase3-pareto.png
```
Expected: `bench/results/*.json: consistent`, a nonzero table-row count, and the plot file present.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including the new `py/dense/tests/test_subset.py` (5) and `py/dense/tests/test_fusion.py` (5) and `py/dense/tests/test_encode.py` (2) — 12 new tests on top of the existing 53.

- [ ] **Step 3: Add the README section**

Add after the existing "## Running Phase 2 (sub-project A: single-node server)" section, before "## Ground rules":

```markdown
## Running Phase 3 (dense recall and the ANN Pareto frontier)

Needs the `dense` extra (`sentence-transformers`, `torch`, `faiss-cpu`,
`hnswlib`, `matplotlib`) and Phase 1's WAND run files for dl19/dl20
(`uv run python -m baselines.cascade_bm25` if not already present).
`hnswlib` has no macOS wheel and needs the SDK's libc++ headers pointed at
explicitly to build from source:

```bash
SDK_PATH=$(xcrun --show-sdk-path)
CXXFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
CPPFLAGS="-isystem $SDK_PATH/usr/include/c++/v1" \
uv sync --extra dense
```

```bash
export PYTHONPATH=py
uv run python -m dense.subset          # 1M-passage qrels-preserving subset
uv run python -m dense.encode          # bge-small-en-v1.5 encoding, ~60 min
uv run python -m dense.ground_truth    # exact brute-force top-k -> bench/results/dense-ground-truth-*.json
uv run python -m dense.ann_sweep       # HNSW + IVF-PQ sweep -> bench/results/dense-ann-sweep.json
uv run python -m dense.fusion          # RRF / score fusion -> bench/results/dense-fusion.json
uv run python -m dense.report          # -> bench/phase3.md, bench/plots/phase3-pareto.png
```

This machine has 8GB RAM and no GPU, so this phase subsets to 1M passages
rather than the full 8.8M-passage corpus — see `bench/phase3.md`'s own
opening section for why.
```

- [ ] **Step 4: Update the status table**

Change the Phase 3 row in the `## Status` table from:
```
| 3 — Dense recall | ANN Pareto frontier | not started |
```
to:
```
| 3 — Dense recall | ANN Pareto frontier | **done** |
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "phase3: document setup and add exit-artifact status"
```
