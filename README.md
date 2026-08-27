# Cascade

A multi-stage search engine built under a fixed end-to-end latency budget, where
every stage is measured, ablated, and defended with numbers. The deliverable is
`bench/REPORT.md`, not a demo.

Full design: [`CASCADE_SEARCH_PLAN.md`](CASCADE_SEARCH_PLAN.md).

## Status

| Phase | What it produces | State |
|---|---|---|
| 0 — Harness | `bench/baselines.md` | **done** |
| 1 — C++ index + BlockMax-WAND | `bench/phase1.md` | **done** |
| 2 — Serving, sharding, tail latency | capacity statement | sub-project A done |
| 3 — Dense recall | ANN Pareto frontier | **done** |
| 4 — Ranking cascade | batching + precision tables | **done** |
| 5 — Video, multi-field | field ablations | not started |
| 6 — Near-real-time indexing | freshness/latency tradeoff | not started |

## Layout

```
py/harness/     metrics, open-loop load generation, latency recording, run files
py/baselines/   Lucene (Anserini) and Tantivy BM25 baselines, cascade C++ index driver
py/dense/       dense encoding, ANN sweep (HNSW/IVF-PQ), hybrid fusion
py/tests/       pytest suite
cpp/index/      analyzer, Porter stemmer, postings codec, index builder
cpp/query/      DAAT-OR, WAND, BlockMax-WAND over a shared cursor
cpp/bindings/   pybind11 module (one analyzer, one BM25, no offline/online skew)
cpp/tests/      correctness: BMW/WAND top-k == exhaustive top-k, on a synthetic corpus
cpp/server/     gRPC service: bounded queue + worker pool + LRU cache
py/server/      server client, process manager, throughput/cache experiments
bench/results/  raw JSON per run — committed
bench/          baselines.md, phase1.md, phase2a.md, phase3.md, plots, REPORT.md
.tools/         JDK 21 and the Anserini fatjar (gitignored)
data/ indexes/ runs/   corpora, built indexes, and run files (gitignored)
```

Build the C++ side with `make -C cpp`. It needs no dependencies beyond
pybind11's headers, which come from the venv.

## Setup

```bash
uv sync --extra dev --extra baselines
uv run pytest                      # 96 pass, 3 skip, no corpus needed
```

The 3 skips are `py/server/tests/test_server_integration.py`, which needs the
built cascade index and generated proto stubs (Phase 1 and Phase 2 setup below).
`py/rank/tests` needs network access: it downloads the real
`cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face
(`AutoTokenizer.from_pretrained` / `AutoModelForSequenceClassification.from_pretrained`)
rather than using a mocked or vendored model, and exports it to ONNX on the fly.
Nothing there is mocked, which is the point — the ONNX export path is what the
Kaggle run depends on, so it is exercised for real, just at CPU scale.

The baselines additionally need an arm64 JDK 21 and the Anserini fatjar in
`.tools/`, and roughly 10 GB of disk for the corpus and indexes:

```bash
curl -sL -o .tools/jdk21.tar.gz \
  "https://api.adoptium.net/v3/binary/latest/21/ga/mac/aarch64/jdk/hotspot/normal/eclipse"
tar xzf .tools/jdk21.tar.gz -C .tools && rm .tools/jdk21.tar.gz
curl -sL -o .tools/anserini-fatjar.jar \
  "https://repo1.maven.org/maven2/io/anserini/anserini/1.0.0/anserini-1.0.0-fatjar.jar"
```

## Running Phase 0

Run these as modules, not as files: `python py/baselines/export.py` puts
`py/baselines/` on the path instead of `py/`, and the imports fail.

```bash
export PYTHONPATH=py
uv run python -m baselines.export        # ir_datasets -> JSONL + query TSVs
uv run python -m baselines.lucene        # index + search with Anserini
uv run python -m baselines.tantivy_bm25  # second independent BM25
uv run python -m baselines.latency       # open-loop latency against Tantivy
uv run python -m baselines.report        # -> bench/results/*.json, bench/baselines.md
```

## Running Phase 1

Needs `data/msmarco-passage.tsv` (from `baselines.export` above) and the C++ side
built (`make -C cpp all`, needs `pybind11` from `uv sync --extra dev`).

```bash
export PYTHONPATH=py
uv run python -m baselines.cascade_bm25        # build index, WAND+BMW NDCG runs,
                                                # DAAT-OR on dl19+dl20 for the
                                                # equivalence check
uv run python -m baselines.cascade_query_cost  # postings/evaluations/p99 table
uv run python -m baselines.phase1_report       # -> bench/results/cascade-*.json,
                                                #    bench/phase1.md
```

`cascade_bm25` must run first: `phase1_report` reads `runs/manifest.json`, which
`cascade_bm25` (not tracked in git — it's under `/runs/`) is what populates it.

## Running Phase 2 (sub-project A: single-node server)

Needs the Phase 1 index and `pkg-config` pointed at Anaconda's grpc/protobuf
(`brew install pkg-config` if you don't have it; the Makefile defaults
`PKG_CONFIG_PATH` to `/opt/anaconda3/lib/pkgconfig`, override it if your
grpc/protobuf live elsewhere).

```bash
export PYTHONPATH=py
uv run python -m server.gen_proto        # generates py/server/generated/*.py
make -C cpp all                          # builds cpp/build/server_bin
uv run python -m server.throughput_knee  # -> bench/results/server-throughput-knee.json
uv run python -m server.cache_sensitivity  # -> bench/results/server-cache-sensitivity.json
uv run python -m server.report           # -> bench/phase2a.md
```

The server itself (`cpp/build/server_bin <index_dir> [--port=P] [--workers=N]
[--queue-depth=D] [--cache-capacity=C] [--algorithm=wand|blockmax-wand|daat-or]`)
can also be run standalone for manual testing.

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

## Ground rules

These are enforced in code, not just documented:

- **Open-loop load generation.** Arrival deadlines are computed up front and
  latency is measured from the scheduled arrival, not from dispatch. A
  closed-loop generator cannot build a queue and so never finds the latency knee.
- **No mean latency.** `LatencyRecorder` reports p50/p95/p99/p99.9 and does not
  expose a mean.
- **Metrics match `trec_eval`.** Verified against `pytrec_eval` to 4 decimals,
  including its linear NDCG gain and its descending-docid tie-break.
- **A number without its config is not a result.** Every recorded run carries
  git SHA, hardware, and the full engine configuration.
