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
| 3 — Dense recall | ANN Pareto frontier | not started |
| 4 — Ranking cascade | batching + precision tables | not started |
| 5 — Video, multi-field | field ablations | not started |
| 6 — Near-real-time indexing | freshness/latency tradeoff | not started |

## Layout

```
py/harness/     metrics, open-loop load generation, latency recording, run files
py/baselines/   Lucene (Anserini) and Tantivy BM25 baselines, cascade C++ index driver
py/tests/       pytest suite
cpp/index/      analyzer, Porter stemmer, postings codec, index builder
cpp/query/      DAAT-OR, WAND, BlockMax-WAND over a shared cursor
cpp/bindings/   pybind11 module (one analyzer, one BM25, no offline/online skew)
cpp/tests/      correctness: BMW/WAND top-k == exhaustive top-k, on a synthetic corpus
cpp/server/     gRPC service: bounded queue + worker pool + LRU cache
py/server/      server client, process manager, throughput/cache experiments
bench/results/  raw JSON per run — committed
bench/          baselines.md, phase1.md, phase2a.md, plots, REPORT.md
.tools/         JDK 21 and the Anserini fatjar (gitignored)
data/ indexes/ runs/   corpora, built indexes, and run files (gitignored)
```

Build the C++ side with `make -C cpp`. It needs no dependencies beyond
pybind11's headers, which come from the venv.

## Setup

```bash
uv sync --extra dev --extra baselines
uv run pytest                      # 50 tests, no corpus needed
```

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
