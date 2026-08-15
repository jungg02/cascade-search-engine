# Cascade

A multi-stage search engine built under a fixed end-to-end latency budget, where
every stage is measured, ablated, and defended with numbers. The deliverable is
`bench/REPORT.md`, not a demo.

Full design: [`CASCADE_SEARCH_PLAN.md`](CASCADE_SEARCH_PLAN.md).

## Status

| Phase | What it produces | State |
|---|---|---|
| 0 — Harness | `bench/baselines.md` | **done** |
| 1 — C++ index + BlockMax-WAND | pruning/latency table | in progress |
| 2 — Serving, sharding, tail latency | capacity statement | not started |
| 3 — Dense recall | ANN Pareto frontier | not started |
| 4 — Ranking cascade | batching + precision tables | not started |
| 5 — Video, multi-field | field ablations | not started |
| 6 — Near-real-time indexing | freshness/latency tradeoff | not started |

## Layout

```
py/harness/     metrics, open-loop load generation, latency recording, run files
py/baselines/   Lucene (Anserini) and Tantivy BM25 baselines
py/tests/       pytest suite
cpp/index/      analyzer, Porter stemmer, postings codec, index builder
cpp/query/      DAAT-OR, WAND, BlockMax-WAND over a shared cursor
cpp/bindings/   pybind11 module (one analyzer, one BM25, no offline/online skew)
cpp/tests/      correctness: BMW top-k == exhaustive top-k
bench/results/  raw JSON per run — committed
bench/          baselines.md, plots, REPORT.md
.tools/         JDK 21 and the Anserini fatjar (gitignored)
data/ indexes/ runs/   corpora and derived artifacts (gitignored)
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
