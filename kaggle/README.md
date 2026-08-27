# Running Phase 4 on Kaggle

1. Create a new Kaggle Notebook, enable a GPU (Settings -> Accelerator -> GPU T4 x2 or x1).
2. Upload as a Kaggle Dataset (Add Data -> Upload), preserving this directory layout:
   - The whole `py/rank/` directory (all `.py` files, not `tests/`).
   - `py/harness/__init__.py`, `py/harness/histogram.py`, `py/harness/runmeta.py`,
     `py/harness/datasets.py`, `py/harness/metrics.py`, `py/harness/runfile.py`,
     and `py/harness/loadgen.py`
     (all six `harness` modules `phase4_driver.py` imports, directly or via
     `rank.crossencoder_harness`/`rank.prerank_mlp` — `datasets.py` is needed for
     `load_queries`/`load_qrels`, which pull from `ir_datasets`; `ir_datasets` itself
     must also be installed on Kaggle: `!pip install ir_datasets`. `loadgen.py` is
     the open-loop load generator `rank.crossencoder_harness`'s batching/queue
     harness is built on).
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
