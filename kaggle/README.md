# Running Phase 4 on Kaggle

1. Create a new Kaggle Notebook, enable a GPU (Settings -> Accelerator -> GPU T4 x2 or x1).
2. Upload as a Kaggle Dataset (Add Data -> Upload), preserving this directory layout
   (`phase4_driver.py` does a plain `import rank.foo` / `import harness.foo`, so both
   packages must be importable from the notebook's working directory — e.g. upload
   into `/kaggle/working/py/rank/...` and `/kaggle/working/py/harness/...`, then add
   `import sys; sys.path.insert(0, "/kaggle/working/py")` as the first cell before
   running the driver):
   - The whole `py/rank/` directory (all `.py` files, not `tests/`).
   - `py/harness/__init__.py`, `py/harness/histogram.py`, `py/harness/runmeta.py`,
     `py/harness/datasets.py`, `py/harness/metrics.py`, and `py/harness/runfile.py`
     (all five `harness` modules `phase4_driver.py` imports, directly or via
     `rank.crossencoder_harness`/`rank.prerank_mlp` — `datasets.py` is needed for
     `load_queries`/`load_qrels`, which pull from `ir_datasets`; `ir_datasets` itself
     must also be installed on Kaggle: `!pip install ir_datasets`).
   - `data/rank-dense-scores.jsonl` and `data/rank-candidate-texts.json`
     (from Task 2's local run), at `/kaggle/working/data/...` (or adjust
     `phase4_driver.py`'s relative paths to wherever you place them).
   - `runs/cascade-wand.dev.txt`, `runs/cascade-wand.dl19.txt`, `runs/cascade-wand.dl20.txt`,
     at `/kaggle/working/runs/...` (same note as above).

   Note: `py/dense/` is **not** needed here — `rank.encode_candidates` (imported by
   the driver only for its pure `DEV_NEGATIVE_POOL_DEPTH`/`truncate_to_top_k` helpers)
   imports `sentence_transformers`/`dense.encode` lazily, inside its own `main()`,
   specifically so this Kaggle driver's import of it doesn't need `py/dense/` uploaded.
3. In a notebook cell:
   ```bash
   !pip uninstall -y onnxruntime  # Kaggle images may preinstall the CPU build
   !pip install onnxruntime-gpu onnxconverter-common
   ```
4. Before running, edit `kaggle/phase4_driver.py`'s `SOURCE_GIT_SHA` constant
   — run `git rev-parse HEAD` locally and paste the result in, since Kaggle
   has no git repo to read it from (see this phase's design spec, "Kaggle
   provenance gap").
5. Run `kaggle/phase4_driver.py` (paste its contents into a cell, or
   `%run phase4_driver.py` if uploaded as a file) — expect roughly an hour
   of GPU time across all four sub-experiments (see the design spec's
   session-budget reasoning).
6. Download `bench/results/rank-*.json` from the notebook's output files and
   drop them into this repo's `bench/results/` directory.
