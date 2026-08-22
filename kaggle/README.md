# Running Phase 4 on Kaggle

1. Create a new Kaggle Notebook, enable a GPU (Settings -> Accelerator -> GPU T4 x2 or x1).
2. Upload as a Kaggle Dataset (Add Data -> Upload):
   - The whole `py/rank/` directory (all `.py` files, not `tests/`).
   - `py/harness/histogram.py`, `py/harness/runmeta.py`, and `py/harness/datasets.py`
     (the three `harness` modules `phase4_driver.py` needs — `datasets.py` is
     needed for `load_queries`, which pulls query text from `ir_datasets`;
     `ir_datasets` itself must also be installed on Kaggle:
     `!pip install ir_datasets`).
   - `data/rank-dense-scores.jsonl` and `data/rank-candidate-texts.json`
     (from Task 2's local run).
   - `runs/cascade-wand.dev.txt`, `runs/cascade-wand.dl19.txt`, `runs/cascade-wand.dl20.txt`.
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
