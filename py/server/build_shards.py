"""Builds one index per shard by invoking the existing, unmodified
cpp/build/build_index binary once per shard TSV -- sharding needs no
change to the C++ builder, only N separate inputs (see partition_corpus.py).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"
INDEXES_DIR = REPO_ROOT / "indexes" / "shards"


def shard_index_dirs(n: int, indexes_dir: Path = INDEXES_DIR) -> list[Path]:
    return [indexes_dir / f"n{n}" / f"shard{i}" for i in range(n)]


def is_built(index_dir: Path) -> bool:
    # index.terms is the builder's last write (cpp/index/builder.cc), so its
    # presence means a previous run finished rather than crashed partway.
    return (index_dir / "index.terms").exists()


def build_shards(
    shard_tsv_paths: list[Path],
    n: int,
    indexes_dir: Path = INDEXES_DIR,
    build_index_bin: Path = BUILD_INDEX_BIN,
) -> list[Path]:
    if not build_index_bin.exists():
        raise FileNotFoundError(f"{build_index_bin} not built; run `make -C cpp all` first")
    index_dirs = shard_index_dirs(n, indexes_dir)
    for shard_tsv, index_dir in zip(shard_tsv_paths, index_dirs):
        if is_built(index_dir):
            continue
        index_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(build_index_bin), str(shard_tsv), str(index_dir)], check=True)
    return index_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    from server.partition_corpus import CORPUS_PATH, partition

    for n in args.n:
        shard_tsvs = partition(CORPUS_PATH, n)
        index_dirs = build_shards(shard_tsvs, n)
        print(f"n={n}: {len(index_dirs)} shard indexes ready under {index_dirs[0].parent}")


if __name__ == "__main__":
    main()
