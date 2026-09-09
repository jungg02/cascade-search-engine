"""Splits the corpus TSV into N contiguous, document-partitioned shards.

cpp/index/builder.cc has no doc-id-range or offset flag -- it always reads
its input TSV from line 0 -- so sharding is done by writing N separate TSV
files rather than by changing the (proven, frozen) C++ builder. Splitting by
line count keeps each shard's underlying documents contiguous and disjoint,
matching the plan's "document-partitioned sharding," and concatenating the
N output files in shard order reproduces the input exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "msmarco-passage.tsv"
SHARDS_DIR = REPO_ROOT / "data" / "shards"


def shard_paths(n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]:
    return [shards_dir / f"n{n}" / f"shard{i}.tsv" for i in range(n)]


def partition(corpus_path: Path, n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]:
    """Writes n shard files under shards_dir/n{n}/, skipping if all already
    exist. Returns the n shard paths in shard order."""
    paths = shard_paths(n, shards_dir)
    if all(p.exists() for p in paths):
        return paths

    lines = corpus_path.read_text().splitlines(keepends=True)
    total = len(lines)
    base, extra = divmod(total, n)
    # The first `extra` shards get one extra line each, so sizes differ by
    # at most one line -- "roughly equal," not "equal except a giant last
    # shard absorbing the whole remainder."
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    start = 0
    for i, path in enumerate(paths):
        size = base + (1 if i < extra else 0)
        path.write_text("".join(lines[start:start + size]))
        start += size
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()
    for n in args.n:
        paths = partition(args.corpus, n)
        print(f"n={n}: {len(paths)} shards written under {paths[0].parent}")


if __name__ == "__main__":
    main()
