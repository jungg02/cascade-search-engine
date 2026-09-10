"""partition_corpus splits a corpus TSV into N contiguous, document-
partitioned shard files. These tests use a small synthetic corpus, never
the real 8.8M-line one -- that's exercised for real in Task 9."""

from __future__ import annotations

from server.partition_corpus import partition, shard_paths


def test_partition_covers_every_line_exactly_once(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    lines = [f"{i}\tpassage number {i}\n" for i in range(23)]
    corpus.write_text("".join(lines))

    paths = partition(corpus, 4, shards_dir=tmp_path / "shards")
    assert len(paths) == 4

    seen: list[str] = []
    for path in paths:
        seen.extend(path.read_text().splitlines(keepends=True))
    assert seen == lines


def test_shard_sizes_differ_by_at_most_one_line(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text("".join(f"{i}\tx\n" for i in range(23)))

    paths = partition(corpus, 4, shards_dir=tmp_path / "shards")
    sizes = [len(p.read_text().splitlines()) for p in paths]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 23


def test_partition_is_idempotent(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text("0\ta\n1\tb\n2\tc\n3\td\n")
    shards_dir = tmp_path / "shards"

    first = partition(corpus, 2, shards_dir=shards_dir)
    first_contents = [p.read_text() for p in first]

    # Corrupt the corpus after the first call: a second call, seeing the
    # shard files already exist, must not touch them again.
    corpus.write_text("garbage")
    second = partition(corpus, 2, shards_dir=shards_dir)
    assert [p.read_text() for p in second] == first_contents


def test_shard_paths_names_files_by_index(tmp_path):
    paths = shard_paths(3, shards_dir=tmp_path)
    assert [p.name for p in paths] == ["shard0.tsv", "shard1.tsv", "shard2.tsv"]
    assert paths[0].parent.name == "n3"
