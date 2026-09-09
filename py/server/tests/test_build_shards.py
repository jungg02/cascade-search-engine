"""build_shards invokes cpp/build/build_index once per shard TSV, skipping
shards that already have a complete index (index.terms exists -- the
builder's last write, per cpp/index/builder.cc). The real binary is never
invoked here: subprocess.run is replaced by a small fake so this test runs
in milliseconds and doesn't need the C++ build."""

from __future__ import annotations

from pathlib import Path

import pytest

import server.build_shards as build_shards_module
from server.build_shards import build_shards, is_built


class _FakeSubprocess:
    """Stands in for the real `subprocess` module inside build_shards'
    namespace only -- monkeypatch.setattr(build_shards_module, "subprocess",
    ...) replaces the *name* build_shards.py looks up, not the process-wide
    subprocess module, so no other code in the test process is affected."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], check: bool) -> None:
        assert check is True
        self.calls.append(args)
        _, shard_tsv, index_dir = args
        Path(index_dir, "index.terms").write_bytes(b"")


def test_build_shards_invokes_binary_once_per_shard(tmp_path, monkeypatch):
    fake = _FakeSubprocess()
    monkeypatch.setattr(build_shards_module, "subprocess", fake)

    fake_bin = tmp_path / "build_index"
    fake_bin.write_text("#!/bin/sh\n")
    shard_tsvs = [tmp_path / f"shard{i}.tsv" for i in range(3)]
    for p in shard_tsvs:
        p.write_text("0\tx\n")

    index_dirs = build_shards(
        shard_tsvs, 3, indexes_dir=tmp_path / "idx", build_index_bin=fake_bin
    )

    assert len(fake.calls) == 3
    assert [c[1] for c in fake.calls] == [str(p) for p in shard_tsvs]
    assert all(d.exists() for d in index_dirs)


def test_build_shards_skips_already_built(tmp_path, monkeypatch):
    fake = _FakeSubprocess()
    monkeypatch.setattr(build_shards_module, "subprocess", fake)

    fake_bin = tmp_path / "build_index"
    fake_bin.write_text("#!/bin/sh\n")
    index_dir = tmp_path / "idx" / "n1" / "shard0"
    index_dir.mkdir(parents=True)
    (index_dir / "index.terms").write_bytes(b"")

    shard_tsv = tmp_path / "shard0.tsv"
    shard_tsv.write_text("0\tx\n")
    build_shards([shard_tsv], 1, indexes_dir=tmp_path / "idx", build_index_bin=fake_bin)

    assert fake.calls == []


def test_is_built_checks_index_terms(tmp_path):
    assert not is_built(tmp_path / "missing")
    (tmp_path / "index.terms").write_bytes(b"")
    assert is_built(tmp_path)


def test_build_shards_raises_if_binary_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_shards(
            [tmp_path / "shard0.tsv"], 1, indexes_dir=tmp_path / "idx",
            build_index_bin=tmp_path / "does-not-exist",
        )
