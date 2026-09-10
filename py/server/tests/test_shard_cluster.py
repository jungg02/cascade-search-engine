"""Integration test: build two tiny real shard indexes from a synthetic
corpus, start a real ShardCluster, dispatch one real query through a real
Broker end to end. Mirrors test_server_integration.py's skip-if-absent
convention -- skips (not a collection failure) if cpp/build/build_index,
server_bin, or the generated proto stubs aren't built."""

from __future__ import annotations

import subprocess
from pathlib import Path

import grpc
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"
SERVER_BIN = REPO_ROOT / "cpp" / "build" / "server_bin"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
_READY = (
    BUILD_INDEX_BIN.exists()
    and SERVER_BIN.exists()
    and (GENERATED_DIR / "search_pb2.py").exists()
)

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="needs cpp/build/build_index, server_bin, and generated proto stubs "
    "(see README's Phase 1 and Phase 2 setup)",
)

if _READY:
    from server.broker import Broker
    from server.shard_cluster import ShardCluster


def _build_shard(tmp_path: Path, name: str, docs: list[tuple[str, str]]) -> Path:
    corpus = tmp_path / f"{name}.tsv"
    corpus.write_text("".join(f"{docid}\t{text}\n" for docid, text in docs))
    index_dir = tmp_path / name
    subprocess.run([str(BUILD_INDEX_BIN), str(corpus), str(index_dir)], check=True)
    return index_dir


def test_broker_merges_across_two_real_shards(tmp_path):
    shard0 = _build_shard(tmp_path, "shard0", [
        ("1", "bank teller job description"),
        ("2", "river bank erosion"),
    ])
    shard1 = _build_shard(tmp_path, "shard1", [
        ("3", "teller machine withdrawal"),
        ("4", "unrelated passage about weather"),
    ])

    with ShardCluster([shard0, shard1], base_port=50251) as cluster:
        addresses = cluster.addresses
        broker = Broker.for_addresses(addresses)
        response = broker.dispatch("bank teller", k=10)
        assert 0 < len(response.results) <= 10
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)
        broker.close()

    # Processes are down: connecting to either address now fails fast
    # rather than hanging or succeeding.
    for address in addresses:
        channel = grpc.insecure_channel(address)
        with pytest.raises(grpc.FutureTimeoutError):
            grpc.channel_ready_future(channel).result(timeout=1.0)
        channel.close()
