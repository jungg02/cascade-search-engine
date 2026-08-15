"""Integration test: start the real C++ server as a subprocess, talk to it
over real gRPC. This proves the network path works end to end — the pure
C++ unit test of SearchServiceImpl (cpp/server/tests/test_search_service.cc)
only proves the request-handling logic, not that grpc::Server, the generated
stubs, and this Python client actually agree on the wire format.

Needs the Phase 1 index and the generated proto stubs; skips gracefully
(not a collection failure) when either is absent, so a fresh clone's default
`uv run pytest` still passes without the ~900MB corpus.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
_READY = INDEX_DIR.exists() and (GENERATED_DIR / "search_pb2.py").exists()

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="needs the built cascade index and generated proto stubs "
    "(see README's Phase 1 and Phase 2 setup)",
)

if _READY:
    import grpc

    from server.client import SearchClient
    from server.process import ServerProcess


def test_single_query_round_trip():
    with ServerProcess(INDEX_DIR, port=50151) as server:
        client = SearchClient(server.address)
        response = client.dispatch("what is a bank teller", k=10)
        assert len(response.results) > 0
        assert len(response.results) <= 10
        client.close()


def test_cache_hit_on_repeated_query():
    with ServerProcess(INDEX_DIR, port=50152, cache_capacity=10) as server:
        client = SearchClient(server.address)
        first = client.dispatch("bank teller")
        assert not first.cache_hit
        second = client.dispatch("bank teller")
        assert second.cache_hit
        assert [(r.docid, r.score) for r in second.results] == [
            (r.docid, r.score) for r in first.results
        ]
        client.close()


def test_saturated_queue_sheds_load():
    # One worker, one queue slot: at most 2 requests in flight at once. Firing
    # 20 concurrently on one shared channel makes it near-certain some arrive
    # before the first two finish, so at least one comes back
    # RESOURCE_EXHAUSTED rather than succeeding or hanging — this is what a
    # synthetic in-process call (test_search_service.cc) can't exercise:
    # there's no network latency there to spread 20 calls out in wall time.
    with ServerProcess(INDEX_DIR, port=50153, workers=1, queue_depth=1) as server:
        client = SearchClient(server.address)
        results = []
        lock = threading.Lock()

        def fire():
            try:
                client.dispatch("the")
                with lock:
                    results.append("ok")
            except grpc.RpcError as exc:
                with lock:
                    results.append(exc.code())

        threads = [threading.Thread(target=fire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.close()

        assert "ok" in results, "at least some requests should succeed"
        assert grpc.StatusCode.RESOURCE_EXHAUSTED in results, (
            "at least one request should be shed under this saturating burst"
        )
