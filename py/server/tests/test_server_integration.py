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
    # One worker, one queue slot: at most 2 requests in flight at once.
    # Firing a big enough burst on one shared channel makes it near-certain
    # some arrive before the first two finish, so at least one comes back
    # RESOURCE_EXHAUSTED rather than succeeding or hanging — this is what a
    # synthetic in-process call (test_search_service.cc) can't exercise:
    # there's no network latency there to spread calls out in wall time.
    #
    # Two things matter for this to reliably hit BoundedQueue's own "queue
    # full" rejection rather than being a coin flip: a Barrier so every
    # thread calls dispatch() at effectively the same instant instead of
    # however plain sequential thread.start() calls happen to interleave
    # (with "the" a sub-millisecond query, a loose burst lets earlier
    # threads finish and free the queue slot before later ones even start),
    # and enough threads. Measured directly: 20 threads + barrier still saw
    # all-succeed runs (4/8 trials with zero rejections); 100 threads +
    # barrier was reliable (0/8 fails across two separate 8-trial batches).
    with ServerProcess(INDEX_DIR, port=50153, workers=1, queue_depth=1) as server:
        client = SearchClient(server.address)
        results = []
        lock = threading.Lock()
        num_threads = 100
        barrier = threading.Barrier(num_threads)

        def fire():
            barrier.wait()
            try:
                client.dispatch("the")
                with lock:
                    results.append(("ok", None))
            except grpc.RpcError as exc:
                with lock:
                    results.append((exc.code(), exc.details()))

        threads = [threading.Thread(target=fire) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.close()

        assert ("ok", None) in results, "at least some requests should succeed"
        # gRPC's own ResourceQuota now carries +8 headroom above
        # workers + queue_depth specifically so this assertion can be
        # precise: with no headroom, gRPC's own admission control exactly
        # coincided with BoundedQueue's capacity, so gRPC always rejected
        # the (N+1)th concurrent call with its own RESOURCE_EXHAUSTED
        # ("Server Threadpool Exhausted") before that call could ever reach
        # try_submit() — BoundedQueue's own "queue full" rejection was
        # unreachable via the network for any config (measured: 150
        # concurrent requests over 150 separate channels against this same
        # workers=1/queue_depth=1 config produced 14 RESOURCE_EXHAUSTED
        # responses, 0 of 14 with detail "queue full"). See main.cc's
        # ResourceQuota comment for the fix.
        assert any(
            code == grpc.StatusCode.RESOURCE_EXHAUSTED and details == "queue full"
            for code, details in results
        ), "at least one request should be shed by BoundedQueue specifically"
