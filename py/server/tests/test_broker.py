"""Broker.dispatch fans out to every client, waits for all of them
(ALL_COMPLETED -- no shard is optional), and merges by score. Fake
client-shaped stubs, not real gRPC/server_bin: this test is about the
merge/wait logic, not the network -- see Task 5 for the real end-to-end
integration test."""

from __future__ import annotations

import time

import pytest

from server.broker import Broker


class _FakeResult:
    def __init__(self, docid: int, score: float) -> None:
        self.docid = docid
        self.score = score


class _FakeResponse:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.results = results


class _FakeClient:
    def __init__(self, results: list[_FakeResult], delay_s: float = 0.0) -> None:
        self._results = results
        self._delay_s = delay_s
        self.closed = False

    def dispatch(self, query: str, k: int = 10, timeout_s: float = 30.0):
        time.sleep(self._delay_s)
        return _FakeResponse(self._results[:k])

    def close(self) -> None:
        self.closed = True


def test_dispatch_merges_and_sorts_by_score_descending():
    client_a = _FakeClient([_FakeResult(1, 0.5), _FakeResult(2, 0.9)])
    client_b = _FakeClient([_FakeResult(3, 0.7)])
    broker = Broker([client_a, client_b])

    response = broker.dispatch("q", k=10)

    assert [r.docid for r in response.results] == [2, 3, 1]
    assert [r.score for r in response.results] == [0.9, 0.7, 0.5]
    broker.close()


def test_dispatch_truncates_to_k():
    client_a = _FakeClient([_FakeResult(1, 0.1), _FakeResult(2, 0.2)])
    client_b = _FakeClient([_FakeResult(3, 0.3), _FakeResult(4, 0.4)])
    broker = Broker([client_a, client_b])

    response = broker.dispatch("q", k=2)

    assert [r.docid for r in response.results] == [4, 3]
    broker.close()


def test_dispatch_waits_for_every_shard_even_a_slow_one():
    fast = _FakeClient([_FakeResult(1, 1.0)])
    slow = _FakeClient([_FakeResult(2, 2.0)], delay_s=0.2)
    broker = Broker([fast, slow])

    response = broker.dispatch("q", k=10)

    assert {r.docid for r in response.results} == {1, 2}
    broker.close()


def test_close_closes_every_client():
    client_a = _FakeClient([])
    client_b = _FakeClient([])
    broker = Broker([client_a, client_b])
    broker.close()
    assert client_a.closed
    assert client_b.closed


def test_constructor_rejects_empty_client_list():
    with pytest.raises(ValueError):
        Broker([])
