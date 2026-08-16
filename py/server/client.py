"""Thin gRPC client wrapper around the cascade server's Search service.

The generated stubs (search_pb2_grpc.py) use a bare `import search_pb2`,
which only resolves if the generated/ directory itself is on sys.path — not
just `py/` per the project's usual PYTHONPATH convention. Every module that
needs the stubs inserts the path itself (idempotent — sys.path membership
doesn't duplicate meaningfully at this scale) rather than depending on
import order between modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_GENERATED_DIR = Path(__file__).resolve().parent / "generated"
if str(_GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATED_DIR))

import grpc

try:
    import search_pb2
    import search_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "generated proto stubs not found; run `python -m server.gen_proto` first"
    ) from exc


class SearchClient:
    """One gRPC channel + stub. dispatch() is the shape harness.loadgen wants:
    a Callable[[str], object]."""

    def __init__(self, address: str, timeout_s: float = 5.0) -> None:
        self._channel = grpc.insecure_channel(address)
        grpc.channel_ready_future(self._channel).result(timeout=timeout_s)
        self._stub = search_pb2_grpc.SearchStub(self._channel)

    def dispatch(self, query: str, k: int = 10, timeout_s: float = 30.0):
        # Server-side Query() waits on future.wait() with no exception safety
        # around it (a parked finding from Task 5) — an unbounded client-side
        # wait would let one hung worker exception hang this call forever,
        # and with it the whole load generator's dispatch thread pool.
        # Generous but finite bounds the blast radius without cutting off
        # legitimately slow requests.
        return self._stub.Query(search_pb2.SearchRequest(query=query, k=k), timeout=timeout_s)

    def close(self) -> None:
        self._channel.close()
