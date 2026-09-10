"""Context manager for N ServerProcess instances -- one per shard index
directory, all started concurrently so N startups don't serialize. Used
both for a plain N-shard cluster (tail_latency.py) and, called twice at
disjoint port ranges over the *same* index directories, for a
primary/replica pair (hedging.py) -- see design spec §5.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from server.process import ServerProcess


class ShardCluster:
    def __init__(self, index_dirs: list[Path], base_port: int, **server_kwargs) -> None:
        self._index_dirs = index_dirs
        self._base_port = base_port
        self._server_kwargs = server_kwargs
        self._servers: list[ServerProcess] = []

    @property
    def addresses(self) -> list[str]:
        return [s.address for s in self._servers]

    def __enter__(self) -> "ShardCluster":
        servers = [
            ServerProcess(index_dir, port=self._base_port + i, **self._server_kwargs)
            for i, index_dir in enumerate(self._index_dirs)
        ]
        started: list[ServerProcess] = []
        failure: Exception | None = None
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            futures = [pool.submit(s.__enter__) for s in servers]
            for future in futures:
                try:
                    started.append(future.result())
                except Exception as exc:  # noqa: BLE001 - still must tear down the rest below
                    failure = exc
        if failure is not None:
            # A partial cluster is not a usable cluster: tear down whatever
            # did start before propagating, so a failed __enter__ never
            # leaks running processes.
            for server in started:
                server.__exit__(None, None, None)
            raise failure
        self._servers = started
        return self

    def __exit__(self, *exc_info) -> None:
        with ThreadPoolExecutor(max_workers=max(len(self._servers), 1)) as pool:
            list(pool.map(lambda s: s.__exit__(*exc_info), self._servers))
        self._servers = []
