"""Starts/stops the cascade C++ server binary as a subprocess — used by both
the integration test and the experiment scripts, so there's exactly one
place that knows the binary's flag names and readiness-wait logic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO_ROOT / "cpp" / "build" / "server_bin"


class ServerProcess:
    """Context manager: `with ServerProcess(index_dir) as server: ...` — use
    server.address to connect."""

    def __init__(
        self,
        index_dir: Path,
        port: int = 50051,
        workers: int = 4,
        queue_depth: int = 64,
        cache_capacity: int = 200,
        algorithm: str = "wand",
        ready_timeout_s: float = 30.0,
    ) -> None:
        if not SERVER_BIN.exists():
            raise FileNotFoundError(f"{SERVER_BIN} not built; run `make -C cpp all` first")
        self._address = f"127.0.0.1:{port}"
        self._args = [
            str(SERVER_BIN),
            str(index_dir),
            f"--port={port}",
            f"--workers={workers}",
            f"--queue-depth={queue_depth}",
            f"--cache-capacity={cache_capacity}",
            f"--algorithm={algorithm}",
        ]
        self._ready_timeout_s = ready_timeout_s
        self._process: subprocess.Popen | None = None

    @property
    def address(self) -> str:
        return self._address

    def __enter__(self) -> "ServerProcess":
        self._process = subprocess.Popen(self._args)
        channel = grpc.insecure_channel(self._address)
        try:
            grpc.channel_ready_future(channel).result(timeout=self._ready_timeout_s)
        except grpc.FutureTimeoutError:
            self.__exit__(None, None, None)
            raise TimeoutError(
                f"server did not become ready within {self._ready_timeout_s}s "
                f"(args: {self._args})"
            )
        finally:
            channel.close()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
