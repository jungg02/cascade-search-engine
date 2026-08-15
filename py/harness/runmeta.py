"""Provenance attached to every recorded benchmark result.

Per the plan's instrumentation spec: a number without its config is not a result.
Anything that can move a latency figure — hardware, thermal state, git SHA — is
captured here so a committed JSON file can be re-read a month later and trusted.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_metadata() -> dict:
    dirty = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty != "unknown" else None,
        "hardware": {
            "model": _sysctl("hw.model"),
            "cpu": _sysctl("machdep.cpu.brand_string"),
            "cores": _sysctl("hw.ncpu"),
            "memory_bytes": _sysctl("hw.memsize"),
            "platform": platform.platform(),
        },
        "python": platform.python_version(),
    }
