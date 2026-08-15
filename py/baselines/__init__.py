"""BM25 baselines. Each engine appends what it ran to runs/manifest.json so the
report generator never has to guess which config produced which run file."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "runs" / "manifest.json"


def append_manifest(entry: dict) -> None:
    """Record one completed run, replacing any earlier entry for the same key."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    key = (entry["engine"], entry["query_set"], entry.get("k1"), entry.get("b"))
    entries = [
        e for e in entries
        if (e["engine"], e["query_set"], e.get("k1"), e.get("b")) != key
    ]
    entries.append(entry)
    MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
