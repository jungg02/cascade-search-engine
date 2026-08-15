"""TREC run file I/O.

Format: `qid Q0 docid rank score tag`, one line per retrieved document, ordered
by descending score. This is what trec_eval consumes and what Anserini emits, so
every engine in this project reports through it.
"""

from __future__ import annotations

from pathlib import Path

Run = dict[str, dict[str, float]]


def write_run(run: Run, path: str | Path, tag: str, depth: int | None = None) -> None:
    """Write a run, truncated to `depth` documents per query if given."""
    with open(path, "w") as out:
        for qid in run:
            ranked = sorted(run[qid].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            if depth is not None:
                ranked = ranked[:depth]
            for rank, (docid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")


def read_run(path: str | Path) -> Run:
    run: Run = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            qid, _, docid, _, score, *_ = fields
            run.setdefault(qid, {})[docid] = float(score)
    return run
