"""ONNX export/quantization and the batching/queue harness are exercised
here on the real (tiny) cross-encoder model, fp32-only, CPU-only -- slow
compared to a GPU but correct, and small enough to run in a normal test
suite. The full fp16/INT8/GPU run happens on Kaggle (Task 8); this test
validates the same code path at a scale that fits in CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rank.batcher import DynamicBatcher
from rank.crossencoder_harness import (
    CrossEncoderSession,
    export_onnx,
    run_batching_sweep,
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="module")
def fp32_onnx_path(tmp_path_factory) -> Path:
    onnx_dir = tmp_path_factory.mktemp("onnx")
    path = onnx_dir / "model_fp32.onnx"
    export_onnx(MODEL_NAME, path)
    return path


def test_export_onnx_produces_a_file(fp32_onnx_path: Path):
    assert fp32_onnx_path.exists()
    assert fp32_onnx_path.stat().st_size > 0


def test_session_scores_a_pair(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    scores = session.score([("what is python", "python is a programming language")])
    assert len(scores) == 1
    assert isinstance(scores[0], float)


def test_session_reports_active_providers(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    assert session.active_providers == ["CPUExecutionProvider"]


def test_score_chunks_batches_larger_than_chunk_size(fp32_onnx_path: Path):
    # A real (if small-scale) regression test for the OOM this chunking
    # fixes: scoring more pairs than chunk_size must still score every pair
    # (via multiple internal session.run() calls), producing the same count
    # and the same per-pair values a single unchunked call would.
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("what is python", f"passage number {i}") for i in range(10)]
    chunked = session.score(pairs, chunk_size=3)
    unchunked = session.score(pairs, chunk_size=len(pairs))
    assert len(chunked) == len(pairs)
    assert chunked == pytest.approx(unchunked)


def test_batching_sweep_tiny_grid(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    pairs = [("query text", f"passage number {i}") for i in range(6)]
    points = run_batching_sweep(
        session_factory=lambda: session,
        grid=[(1, 0.0), (4, 5.0)],
        request_pairs=pairs,
    )
    assert len(points) == 2
    for point in points:
        assert "max_batch_size" in point and "max_wait_ms" in point
        assert "latency_us" in point and "throughput_qps" in point
