"""ONNX export/quantization and the batching/queue harness are exercised
here on the real (tiny) cross-encoder model, fp32-only, CPU-only -- slow
compared to a GPU but correct, and small enough to run in a normal test
suite. The full fp16/INT8/GPU run happens on Kaggle (Task 8); this test
validates the same code path at a scale that fits in CI."""

from __future__ import annotations

import math
from pathlib import Path

import onnx
import pytest

from rank.crossencoder_harness import (
    CrossEncoderSession,
    convert_to_fp16,
    export_onnx,
    quantize_int8,
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


def test_session_exposes_token_type_ids_input(fp32_onnx_path: Path):
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session._session.get_inputs()]
    assert "token_type_ids" in input_names


def test_session_scores_relevant_pair_above_floor(fp32_onnx_path: Path):
    # Value-pinned regression test for the token_type_ids omission: verified
    # by hand against real cross-encoder scores on this exact pair --
    # broken (token_type_ids omitted) scores it -0.4496, correct scores it
    # +7.4720. 5.0 separates cleanly from both. isinstance(score, float),
    # which is all the older smoke test checked, passes either way (and
    # would pass on NaN too), which is why this bug survived to review.
    session = CrossEncoderSession(fp32_onnx_path, MODEL_NAME, providers=["CPUExecutionProvider"])
    scores = session.score(
        [("what is the capital of france",
          "paris is the capital and most populous city of france")]
    )
    assert scores[0] > 5.0


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_converted_models_keep_three_inputs_and_score_finitely(
    fp32_onnx_path: Path, tmp_path: Path, precision: str
):
    # Neither conversion was covered by any test, and both rewrite the graph
    # the token_type_ids fix just changed: float16 conversion must leave the
    # *integer* inputs (input_ids/attention_mask/token_type_ids) alone, and
    # dynamic int8 quantization must not drop the newly-added third input. A
    # dropped or retyped token_type_ids would resurface exactly that bug in
    # the precision half of this phase, where it is hardest to notice.
    converted = tmp_path / f"model_{precision}.onnx"
    (convert_to_fp16 if precision == "fp16" else quantize_int8)(fp32_onnx_path, converted)

    graph_inputs = onnx.load(str(converted)).graph.input
    assert [i.name for i in graph_inputs] == ["input_ids", "attention_mask", "token_type_ids"]
    assert all(i.type.tensor_type.elem_type == onnx.TensorProto.INT64 for i in graph_inputs)

    session = CrossEncoderSession(converted, MODEL_NAME, providers=["CPUExecutionProvider"])
    score = session.score(
        [("what is the capital of france",
          "paris is the capital and most populous city of france")]
    )[0]
    assert math.isfinite(score)
    assert score > 5.0


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
