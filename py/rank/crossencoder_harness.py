"""Cross-encoder serving harness: ONNX export/precision variants, an
in-process async dynamic-batching harness (no real network hop -- an
open-loop request generator feeds rank.batcher.DynamicBatcher directly,
matching the design spec's choice to keep this runnable inside one Kaggle
notebook), and queue discipline (load-shedding vs. unbounded).

INT8 is dynamic quantization (onnxruntime.quantization.quantize_dynamic) --
no calibration dataset needed, simpler than static/calibrated quantization,
at a real accuracy-vs-effort tradeoff this project measures rather than
assumes. fp16 is a separate converted ONNX graph
(onnxconverter_common.float16), not a runtime cast. Whether
CUDAExecutionProvider actually accelerates the dynamically-quantized INT8
ops is an open, measured question, not an assumption -- if it silently falls
back to slower ops, that IS the finding to report.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from harness.histogram import LatencyRecorder


def export_onnx(model_name: str, onnx_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    dummy = tokenizer("dummy query", "dummy passage", return_tensors="pt")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # token_type_ids is not optional for a BERT-family cross-encoder: it is
    # the only input carrying the query/passage segment boundary (0 for query
    # tokens, 1 for passage tokens). Exporting a 2-input graph makes ORT run
    # the model with an all-zero segment embedding, which silently destroys
    # the model's discriminative power rather than failing -- measured on a
    # real pair, a relevant passage scores +7.4720 with token_type_ids and
    # -0.4496 without it.
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        # torch>=2.9 defaults to the dynamo/torch.export-based exporter, which
        # requires the optional `onnxscript` package (not a project dependency)
        # and takes `dynamic_shapes` instead of `dynamic_axes`. This module's
        # code was written for the legacy TorchScript-based exporter (it passes
        # `dynamic_axes`, which torch's own docs say "is used when dynamo is
        # False"), so pin that behavior explicitly rather than pull in a new
        # dependency for the new exporter.
        dynamo=False,
    )


def convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    model = onnx.load(str(fp32_path))
    fp16_model = float16.convert_float_to_float16(model)
    onnx.save(fp16_model, str(fp16_path))


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), weight_type=QuantType.QInt8)


DEFAULT_SCORE_CHUNK_SIZE = 32


class CrossEncoderSession:
    def __init__(self, onnx_path: Path, tokenizer_name: str, providers: list[str]) -> None:
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def active_providers(self) -> list[str]:
        """The execution providers ONNX Runtime actually initialized -- may
        silently differ from what the caller requested. A CUDA/cuDNN version
        mismatch makes ORT log a warning and fall back to CPU rather than
        raise, so a caller that only checks for an exception at session
        creation would never notice it ran the "GPU" experiment on CPU.
        Callers must record this in their result's provenance rather than
        assume the requested provider is the one that ran."""
        return self._session.get_providers()

    def score(
        self, pairs: list[tuple[str, str]], chunk_size: int = DEFAULT_SCORE_CHUNK_SIZE
    ) -> list[float]:
        # Chunked so a caller scoring an unbounded candidate pool (e.g. every
        # WAND candidate for a query, up to depth 1000) never hands a single
        # unbounded batch to the tokenizer/session in one call -- padding to
        # the batch's longest sequence times an unbounded batch size is an
        # uncontrolled memory spike, worse still if CUDA silently fell back
        # to CPU (see active_providers above) and the "GPU" run is actually
        # consuming system RAM instead of VRAM.
        scores: list[float] = []
        for start in range(0, len(pairs), chunk_size):
            chunk = pairs[start : start + chunk_size]
            queries, passages = zip(*chunk)
            encoded = self._tokenizer(
                list(queries), list(passages), padding=True, truncation=True, return_tensors="np"
            )
            outputs = self._session.run(
                ["logits"],
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    # A two-sequence tokenizer call already returns this; the
                    # bug this replaces was discarding it, leaving the segment
                    # embedding zeroed. See export_onnx's note.
                    "token_type_ids": encoded["token_type_ids"],
                },
            )
            scores.extend(float(x) for x in np.asarray(outputs[0]).reshape(-1))
        return scores


async def _run_open_loop(
    session,
    batcher_factory: Callable[[], "DynamicBatcher"],
    request_pairs: list[tuple[str, str]],
    duration_s: float,
    arrival_interval_s: float,
    on_shed: Callable[[], None] | None = None,
    max_queue_depth: int | None = None,
) -> LatencyRecorder:
    from rank.batcher import DynamicBatcher  # local import avoids a hard cycle at module load

    batcher: DynamicBatcher = batcher_factory()
    recorder = LatencyRecorder()
    pending: dict[int, float] = {}
    next_id = 0
    start = time.perf_counter()
    request_index = 0

    async def generator():
        nonlocal next_id, request_index
        while time.perf_counter() - start < duration_s:
            if max_queue_depth is not None and len(batcher) >= max_queue_depth:
                if on_shed is not None:
                    on_shed()
            else:
                pending[next_id] = time.perf_counter()
                batcher.add((next_id, request_pairs[request_index % len(request_pairs)]))
                next_id += 1
                request_index += 1
            await asyncio.sleep(arrival_interval_s)

    async def consumer():
        while time.perf_counter() - start < duration_s + batcher.max_wait_ms / 1000.0:
            if batcher.should_flush():
                batch = batcher.flush()
                ids = [item[0] for item in batch]
                pairs = [item[1] for item in batch]
                scores = session.score(pairs)
                assert len(scores) == len(pairs)
                now = time.perf_counter()
                for request_id in ids:
                    recorder.record((now - pending.pop(request_id)) * 1_000_000)
            await asyncio.sleep(0.001)

    await asyncio.gather(generator(), consumer())
    return recorder


def run_batching_sweep(
    session_factory: Callable[[], CrossEncoderSession],
    grid: list[tuple[int, float]],
    request_pairs: list[tuple[str, str]],
    duration_s: float = 2.0,
) -> list[dict]:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    points = []
    for max_batch_size, max_wait_ms in grid:
        recorder = asyncio.run(
            _run_open_loop(
                session,
                # DynamicBatcher's clock contract is milliseconds (its
                # max_wait_ms comparison assumes clock() ticks in ms) --
                # time.perf_counter() ticks in seconds, so it must be scaled
                # here or a "5ms" wait budget silently becomes 5 seconds.
                batcher_factory=lambda: DynamicBatcher(
                    max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
                ),
                request_pairs=request_pairs,
                duration_s=duration_s,
                arrival_interval_s=0.001,
            )
        )
        summary = recorder.summary()
        points.append(
            {
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "latency_us": summary,
                "throughput_qps": summary.get("count", 0) / duration_s,
            }
        )
    return points


def run_queue_discipline(
    session_factory: Callable[[], CrossEncoderSession],
    discipline: str,
    arrival_rate_qps: float,
    batcher_config: tuple[int, float],
    request_pairs: list[tuple[str, str]],
    duration_s: float = 2.0,
    max_queue_depth: int = 64,
) -> dict:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    max_batch_size, max_wait_ms = batcher_config
    shed_count = 0

    def on_shed():
        nonlocal shed_count
        shed_count += 1

    recorder = asyncio.run(
        _run_open_loop(
            session,
            # See run_batching_sweep's comment: DynamicBatcher expects a
            # millisecond clock, not seconds.
            batcher_factory=lambda: DynamicBatcher(
                max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
            ),
            request_pairs=request_pairs,
            duration_s=duration_s,
            arrival_interval_s=1.0 / arrival_rate_qps,
            on_shed=on_shed if discipline == "shed" else None,
            max_queue_depth=max_queue_depth if discipline == "shed" else None,
        )
    )
    return {
        "discipline": discipline,
        "arrival_rate_qps": arrival_rate_qps,
        "latency_us": recorder.summary(),
        "shed_count": shed_count,
    }
