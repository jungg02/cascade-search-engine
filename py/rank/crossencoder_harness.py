"""Cross-encoder serving harness: ONNX export/precision variants, an
in-process dynamic-batching harness (no real network hop -- harness.loadgen's
open-loop request generator feeds rank.batcher.DynamicBatcher directly,
matching the design spec's choice to keep this runnable inside one Kaggle
notebook), and queue discipline (load-shedding vs. unbounded).

The batching harness is thread-based, not asyncio-based, and that is load-
bearing: session.score() is a blocking call with no await in it, so an
asyncio generator sharing its event loop could never advance while a batch
was being scored -- arrivals silently throttled to the scorer's own rate and
the generator became closed-loop in everything but name (measured: a nominal
1992 qps target achieved ~719 qps, and no run ever shed a single request).
Scoring now runs on a dedicated flusher thread while harness.loadgen.
run_open_loop drives arrivals off absolute, precomputed deadlines.

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

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from harness.loadgen import LoadResult, run_open_loop


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


class _QueueFullError(Exception):
    """Raised by dispatch() under the shed discipline when the batcher is at
    capacity. run_open_loop's own dispatch wrapper catches any exception and
    counts it as a LoadResult error rather than aborting the run, which is
    exactly what "shed" should mean -- but sheds are counted explicitly here
    too (see _BatchedRun.shed_count), because result.errors would also absorb
    a genuine scoring failure and let a broken run masquerade as successful
    load-shedding."""


class _ScorerFailed(Exception):
    """Raised by dispatch() once the flusher thread has died, so in-flight and
    subsequent arrivals fail fast instead of blocking forever on an Event that
    nothing will ever set."""


@dataclass
class _BatchedRun:
    """One open-loop run's outcome: harness.loadgen's own LoadResult plus the
    shed count, which loadgen has no concept of."""

    result: LoadResult
    shed_count: int


def _run_batched_open_loop(
    session,
    batcher_factory: Callable[[], "DynamicBatcher"],
    request_pairs: list[tuple[str, str]],
    duration_s: float,
    qps: float,
    workers: int,
    max_queue_depth: int | None = None,
    warmup_requests: int = 0,
) -> _BatchedRun:
    """Drive `session` through one shared DynamicBatcher using harness.loadgen's
    real open-loop generator: arrivals run on their own schedule regardless of
    how long a batch takes to score, because session.score() runs on a
    dedicated flusher thread, not on the generator's critical path. Each
    dispatched request enqueues itself into the batcher and blocks on its own
    threading.Event until the flusher scores its batch and wakes it -- this is
    what lets run_open_loop's scheduled/start/done timestamps stay meaningful
    (queue_delay = start - scheduled genuinely reflects batcher/queue wait, not
    an artifact of a stalled event loop).

    Concurrency contract, since getting this wrong deadlocks rather than fails:
      * `lock` guards the batcher, the shed counter and the round-robin index,
        and nothing else. It is never held across session.score() -- the flush
        happens under the lock, the scoring does not.
      * A request's Event travels *inside* the batched item, so the flusher
        only ever wakes events it was handed. There is no shared id->event map
        for the two sides to race over.
      * If session.score() raises, the flusher records the exception, wakes
        every request it is holding, and keeps draining-and-waking until stop
        is set, so no dispatch thread can block forever. dispatch() then fails
        fast, run_open_loop returns, and the original exception is re-raised
        here rather than surfacing as a silent hang.
    """
    from rank.batcher import DynamicBatcher  # local import avoids a hard cycle at module load

    batcher: DynamicBatcher = batcher_factory()
    lock = threading.Lock()
    stop = threading.Event()
    failure: list[BaseException] = []
    counters = {"shed": 0, "next_index": 0}

    def flusher() -> None:
        while not stop.is_set():
            with lock:
                batch = batcher.flush() if batcher.should_flush() else None
            if not batch:
                time.sleep(0.0005)
                continue
            try:
                scores = session.score([pair for _, pair in batch])
                assert len(scores) == len(batch)
            except BaseException as exc:  # noqa: BLE001 -- re-raised by the caller
                failure.append(exc)
                for event, _ in batch:
                    event.set()
                # Keep releasing whatever arrives until the run is torn down:
                # dispatch() is failing fast by now, but requests already
                # enqueued (or enqueued in the race window) still need waking.
                while not stop.is_set():
                    with lock:
                        orphans = batcher.flush()
                    for event, _ in orphans:
                        event.set()
                    time.sleep(0.0005)
                return
            for event, _ in batch:
                event.set()

    flusher_thread = threading.Thread(target=flusher, daemon=True)
    flusher_thread.start()

    def dispatch(_query: str) -> None:
        # _query (the Zipf-sampled text run_open_loop hands us) is not used to
        # pick the pair -- request_pairs is round-robined independently under
        # the lock, since real pair identity for this synthetic benchmark
        # workload doesn't need to correlate with the sampler's popularity
        # draw, only arrival *timing* does.
        if failure:
            raise _ScorerFailed()
        event = threading.Event()
        with lock:
            if max_queue_depth is not None and len(batcher) >= max_queue_depth:
                counters["shed"] += 1
                raise _QueueFullError()
            pair = request_pairs[counters["next_index"] % len(request_pairs)]
            counters["next_index"] += 1
            batcher.add((event, pair))
        event.wait()
        if failure:
            raise _ScorerFailed()

    try:
        for _ in range(warmup_requests):
            dispatch("warmup")
        result = run_open_loop(
            dispatch=dispatch,
            queries=[q for q, _ in request_pairs],
            qps=qps,
            duration_s=duration_s,
            workers=workers,
        )
    finally:
        stop.set()
        flusher_thread.join(timeout=5.0)
    # Checked after the finally block (so it can't mask an in-flight
    # exception) and raised rather than ignored: a flusher that outlives its
    # run is a daemon thread still holding an ONNX Runtime session while the
    # interpreter tears down, which surfaces later as an unrelated-looking
    # native crash. Fail loudly here instead.
    if flusher_thread.is_alive():
        raise RuntimeError("flusher thread did not stop within 5s of the run ending")
    if failure:
        raise failure[0]
    return _BatchedRun(result=result, shed_count=counters["shed"])


# Requests offered per batching-sweep probe. The probe is a saturating burst:
# qps is set far above any config's plausible throughput, so what the offered
# rate actually buys is "every request is already waiting", and achieved_qps
# becomes that config's real sustained throughput. The *budget* has to be the
# bounded quantity rather than the rate, because run_open_loop precomputes the
# whole arrival list up front and its ThreadPoolExecutor drains every submitted
# request before returning -- a fire-hose bounded only by duration_s runs for
# offered_requests/capacity seconds (at 100k qps for 10s: hours), not for
# duration_s. Bounding the budget instead terminates in budget/capacity
# seconds, identically on this repo's CPU fixture and on a Kaggle T4, with no
# hardware-specific rate guess baked in. Consequence, stated where it can't be
# missed: under a saturating probe the sweep's p99 is drain-dominated --
# comparable across configs at equal budget, but not an absolute client-side
# latency. The queue-discipline experiment below, which offers a real specified
# arrival rate, is where latency/queue_delay are absolute numbers.
DEFAULT_PROBE_REQUESTS = 2000
_SATURATING_QPS = 100_000.0


def run_batching_sweep(
    session_factory: Callable[[], CrossEncoderSession],
    grid: list[tuple[int, float]],
    request_pairs: list[tuple[str, str]],
    probe_requests: int = DEFAULT_PROBE_REQUESTS,
    warmup_requests: int = 20,
) -> list[dict]:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    points = []
    for max_batch_size, max_wait_ms in grid:
        run = _run_batched_open_loop(
            session,
            # DynamicBatcher's clock contract is milliseconds (its max_wait_ms
            # comparison assumes clock() ticks in ms) -- time.perf_counter()
            # ticks in seconds, so it must be scaled here or a "5ms" wait
            # budget silently becomes 5 seconds.
            batcher_factory=lambda: DynamicBatcher(
                max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
            ),
            request_pairs=request_pairs,
            duration_s=probe_requests / _SATURATING_QPS,
            qps=_SATURATING_QPS,
            # Sized well above max_batch_size so the thread pool is never what
            # stops a batch from filling -- the batcher's own max_batch_size/
            # max_wait_ms must be the only thing shaping throughput.
            workers=max(max_batch_size * 8, 64),
            warmup_requests=warmup_requests,
        )
        summary = run.result.summary()
        points.append(
            {
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "latency_us": summary["latency"],
                "throughput_qps": run.result.achieved_qps,
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
    warmup_requests: int = 20,
) -> dict:
    from rank.batcher import DynamicBatcher

    session = session_factory()
    max_batch_size, max_wait_ms = batcher_config
    # Generously sized so the thread pool is never the binding constraint --
    # only the batcher's max_queue_depth (shed) or real scorer throughput
    # (unbounded) should be able to cause backlog/shedding.
    workers = max(max_queue_depth * 4, 256)

    run = _run_batched_open_loop(
        session,
        # See run_batching_sweep's comment: DynamicBatcher expects a
        # millisecond clock, not seconds.
        batcher_factory=lambda: DynamicBatcher(
            max_batch_size, max_wait_ms, lambda: time.perf_counter() * 1000.0
        ),
        request_pairs=request_pairs,
        duration_s=duration_s,
        qps=arrival_rate_qps,
        workers=workers,
        max_queue_depth=max_queue_depth if discipline == "shed" else None,
        warmup_requests=warmup_requests,
    )
    summary = run.result.summary()
    return {
        "discipline": discipline,
        "arrival_rate_qps": arrival_rate_qps,
        "achieved_qps": run.result.achieved_qps,
        "latency_us": summary["latency"],
        "queue_delay_us": summary["queue_delay"],
        "shed_count": run.shed_count,
    }
