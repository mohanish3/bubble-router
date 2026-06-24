from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder


PAIRS = [
    (
        "Implement a Python LRU cache and add regression tests.",
        "This task is general reasoning, ordinary questions, planning, or tool use.",
    ),
    (
        "Implement a Python LRU cache and add regression tests.",
        "This task is complex multi-step reasoning, difficult analysis, proofs, or deep research.",
    ),
    (
        "Implement a Python LRU cache and add regression tests.",
        "This task is software engineering, coding, debugging, code review, or repository work.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = CrossEncoder(
        "cross-encoder/nli-MiniLM2-L6-H768",
        device=args.device,
        activation_fn=None,
        model_kwargs={"torch_dtype": "float16"} if args.device == "cuda" else None,
    )
    load_seconds = time.perf_counter() - load_started

    for _ in range(5):
        model.predict(PAIRS, apply_softmax=True, show_progress_bar=False)
    if args.device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        model.predict(PAIRS, apply_softmax=True, show_progress_bar=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)

    result = {
        "device": args.device,
        "iterations": args.iterations,
        "pairs_per_request": len(PAIRS),
        "load_seconds": round(load_seconds, 4),
        "latency_ms_mean": round(statistics.mean(latencies), 3),
        "latency_ms_p50": round(statistics.median(latencies), 3),
        "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "requests_per_second": round(1000 / statistics.mean(latencies), 2),
    }
    if args.device == "cuda":
        result["peak_cuda_memory_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 2
        )
        result["reserved_cuda_memory_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024, 2
        )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
