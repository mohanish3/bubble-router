from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parents[1]))

from model_router.config import load_config
from model_router.process_manager import ModelProcessManager


async def generate(
    client: httpx.AsyncClient, model: str, prompt: str
) -> dict[str, object]:
    started = time.perf_counter()
    first_token_at = None
    chunks = []
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 64,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            if first_token_at is None:
                first_token_at = time.perf_counter()
            event = json.loads(payload)
            chunks.append(event)
    ended = time.perf_counter()
    usage = next(
        (chunk.get("usage") for chunk in reversed(chunks) if chunk.get("usage")),
        {},
    )
    timings = next(
        (chunk.get("timings") for chunk in reversed(chunks) if chunk.get("timings")),
        {},
    )
    completion_tokens = int((usage or {}).get("completion_tokens", 0))
    generation_seconds = (
        ended - first_token_at if first_token_at is not None else ended - started
    )
    return {
        "request_seconds": round(ended - started, 3),
        "time_to_first_token_seconds": round(
            (first_token_at or ended) - started, 3
        ),
        "completion_tokens": completion_tokens,
        "generation_seconds": round(generation_seconds, 3),
        "tokens_per_second": round(
            completion_tokens / generation_seconds, 2
        ) if completion_tokens and generation_seconds else None,
        "server_tokens_per_second": (
            round(float(timings["predicted_per_second"]), 2)
            if timings and timings.get("predicted_per_second") is not None
            else None
        ),
    }


async def main() -> None:
    config = load_config()
    manager = ModelProcessManager(config)
    client = httpx.AsyncClient(
        base_url=config.server["upstream_base_url"],
        timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
    )
    report: dict[str, object] = {"runs": []}
    try:
        for model_key, prompt in [
            ("gemma", "In one sentence, explain why the sky is blue."),
            ("omnicoder", "Write a Python function add(a, b) that returns their sum."),
        ]:
            load_started = time.perf_counter()
            await manager.ensure(model_key)
            load_seconds = time.perf_counter() - load_started
            generation = await generate(
                client, config.models[model_key].public_id, prompt
            )
            report["runs"].append({
                "model": model_key,
                "load_or_switch_seconds": round(load_seconds, 3),
                **generation,
            })
        report["switch_count"] = manager.switch_count
        report["load_count"] = manager.load_count
    finally:
        await client.aclose()
        await manager.close()

    output = Path(__file__).with_name("router_dry_run_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
