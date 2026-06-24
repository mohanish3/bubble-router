import asyncio
import json
from pathlib import Path

import httpx
import pytest

from model_router.backends.base import Backend, RoutedResponse, _http_forward
from model_router.config import load_config
from model_router.service import RouterService

FIXTURE = Path(__file__).parent / "fixtures" / "model-router.test.json"


class FakeBackend(Backend):
    """Test double: tracks ensure() calls, forwards via a mock httpx client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._active_model: str | None = None
        self._load_count = 0
        self._switch_count = 0
        self._last_load_seconds = 0.01
        self.ensure_calls: list[str] = []

    @property
    def active_model(self) -> str | None:
        return self._active_model

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def last_load_seconds(self) -> float:
        return self._last_load_seconds

    async def ensure(self, model_key: str, model_config) -> None:
        self.ensure_calls.append(model_key)
        if self._active_model != model_key:
            if self._active_model is not None:
                self._switch_count += 1
            self._active_model = model_key
            self._load_count += 1

    async def forward(self, payload: dict, wants_stream: bool) -> RoutedResponse:
        return await _http_forward(self._client, payload, wants_stream)

    async def stop(self) -> None:
        self._active_model = None

    def is_alive(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_one_active_inference_and_model_reuse():
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "model": payload["model"],
                "usage": {"total_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream")
    fake = FakeBackend(client)
    service = RouterService(load_config(FIXTURE), backend=fake)
    await service.start(warm=False)
    try:
        jobs = [
            await service.submit(
                {"model": model, "messages": [{"role": "user", "content": "x"}]}, {}
            )
            for model in ("general", "general", "coding")
        ]
        for job in jobs:
            response = await asyncio.wait_for(job.result, 1)
            assert response.status_code == 200
            job.finished.set()
        await asyncio.sleep(0.05)
        assert peak == 1
        assert fake.ensure_calls[:2] == ["gemma", "gemma"]
        assert fake.load_count == 2
    finally:
        await service.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_direct_override_preserves_openai_response_fields():
    expected = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": "",
                "reasoning": "inspect",
                "tool_calls": [{"id": "1", "type": "function"}],
            },
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    async def handler(_):
        return httpx.Response(200, json=expected)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream")
    service = RouterService(load_config(FIXTURE), backend=FakeBackend(client))
    await service.start(warm=False)
    try:
        job = await service.submit(
            {"model": "complex-reasoning", "messages": [{"role": "user", "content": "x"}]},
            {},
        )
        response = await job.result
        assert json.loads(response.body) == expected
        job.finished.set()
    finally:
        await service.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_model_crash_response_restarts_and_retries_once():
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "model unavailable"})
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream")
    fake = FakeBackend(client)
    service = RouterService(load_config(FIXTURE), backend=fake)
    await service.start(warm=False)
    try:
        job = await service.submit(
            {"model": "general", "messages": [{"role": "user", "content": "x"}]}, {}
        )
        response = await job.result
        assert response.status_code == 200
        assert calls == 2
        assert fake.load_count == 2
        job.finished.set()
    finally:
        await service.close()
        await client.aclose()
