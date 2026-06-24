from pathlib import Path

import httpx
import pytest

from model_router.app import create_app
from model_router.config import load_config
from model_router.service import RouterService
from test_service import FakeBackend

FIXTURE = Path(__file__).parent / "fixtures" / "model-router.test.json"
AUTH = {"Authorization": "Bearer local-model-router"}


class SSEStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_auth_models_and_malformed_request(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_API_KEY", "local-model-router")

    async def upstream(_):
        return httpx.Response(200, json={"choices": []})

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="http://upstream"
    )
    config = load_config(FIXTURE)
    service = RouterService(config, backend=FakeBackend(upstream_client))
    await service.start(warm=False)
    app = create_app(config, service=service, warm=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/v1/models")).status_code == 401
        models = await client.get("/v1/models", headers=AUTH)
        assert models.status_code == 200
        assert any(item["id"] == "auto" for item in models.json()["data"])
        malformed = await client.post(
            "/v1/chat/completions",
            headers={**AUTH, "Content-Type": "application/json"},
            content=b"{bad",
        )
        assert malformed.status_code == 400
        invalid = await client.post(
            "/v1/chat/completions", headers=AUTH, json={"model": "auto"}
        )
        assert invalid.status_code == 400
    await service.close()
    await upstream_client.aclose()


@pytest.mark.asyncio
async def test_empty_api_key_disables_auth(monkeypatch):
    monkeypatch.delenv("MODEL_ROUTER_API_KEY", raising=False)

    async def upstream(_):
        return httpx.Response(200, json={"choices": []})

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="http://upstream"
    )
    config = load_config(FIXTURE)
    service = RouterService(config, backend=FakeBackend(upstream_client))
    await service.start(warm=False)
    app = create_app(config, service=service, warm=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/v1/models")).status_code == 200
    await service.close()
    await upstream_client.aclose()


@pytest.mark.asyncio
async def test_sse_passthrough():
    async def upstream(_):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=SSEStream(),
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="http://upstream"
    )
    config = load_config(FIXTURE)
    service = RouterService(config, backend=FakeBackend(upstream_client))
    await service.start(warm=False)
    app = create_app(config, service=service, warm=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "coding",
                "stream": True,
                "messages": [{"role": "user", "content": "code"}],
            },
        )
        assert response.status_code == 200
        assert "data: [DONE]" in response.text
    await service.close()
    await upstream_client.aclose()
