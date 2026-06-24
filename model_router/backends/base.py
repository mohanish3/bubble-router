from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


@dataclass
class RoutedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes | None = None
    stream: AsyncIterator[bytes] | None = None


async def _http_forward(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    wants_stream: bool,
) -> RoutedResponse:
    request = client.build_request("POST", "/v1/chat/completions", json=payload)
    response = await client.send(request, stream=wants_stream)
    headers = {
        key: value for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP
    }
    if not wants_stream:
        await response.aread()
        if response.status_code >= 500:
            raise RuntimeError(f"upstream {response.status_code}: {response.content[:200]!r}")
        return RoutedResponse(response.status_code, headers, body=response.content)
    if response.status_code >= 500:
        body = await response.aread()
        await response.aclose()
        raise RuntimeError(f"upstream {response.status_code}: {body[:200]!r}")
    iterator = response.aiter_raw()
    try:
        first = await anext(iterator)
    except StopAsyncIteration:
        first = b""
    except Exception:
        await response.aclose()
        raise

    async def stream() -> AsyncIterator[bytes]:
        try:
            if first:
                yield first
            async for chunk in iterator:
                yield chunk
        finally:
            await response.aclose()

    return RoutedResponse(response.status_code, headers, stream=stream())


class Backend(ABC):
    @property
    @abstractmethod
    def active_model(self) -> str | None: ...

    @property
    def load_count(self) -> int:
        return 0

    @property
    def switch_count(self) -> int:
        return 0

    @property
    def last_load_seconds(self) -> float:
        return 0.0

    @abstractmethod
    async def ensure(self, model_key: str, model_config: Any) -> None: ...

    @abstractmethod
    async def forward(self, payload: dict[str, Any], wants_stream: bool) -> RoutedResponse: ...

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {
            "active_model": self.active_model,
            "load_count": self.load_count,
            "switch_count": self.switch_count,
            "last_load_seconds": round(self.last_load_seconds, 3),
        }


class HTTPProxyBackend(Backend):
    """Base for backends that proxy to an OpenAI-compatible HTTP server."""

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
        )
        self._active_model: str | None = None
        self._load_count = 0
        self._switch_count = 0
        self._last_load_seconds = 0.0

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

    async def close(self) -> None:
        await self._client.aclose()

    async def forward(self, payload: dict[str, Any], wants_stream: bool) -> RoutedResponse:
        return await _http_forward(self._client, payload, wants_stream)
