from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .base import Backend, RoutedResponse, _http_forward

logger = logging.getLogger("model_router.backends.api")

_PROVIDER_URLS = {
    "claude":  "https://api.anthropic.com/v1",
    "openai":  "https://api.openai.com/v1",
    "gemini":  "https://generativelanguage.googleapis.com/v1beta/openai",
}

_DEFAULT_KEY_ENVS = {
    "claude":  "ANTHROPIC_API_KEY",
    "openai":  "OPENAI_API_KEY",
    "gemini":  "GEMINI_API_KEY",
}

_CLAUDE_EXTRA_HEADERS = {
    "anthropic-version": "2023-06-01",
}


class APIBackend(Backend):
    """Stateless HTTP backend for cloud LLM APIs (Claude, OpenAI, Gemini).

    No process management or model switching — each request is a direct API call.
    Set the relevant env var before starting: ANTHROPIC_API_KEY, OPENAI_API_KEY,
    or GEMINI_API_KEY.
    """

    def __init__(self, provider: str, model_config: Any) -> None:
        self._provider = provider
        self._model_config = model_config
        base_url = model_config.base_url or _PROVIDER_URLS.get(provider, "")
        if not base_url:
            raise ValueError(f"Unknown API provider: {provider!r}. Set base_url in config.")
        key_env = model_config.api_key_env or _DEFAULT_KEY_ENVS.get(provider, "")
        api_key = os.getenv(key_env, "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if provider == "claude":
            headers.update(_CLAUDE_EXTRA_HEADERS)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
        )
        self._active: str | None = None

    @property
    def active_model(self) -> str | None:
        return self._active

    async def ensure(self, model_key: str, model_config: Any) -> None:
        self._active = model_key

    async def forward(self, payload: dict[str, Any], wants_stream: bool) -> RoutedResponse:
        return await _http_forward(self._client, payload, wants_stream)

    async def close(self) -> None:
        await self._client.aclose()

    def is_alive(self) -> bool:
        return True
