from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .base import HTTPProxyBackend, RoutedResponse

logger = logging.getLogger("model_router.backends.ollama")

_OLLAMA_DEFAULT = "http://127.0.0.1:11434"


class OllamaBackend(HTTPProxyBackend):
    """Routes to an Ollama server. Auto-pulls models not yet downloaded.

    Pins one model at a time by unloading the previous via keep_alive=0.
    Configure Ollama with OLLAMA_MAX_LOADED_MODELS=1 for hard enforcement.
    """

    def __init__(self, base_url: str = _OLLAMA_DEFAULT) -> None:
        super().__init__(base_url)
        self._api_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10, read=None, write=None, pool=10),
        )

    async def close(self) -> None:
        await self._client.aclose()
        await self._api_client.aclose()

    async def _is_local(self, model_id: str) -> bool:
        try:
            response = await self._api_client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name") == model_id or m.get("model") == model_id for m in models)
        except httpx.HTTPError:
            return False

    async def _pull(self, model_id: str) -> None:
        logger.info("ollama_pull model=%s (this may take a while)", model_id)
        async with self._api_client.stream(
            "POST", "/api/pull", json={"name": model_id, "stream": True}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    logger.debug("pull_progress %s", line[:120])
        logger.info("ollama_pull_done model=%s", model_id)

    async def _unload_previous(self, model_id: str) -> None:
        """Ask Ollama to evict the model from memory (keep_alive=0)."""
        try:
            await self._api_client.post(
                "/api/generate",
                json={"model": model_id, "keep_alive": 0},
                timeout=5,
            )
        except httpx.HTTPError:
            pass

    async def ensure(self, model_key: str, model_config: Any) -> None:
        model_id = model_config.model_id or model_config.public_id
        if self._active_model == model_key:
            return
        previous_model_id = None
        if self._active_model:
            previous_model_id = model_config  # can't easily get previous config here
        started = time.monotonic()
        if not await self._is_local(model_id):
            await self._pull(model_id)
        if self._active_model:
            await self._unload_previous(model_id)
        previous = self._active_model
        self._active_model = model_key
        self._load_count += 1
        if previous and previous != model_key:
            self._switch_count += 1
        self._last_load_seconds = time.monotonic() - started
        logger.info("ollama_ready model=%s latency_seconds=%.3f", model_key, self._last_load_seconds)
