from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import HTTPProxyBackend

logger = logging.getLogger("model_router.backends.vllm")

_VLLM_DEFAULT = "http://127.0.0.1:8000"


class VLLMBackend(HTTPProxyBackend):
    """Routes to a running vLLM server (OpenAI-compatible).

    vLLM starts with a fixed model; no hot-swapping. The router treats
    all vllm-backed models as always ready. Use separate vLLM instances
    if you need multiple models simultaneously.
    """

    def __init__(self, base_url: str = _VLLM_DEFAULT) -> None:
        super().__init__(base_url)
        self._validated = False

    async def ensure(self, model_key: str, model_config: Any) -> None:
        if self._active_model == model_key:
            return
        if not self._validated:
            try:
                resp = await self._client.get("/v1/models")
                resp.raise_for_status()
                self._validated = True
                logger.info("vllm_connected base_url=%s", self._client.base_url)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"vLLM server not reachable at {self._client.base_url}: {exc}"
                ) from exc
        self._active_model = model_key
        self._load_count += 1
