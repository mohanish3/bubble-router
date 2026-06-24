from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import HTTPProxyBackend

logger = logging.getLogger("model_router.backends.lmstudio")

_LMSTUDIO_DEFAULT = "http://127.0.0.1:1234"


class LMStudioBackend(HTTPProxyBackend):
    """Routes to LM Studio's local server (OpenAI-compatible).

    Model must be loaded manually in the LM Studio UI before routing.
    The router validates connectivity on first use and warns if the
    expected model is not active.
    """

    def __init__(self, base_url: str = _LMSTUDIO_DEFAULT) -> None:
        super().__init__(base_url)

    async def ensure(self, model_key: str, model_config: Any) -> None:
        if self._active_model == model_key:
            return
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            available = [m.get("id") for m in resp.json().get("data", [])]
            expected = model_config.model_id or model_config.public_id
            if expected and expected not in available:
                logger.warning(
                    "lmstudio_model_mismatch expected=%s available=%s — "
                    "load the model in LM Studio first",
                    expected, available,
                )
            logger.info("lmstudio_connected model=%s", model_key)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"LM Studio not reachable at {self._client.base_url}: {exc}"
            ) from exc
        self._active_model = model_key
        self._load_count += 1
