from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from typing import Any

import httpx

from .base import Backend, HTTPProxyBackend, RoutedResponse, _http_forward

logger = logging.getLogger("model_router.backends.llamacpp")


class LlamaCppBackend(HTTPProxyBackend):
    """Manages a llama-server child process, hot-swapping GGUF models on demand."""

    def __init__(self, config: Any) -> None:
        upstream_url = config.server["upstream_base_url"]
        super().__init__(upstream_url)
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._health_client = httpx.AsyncClient(
            base_url=upstream_url, timeout=httpx.Timeout(5)
        )

    def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def close(self) -> None:
        await self.stop()
        await self._client.aclose()
        await self._health_client.aclose()

    async def stop(self) -> None:
        if not self.process or self.process.returncode is not None:
            self.process = None
            self._active_model = None
            return
        logger.info("model_stop model=%s pid=%s", self._active_model, self.process.pid)
        if os.name == "nt":
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.process.terminate()
        try:
            await asyncio.wait_for(
                self.process.wait(), self.config.server["stop_timeout_seconds"]
            )
        except asyncio.TimeoutError:
            logger.warning("model_force_kill model=%s pid=%s", self._active_model, self.process.pid)
            self.process.kill()
            await self.process.wait()
        self.process = None
        self._active_model = None

    async def ensure(self, model_key: str, model_config: Any) -> None:
        if (
            self._active_model == model_key
            and self.process
            and self.process.returncode is None
        ):
            return
        previous = self._active_model
        await self.stop()
        started = time.monotonic()
        command = self.config.command(model_key)
        log_dir = self.config.root / "logs" / "model-router"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / f"{model_key}.out.log").open("ab")
        stderr = (log_dir / f"{model_key}.err.log").open("ab")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = await asyncio.create_subprocess_exec(
            *command, stdout=stdout, stderr=stderr, **kwargs
        )
        stdout.close()
        stderr.close()
        self._active_model = model_key
        self._load_count += 1
        if previous and previous != model_key:
            self._switch_count += 1
        deadline = time.monotonic() + self.config.server["ready_timeout_seconds"]
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited during startup with code {self.process.returncode}"
                )
            try:
                response = await self._health_client.get("/v1/models")
                if response.is_success:
                    self._last_load_seconds = time.monotonic() - started
                    logger.info(
                        "model_ready model=%s latency_seconds=%.3f",
                        model_key,
                        self._last_load_seconds,
                    )
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        await self.stop()
        raise TimeoutError(f"llama-server not ready for {model_key}")
