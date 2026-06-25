from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import RouterConfig, load_config
from .service import RouterService

logger = logging.getLogger("model_router.app")


def openai_error(message: str, status: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "router_error", "code": code}},
    )


def create_app(
    config: RouterConfig | None = None,
    service: RouterService | None = None,
    warm: bool = True,
) -> FastAPI:
    config = config or load_config()
    service = service or RouterService(config)
    key = os.getenv(
        config.server["api_key_env"], config.server.get("default_api_key", "")
    )
    _metrics: dict = defaultdict(lambda: {"count": 0, "total_latency_s": 0.0})

    async def authorize(request: Request) -> None:
        if not key:
            return
        bearer = request.headers.get("authorization", "")
        supplied = bearer[7:] if bearer.lower().startswith("bearer ") else ""
        supplied = supplied or request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied, key):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start(warm=warm)
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="bubble-router", lifespan=lifespan)
    app.state.router_service = service

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return openai_error(str(exc.detail), exc.status_code, "authentication_error")

    @app.get("/health")
    async def health(_: None = Depends(authorize)) -> dict:
        pb = service.primary_backend
        return {
            "status": "ok" if service.worker_task and not service.worker_task.done() else "error",
            "active_model": pb.active_model,
            "upstream_alive": pb.is_alive(),
        }

    @app.get("/v1/models")
    async def models(_: None = Depends(authorize)) -> dict:
        aliases = [
            {"id": alias, "object": "model", "owned_by": "bubble-router"}
            for alias in config.aliases
        ]
        concrete = [
            {"id": model.public_id, "object": "model", "owned_by": model.backend}
            for model in config.models.values()
        ]
        return {"object": "list", "data": aliases + concrete}

    @app.get("/router/status")
    async def status(_: None = Depends(authorize)) -> dict:
        return service.status()

    @app.get("/metrics")
    async def metrics_endpoint(_: None = Depends(authorize)) -> dict:
        rows = []
        for model_key, m in _metrics.items():
            avg_lat = m["total_latency_s"] / m["count"] if m["count"] else 0.0
            rows.append({"model": model_key, "requests": m["count"], "avg_latency_s": round(avg_lat, 3)})
        return {"routing": rows, "queue_depths": service.scheduler.depths()}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, _: None = Depends(authorize)) -> Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return openai_error("Malformed JSON request", 400, "invalid_json")
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return openai_error("messages must be an array", 400, "invalid_request")
        req_meta = {
            k: v
            for k, v in {
                "agent_id": request.headers.get("x-bubble-agent-id"),
                "session_id": request.headers.get("x-bubble-session-id"),
            }.items()
            if v
        }
        try:
            job = await service.submit(payload, req_meta)
        except KeyError:
            return openai_error(
                f"Unknown model: {payload.get('model')}", 400, "model_not_found"
            )

        t0 = time.monotonic()
        while job.result and not job.result.done():
            if await request.is_disconnected():
                await service.scheduler.cancel(job)
                return Response(status_code=499)
            await asyncio.sleep(0.05)
        try:
            routed = await job.result
        except asyncio.CancelledError:
            return Response(status_code=499)
        except Exception as exc:
            job.finished.set()
            return openai_error(str(exc), 502, "upstream_error")

        elapsed = time.monotonic() - t0
        router_model = job.classification_model or job.model
        top_score = (
            max(job.classification_scores.values()) if job.classification_scores else 1.0
        )
        _metrics[router_model]["count"] += 1
        _metrics[router_model]["total_latency_s"] += elapsed
        extra_headers = {
            "X-Router-Model": router_model,
            "X-Router-Confidence": f"{top_score:.3f}",
        }

        if routed.stream:
            async def stream_body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in routed.stream:
                        if await request.is_disconnected():
                            break
                        yield chunk
                finally:
                    await routed.stream.aclose()
                    job.finished.set()

            return StreamingResponse(
                stream_body(),
                status_code=routed.status_code,
                headers={**dict(routed.headers), **extra_headers},
                media_type=routed.headers.get("content-type", "text/event-stream"),
            )
        job.finished.set()
        return Response(
            content=routed.body,
            status_code=routed.status_code,
            headers={**dict(routed.headers), **extra_headers},
            media_type=routed.headers.get("content-type", "application/json"),
        )

    return app
