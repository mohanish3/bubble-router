from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import Any

import httpx

from .backends import Backend, LlamaCppBackend, RoutedResponse, create_backends
from .classifier import Classification, TaskClassifier
from .config import RouterConfig
from .scheduler import Job, Scheduler

logger = logging.getLogger("model_router.service")


class RouterService:
    def __init__(
        self,
        config: RouterConfig,
        classifier: TaskClassifier | None = None,
        backend: Backend | dict[str, Backend] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier or TaskClassifier(
            config.classifier["model"],
            config.classifier["labels"],
            float(config.classifier["confidence_threshold"]),
        )
        if backend is None:
            self._backends: dict[str, Backend] = create_backends(config)
        elif isinstance(backend, dict):
            self._backends = backend
        else:
            self._backends = {key: backend for key in config.models}
        self.scheduler = Scheduler(
            list(config.models),
            int(config.server["max_loaded_jobs"]),
            float(config.server["max_loaded_seconds"]),
        )
        self._client = client  # optional: only used by _llm_classify
        self._owns_client = client is not None
        self.worker_task: asyncio.Task[None] | None = None
        self.active_job: Job | None = None
        self.completed: Counter[str] = Counter()
        self.classifications: Counter[str] = Counter()
        self.queue_wait_seconds: list[float] = []
        self.inference_seconds: list[float] = []

    @property
    def primary_backend(self) -> Backend:
        """Returns the LlamaCpp backend if one exists, otherwise the first backend."""
        for b in self._backends.values():
            if isinstance(b, LlamaCppBackend):
                return b
        return next(iter(self._backends.values()))

    async def start(self, warm: bool = True) -> None:
        if not self.worker_task:
            self.worker_task = asyncio.create_task(self._worker(), name="model-router-worker")
        if warm:
            warm_key = self.config.server["warm_model"]
            await self._backends[warm_key].ensure(warm_key, self.config.models[warm_key])

    async def close(self) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
        for queue in self.scheduler.queues.values():
            while queue:
                job = queue.popleft()
                if job.result and not job.result.done():
                    job.result.set_exception(RuntimeError("router shutting down"))
        closed: set[int] = set()
        for b in self._backends.values():
            if id(b) not in closed:
                await b.close()
                closed.add(id(b))
        if self._owns_client and self._client:
            await self._client.aclose()

    async def classify(
        self, payload: dict[str, Any], metadata: dict[str, str]
    ) -> Classification:
        requested = payload.get("model", "auto")
        resolved = self.config.resolve_model(requested)
        if resolved != "auto":
            result = Classification(resolved, {resolved: 1.0}, False)
        else:
            result = await self.classifier.classify(payload, metadata)
        self.classifications[result.model] += 1
        return result

    async def submit(self, payload: dict[str, Any], metadata: dict[str, str]) -> Job:
        classification = await self.classify(payload, metadata)
        forwarded = dict(payload)
        forwarded["model"] = self.config.models[classification.model].public_id
        job = Job(classification.model, forwarded, metadata, classification.model, classification.scores)
        await self.scheduler.enqueue(job)
        depths = self.scheduler.depths()
        if depths[classification.model] >= int(self.config.server["queue_warning_depth"]):
            logger.warning("queue_depth_warning model=%s depths=%s", classification.model, depths)
        return job

    async def _llm_classify(self, payload: dict[str, Any]) -> str | None:
        if not self._client:
            return None
        active = self.primary_backend.active_model
        if not active:
            return None
        model_id = self.config.models[active].public_id
        labels: dict[str, str] = self.config.classifier["labels"]
        label_list = ", ".join(labels)
        rubric = "\n".join(f"- {k}: {v}" for k, v in labels.items())
        grammar = "root ::= (" + " | ".join(f'"{k}"' for k in labels) + ")"
        classify_payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a request router. Classify the conversation to exactly one label.\n"
                        f"{rubric}\nOutput the label name only. No explanation."
                    ),
                },
                *payload.get("messages", []),
                {"role": "user", "content": f"Route to one of: {label_list}"},
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": False,
            "grammar": grammar,
        }
        request = self._client.build_request("POST", "/v1/chat/completions", json=classify_payload)
        response = await self._client.send(request)
        response.raise_for_status()
        msg = response.json()["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning_content") or "").strip().lower()
        for key in labels:
            if key in content:
                return key
        return None

    async def _worker(self) -> None:
        llm_classify_enabled = bool(self.config.server.get("llm_classify", False))
        llm_classify_timeout = float(self.config.server.get("llm_classify_timeout", 2.0))
        while True:
            job = await self.scheduler.next_job()
            if job.cancelled:
                continue
            self.active_job = job
            backend = self._backends[job.model]
            if llm_classify_enabled and self.primary_backend.active_model:
                try:
                    reclassified = await asyncio.wait_for(
                        self._llm_classify(job.payload), timeout=llm_classify_timeout
                    )
                    if reclassified and reclassified != job.model:
                        logger.info(
                            "llm_reclassify id=%s old=%s new=%s", job.id, job.model, reclassified
                        )
                        job.model = reclassified
                        job.payload = {
                            **job.payload,
                            "model": self.config.models[reclassified].public_id,
                        }
                        self.classifications[reclassified] += 1
                        backend = self._backends[job.model]
                except asyncio.TimeoutError:
                    logger.warning("llm_classify_timeout id=%s", job.id)
                except Exception:
                    logger.exception("llm_classify_error id=%s", job.id)
            queued = time.monotonic() - job.enqueued_at
            self.queue_wait_seconds.append(queued)
            started = time.monotonic()
            try:
                await backend.ensure(job.model, self.config.models[job.model])
                response = await self._forward_with_retry(job, backend)
                if job.cancelled:
                    if response.stream:
                        await response.stream.aclose()
                    continue
                if job.result and not job.result.done():
                    job.result.set_result(response)
                await job.finished.wait()
                self.completed[job.model] += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("job_failed id=%s model=%s", job.id, job.model)
                if job.result and not job.result.done():
                    job.result.set_exception(exc)
                job.finished.set()
            finally:
                self.inference_seconds.append(time.monotonic() - started)
                self.active_job = None

    async def _forward_with_retry(self, job: Job, backend: Backend) -> RoutedResponse:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await backend.forward(job.payload, job.payload.get("stream") is True)
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt:
                    break
                logger.warning("upstream_retry error=%s", type(exc).__name__)
                active = backend.active_model
                await backend.stop()
                if active:
                    await backend.ensure(active, self.config.models[active])
        raise RuntimeError(f"upstream request failed after retry: {last_error}") from last_error

    def status(self) -> dict[str, Any]:
        pb = self.primary_backend
        avg = lambda values: round(sum(values) / len(values), 3) if values else 0.0
        return {
            "active_model": pb.active_model,
            "active_job": self.active_job.id if self.active_job else None,
            "queue_depths": self.scheduler.depths(),
            "oldest_wait_seconds": self.scheduler.oldest_waits(),
            "load_count": pb.load_count,
            "switch_count": pb.switch_count,
            "last_model_load_seconds": pb.last_load_seconds,
            "completed_jobs": dict(self.completed),
            "classifications": dict(self.classifications),
            "avg_queue_wait_seconds": avg(self.queue_wait_seconds),
            "avg_inference_seconds": avg(self.inference_seconds),
        }
