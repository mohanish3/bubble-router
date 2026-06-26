from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    model: str
    payload: dict[str, Any]
    metadata: dict[str, str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)
    classification_model: str | None = None
    classification_scores: dict[str, float] | None = None
    result: asyncio.Future[Any] | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class Scheduler:
    def __init__(
        self, model_keys: list[str], max_loaded_jobs: int = 4, max_loaded_seconds: float = 30
    ) -> None:
        self.queues = {key: deque() for key in model_keys}
        self.max_loaded_jobs = max_loaded_jobs
        self.max_loaded_seconds = max_loaded_seconds
        self.loaded_model: str | None = None
        self.loaded_since = time.monotonic()
        self.loaded_jobs = 0
        self.condition = asyncio.Condition()

    async def enqueue(self, job: Job) -> None:
        job.result = asyncio.get_running_loop().create_future()
        async with self.condition:
            self.queues[job.model].append(job)
            self.condition.notify()

    async def cancel(self, job: Job) -> bool:
        async with self.condition:
            queue = self.queues[job.model]
            try:
                queue.remove(job)
                job.cancelled = True
                if job.result and not job.result.done():
                    job.result.cancel()
                return True
            except ValueError:
                job.cancelled = True
                return False

    def _oldest_model(self) -> str | None:
        candidates = [
            (queue[0].enqueued_at, key)
            for key, queue in self.queues.items()
            if queue
        ]
        return min(candidates)[1] if candidates else None

    def choose(self, now: float | None = None) -> str | None:
        now = now or time.monotonic()
        loaded_queue = self.queues.get(self.loaded_model or "")
        favor_loaded = (
            loaded_queue
            and self.loaded_jobs < self.max_loaded_jobs
            and now - self.loaded_since < self.max_loaded_seconds
        )
        return self.loaded_model if favor_loaded else self._oldest_model()

    async def next_job(self) -> Job:
        async with self.condition:
            await self.condition.wait_for(lambda: any(self.queues.values()))
            model = self.choose()
            if model != self.loaded_model:
                self.loaded_model = model
                self.loaded_since = time.monotonic()
                self.loaded_jobs = 0
            job = self.queues[model].popleft()
            self.loaded_jobs += 1
            return job

    def depths(self) -> dict[str, int]:
        return {key: len(queue) for key, queue in self.queues.items()}

    def oldest_waits(self) -> dict[str, float]:
        now = time.monotonic()
        return {
            key: round(now - queue[0].enqueued_at, 3) if queue else 0.0
            for key, queue in self.queues.items()
        }