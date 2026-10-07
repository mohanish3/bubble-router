"""Durable, priority-aware scheduler for Mac model resources.

SQLite stores job identifiers, resource/priority, caller-supplied metadata and
state only. Job callables, results, prompts and generated content stay in
memory and are never replayed after a restart.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
from collections import Counter
import inspect
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping


class Priority(IntEnum):
    P0_CONTROL = 0
    P1_INTERACTIVE = 1
    P2_VISION = 2
    P3_BACKGROUND_SPEECH = 3
    P4_IMAGE = 4
    P5_VIDEO = 5


class SchedulerError(RuntimeError):
    pass


class QueueFullError(SchedulerError):
    pass


class SchedulerDrainingError(SchedulerError):
    pass


JobCallable = Callable[["Job"], Any | Awaitable[Any]]


@dataclass
class Job:
    id: str
    resource: str
    priority: int
    metadata: dict[str, Any]
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: Any = None
    future: asyncio.Future[Any] | None = field(default=None, repr=False)
    _callable: JobCallable | None = field(default=None, repr=False)
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _scheduler: "Scheduler | None" = field(default=None, repr=False)

    async def cancel(self) -> bool:
        if self._scheduler is None:
            return False
        return await self._scheduler.cancel(self.id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "resource": self.resource,
            "priority": self.priority,
            "metadata": dict(self.metadata),
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class Scheduler:
    """A single execution worker with durable metadata and in-memory callables."""

    TERMINAL = {"succeeded", "failed", "cancelled"}
    METADATA_FIELDS = {
        "agent-id", "agent_id", "session-id", "session_id", "source", "source_channel", "channel",
        "authenticated", "interactive", "estimated_peak_gib", "security_summary",
    }
    SECURITY_SUMMARY_FIELDS = {"decision", "score", "policy_version", "reason", "scanner"}

    def __init__(
        self,
        db_path: str | Path,
        resource_manager: Any = None,
        *,
        aging_seconds: float = 30.0,
        affinity_batch: int = 2,
        max_queue_size: int = 128,
    ) -> None:
        if aging_seconds <= 0 or affinity_batch < 1 or max_queue_size < 1:
            raise ValueError("aging_seconds, affinity_batch and max_queue_size must be positive")
        self.db_path = str(db_path)
        self.resource_manager = resource_manager
        self.aging_seconds = float(aging_seconds)
        self.affinity_batch = int(affinity_batch)
        self.max_queue_size = int(max_queue_size)
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                resource TEXT NOT NULL,
                priority INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                error_class TEXT
            )"""
        )
        # Scrub rows written by older versions that allowed arbitrary caller
        # metadata before marking interrupted jobs failed.
        for row in self._db.execute("SELECT id, metadata_json FROM jobs").fetchall():
            try:
                previous = json.loads(row["metadata_json"])
            except (TypeError, ValueError):
                previous = {}
            safe = self._safe_metadata(previous, allow_artifact=True)
            encoded = json.dumps(safe, separators=(",", ":"), ensure_ascii=False)
            if encoded != row["metadata_json"]:
                self._db.execute("UPDATE jobs SET metadata_json=? WHERE id=?", (encoded, row["id"]))
        # A fresh process has no callable or in-memory request body to resume.
        self._db.execute(
            "UPDATE jobs SET state='failed', finished_at=?, error_class='InterruptedByRestart' "
            "WHERE state IN ('queued','running')",
            (time.time(),),
        )
        self._db.commit()
        self._jobs: dict[str, Job] = {}
        self._pending: list[Job] = []
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker: asyncio.Task[None] | None = None
        self._active: Job | None = None
        self._accepting = True
        self._affinity_resource: str | None = None
        self._affinity_count = 0

    def _persist(self, job: Job, *, error_class: str | None = None) -> None:
        self._db.execute(
            """INSERT INTO jobs(id,resource,priority,metadata_json,state,created_at,started_at,finished_at,error_class)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET resource=excluded.resource, priority=excluded.priority,
               metadata_json=excluded.metadata_json, state=excluded.state, created_at=excluded.created_at,
               started_at=excluded.started_at, finished_at=excluded.finished_at, error_class=excluded.error_class""",
            (
                job.id,
                job.resource,
                int(job.priority),
                json.dumps(job.metadata, separators=(",", ":"), ensure_ascii=False),
                job.state,
                job.created_at,
                job.started_at,
                job.finished_at,
                error_class,
            ),
        )
        self._db.commit()

    @classmethod
    def _safe_metadata(cls, metadata: Mapping[str, Any] | None, *, allow_artifact: bool = False) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in dict(metadata or {}).items():
            if key not in cls.METADATA_FIELDS:
                if key == "artifact" and allow_artifact and isinstance(value, Mapping):
                    allowed = {"artifact_path", "artifact_id", "content_type", "filename", "size_bytes"}
                    artifact = {
                        str(subkey): subvalue for subkey, subvalue in value.items()
                        if subkey in allowed and isinstance(subvalue, (str, int, float, bool, type(None)))
                    }
                    if artifact:
                        safe[key] = artifact
                continue
            if key == "security_summary":
                if isinstance(value, Mapping):
                    summary = {
                        str(subkey): (subvalue[:120] if isinstance(subvalue, str) else subvalue)
                        for subkey, subvalue in value.items()
                        if subkey in cls.SECURITY_SUMMARY_FIELDS
                        and isinstance(subvalue, (str, int, float, bool, type(None)))
                        and (not isinstance(subvalue, float) or math.isfinite(subvalue))
                    }
                    if summary:
                        safe[key] = summary
                continue
            if key == "estimated_peak_gib":
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 < float(value) <= 128:
                    safe[key] = float(value)
                continue
            if key in {"authenticated", "interactive"}:
                if isinstance(value, bool):
                    safe[key] = value
                continue
            if isinstance(value, str):
                safe[key] = value[:120]
        return safe

    def _publish_queue_resources(self) -> None:
        setter = getattr(self.resource_manager, "set_queued_resources", None)
        if not callable(setter):
            return
        counts = Counter(job.resource for job in self._pending)
        if self._active is not None:
            counts[self._active.resource] += 1
        setter(counts)

    @staticmethod
    def _priority_value(priority: int | Priority | str) -> int:
        if isinstance(priority, str):
            normalized = priority.upper().replace("-", "_")
            if not normalized.startswith("P"):
                normalized = "P" + normalized
            names = {
                "P0": 0, "P0_CONTROL": 0,
                "P1": 1, "P1_INTERACTIVE": 1,
                "P2": 2, "P2_VISION": 2,
                "P3": 3, "P3_BACKGROUND_SPEECH": 3,
                "P4": 4, "P4_IMAGE": 4,
                "P5": 5, "P5_VIDEO": 5,
            }
            if normalized not in names:
                raise ValueError(f"invalid priority: {priority}")
            return names[normalized]
        value = int(priority)
        if not 0 <= value <= 5:
            raise ValueError("priority must be between 0 (highest) and 5 (lowest)")
        return value

    async def submit(
        self,
        resource: str,
        priority: int | Priority | str,
        callable: JobCallable,
        metadata: Mapping[str, Any] | None = None,
    ) -> Job:
        if not builtins.callable(callable):
            raise TypeError("callable must accept one Job argument")
        clean_metadata = self._safe_metadata(metadata)
        priority_value = self._priority_value(priority)
        clean_metadata["interactive"] = priority_value in (1, 2)
        if isinstance(clean_metadata.get("source"), str):
            clean_metadata.setdefault("source_channel", clean_metadata["source"])
        profiles = getattr(self.resource_manager, "backends", None)
        if isinstance(profiles, Mapping) and resource in profiles:
            profile = profiles[resource]
            estimate_fn = getattr(profile, "estimate", None)
            try:
                estimate = estimate_fn(resource) if builtins.callable(estimate_fn) else getattr(profile, "estimate_gib", None)
                if estimate is not None and math.isfinite(float(estimate)) and float(estimate) > 0:
                    metadata_estimate = clean_metadata.get("estimated_peak_gib", 0.0)
                    if (isinstance(metadata_estimate, (int, float)) and not isinstance(metadata_estimate, bool)
                            and math.isfinite(float(metadata_estimate)) and float(metadata_estimate) > 0):
                        estimate = max(float(estimate), float(metadata_estimate))
                    clean_metadata["estimated_peak_gib"] = float(estimate)
            except (TypeError, ValueError, AttributeError):
                pass
        async with self._lock:
            if not self._accepting:
                raise SchedulerDrainingError("scheduler is draining and not accepting jobs")
            if len(self._pending) + int(self._active is not None) >= self.max_queue_size:
                raise QueueFullError("scheduler queue is full")
            loop = asyncio.get_running_loop()
            job = Job(
                id=uuid.uuid4().hex,
                resource=str(resource),
                priority=priority_value,
                metadata=clean_metadata,
                future=loop.create_future(),
                _callable=callable,
                _scheduler=self,
            )
            self._jobs[job.id] = job
            self._pending.append(job)
            self._publish_queue_resources()
            self._persist(job)
            self._idle.clear()
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run(), name="mac-resource-scheduler")
            return job

    async def start(self) -> None:
        """Mark the scheduler ready for the application lifespan.

        Work starts lazily when the first job is admitted, so startup does not
        create an idle task or replay rows left by an earlier process.
        """
        if self._db is None:
            raise SchedulerError("scheduler is closed")

    def _effective_priority(self, job: Job, now: float) -> int:
        waited = max(0.0, now - job.created_at)
        aging_steps = int(waited // self.aging_seconds)
        return max(0, job.priority - aging_steps)

    def _choose(self) -> Job:
        now = time.time()
        pending = self._pending
        overall = min(pending, key=lambda job: (self._effective_priority(job, now), job.created_at))
        if (
            self._affinity_resource is not None
            and self._affinity_count < self.affinity_batch
        ):
            affinity = [job for job in pending if job.resource == self._affinity_resource]
            if affinity:
                preferred = min(affinity, key=lambda job: (self._effective_priority(job, now), job.created_at))
                # Affinity can batch comparable work but cannot delay a higher aged priority.
                if self._effective_priority(preferred, now) <= self._effective_priority(overall, now):
                    overall = preferred
        pending.remove(overall)
        if overall.resource == self._affinity_resource:
            self._affinity_count += 1
        else:
            self._affinity_resource = overall.resource
            self._affinity_count = 1
        return overall

    async def _run(self) -> None:
        while True:
            async with self._lock:
                if not self._pending:
                    self._active = None
                    self._publish_queue_resources()
                    self._idle.set()
                    return
                job = self._choose()
                self._active = job
                self._publish_queue_resources()
                job.state = "running"
                job.started_at = time.time()
                self._persist(job)
                task = asyncio.create_task(self._execute(job), name=f"mac-job-{job.id}")
                job._task = task
            try:
                result = await task
            except asyncio.CancelledError:
                await self._finish_cancelled(job)
            except BaseException as exc:
                job.error = str(exc)
                job.state = "failed"
                job.finished_at = time.time()
                self._persist(job, error_class=type(exc).__name__)
                if job.future and not job.future.done():
                    job.future.set_exception(exc)
            else:
                job.result = result
                job.state = "succeeded"
                job.finished_at = time.time()
                self._persist_artifact_metadata(job, result)
                self._persist(job)
                if job.future and not job.future.done():
                    job.future.set_result(result)
            finally:
                async with self._lock:
                    self._active = None
                    self._publish_queue_resources()
                    if not self._pending:
                        self._idle.set()

    async def _execute(self, job: Job) -> Any:
        assert job._callable is not None
        result = job._callable(job)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _persist_artifact_metadata(job: Job, result: Any) -> None:
        if not isinstance(result, Mapping):
            return
        allowed = {"artifact_path", "artifact_id", "content_type", "filename", "size_bytes"}
        artifact = {key: result[key] for key in allowed if key in result}
        if not artifact:
            return
        if "artifact_path" in artifact:
            artifact["artifact_path"] = os.fspath(artifact["artifact_path"])
        job.metadata["artifact"] = artifact

    async def _finish_cancelled(self, job: Job) -> None:
        job.state = "cancelled"
        job.finished_at = time.time()
        self._persist(job, error_class="CancelledError")
        if job.future and not job.future.done():
            job.future.cancel()

    async def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        row = self._db.execute(
            "SELECT id,resource,priority,metadata_json,state,created_at,started_at,finished_at,error_class "
            "FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        job = Job(
            id=row["id"],
            resource=row["resource"],
            priority=int(row["priority"]),
            metadata=json.loads(row["metadata_json"]),
            state=row["state"],
            created_at=float(row["created_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=row["error_class"],
            _scheduler=self,
        )
        self._jobs[job_id] = job
        return job

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in self.TERMINAL:
                return False
            if job.state == "queued":
                with contextlib.suppress(ValueError):
                    self._pending.remove(job)
                job.state = "cancelled"
                job.finished_at = time.time()
                self._persist(job, error_class="CancelledError")
                if job.future and not job.future.done():
                    job.future.cancel()
                if not self._pending and self._active is None:
                    self._idle.set()
                self._publish_queue_resources()
                return True
            task = job._task
            if task is not None:
                task.cancel()
        return True

    async def queues(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        visible = list(self._pending)
        if self._active is not None:
            visible.append(self._active)
        for job in sorted(visible, key=lambda item: (item.state != "running", item.priority, item.created_at)):
            grouped.setdefault(job.resource, []).append(job.snapshot())
        return grouped

    async def metrics(self) -> dict[str, Any]:
        """Return durable job counts and body-free wait/execution aggregates."""
        rows = self._db.execute(
            """SELECT resource, state, COUNT(*) AS count,
                      AVG(CASE WHEN started_at IS NOT NULL THEN started_at - created_at END) AS avg_queue_wait_seconds,
                      AVG(CASE WHEN finished_at IS NOT NULL AND started_at IS NOT NULL THEN finished_at - started_at END) AS avg_execution_seconds
               FROM jobs GROUP BY resource, state ORDER BY resource, state"""
        ).fetchall()
        resources: dict[str, dict[str, Any]] = {}
        totals: Counter[str] = Counter()
        queue_weighted: dict[str, float] = {}
        execution_weighted: dict[str, float] = {}
        queue_samples: dict[str, int] = Counter()
        execution_samples: dict[str, int] = Counter()
        for row in rows:
            resource = str(row["resource"])
            state = str(row["state"])
            count = int(row["count"])
            item = resources.setdefault(resource, {"states": {}, "avg_queue_wait_seconds": None, "avg_execution_seconds": None})
            item["states"][state] = count
            totals[state] += count
            if row["avg_queue_wait_seconds"] is not None:
                queue_weighted[resource] = queue_weighted.get(resource, 0.0) + float(row["avg_queue_wait_seconds"]) * count
                queue_samples[resource] += count
            if row["avg_execution_seconds"] is not None:
                execution_weighted[resource] = execution_weighted.get(resource, 0.0) + float(row["avg_execution_seconds"]) * count
                execution_samples[resource] += count
        for resource, item in resources.items():
            if resource in queue_weighted:
                item["avg_queue_wait_seconds"] = queue_weighted[resource] / queue_samples[resource]
            if resource in execution_weighted:
                item["avg_execution_seconds"] = execution_weighted[resource] / execution_samples[resource]
        return {
            "totals_by_state": dict(totals),
            "resources": resources,
            "queue_depth": len(self._pending),
            "running": int(self._active is not None),
        }

    async def drain(self) -> None:
        """Stop admission and wait until all already-accepted work finishes."""
        self._accepting = False
        await self._idle.wait()

    async def resume(self) -> None:
        self._accepting = True

    async def close(self) -> None:
        await self.drain()
        worker = self._worker
        if worker is not None:
            await worker
        self._db.close()
