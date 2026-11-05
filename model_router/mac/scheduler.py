"""Durable, priority-aware scheduler for Mac model resources.

SQLite stores job identifiers, resource/priority, caller-supplied metadata,
state, media prompts and a finished job's artifact path/measurements. Job
callables, other results and generated content are never replayed after restart.
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
from pathlib import Path
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

# Artifact fields kept in SQLite so a finished media job can still be polled after a restart.
ARTIFACT_FIELDS = frozenset({"artifact_path", "artifact_id", "content_type", "filename", "size_bytes",
                             "actual_duration_seconds", "actual_frames", "actual_fps", "width", "height",
                             "render_width", "render_height", "upscale",
                             "original_artifact_path", "original_content_type", "generation_seconds", "peak_rss_gib", "peak_metal_gib", "music_model", "music_quality_json",
                             "backend", "model", "mode", "steps", "cfg_strength", "rescale_t", "seed", "num_gaussians",
                             "device", "offline", "load_seconds", "generation_seconds", "total_seconds",
                             "metal_allocated_gib_at_finish", "input_artifact", "camera_artifact",
                             "encoder_sources_json"})
# Token counts kept from a finished chat job's reply (reasoning comes from completion_tokens_details).
USAGE_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "reasoning_tokens"})


class Priority(IntEnum):
    P0_CONTROL = 0
    P1_INTERACTIVE = 1
    P2_VISION = 2
    P3_BACKGROUND_SPEECH = 3
    P4_IMAGE = 4
    P5_VIDEO = 5


# Lanes for chat work. A manual job (a person is waiting) shelves background
# jobs (scheduled agent work) until manual traffic has been quiet for the grace
# window; a background job held longer than max_hold runs anyway so its client
# does not time out. Jobs without a lane are never held and never shelve.
LANES = ("manual", "background")


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
    """One scheduling worker with durable metadata and in-memory callables.

    Jobs run one at a time, except that a resource with more than one slot
    (resource_slots, e.g. a vLLM chat server) runs comparable jobs together.
    Other resources wait until those jobs finish, so only one resource is
    active at a time."""

    TERMINAL = {"succeeded", "failed", "cancelled"}
    METADATA_FIELDS = {
        "agent-id", "agent_id", "session-id", "session_id", "source", "source_channel", "channel", "agent_name",
        "authenticated", "interactive", "estimated_peak_gib", "security_summary", "lane", "kv_tokens",
        # Display origin (app.job_origin); scheduling reads only "lane".
        "origin", "origin_label", "loop_id", "loop_run_id", "loop_name", "cron_id", "cron_name", "ready_at",
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
        shelf_grace_seconds: float = 120.0,
        shelf_max_hold_seconds: float = 900.0,
    ) -> None:
        if aging_seconds <= 0 or affinity_batch < 1 or max_queue_size < 1:
            raise ValueError("aging_seconds, affinity_batch and max_queue_size must be positive")
        if shelf_grace_seconds < 0 or shelf_max_hold_seconds <= 0:
            raise ValueError("shelf_grace_seconds must be non-negative and shelf_max_hold_seconds positive")
        self.db_path = str(db_path)
        self.resource_manager = resource_manager
        self.failure_advisor = None
        self.aging_seconds = float(aging_seconds)
        self.affinity_batch = int(affinity_batch)
        self.max_queue_size = int(max_queue_size)
        self.shelf_grace_seconds = float(shelf_grace_seconds)
        self.shelf_max_hold_seconds = float(shelf_max_hold_seconds)
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
        self._db.execute("CREATE TABLE IF NOT EXISTS job_prompts (job_id TEXT PRIMARY KEY, prompt TEXT NOT NULL)")
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
        self._running: dict[str, Job] = {}
        # Concurrent request slots per resource; unlisted resources get one.
        self.resource_slots: dict[str, int] = {}
        # KV cache tokens per resource and the extra each request takes (block
        # rounding, recurrent state). Until a resource reports its capacity,
        # its jobs run one at a time.
        self.resource_kv_tokens: dict[str, tuple[int, int]] = {}
        self._accepting = True
        self._affinity_resource: str | None = None
        self._affinity_count = 0
        self._manual_seen = 0.0
        self._operator_hold_until = 0.0
        self._wakeup = asyncio.Event()

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
                if key == 'failure_advice' and allow_artifact and isinstance(value, Mapping):
                    safe[key] = {k:(v[:300] if isinstance(v,str) else v) for k,v in value.items()
                        if k in {'action','available','message','automatic_retry','confidence_calibrated'}
                        and isinstance(v,(str,bool))}
                if key == "usage" and allow_artifact and isinstance(value, Mapping):
                    usage = {str(k): v for k, v in value.items() if k in USAGE_FIELDS and type(v) is int and v >= 0}
                    if usage:
                        safe[key] = usage
                if key == "artifact" and allow_artifact and isinstance(value, Mapping):
                    artifact = {
                        str(subkey): subvalue for subkey, subvalue in value.items()
                        if subkey in ARTIFACT_FIELDS and isinstance(subvalue, (str, int, float, bool, type(None)))
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
            if key == "kv_tokens":
                if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 10_000_000:
                    safe[key] = value
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
        counts.update(job.resource for job in self._running.values())
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
        *,
        prompt: str | None = None,
        lane: str | None = None,
    ) -> Job:
        if not builtins.callable(callable):
            raise TypeError("callable must accept one Job argument")
        if lane is not None and lane not in LANES:
            raise ValueError(f"invalid lane: {lane}")
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 393216):
            raise ValueError("Invalid media prompt")
        clean_metadata = self._safe_metadata(metadata)
        clean_metadata.pop("lane", None)
        if lane is not None:
            clean_metadata["lane"] = lane
        priority_value = self._priority_value(priority)
        clean_metadata["interactive"] = priority_value in (1, 2)
        if isinstance(clean_metadata.get("source"), str):
            clean_metadata.setdefault("source_channel", clean_metadata.get("channel") or clean_metadata["source"])
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
            if len(self._pending) + len(self._running) >= self.max_queue_size:
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
            if prompt is not None:
                self._db.execute("INSERT INTO job_prompts VALUES (?,?)", (job.id, prompt))
                self._db.commit()
            if lane == "manual":
                self._manual_seen = time.time()
            self._idle.clear()
            self._wakeup.set()
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

    def _manual_active(self, now: float) -> bool:
        """Manual work is queued or running, inside its grace window, or an operator hold is on."""
        if any(job.metadata.get("lane") == "manual" for job in self._pending):
            return True
        if any(job.metadata.get("lane") == "manual" for job in self._running.values()):
            return True
        return now < self._manual_seen + self.shelf_grace_seconds or now < self._operator_hold_until

    def _held(self, job: Job, now: float) -> bool:
        # A draining scheduler runs shelved work instead of holding shutdown
        # or a serving-mode switch for the grace window.
        return (job.metadata.get("lane") == "background" and self._accepting
                and now - job.created_at < self.shelf_max_hold_seconds
                and self._manual_active(now))

    def _rank(self, job: Job, now: float) -> tuple[Any, ...]:
        # Control first, then manual work and jobs held past the cap, then the
        # rest by aged priority. Manual work does not wait for older aged jobs.
        urgent = job.metadata.get("lane") == "manual" or now - job.created_at >= self.shelf_max_hold_seconds
        return (job.priority != 0, not urgent, self._effective_priority(job, now), job.created_at)

    def _choose(self) -> Job | None:
        now = time.time()
        eligible = [job for job in self._pending if not self._held(job, now)]
        if not eligible:
            return None
        overall = min(eligible, key=lambda job: self._rank(job, now))
        if (
            self._affinity_resource is not None
            and self._affinity_count < self.affinity_batch
        ):
            affinity = [job for job in eligible if job.resource == self._affinity_resource]
            if affinity:
                preferred = min(affinity, key=lambda job: self._rank(job, now))
                # Affinity can batch comparable work but cannot delay a higher
                # aged priority, a manual job or a job held past the cap.
                if self._rank(preferred, now)[:3] <= self._rank(overall, now)[:3]:
                    overall = preferred
        self._pending.remove(overall)
        if overall.resource == self._affinity_resource:
            self._affinity_count += 1
        else:
            self._affinity_resource = overall.resource
            self._affinity_count = 1
        return overall

    def _slots(self, resource: str) -> int:
        return max(1, int(self.resource_slots.get(resource, 1)))

    def mark_ready(self, job: Job) -> None:
        """Record when a running job's model finished loading (metadata ready_at); run time
        before it was spent loading, not generating."""
        if job.state == "running" and "ready_at" not in job.metadata:
            job.metadata["ready_at"] = time.time()
            self._persist(job)

    def set_kv_capacity(self, resource: str, tokens: int, per_request: int = 0) -> None:
        """Record a resource's KV cache size and admit any jobs that now fit."""
        self.resource_kv_tokens[resource] = (int(tokens), int(per_request))
        self._wakeup.set()

    def _fits_alongside(self, job: Job) -> bool:
        """Whether job's KV estimate fits beside the jobs running on its resource,
        so concurrent requests do not evict each other's cached prompts."""
        capacity = self.resource_kv_tokens.get(job.resource)
        if capacity is None or not isinstance(job.metadata.get("kv_tokens"), int):
            return False
        tokens, per_request = capacity
        running = [other for other in self._running.values() if other.resource == job.resource]
        if not all(isinstance(other.metadata.get("kv_tokens"), int) for other in running):
            return False
        used = sum(other.metadata["kv_tokens"] + per_request for other in running)
        return used + job.metadata["kv_tokens"] + per_request <= tokens

    def _choose_alongside(self, resource: str) -> Job | None:
        """A pending job for the running resource that may start now.

        It must rank as high as the best waiting job, ignoring arrival order.
        Aging lifts a job that waits for another resource above new arrivals,
        so it stops further admission and runs once the running jobs finish.
        When the best same-resource job does not fit the KV cache, nothing
        smaller starts in its place; it runs once enough running jobs finish."""
        now = time.time()
        eligible = [job for job in self._pending if not self._held(job, now)]
        same = [job for job in eligible if job.resource == resource]
        if not same:
            return None
        overall = min(eligible, key=lambda job: self._rank(job, now))
        preferred = min(same, key=lambda job: self._rank(job, now))
        if self._rank(preferred, now)[:3] > self._rank(overall, now)[:3]:
            return None
        if not self._fits_alongside(preferred):
            return None
        self._pending.remove(preferred)
        return preferred

    def _next_release(self, now: float) -> float:
        """Seconds until a held background job may run without new manual work."""
        ends = [self._manual_seen + self.shelf_grace_seconds, self._operator_hold_until]
        quiet = max(ends)
        caps = [job.created_at + self.shelf_max_hold_seconds for job in self._pending if self._held(job, now)]
        return max(0.05, min([quiet, *caps]) - now)

    def _start(self, job: Job) -> None:
        self._running[job.id] = job
        job.state = "running"
        job.started_at = time.time()
        self._persist(job)
        job._task = asyncio.create_task(self._execute(job), name=f"mac-job-{job.id}")
        asyncio.create_task(self._finish(job), name=f"mac-job-finish-{job.id}")

    async def _run(self) -> None:
        while True:
            delay: float | None = None
            async with self._lock:
                if not self._pending and not self._running:
                    self._publish_queue_resources()
                    self._idle.set()
                    return
                self._wakeup.clear()
                if not self._running:
                    job = self._choose()
                    if job is None:
                        # Only shelved background work is left: sleep until the
                        # grace window or a hold cap ends, or new work arrives.
                        delay = self._next_release(time.time())
                    else:
                        self._start(job)
                while self._running:
                    resource = next(iter(self._running.values())).resource
                    if len(self._running) >= self._slots(resource):
                        break
                    job = self._choose_alongside(resource)
                    if job is None:
                        break
                    self._start(job)
                if self._running and any(self._held(job, time.time()) for job in self._pending):
                    delay = self._next_release(time.time())
                self._publish_queue_resources()
            # Wake on a finished job, new work, a cancel, or a hold ending.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), delay)

    async def _finish(self, job: Job) -> None:
        try:
            result = await job._task
        except asyncio.CancelledError:
            await self._finish_cancelled(job)
        except BaseException as exc:
            job.error = str(exc)
            job.state = "failed"
            job.finished_at = time.time()
            if self.failure_advisor:
                try:
                    advice = await self.failure_advisor(job, exc)
                    if advice:
                        job.metadata['failure_advice'] = advice
                except Exception:
                    pass  # Advisory review must never replace the actual failure.
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
                if job.metadata.get("lane") == "manual":
                    self._manual_seen = time.time()
                self._running.pop(job.id, None)
                self._publish_queue_resources()
                if not self._pending and not self._running:
                    self._idle.set()
                self._wakeup.set()

    async def _execute(self, job: Job) -> Any:
        assert job._callable is not None
        result = job._callable(job)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _persist_artifact_metadata(job: Job, result: Any) -> None:
        if not isinstance(result, Mapping):
            return
        usage = result.get("usage")
        if isinstance(usage, Mapping):
            details = usage.get("completion_tokens_details")
            reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
            kept = {key: value for key, value in {**usage, "reasoning_tokens": reasoning}.items()
                    if key in USAGE_FIELDS and type(value) is int and value >= 0}
            if kept:
                job.metadata["usage"] = kept
        artifact = {key: result[key] for key in ARTIFACT_FIELDS if key in result}
        encoder_sources = result.get("encoder_sources")
        if isinstance(encoder_sources, Mapping):
            artifact["encoder_sources_json"] = json.dumps(encoder_sources, separators=(",", ":"), ensure_ascii=False)
        if "artifact_path" not in artifact and isinstance(result.get("artifact"), (str, os.PathLike)):
            artifact["artifact_path"] = result["artifact"]
        if not artifact:
            return
        if "artifact_path" in artifact:
            artifact["artifact_path"] = os.fspath(artifact["artifact_path"])
        job.metadata["artifact"] = artifact

    @staticmethod
    def _restored_result(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
        """Rebuild a finished media job's result from its persisted artifact metadata."""
        artifact = metadata.get("artifact")
        if not isinstance(artifact, Mapping) or "artifact_path" not in artifact:
            return None
        restored = {"artifact": artifact["artifact_path"],
                    **{key: value for key, value in artifact.items()
                       if key not in {"artifact_path", "encoder_sources_json"}}}
        sources_json = artifact.get("encoder_sources_json")
        if isinstance(sources_json, str):
            try:
                sources = json.loads(sources_json)
            except ValueError:
                sources = None
            if isinstance(sources, Mapping):
                restored["encoder_sources"] = dict(sources)
        return restored

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
        if job.state == "succeeded":
            job.result = self._restored_result(job.metadata)
        self._jobs[job_id] = job
        return job

    def get_prompt(self, job_id: str) -> str | None:
        row = self._db.execute("SELECT prompt FROM job_prompts WHERE job_id=?", (job_id,)).fetchone()
        return row[0] if row else None

    async def delete_media(self, job_id: str, output_root: Path) -> dict[str, Any]:
        """Delete a terminal media job, its prompt and its associated output files."""
        async with self._lock:
            job = await self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            # A story also owns text planning and quality-review history.
            # Keep ordinary chat jobs outside this media-cleanup API.
            story_session = job.metadata.get("session-id") or job.metadata.get("session_id")
            story_job = (job.metadata.get("agent-id") == "bubble-dashboard"
                         and job.metadata.get("source") == "dashboard"
                         and isinstance(story_session, str) and len(story_session) == 32
                         and all(c in "0123456789abcdef" for c in story_session))
            if job.resource not in ("qwen-image", "qwen-audio", "minimax-h3", "ace-music", "hf-space:speridlabs/idol", "idol-local") and not story_job:
                raise ValueError("Only media or dashboard story jobs can be deleted")
            if job.state not in self.TERMINAL or job.id in self._running:
                raise ValueError("Finish or cancel the job before deleting it")
            root = output_root.resolve()
            candidates = set()
            result = job.result if isinstance(job.result, Mapping) else {}
            artifact = job.metadata.get("artifact", {})
            for value in (result.get("artifact"), artifact.get("artifact_path"),
                          result.get("original_artifact_path"), artifact.get("original_artifact_path"),
                          result.get("input_artifact"), artifact.get("input_artifact"),
                          result.get("camera_artifact"), artifact.get("camera_artifact")):
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError("Invalid media artifact path")
                path = Path(value.removeprefix("file://")).expanduser().resolve()
                if not path.is_relative_to(root) or path == root:
                    raise ValueError("Media artifact must be inside the output directory")
                candidates.add(path)
            # Older rows may lack an artifact path. Only exact job-ID filenames
            # in the known output directories are eligible for removal.
            if len(job_id) == 32 and all(c in "0123456789abcdef" for c in job_id):
                for folder, extensions in (("images", ("png", "jpg", "jpeg", "webp")),
                                           ("audio", ("wav", "mp3", "flac")), ("video", ("mp4", "mov"))):
                    for extension in extensions:
                        path = root / folder / f"{job_id}.{extension}"
                        if path.exists() or path.is_symlink():
                            resolved = path.resolve()
                            if not resolved.is_relative_to(root) or path.is_symlink():
                                raise ValueError("Invalid media artifact path")
                            candidates.add(resolved)
            for path in candidates:
                if path.exists() and not path.is_file():
                    raise ValueError("Media artifact is not a file")
                # Shared artifacts must survive until their final owning job is removed.
                for row in self._db.execute("SELECT metadata_json FROM jobs WHERE id!=?", (job_id,)):
                    other = json.loads(row[0]).get("artifact", {})
                    for field in ("artifact_path", "original_artifact_path", "input_artifact", "camera_artifact"):
                        value = other.get(field)
                        if isinstance(value, str) and Path(value.removeprefix("file://")).expanduser().resolve() == path:
                            raise ValueError("Another job references this media file")
            removed = 0
            with self._db:
                self._db.execute("DELETE FROM job_prompts WHERE job_id=?", (job_id,))
                self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
                for path in candidates:
                    if path.exists():
                        path.unlink()
                        removed += 1
            self._jobs.pop(job_id, None)
            job.result = None
            job.metadata.clear()
            job._callable = None
            return {"deleted": True, "media_files_deleted": removed}

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
                if not self._pending and not self._running:
                    self._idle.set()
                self._publish_queue_resources()
                self._wakeup.set()
                return True
            task = job._task
            if task is not None:
                task.cancel()
        return True

    async def queues(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        visible = [*self._pending, *self._running.values()]
        now = time.time()
        for job in sorted(visible, key=lambda item: (item.state != "running", item.priority, item.created_at)):
            grouped.setdefault(job.resource, []).append({**job.snapshot(), "held": self._held(job, now)})
        return grouped

    def shelf_status(self) -> dict[str, Any]:
        now = time.time()
        held = [job for job in self._pending if self._held(job, now)]
        return {
            "shelved": self._manual_active(now),
            "held_background_jobs": len(held),
            "oldest_held_seconds": round(max((now - job.created_at for job in held), default=0.0), 1),
            "grace_seconds": self.shelf_grace_seconds,
            "max_hold_seconds": self.shelf_max_hold_seconds,
            "operator_hold_seconds": round(max(0.0, self._operator_hold_until - now), 1),
        }

    async def shelve(self, seconds: float) -> dict[str, Any]:
        """Hold background work for an interactive session the Router cannot see.
        The per-job max_hold cap still applies."""
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 0 < seconds <= 86400:
            raise ValueError("seconds must be between 0 and 86400")
        self._operator_hold_until = time.time() + seconds
        return self.shelf_status()

    async def unshelve(self) -> dict[str, Any]:
        """End an operator hold and the manual grace window now."""
        self._operator_hold_until = 0.0
        self._manual_seen = 0.0
        self._wakeup.set()
        return self.shelf_status()

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
            "running": len(self._running),
        }

    async def drain(self) -> None:
        """Stop admission and wait until all already-accepted work finishes."""
        self._accepting = False
        self._wakeup.set()
        await self._idle.wait()

    async def resume(self) -> None:
        self._accepting = True

    async def close(self) -> None:
        await self.drain()
        worker = self._worker
        if worker is not None:
            await worker
        self._db.close()
