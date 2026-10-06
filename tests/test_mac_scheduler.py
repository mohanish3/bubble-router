import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from model_router.mac.scheduler import (
    Priority,
    QueueFullError,
    Scheduler,
    SchedulerDrainingError,
)


@pytest.mark.asyncio
async def test_priority_order_and_bounded_resource_affinity(tmp_path):
    scheduler = Scheduler(tmp_path / "jobs.sqlite", affinity_batch=2)
    gate = asyncio.Event()
    order = []

    async def hold(job):
        order.append(job.resource)
        if len(order) == 1:
            await gate.wait()

    first = await scheduler.submit("qwen-vlm", Priority.P1_INTERACTIVE, hold)
    await asyncio.sleep(0)
    await scheduler.submit("qwen-vlm", Priority.P2_VISION, hold)
    await scheduler.submit("qwen-vlm", Priority.P2_VISION, hold)
    await scheduler.submit("qwen-audio", Priority.P0_CONTROL, hold)
    gate.set()
    await first.future
    await scheduler._idle.wait()
    assert order == ["qwen-vlm", "qwen-audio", "qwen-vlm", "qwen-vlm"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancel_queued_job_and_drain_rejects_new_work(tmp_path):
    scheduler = Scheduler(tmp_path / "jobs.sqlite", max_queue_size=2)
    gate = asyncio.Event()

    async def hold(job):
        await gate.wait()

    running = await scheduler.submit("qwen-vlm", 1, hold)
    await asyncio.sleep(0)
    queued = await scheduler.submit("qwen-audio", 1, hold)
    queued_state = await scheduler.queues()
    assert queued_state["qwen-vlm"][0]["state"] == "running"
    assert queued_state["qwen-audio"][0]["state"] == "queued"
    assert await scheduler.cancel(queued.id)
    assert queued.state == "cancelled"
    drain = asyncio.create_task(scheduler.drain())
    await asyncio.sleep(0)
    with pytest.raises(SchedulerDrainingError):
        await scheduler.submit("qwen-vlm", 1, hold)
    gate.set()
    await running.future
    await drain
    await scheduler.close()


@pytest.mark.asyncio
async def test_start_is_lifespan_safe_and_artifact_metadata_is_durable(tmp_path):
    path = tmp_path / "artifact.sqlite"
    scheduler = Scheduler(path)
    await scheduler.start()

    async def make_artifact(job):
        return {
            "artifact_path": tmp_path / "output.mp4",
            "content_type": "video/mp4",
            "bytes": b"private output bytes",
        }

    job = await scheduler.submit("minimax-h3", Priority.P5_VIDEO, make_artifact)
    result = await job.future
    assert result["bytes"] == b"private output bytes"
    assert job.metadata["artifact"] == {
        "artifact_path": str(tmp_path / "output.mp4"),
        "content_type": "video/mp4",
    }
    persisted = sqlite3.connect(path).execute(
        "SELECT metadata_json FROM jobs WHERE id=?", (job.id,)
    ).fetchone()[0]
    assert "private output bytes" not in persisted
    assert "output.mp4" in persisted
    await scheduler.close()


@pytest.mark.asyncio
async def test_submit_persists_priority_channel_and_resource_peak_metadata(tmp_path):
    profile = SimpleNamespace(estimate=lambda name: {"qwen-vlm": 20.0}[name])
    manager = SimpleNamespace(backends={"qwen-vlm": profile})
    scheduler = Scheduler(tmp_path / "metadata.sqlite", resource_manager=manager)
    gate = asyncio.Event()

    async def wait(job):
        await gate.wait()

    job = await scheduler.submit(
        "qwen-vlm", Priority.P1_INTERACTIVE, wait, metadata={"source": "telegram"}
    )
    assert job.metadata["interactive"] is True
    assert job.metadata["source_channel"] == "telegram"
    assert job.metadata["estimated_peak_gib"] == 20.0
    gate.set()
    await job.future
    await scheduler.close()


@pytest.mark.asyncio
async def test_background_priority_is_not_interactive(tmp_path):
    scheduler = Scheduler(tmp_path / "priority.sqlite")
    job = await scheduler.submit("qwen-audio", Priority.P3_BACKGROUND_SPEECH, lambda job: None)
    assert job.metadata["interactive"] is False
    await job.future
    await scheduler.close()


@pytest.mark.asyncio
async def test_queue_limit_and_restart_marks_jobs_failed_without_replay(tmp_path):
    path = tmp_path / "durable.sqlite"
    scheduler = Scheduler(path, max_queue_size=1)
    gate = asyncio.Event()

    async def hold(job):
        await gate.wait()

    active = await scheduler.submit("qwen-vlm", 1, hold, metadata={"session-id": "r-1", "prompt": "PRIVATE_BODY"})
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        await scheduler.submit("qwen-audio", 1, hold)
    assert await scheduler.cancel(active.id)
    await scheduler._idle.wait()
    await scheduler.close()

    # A prior process leaves queued/running metadata, but no callable or body.
    db = sqlite3.connect(path)
    db.execute(
        "INSERT INTO jobs(id,resource,priority,metadata_json,state,created_at) VALUES(?,?,?,?,?,?)",
        ("interrupted", "qwen-vlm", 1, '{"session-id":"r-1","prompt":"OLD_PRIVATE_BODY"}', "running", 1.0),
    )
    db.commit()
    db.close()
    restarted = Scheduler(path)
    row = sqlite3.connect(path).execute(
        "SELECT state, error_class, metadata_json FROM jobs WHERE id=?", ("interrupted",)
    ).fetchone()
    assert row[0] == "failed"
    assert row[1] == "InterruptedByRestart"
    assert row[2] == '{"session-id":"r-1"}'
    durable = await restarted.get("interrupted")
    assert durable is not None
    assert durable.state == "failed"
    assert durable.metadata == {"session-id": "r-1"}
    await restarted.close()


@pytest.mark.asyncio
async def test_metadata_whitelist_prevents_body_persistence_and_publishes_resource_demand(tmp_path):
    class Manager:
        def __init__(self):
            self.queued = {}
            self.backends = {}

        def set_queued_resources(self, counts):
            self.queued = dict(counts)

    manager = Manager()
    scheduler = Scheduler(tmp_path / "safe.sqlite", resource_manager=manager)
    gate = asyncio.Event()

    async def hold(job):
        await gate.wait()

    job = await scheduler.submit(
        "qwen-vlm", 1, hold,
        metadata={
            "agent-id": "agent-7",
            "session-id": "session-2",
            "source": "telegram",
            "authenticated": True,
            "prompt": "NEVER_WRITE_THIS_BODY",
            "messages": [{"content": "ALSO_PRIVATE"}],
            "security_summary": {"decision": "allow", "score": 0.12, "payload": "DROP_ME"},
        },
    )
    await asyncio.sleep(0)
    persisted = sqlite3.connect(tmp_path / "safe.sqlite").execute(
        "SELECT metadata_json FROM jobs WHERE id=?", (job.id,)
    ).fetchone()[0]
    assert "NEVER_WRITE_THIS_BODY" not in persisted
    assert "ALSO_PRIVATE" not in persisted
    assert "DROP_ME" not in persisted
    assert job.metadata["security_summary"] == {"decision": "allow", "score": 0.12}
    assert manager.queued == {"qwen-vlm": 1}
    gate.set()
    await job.future
    assert manager.queued == {}
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_metrics_are_body_free_and_include_modality_latency(tmp_path):
    scheduler = Scheduler(tmp_path / "metrics.sqlite")
    job = await scheduler.submit(
        "qwen-audio", 3, lambda job: {"ok": True}, metadata={"source": "telegram", "prompt": "SECRET"}
    )
    await job.future
    metrics = await scheduler.metrics()
    assert metrics["totals_by_state"]["succeeded"] == 1
    assert metrics["resources"]["qwen-audio"]["states"]["succeeded"] == 1
    assert metrics["resources"]["qwen-audio"]["avg_execution_seconds"] is not None
    assert metrics["running"] == 0
    assert metrics["queue_depth"] == 0
    assert "SECRET" not in str(metrics)
    await scheduler.close()
