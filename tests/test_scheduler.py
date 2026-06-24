import asyncio
import time

import pytest

from model_router.scheduler import Job, Scheduler


@pytest.mark.asyncio
async def test_fifo_within_model():
    scheduler = Scheduler(["gemma", "qwen-opus"], max_loaded_jobs=4)
    first = Job("gemma", {}, {}, enqueued_at=1)
    second = Job("gemma", {}, {}, enqueued_at=2)
    await scheduler.enqueue(first)
    await scheduler.enqueue(second)
    assert await scheduler.next_job() is first
    assert await scheduler.next_job() is second


@pytest.mark.asyncio
async def test_switches_after_four_loaded_jobs_to_globally_oldest():
    scheduler = Scheduler(["gemma", "qwen-opus"], max_loaded_jobs=4)
    scheduler.loaded_model = "gemma"
    scheduler.loaded_since = time.monotonic()
    scheduler.loaded_jobs = 3
    await scheduler.enqueue(Job("gemma", {}, {}, enqueued_at=20))
    await scheduler.enqueue(Job("qwen-opus", {}, {}, enqueued_at=10))
    assert (await scheduler.next_job()).model == "gemma"
    assert (await scheduler.next_job()).model == "qwen-opus"


def test_switches_after_time_limit():
    scheduler = Scheduler(["gemma", "qwen-opus"], max_loaded_seconds=30)
    scheduler.loaded_model = "gemma"
    scheduler.loaded_since = 10
    scheduler.queues["gemma"].append(Job("gemma", {}, {}, enqueued_at=20))
    scheduler.queues["qwen-opus"].append(Job("qwen-opus", {}, {}, enqueued_at=15))
    assert scheduler.choose(now=41) == "qwen-opus"


@pytest.mark.asyncio
async def test_cancel_removes_queued_job():
    scheduler = Scheduler(["gemma"])
    job = Job("gemma", {}, {})
    await scheduler.enqueue(job)
    assert await scheduler.cancel(job)
    assert scheduler.depths()["gemma"] == 0
    assert job.result.cancelled()
