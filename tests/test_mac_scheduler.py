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
async def test_media_prompt_is_durable_and_deleted_with_job_and_output(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    output = tmp_path / 'outputs' / 'images' / 'fixture.png'
    output.parent.mkdir(parents=True)
    output.write_bytes(b'disposable media fixture')
    async def generate(job):
        return {'artifact': str(output), 'content_type': 'image/png'}
    job = await scheduler.submit('qwen-image', 4, generate, prompt='Exact media prompt <test>')
    await job.future
    await scheduler._idle.wait()
    assert scheduler.get_prompt(job.id) == 'Exact media prompt <test>'
    await scheduler.close()
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    assert scheduler.get_prompt(job.id) == 'Exact media prompt <test>'
    assert (await scheduler.get(job.id)).state == 'succeeded'
    result = await scheduler.delete_media(job.id, tmp_path / 'outputs')
    assert result == {'deleted': True, 'media_files_deleted': 1}
    assert not output.exists()
    assert await scheduler.get(job.id) is None
    assert scheduler.get_prompt(job.id) is None
    assert job.id not in scheduler._jobs
    await scheduler.close()


@pytest.mark.asyncio
async def test_delete_rejects_active_jobs_and_artifacts_outside_outputs(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    gate = asyncio.Event()
    async def hold(job):
        await gate.wait()
    active = await scheduler.submit('minimax-h3', 5, hold, prompt='Active prompt')
    with pytest.raises(ValueError, match='Finish or cancel'):
        await scheduler.delete_media(active.id, tmp_path / 'outputs')
    assert await scheduler.get(active.id) is active
    gate.set()
    await active.future
    await scheduler._idle.wait()
    outside = tmp_path / 'private.txt'
    outside.write_text('Protected fixture')
    active.result = {'artifact': str(outside)}
    with pytest.raises(ValueError, match='inside the output'):
        await scheduler.delete_media(active.id, tmp_path / 'outputs')
    assert outside.exists()
    assert await scheduler.get(active.id) is active
    await scheduler.close()


@pytest.mark.asyncio
async def test_delete_failed_job_finds_legacy_media_by_exact_id(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    async def fail(job):
        raise RuntimeError('Fixture failure')
    job = await scheduler.submit('qwen-image', 4, fail, prompt='Failed prompt')
    with pytest.raises(RuntimeError):
        await job.future
    await scheduler._idle.wait()
    output = tmp_path / 'outputs' / 'images' / (job.id + '.png')
    output.parent.mkdir(parents=True)
    output.write_bytes(b'partial output fixture')
    unrelated = output.with_name('unrelated.png')
    unrelated.write_bytes(b'protected media fixture')
    assert (await scheduler.delete_media(job.id, tmp_path / 'outputs'))['media_files_deleted'] == 1
    assert not output.exists()
    assert unrelated.exists()
    await scheduler.close()


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


@pytest.mark.asyncio
async def test_media_result_survives_restart(tmp_path):
    path = tmp_path / "restart.sqlite"
    scheduler = Scheduler(path)
    await scheduler.start()

    async def render(job):
        return {"artifact": str(tmp_path / "clip.mp4"), "content_type": "video/mp4",
                "actual_duration_seconds": 5.0, "actual_frames": 120, "prompt": "private"}

    job = await scheduler.submit("minimax-h3", Priority.P5_VIDEO, render)
    await job.future
    await scheduler.close()

    restarted = Scheduler(path)
    await restarted.start()
    loaded = await restarted.get(job.id)
    assert loaded.state == "succeeded"
    assert loaded.result == {"artifact": str(tmp_path / "clip.mp4"), "content_type": "video/mp4",
                             "actual_duration_seconds": 5.0, "actual_frames": 120}
    await restarted.close()


@pytest.mark.asyncio
async def test_manual_work_shelves_background_until_grace_window_ends(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', aging_seconds=0.01, shelf_grace_seconds=0.3)
    order = []
    gate = asyncio.Event()
    def record(name):
        async def run(job):
            if name == 'manual-1':
                await gate.wait()
            order.append(name)
        return run
    manual = await scheduler.submit('swift', 1, record('manual-1'), lane='manual')
    await asyncio.sleep(0.05)
    background = await scheduler.submit('arex', 1, record('background'), lane='background')
    # Background has aged past the manual job's priority but is still held.
    await asyncio.sleep(0.05)
    follow_up = await scheduler.submit('swift', 1, record('manual-2'), lane='manual')
    status = scheduler.shelf_status()
    assert status['shelved'] and status['held_background_jobs'] == 1
    assert [job['held'] for job in (await scheduler.queues())['arex']] == [True]
    gate.set()
    await follow_up.future
    await asyncio.sleep(0.1)
    # Manual traffic stopped less than the grace window ago.
    assert order == ['manual-1', 'manual-2'] and background.state == 'queued'
    await background.future
    assert order == ['manual-1', 'manual-2', 'background']
    assert background.metadata['lane'] == 'background' and manual.metadata['lane'] == 'manual'
    assert not scheduler.shelf_status()['shelved']
    await scheduler.close()


@pytest.mark.asyncio
async def test_held_background_runs_after_max_hold_even_while_manual_is_queued(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', shelf_grace_seconds=60, shelf_max_hold_seconds=0.2)
    order = []
    def record(name):
        async def run(job):
            order.append(name)
            await asyncio.sleep(0.05)
        return run
    background = await scheduler.submit('arex', 1, record('background'), lane='background')
    await scheduler.submit('swift', 1, record('manual-1'), lane='manual')
    await asyncio.sleep(0.1)
    assert order == ['manual-1']
    await asyncio.sleep(0.15)
    await scheduler.submit('swift', 1, record('manual-2'), lane='manual')
    await background.future
    await scheduler._idle.wait()
    # Past the cap the background job runs although manual traffic continues.
    assert order == ['manual-1', 'background', 'manual-2']
    await scheduler.close()


@pytest.mark.asyncio
async def test_operator_shelve_holds_background_but_not_untagged_work(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', shelf_grace_seconds=0)
    with pytest.raises(ValueError):
        await scheduler.shelve(0)
    with pytest.raises(ValueError):
        await scheduler.submit('swift', 1, lambda job: None, lane='urgent')
    assert (await scheduler.shelve(60))['operator_hold_seconds'] > 59
    background = await scheduler.submit('arex', 1, lambda job: 'bg', lane='background')
    other = await scheduler.submit('qwen-image', 4, lambda job: 'image')
    assert await other.future == 'image'
    await asyncio.sleep(0.05)
    assert background.state == 'queued' and 'lane' not in other.metadata
    status = await scheduler.unshelve()
    assert status['operator_hold_seconds'] == 0
    assert await background.future == 'bg'
    await scheduler.close()


@pytest.mark.asyncio
async def test_drain_releases_shelved_background_work(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', shelf_grace_seconds=60)
    await scheduler.submit('swift', 1, lambda job: None, lane='manual')
    background = await scheduler.submit('arex', 1, lambda job: 'bg', lane='background')
    await asyncio.wait_for(scheduler.drain(), 2)
    assert background.state == 'succeeded'
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancelling_a_held_background_job_leaves_worker_idle(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', shelf_grace_seconds=60)
    await (await scheduler.submit('swift', 1, lambda job: None, lane='manual')).future
    background = await scheduler.submit('arex', 1, lambda job: 'bg', lane='background')
    await asyncio.sleep(0.05)
    assert await scheduler.cancel(background.id)
    await asyncio.wait_for(scheduler._idle.wait(), 2)
    assert background.state == 'cancelled'
    await scheduler.close()


async def _until(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, 'condition not reached'
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_resource_with_slots_runs_comparable_jobs_together(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    scheduler.resource_slots = {'swift': 2}
    scheduler.set_kv_capacity('swift', 100_000)
    gate = asyncio.Event(); running = set(); peak = []
    async def work(job):
        running.add(job.id); peak.append(len(running))
        await gate.wait(); running.discard(job.id)
        return job.resource
    jobs = [await scheduler.submit(r, 1, work, {'kv_tokens': 10_000}) for r in ('swift', 'swift', 'swift', 'arex')]
    await _until(lambda: len(running) == 2)
    await asyncio.sleep(0.05)
    # Two slots: the third Swift job and the AREX job wait.
    assert len(running) == 2 and jobs[2].state == 'queued' and jobs[3].state == 'queued'
    gate.set()
    assert [await job.future for job in jobs] == ['swift', 'swift', 'swift', 'arex']
    assert max(peak) == 2
    # AREX started only after every Swift job finished: one resource at a time.
    assert jobs[3].started_at >= max(job.finished_at for job in jobs[:3])
    await scheduler.close()


@pytest.mark.asyncio
async def test_resources_without_slots_stay_serial(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    running = set(); peak = []
    async def work(job):
        running.add(job.id); peak.append(len(running))
        await asyncio.sleep(0.01); running.discard(job.id)
    jobs = [await scheduler.submit('swift', 1, work) for _ in range(3)]
    for job in jobs: await job.future
    assert max(peak) == 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_waiting_job_for_another_resource_stops_new_admissions(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', aging_seconds=0.1)
    scheduler.resource_slots = {'swift': 4}
    gate = asyncio.Event()
    async def work(job):
        await gate.wait(); return job.resource
    first = await scheduler.submit('swift', 4, work)
    await _until(lambda: first.state == 'running')
    other = await scheduler.submit('arex', 4, work)
    await asyncio.sleep(0.35)  # AREX has aged three steps past fresh Swift work
    late = await scheduler.submit('swift', 4, work)
    await asyncio.sleep(0.03)
    assert late.state == 'queued' and other.state == 'queued'
    gate.set()
    await late.future
    # The aged AREX job ran before the later Swift job.
    assert other.started_at <= late.started_at
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancelling_one_concurrent_job_leaves_the_other_running(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    scheduler.resource_slots = {'swift': 2}
    scheduler.set_kv_capacity('swift', 100_000)
    gate = asyncio.Event()
    async def work(job):
        await gate.wait(); return 'done'
    a = await scheduler.submit('swift', 1, work, {'kv_tokens': 10_000})
    b = await scheduler.submit('swift', 1, work, {'kv_tokens': 10_000})
    await _until(lambda: a.state == 'running' and b.state == 'running')
    assert await scheduler.cancel(a.id)
    with pytest.raises(asyncio.CancelledError):
        await a.future
    assert b.state == 'running'
    gate.set()
    assert await b.future == 'done'
    metrics = await scheduler.metrics()
    assert metrics['running'] == 0
    await scheduler.close()


@pytest.mark.asyncio
async def test_jobs_that_overflow_the_kv_cache_wait_in_order(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    scheduler.resource_slots = {'arex': 4}
    scheduler.set_kv_capacity('arex', 73_000, per_request=1_000)
    gates = {}; running = set(); peak = []
    async def work(job):
        running.add(job.id); peak.append(len(running))
        gates[job.id] = asyncio.Event()
        await gates[job.id].wait(); running.discard(job.id)
    big = [await scheduler.submit('arex', 1, work, {'kv_tokens': 40_000}) for _ in range(2)]
    small = await scheduler.submit('arex', 1, work, {'kv_tokens': 5_000})
    await _until(lambda: big[0].state == 'running')
    await asyncio.sleep(0.05)
    # 41k + 41k > 73k: the second long prompt waits, and the short one behind
    # it does not start in its place.
    assert big[1].state == 'queued' and small.state == 'queued'
    gates[big[0].id].set()
    await _until(lambda: big[1].state == 'running' and small.state == 'running')
    assert max(peak) == 2
    for job in (big[1], small):
        gates[job.id].set()
    await asyncio.gather(*(job.future for job in big + [small]))
    await scheduler.close()


@pytest.mark.asyncio
async def test_unknown_kv_capacity_runs_one_at_a_time_until_reported(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite')
    scheduler.resource_slots = {'arex': 4}
    gate = asyncio.Event()
    async def work(job):
        await gate.wait()
    a = await scheduler.submit('arex', 1, work, {'kv_tokens': 1_000})
    b = await scheduler.submit('arex', 1, work, {'kv_tokens': 1_000})
    await _until(lambda: a.state == 'running')
    await asyncio.sleep(0.05)
    assert b.state == 'queued'
    scheduler.set_kv_capacity('arex', 10_000)
    await _until(lambda: b.state == 'running')
    gate.set()
    await asyncio.gather(a.future, b.future)
    await scheduler.close()


@pytest.mark.asyncio
async def test_queue_view_and_restart_keep_job_origin(tmp_path):
    scheduler = Scheduler(tmp_path / 'jobs.sqlite', shelf_grace_seconds=0.05)
    gate = asyncio.Event()
    async def wait(job):
        await gate.wait()
    loop_origin = {'source': 'loop', 'origin': 'loop', 'origin_label': 'Loop · Daily AI & business trends',
                   'loop_id': 'loop_daily-trends', 'loop_run_id': 'run_1', 'loop_name': 'Daily AI & business trends',
                   'prompt': 'never stored'}
    manual = await scheduler.submit('swift', 1, wait, {'source': '', 'origin': 'manual', 'origin_label': 'Manual'}, lane='manual')
    loop = await scheduler.submit('swift', 1, wait, loop_origin, lane='background')
    listed = {job['id']: job for job in (await scheduler.queues())['swift']}
    # The loop job is held behind the manual chat yet still says which loop it is.
    assert listed[loop.id]['held'] and listed[loop.id]['metadata']['lane'] == 'background'
    assert listed[loop.id]['metadata']['origin_label'] == 'Loop · Daily AI & business trends'
    assert 'prompt' not in listed[loop.id]['metadata']
    assert listed[manual.id]['metadata']['origin_label'] == 'Manual'
    gate.set()
    await asyncio.gather(manual.future, loop.future)
    await scheduler.close()
    restarted = Scheduler(tmp_path / 'jobs.sqlite')
    stored = (await restarted.get(loop.id)).metadata
    assert (stored['origin'], stored['loop_run_id'], stored['loop_name']) == ('loop', 'run_1', 'Daily AI & business trends')
    await restarted.close()
