import asyncio
import json
import os
import sys
import threading

import pytest

from model_router.mac.resources import (
    BackendConfig,
    MemorySnapshot,
    ResourceAdmissionError,
    ResourceManager,
    reclaimable_available_gib,
)


class FakeTelemetry:
    def __init__(self, available=64.0, pressure="normal", vm_stat=None):
        self.available = available
        self.pressure = pressure
        self.vm_stat = vm_stat or {}

    def snapshot(self):
        return MemorySnapshot(self.available, 0.0, 0.0, self.pressure, self.vm_stat)


class SequenceTelemetry:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            if len(self.snapshots) > 1:
                return self.snapshots.pop(0)
            if self.snapshots:
                return self.snapshots[0]
            raise RuntimeError("telemetry unavailable")


@pytest.mark.asyncio
async def test_manager_starts_and_stops_only_its_owned_process():
    telemetry = FakeTelemetry()
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    status = await manager.status()
    pid = status["resources"]["qwen-vlm"]["pid"]
    assert status["resources"]["qwen-vlm"]["running"]
    await manager.stop("qwen-vlm")
    assert (await manager.status())["resources"]["qwen-vlm"]["pid"] is None
    assert pid is not None


@pytest.mark.asyncio
async def test_low_memory_rejects_without_starting_process():
    telemetry = FakeTelemetry(available=20.0)
    manager = ResourceManager(
        {"qwen-image": BackendConfig(command=(sys.executable, "-c", "pass"), persistent=True)},
        telemetry=telemetry,
    )
    with pytest.raises(ResourceAdmissionError):
        await manager.ensure("qwen-image")
    assert not (await manager.status())["resources"]["qwen-image"]["running"]


@pytest.mark.asyncio
async def test_generator_lease_does_not_load_idle_cores_afterwards():
    manager = ResourceManager(
        {
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-image": BackendConfig(persistent=False, estimate_gib=40),
        },
        telemetry=FakeTelemetry(),
    )
    async with manager.lease("qwen-image"):
        assert (await manager.status())["resources"]["qwen-image"]["leased"] == 1
    status = await manager.status()
    assert not status["resources"]["qwen-audio"]["running"]
    assert status["core_restore"].get("attempted") is not True


@pytest.mark.asyncio
async def test_manager_uses_12_gib_reserve_and_reports_telemetry():
    manager = ResourceManager(
        {"minimax-h3": BackendConfig(persistent=False)},
        telemetry=FakeTelemetry(available=12.0),
    )
    with pytest.raises(ResourceAdmissionError):
        await manager.ensure("minimax-h3")
    assert (await manager.status())["memory"]["reserve_gib"] == 12.0


@pytest.mark.asyncio
async def test_cancelling_a_core_lease_stops_its_owned_server():
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=FakeTelemetry(),
    )

    async def request():
        async with manager.lease("qwen-vlm"):
            await asyncio.sleep(60)

    task = asyncio.create_task(request())
    while not (await manager.status())["resources"]["qwen-vlm"]["running"]:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (await manager.status())["resources"]["qwen-vlm"]["running"]


@pytest.mark.asyncio
async def test_cancelling_one_shared_lease_keeps_the_server_for_the_others():
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=FakeTelemetry(),
    )
    release = asyncio.Event()

    async def request(wait):
        async with manager.lease("qwen-vlm", incremental_gib=0.0):
            await wait()

    first = asyncio.create_task(request(lambda: asyncio.sleep(60)))
    while not (await manager.status())["resources"]["qwen-vlm"]["running"]:
        await asyncio.sleep(0.01)
    second = asyncio.create_task(request(release.wait))
    while manager._leases.get("qwen-vlm", 0) < 2:
        await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await manager.status())["resources"]["qwen-vlm"]["running"]
    release.set()
    await second
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_persistent_generator_stays_loaded_until_memory_is_needed():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=10),
            "qwen-image": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), estimate_gib=40),
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=21),
        },
        telemetry=telemetry,
        reserve_gib=10,
    )
    await manager.ensure("qwen-audio")
    async with manager.lease("qwen-image"):
        pid = (await manager.status())["resources"]["qwen-image"]["pid"]
    status = await manager.status()
    assert status["resources"]["qwen-image"]["running"]
    assert status["resources"]["qwen-audio"]["running"]
    async with manager.lease("qwen-image"):
        assert (await manager.status())["resources"]["qwen-image"]["pid"] == pid
    telemetry.available = 14.0
    original_stop = manager._stop_locked

    async def stop_frees_memory(name):
        await original_stop(name)
        if name == "qwen-image":
            telemetry.available = 54.0

    manager._stop_locked = stop_frees_memory
    async with manager.lease("qwen-vlm"):
        status = await manager.status()
        assert status["resources"]["qwen-vlm"]["running"]
        assert not status["resources"]["qwen-image"]["running"]
        assert status["resources"]["qwen-audio"]["running"]
    for name in ("qwen-vlm", "qwen-audio"):
        await manager.stop(name)


@pytest.mark.asyncio
async def test_manager_does_not_stop_an_unowned_process():
    outsider = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)",
        start_new_session=True,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    manager = ResourceManager({"qwen-vlm": BackendConfig(core=True)}, telemetry=FakeTelemetry())
    await manager.stop("qwen-vlm")
    assert outsider.returncode is None
    outsider.terminate()
    await outsider.wait()


@pytest.mark.asyncio
async def test_ownership_journal_reaps_only_verified_process_group(tmp_path):
    pytest.importorskip("psutil")
    journal = tmp_path / "owned-processes.json"
    config = {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)}
    first = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    await first.ensure("qwen-vlm")
    old_pid = (await first.status())["resources"]["qwen-vlm"]["pid"]
    assert journal.stat().st_mode & 0o777 == 0o600

    second = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    await second.ensure("qwen-vlm")
    new_pid = (await second.status())["resources"]["qwen-vlm"]["pid"]
    assert new_pid != old_pid
    await second.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_one_shot_crash_journal_hashes_argv_and_reaps_owned_group(tmp_path):
    pytest.importorskip("psutil")
    journal = tmp_path / "owned-processes.json"
    prompt = "PRIVATE_PROMPT_MUST_NOT_BE_JOURNALED"
    config = {
        "minimax-h3": BackendConfig(persistent=False, estimate_gib=40),
        "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
    }
    first = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    task = asyncio.create_task(
        first.run_owned("minimax-h3", [sys.executable, "-c", "import time; time.sleep(60)", prompt])
    )
    while not journal.exists():
        await asyncio.sleep(0.01)
    journal_text = journal.read_text()
    assert prompt not in journal_text
    assert "argv_sha256" in journal_text
    record = __import__("json").loads(journal_text)["oneshots"][0]
    old_pid = record["pid"]

    second = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    await second.ensure("qwen-vlm")
    await asyncio.wait_for(task, timeout=5)
    assert not __import__("psutil").pid_exists(old_pid)
    await second.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_one_shot_recovery_leaves_process_alive_when_nonce_does_not_match(tmp_path):
    psutil = pytest.importorskip("psutil")
    outsider = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)",
        start_new_session=True,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        journal = tmp_path / "owned-processes.json"
        argv = [sys.executable, "-c", "import time; time.sleep(60)"]
        journal.write_text(json.dumps({
            "version": 2,
            "persistent": {},
            "oneshots": [{
                "name": "minimax-h3",
                "pid": outsider.pid,
                "create_time": psutil.Process(outsider.pid).create_time(),
                "argv_sha256": ResourceManager._argv_sha256(argv),
                "token": "wrong-owner-token",
            }],
        }))
        config = {
            "minimax-h3": BackendConfig(persistent=False),
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
        }
        manager = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
        await manager.ensure("qwen-vlm")
        assert outsider.returncode is None
        current = json.loads(journal.read_text())
        assert current["oneshots"][0]["token"] == "wrong-owner-token"
        await manager.stop("qwen-vlm")
    finally:
        if outsider.returncode is None:
            outsider.terminate()
            await outsider.wait()


@pytest.mark.asyncio
async def test_recover_owned_orphans_does_not_load_any_resource(tmp_path):
    pytest.importorskip("psutil")
    journal = tmp_path / "owned-processes.json"
    config = {
        "minimax-h3": BackendConfig(persistent=False, estimate_gib=40),
        "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
    }
    first = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    task = asyncio.create_task(
        first.run_owned("minimax-h3", [sys.executable, "-c", "import time; time.sleep(60)"])
    )
    while not journal.exists():
        await asyncio.sleep(0.01)
    pid = json.loads(journal.read_text())["oneshots"][0]["pid"]
    second = ResourceManager(config, telemetry=FakeTelemetry(), ownership_path=journal)
    await second.recover_owned_orphans()
    await asyncio.wait_for(task, timeout=5)
    status = await second.status()
    assert not __import__("psutil").pid_exists(pid)
    assert not any(item["running"] for item in status["resources"].values())


@pytest.mark.asyncio
async def test_cancelling_during_health_wait_stops_starting_server():
    manager = ResourceManager(
        {
            "qwen-vlm": BackendConfig(
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
                health_url="http://127.0.0.1:1/health",
                health_timeout=300,
                core=True,
            )
        },
        telemetry=FakeTelemetry(),
    )
    task = asyncio.create_task(manager.ensure("qwen-vlm"))
    while not (await manager.status())["resources"]["qwen-vlm"]["running"]:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (await manager.status())["resources"]["qwen-vlm"]["running"]


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["available", "pressure", "swap", "telemetry"])
async def test_runtime_guard_stops_only_owned_process_on_memory_trigger(tmp_path, monkeypatch, trigger):
    import model_router.mac.resources as resources

    pytest.importorskip("psutil")
    monkeypatch.setattr(resources, "RUNTIME_GUARD_POLL_SECONDS", 0.01)
    baseline = MemorySnapshot(64.0, 2.0, 0.0, "normal")
    changes = {
        "available": MemorySnapshot(15.9, 2.0, 0.0, "normal"),
        "pressure": MemorySnapshot(64.0, 2.0, 0.0, "warning"),
        "swap": MemorySnapshot(64.0, 3.1, 0.0, "normal"),
    }

    class ErrorAfterBaseline:
        calls = 0
        def snapshot(self):
            self.calls += 1
            if self.calls == 1:
                return baseline
            raise RuntimeError("telemetry fault")

    telemetry = ErrorAfterBaseline() if trigger == "telemetry" else SequenceTelemetry([baseline, changes[trigger]])
    outsider = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)",
        start_new_session=True,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    journal = tmp_path / "owned-processes.json"
    manager = ResourceManager(
        {"minimax-h3": BackendConfig(
            persistent=False,
            runtime_memory_margin_gib=4,
            runtime_swap_growth_gib=1,
        )},
        telemetry=telemetry,
        ownership_path=journal,
    )
    task = asyncio.create_task(manager.run_owned(
        "minimax-h3", [sys.executable, "-c", "import time; time.sleep(60)"]
    ))
    try:
        while not journal.exists():
            await asyncio.sleep(0.005)
        pid = json.loads(journal.read_text())["oneshots"][0]["pid"]
        with pytest.raises(ResourceAdmissionError, match="runtime memory guard"):
            await asyncio.wait_for(task, timeout=5)
        assert not __import__("psutil").pid_exists(pid)
        assert outsider.returncode is None
        assert json.loads(journal.read_text())["oneshots"] == []
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        if outsider.returncode is None:
            outsider.terminate()
            await outsider.wait()


@pytest.mark.asyncio
async def test_runtime_guard_allows_normal_completion_and_drains_output(monkeypatch):
    import model_router.mac.resources as resources

    monkeypatch.setattr(resources, "RUNTIME_GUARD_POLL_SECONDS", 0.01)
    manager = ResourceManager(
        {"minimax-h3": BackendConfig(
            persistent=False,
            runtime_memory_margin_gib=4,
            runtime_swap_growth_gib=1,
        )},
        telemetry=FakeTelemetry(available=64.0),
    )
    result = await asyncio.wait_for(
        manager.run_owned("minimax-h3", [sys.executable, "-c", "print('complete')"]), timeout=5
    )
    assert result.returncode == 0 and result.stdout == b"complete\n"
    assert not (await manager.status())["ownership"]["owned_oneshots"]


@pytest.mark.asyncio
async def test_cancelling_runtime_guard_cleans_owned_child_and_journal(tmp_path, monkeypatch):
    import model_router.mac.resources as resources

    pytest.importorskip("psutil")
    monkeypatch.setattr(resources, "RUNTIME_GUARD_POLL_SECONDS", 0.01)
    journal = tmp_path / "owned-processes.json"
    manager = ResourceManager(
        {"minimax-h3": BackendConfig(
            persistent=False,
            runtime_memory_margin_gib=4,
            runtime_swap_growth_gib=1,
        )},
        telemetry=FakeTelemetry(available=64.0),
        ownership_path=journal,
    )
    task = asyncio.create_task(manager.run_owned(
        "minimax-h3", [sys.executable, "-c", "import time; time.sleep(60)"]
    ))
    while not journal.exists():
        await asyncio.sleep(0.005)
    pid = json.loads(journal.read_text())["oneshots"][0]["pid"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert not __import__("psutil").pid_exists(pid)
    assert json.loads(journal.read_text())["oneshots"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("runtime_memory_margin_gib", 0),
    ("runtime_memory_margin_gib", float("inf")),
    ("runtime_swap_growth_gib", -1),
    ("runtime_swap_growth_gib", True),
])
async def test_runtime_guard_configuration_is_validated_before_spawn(tmp_path, field, value):
    journal = tmp_path / "owned-processes.json"
    manager = ResourceManager(
        {"minimax-h3": BackendConfig(persistent=False, **{field: value})},
        telemetry=FakeTelemetry(),
        ownership_path=journal,
    )
    with pytest.raises(ValueError, match=field):
        await manager.run_owned("minimax-h3", [sys.executable, "-c", "raise SystemExit(0)"])
    assert not journal.exists()


@pytest.mark.asyncio
async def test_existing_core_lease_checks_incremental_headroom_without_killing_it():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    telemetry.available = 15.0  # 12 GiB reserve + default 4 GiB request headroom.
    with pytest.raises(ResourceAdmissionError):
        async with manager.lease("qwen-vlm"):
            pass
    assert (await manager.status())["resources"]["qwen-vlm"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_new_core_load_uses_full_peak_estimate_without_double_counting_incremental():
    telemetry = FakeTelemetry(available=34.0)
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    assert (await manager.status())["resources"]["qwen-vlm"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_existing_core_lease_uses_request_specific_incremental_peak():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    telemetry.available = 20.0
    async with manager.lease("qwen-vlm", incremental_gib=5.0):
        assert (await manager.status())["resources"]["qwen-vlm"]["leased"] == 1
    with pytest.raises(ResourceAdmissionError):
        async with manager.lease("qwen-vlm", incremental_gib=12.0):
            pass
    with pytest.raises(ValueError):
        async with manager.lease("qwen-vlm", incremental_gib=65.0):
            pass
    assert (await manager.status())["resources"]["qwen-vlm"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_cold_start_adds_only_requested_peak_above_profile_floor():
    telemetry = FakeTelemetry(available=37.0)
    manager = ResourceManager(
        {"qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True)},
        telemetry=telemetry,
    )
    async with manager.lease("qwen-vlm", incremental_gib=9.0):
        assert (await manager.status())["resources"]["qwen-vlm"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_generator_artifact_is_unaffected_by_later_memory_drop():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-image": BackendConfig(persistent=False, estimate_gib=40),
        },
        telemetry=telemetry,
    )
    async with manager.lease("qwen-image"):
        telemetry.available = 0.0
        artifact = {"artifact_path": "/tmp/result.png"}
    assert artifact["artifact_path"] == "/tmp/result.png"
    status = await manager.status()
    assert not status["resources"]["qwen-vlm"]["running"]
    assert status["core_restore"].get("attempted") is not True


@pytest.mark.asyncio
async def test_restore_core_reports_partial_readiness_without_evicting_first_core():
    from model_router.mac.resources import ResourceRestoreError

    telemetry = FakeTelemetry(available=39.0)
    manager = ResourceManager(
        {
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
        },
        telemetry=telemetry,
    )
    original_ensure = manager._ensure_locked

    async def ensure_with_lower_headroom(name, *, extra_gib=0.0):
        result = await original_ensure(name, extra_gib=extra_gib)
        if name == "qwen-vlm":
            telemetry.available = 19.0
        return result

    manager._ensure_locked = ensure_with_lower_headroom
    with pytest.raises(ResourceRestoreError):
        await manager.restore_core()
    status = await manager.status()
    assert status["core_restore"]["resources"] == {"qwen-vlm": True, "qwen-audio": False}
    assert set(status["core_restore"]["errors"]) == {"qwen-audio"}
    assert status["resources"]["qwen-vlm"]["running"]
    assert not status["resources"]["qwen-audio"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_evicted_core_reloads_on_its_next_request_not_after_the_lease():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
        },
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    await manager.ensure("qwen-audio")
    telemetry.available = 15.0
    original_stop = manager._stop_locked

    async def stop_audio(name):
        await original_stop(name)
        if name == "qwen-audio":
            telemetry.available = 20.0

    manager._stop_locked = stop_audio
    async with manager.lease("qwen-vlm"):
        status = await manager.status()
        assert status["resources"]["qwen-vlm"]["running"]
        assert not status["resources"]["qwen-audio"]["running"]
    status = await manager.status()
    assert status["resources"]["qwen-vlm"]["running"]
    assert not status["resources"]["qwen-audio"]["running"]
    telemetry.available = 64.0
    async with manager.lease("qwen-audio"):
        assert (await manager.status())["resources"]["qwen-audio"]["running"]
    await manager.stop("qwen-audio")
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_status_includes_vm_stat_owned_orphans_and_resource_counters():
    telemetry = FakeTelemetry(vm_stat={"free_pages": 120, "page_size_bytes": 16384})
    manager = ResourceManager({"qwen-vlm": BackendConfig(core=True)}, telemetry=telemetry)
    manager._unverified_orphans = {"persistent": {"qwen-vlm": {"pid": 22}}, "oneshots": [{"pid": 23}]}
    status = await manager.status()
    assert status["memory"]["vm_stat"]["free_pages"] == 120
    assert status["ownership"] == {
        "unverified_persistent_orphans": 1,
        "unverified_oneshot_orphans": 1,
        "owned_oneshots": 0,
    }
    assert status["metrics"]["failed_admissions"] == 0


@pytest.mark.asyncio
async def test_eviction_ranking_protects_queued_core_and_uses_footprint_reload_cost():
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {
            "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=20, reload_seconds=5),
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=8, reload_seconds=5),
            "qwen-image": BackendConfig(persistent=False, estimate_gib=10),
        },
        telemetry=telemetry,
    )
    await manager.ensure("qwen-vlm")
    await manager.ensure("qwen-audio")
    assert manager._eviction_score("qwen-vlm") > manager._eviction_score("qwen-audio")
    manager.set_queued_resources({"qwen-vlm": 1})
    telemetry.available = 13.0
    async with manager._lock:
        with pytest.raises(ResourceAdmissionError) as error:
            await manager._admit_locked("qwen-image")
    assert error.value.evicted == ["qwen-audio"]
    assert (await manager.status())["metrics"]["evictions"] == 1
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_failed_generator_admission_restores_cores_it_evicted():
    telemetry = FakeTelemetry(available=64.0)
    configs = {
        "qwen-vlm": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=20),
        "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=8),
        "qwen-image": BackendConfig(persistent=False, estimate_gib=40),
    }
    manager = ResourceManager(configs, telemetry=telemetry)
    await manager.ensure("qwen-vlm")
    await manager.ensure("qwen-audio")
    original_stop = manager._stop_locked

    async def stop_and_reclaim(name):
        await original_stop(name)
        telemetry.available += configs[name].estimate(name)

    manager._stop_locked = stop_and_reclaim
    telemetry.available = 30.0
    with pytest.raises(ResourceAdmissionError):
        async with manager.lease("qwen-image", incremental_gib=64.0):
            pass
    status = await manager.status()
    assert status["resources"]["qwen-vlm"]["running"]
    assert status["resources"]["qwen-audio"]["running"]
    assert status["core_restore"]["ready"] is True
    assert status["metrics"]["failed_admissions"] == 1
    assert status["metrics"]["evictions"] == 2
    await manager.stop("qwen-vlm")
    await manager.stop("qwen-audio")


def test_reclaimable_available_counts_active_file_cache():
    # Values observed after an h3.c run: ~40 GiB of file-backed pages stayed
    # active, so free + inactive understated what the kernel could reclaim.
    page = 16384
    vm_stat = {
        "page_size_bytes": page,
        "anonymous_pages": 1_085_267,
        "wired_pages": 203_567,
        "compressor_pages": 219_981,
        "file_backed_pages": 2_613_906,
    }
    available = reclaimable_available_gib(64 * 1024**3, vm_stat)
    held = (1_085_267 + 203_567 + 219_981) * page / 1024**3
    assert available == pytest.approx(64 - held)
    assert 40 < available < 42


def test_reclaimable_available_needs_complete_vm_stat():
    assert reclaimable_available_gib(64 * 1024**3, {"page_size_bytes": 16384, "wired_pages": 1}) is None
    assert reclaimable_available_gib(0, {"page_size_bytes": 16384, "anonymous_pages": 1,
                                         "wired_pages": 1, "compressor_pages": 1}) is None


@pytest.mark.asyncio
async def test_on_demand_core_defers_restore_instead_of_failing():
    telemetry = FakeTelemetry(available=40.0)
    sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
    manager = ResourceManager(
        {
            "ornith": BackendConfig(command=sleeper, core=True, estimate_gib=19),
            "qwen-vlm": BackendConfig(command=sleeper, core=True, restore=False, estimate_gib=21),
        },
        telemetry=telemetry,
    )
    async with manager.lease("qwen-vlm"):
        telemetry.available = 20.0
    await manager.restore_core()
    status = await manager.status()
    assert status["core_restore"]["ready"] is True
    assert set(status["core_restore"]["deferred"]) == {"ornith"}
    assert status["resources"]["qwen-vlm"]["running"]
    assert not status["resources"]["ornith"]["running"]
    await manager.stop("qwen-vlm")


@pytest.mark.asyncio
async def test_on_demand_core_is_not_auto_restored():
    sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
    manager = ResourceManager(
        {
            "ornith": BackendConfig(command=sleeper, core=True, estimate_gib=19),
            "qwen-vlm": BackendConfig(command=sleeper, core=True, restore=False, estimate_gib=21),
        },
        telemetry=FakeTelemetry(available=64.0),
    )
    await manager.restore_core()
    status = await manager.status()
    assert status["resources"]["ornith"]["running"]
    assert not status["resources"]["qwen-vlm"]["running"]
    await manager.stop("ornith")


@pytest.mark.asyncio
async def test_exclusive_group_runs_one_model_at_a_time():
    sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
    manager = ResourceManager(
        {
            "ornith": BackendConfig(command=sleeper, core=True, estimate_gib=19, exclusive_group="chat"),
            "swift": BackendConfig(command=sleeper, core=True, restore=False, estimate_gib=17, exclusive_group="chat"),
            "qwen-audio": BackendConfig(command=sleeper, core=True, estimate_gib=7),
        },
        telemetry=FakeTelemetry(available=64.0),
    )
    async with manager.lease("ornith"):
        pass
    async with manager.lease("qwen-audio"):
        pass
    async with manager.lease("swift"):
        status = await manager.status()
        assert status["resources"]["swift"]["running"]
        assert not status["resources"]["ornith"]["running"]
        assert status["resources"]["qwen-audio"]["running"]

    order = []
    async def hold_swift():
        async with manager.lease("swift"):
            order.append("swift-start"); await asyncio.sleep(0.6); order.append("swift-end")
    async def then_ornith():
        await asyncio.sleep(0.1)
        async with manager.lease("ornith"):
            order.append("ornith")
    await asyncio.gather(hold_swift(), then_ornith())
    assert order == ["swift-start", "swift-end", "ornith"]
    for name in ("ornith", "swift", "qwen-audio"):
        await manager.stop(name)


@pytest.mark.asyncio
async def test_exclusive_switch_waits_for_memory_reclaim():
    sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
    telemetry = FakeTelemetry(available=40.0)
    manager = ResourceManager(
        {"ornith": BackendConfig(command=sleeper, core=True, estimate_gib=19, exclusive_group="chat"),
         "swift": BackendConfig(command=sleeper, core=True, restore=False, estimate_gib=19, exclusive_group="chat")},
        telemetry=telemetry,
    )
    async with manager.lease("ornith", incremental_gib=0):
        pass
    telemetry.available = 20.0          # Ornith still resident
    original_stop = manager._stop_locked
    async def stop_then_reclaim_later(name):
        await original_stop(name)
        async def reclaim():
            await asyncio.sleep(1.0); telemetry.available = 40.0
        asyncio.get_running_loop().create_task(reclaim())
    manager._stop_locked = stop_then_reclaim_later
    async with manager.lease("swift", incremental_gib=0):
        assert (await manager.status())["resources"]["swift"]["running"]
    await manager.stop("swift")


@pytest.mark.asyncio
async def test_lease_on_loaded_model_waits_for_memory_instead_of_failing(monkeypatch):
    from model_router.mac import resources

    monkeypatch.setattr(resources, "ADMISSION_WAIT_SECONDS", 5.0)
    monkeypatch.setattr(resources, "ADMISSION_RETRY_SECONDS", 0.05)
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {"swift": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=19)},
        telemetry=telemetry,
        reserve_gib=10,
    )
    await manager.ensure("swift")
    pid = (await manager.status())["resources"]["swift"]["pid"]
    telemetry.available = 17.0

    async def memory_frees_up():
        await asyncio.sleep(0.3)
        telemetry.available = 20.0

    freer = asyncio.create_task(memory_frees_up())
    async with manager.lease("swift", incremental_gib=8.0):
        status = await manager.status()
        assert status["resources"]["swift"]["pid"] == pid
        assert status["metrics"]["admission_waits"] >= 1
    await freer
    await manager.stop("swift")


@pytest.mark.asyncio
async def test_lease_still_fails_after_the_admission_wait(monkeypatch):
    from model_router.mac import resources

    monkeypatch.setattr(resources, "ADMISSION_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(resources, "ADMISSION_RETRY_SECONDS", 0.05)
    telemetry = FakeTelemetry(available=64.0)
    manager = ResourceManager(
        {"swift": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=19)},
        telemetry=telemetry,
        reserve_gib=10,
    )
    await manager.ensure("swift")
    telemetry.available = 5.0
    with pytest.raises(ResourceAdmissionError):
        async with manager.lease("swift", incremental_gib=8.0):
            pass
    assert (await manager.status())["resources"]["swift"]["running"]
    await manager.stop("swift")


@pytest.mark.asyncio
async def test_lease_unloads_an_idle_model_even_with_queued_work():
    telemetry = FakeTelemetry(available=64.0)
    configs = {
        "swift": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=30),
        "minimax-h3": BackendConfig(persistent=False, estimate_gib=40),
    }
    manager = ResourceManager(configs, telemetry=telemetry, reserve_gib=5)
    await manager.ensure("swift")
    original_stop = manager._stop_locked

    async def stop_and_reclaim(name):
        await original_stop(name)
        telemetry.available += configs[name].estimate(name)

    manager._stop_locked = stop_and_reclaim
    manager.set_queued_resources({"swift": 1})
    telemetry.available = 30.0
    async with manager.lease("minimax-h3"):
        status = await manager.status()
        assert not status["resources"]["swift"]["running"]
        assert status["metrics"]["failed_admissions"] == 0


@pytest.mark.asyncio
async def test_lease_queues_while_another_model_is_in_use_then_unloads_it(monkeypatch):
    from model_router.mac import resources

    monkeypatch.setattr(resources, "ADMISSION_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(resources, "ADMISSION_RETRY_SECONDS", 0.05)
    telemetry = FakeTelemetry(available=64.0)
    configs = {
        "swift": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True, estimate_gib=30),
        "minimax-h3": BackendConfig(persistent=False, estimate_gib=40),
    }
    manager = ResourceManager(configs, telemetry=telemetry, reserve_gib=5)
    await manager.ensure("swift")
    original_stop = manager._stop_locked

    async def stop_and_reclaim(name):
        await original_stop(name)
        telemetry.available += configs[name].estimate(name)

    manager._stop_locked = stop_and_reclaim
    released = asyncio.Event()

    async def chat():
        async with manager.lease("swift"):
            telemetry.available = 30.0
            await asyncio.sleep(0.8)  # longer than the 0.2 s memory wait
        released.set()

    chat_task = asyncio.create_task(chat())
    await asyncio.sleep(0.1)
    async with manager.lease("minimax-h3"):
        assert released.is_set()
        status = await manager.status()
        assert not status["resources"]["swift"]["running"]
        assert status["metrics"]["admission_queued"] >= 1
        assert status["metrics"]["failed_admissions"] == 0
    await chat_task
