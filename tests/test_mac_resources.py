import asyncio
import json
import os
import sys

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
async def test_generator_lease_restores_configured_core_servers():
    manager = ResourceManager(
        {
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-image": BackendConfig(persistent=False, estimate_gib=40),
        },
        telemetry=FakeTelemetry(),
    )
    async with manager.lease("qwen-image"):
        assert (await manager.status())["resources"]["qwen-image"]["leased"] == 1
    assert (await manager.status())["resources"]["qwen-audio"]["running"]
    await manager.stop("qwen-audio")


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
async def test_persistent_generator_stops_before_core_restore():
    manager = ResourceManager(
        {
            "qwen-audio": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), core=True),
            "qwen-image": BackendConfig(command=(sys.executable, "-c", "import time; time.sleep(60)"), estimate_gib=40),
        },
        telemetry=FakeTelemetry(),
    )
    await manager.ensure("qwen-audio")
    async with manager.lease("qwen-image"):
        assert (await manager.status())["resources"]["qwen-image"]["running"]
    status = await manager.status()
    assert not status["resources"]["qwen-image"]["running"]
    assert status["resources"]["qwen-audio"]["running"]
    await manager.stop("qwen-audio")


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
async def test_generator_restoration_failure_is_visible_without_failing_artifact():
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
    assert status["core_restore"]["ready"] is False
    assert "qwen-vlm" in status["core_restore"]["errors"]


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
async def test_incremental_eviction_is_followed_by_best_effort_core_restore():
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
    assert status["resources"]["qwen-audio"]["running"]
    assert status["core_restore"]["ready"] is True
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
