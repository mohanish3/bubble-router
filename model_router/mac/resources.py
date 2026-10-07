"""Memory-aware lifecycle management for the macOS model processes.

The manager only signals process groups it created itself. It never discovers
and kills processes by name or PID.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import tempfile
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


GIB = 1024**3
EXCLUSIVE_WAIT_SECONDS = 900.0
RECLAIM_WAIT_SECONDS = 15.0
# A request that does not fit yet waits in line for memory (earlier requests
# finishing, the OS reclaiming pages) instead of failing on the first check.
ADMISSION_WAIT_SECONDS = 120.0
ADMISSION_RETRY_SECONDS = 1.0
RUNTIME_GUARD_POLL_SECONDS = 1.0
DEFAULT_ESTIMATES_GIB = {
    "qwen-vlm": 20.0,
    "qwen-audio": 8.0,
    "qwen-image": 40.0,
    "minimax-h3": 40.0,
}
DEFAULT_INCREMENTAL_GIB = {"qwen-vlm": 4.0, "qwen-audio": 2.0}
MAX_REQUEST_INCREMENTAL_GIB = 64.0


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for one manager-owned backend process.

    ``command`` is the persistent server command. For one-shot generators it
    may be empty; ``run_owned`` then receives the executable in ``args``.
    Model paths are reported for readiness/status and are never downloaded.
    """

    command: Sequence[str] = ()
    health_url: str = ""
    health_headers: Mapping[str, str] | None = None
    cwd: str | Path | None = None
    model_paths: Sequence[str | Path] = ()
    estimate_gib: float | None = None
    incremental_gib: float | None = None
    reload_seconds: float | None = None
    persistent: bool = True
    core: bool = False
    # Core resources are evictable and shared; restore=False keeps one out of
    # automatic core restoration so it only loads when a request needs it.
    restore: bool = True
    # Resources sharing a group never run together: loading one unloads the
    # idle others, and a request waits while another member is busy.
    exclusive_group: str | None = None
    health_timeout: float = 300.0
    env: Mapping[str, str] | None = None
    # Optional in-flight guard for one-shot jobs. The manager watches system
    # availability below reserve+margin, memory-pressure state, and swap
    # growth from the pre-spawn baseline.
    runtime_memory_margin_gib: float | None = None
    runtime_swap_growth_gib: float | None = None

    def estimate(self, name: str) -> float:
        value = self.estimate_gib
        if value is None:
            value = DEFAULT_ESTIMATES_GIB.get(name, 8.0)
        if value <= 0:
            raise ValueError("estimate_gib must be positive")
        return float(value)

    def incremental(self, name: str) -> float:
        value = self.incremental_gib
        if value is None:
            value = DEFAULT_INCREMENTAL_GIB.get(name, 0.0)
        if value < 0:
            raise ValueError("incremental_gib cannot be negative")
        return float(value)


@dataclass(frozen=True)
class MemorySnapshot:
    available_gib: float
    swap_used_gib: float = 0.0
    process_rss_gib: float = 0.0
    pressure: str = "unknown"
    vm_stat: Mapping[str, int | float] = field(default_factory=dict)


def reclaimable_available_gib(total_bytes: int, vm_stat: Mapping[str, int | float]) -> float | None:
    """Memory not held by anonymous, wired, or compressed pages.

    psutil's macOS ``available`` is free + inactive, so file cache that stays on
    the active list (for example, mmap-loaded weights of an exited h3.c run)
    is reported as used even though the kernel can reclaim it.
    """
    page_size = int(vm_stat.get("page_size_bytes", 0))
    keys = ("anonymous_pages", "wired_pages", "compressor_pages")
    if not total_bytes or not page_size or any(key not in vm_stat for key in keys):
        return None
    held = sum(int(vm_stat[key]) for key in keys) * page_size
    return max(0.0, (total_bytes - held) / GIB)


class SystemTelemetry:
    """Best-effort psutil and macOS memory telemetry."""

    def snapshot(self) -> MemorySnapshot:
        available = swap = rss = 0.0
        total_bytes = 0
        try:
            import psutil  # type: ignore[import-not-found]

            vm = psutil.virtual_memory()
            available = vm.available / GIB
            total_bytes = int(vm.total)
            swap = psutil.swap_memory().used / GIB
            for proc in psutil.process_iter(["memory_info"]):
                try:
                    info = proc.info.get("memory_info")
                    if info is not None:
                        rss += info.rss
                except (psutil.Error, OSError):
                    continue
            rss /= GIB
        except Exception:
            # sysconf is a useful fallback on macOS and POSIX systems.
            try:
                available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
            except (ValueError, OSError, AttributeError):
                pass
        pressure = "unknown"
        vm_stat: dict[str, int | float] = {}
        if os.uname().sysname == "Darwin":
            try:
                result = subprocess.run(
                    ["/usr/bin/memory_pressure", "-Q"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                pressure = (result.stdout or result.stderr).strip()[:256] or "unknown"
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                result = subprocess.run(
                    ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=2, check=False
                )
                output = result.stdout or ""
                page_match = re.search(r"page size of (\d+) bytes", output)
                if page_match:
                    vm_stat["page_size_bytes"] = int(page_match.group(1))
                page_size = int(vm_stat.get("page_size_bytes", 0))
                for label, key in (
                    ("Pages free", "free_pages"),
                    ("Pages active", "active_pages"),
                    ("Pages inactive", "inactive_pages"),
                    ("Pages speculative", "speculative_pages"),
                    ("Pages wired down", "wired_pages"),
                    ("Pages occupied by compressor", "compressor_pages"),
                    ("File-backed pages", "file_backed_pages"),
                    ("Anonymous pages", "anonymous_pages"),
                ):
                    match = re.search(rf"^{re.escape(label)}:\s+([\d.]+)", output, re.MULTILINE)
                    if match:
                        pages = int(float(match.group(1).replace(".", "")))
                        vm_stat[key] = pages
                        if page_size:
                            vm_stat[key.replace("_pages", "_gib")] = pages * page_size / GIB
            except (OSError, subprocess.TimeoutExpired):
                pass
            reclaimable = reclaimable_available_gib(total_bytes, vm_stat)
            if reclaimable is not None:
                available = max(available, reclaimable)
        return MemorySnapshot(available, swap, rss, pressure, vm_stat)


class ResourceError(RuntimeError):
    """Base class for resource lifecycle errors."""


class ResourceBusyError(ResourceError):
    """Raised when a running resource cannot safely be replaced."""


class ResourceAdmissionError(ResourceError):
    """Raised when the host cannot preserve the configured memory reserve."""

    def __init__(self, message: str, *, evicted: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.evicted = list(evicted)


class ResourceRestoreError(ResourceError):
    """Raised when one or more configured core services remain cold."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass
class _OwnedProcess:
    process: asyncio.subprocess.Process
    token: str
    started_at: float = field(default_factory=time.time)


@dataclass
class _OwnedOneShot:
    process: asyncio.subprocess.Process
    name: str
    token: str
    argv_sha256: str
    started_at: float = field(default_factory=time.time)


class ResourceManager:
    """Own persistent model servers and grant memory-aware execution leases."""

    def __init__(
        self,
        backends: Mapping[str, BackendConfig],
        *,
        telemetry: Any = None,
        reserve_gib: float = 12.0,
        ownership_path: str | Path | None = None,
    ) -> None:
        if reserve_gib < 0:
            raise ValueError("reserve_gib cannot be negative")
        self.backends = dict(backends)
        self.telemetry = telemetry or SystemTelemetry()
        self.reserve_gib = float(reserve_gib)
        self.ownership_path = Path(ownership_path).expanduser() if ownership_path is not None else None
        self._lock = asyncio.Lock()
        self._processes: dict[str, _OwnedProcess] = {}
        self._leases: dict[str, int] = {name: 0 for name in self.backends}
        self._last_used: dict[str, float] = {}
        self._reload_seconds: dict[str, float] = {
            name: float(config.reload_seconds) if config.reload_seconds is not None else config.estimate(name) * 5.0
            for name, config in self.backends.items()
        }
        self._queued_resources: dict[str, int] = {}
        self._metrics = {
            "loads": 0,
            "unloads": 0,
            "evictions": 0,
            "failed_admissions": 0,
            "pressure_events": 0,
        }
        self._oneshots: dict[int, _OwnedOneShot] = {}
        self._unverified_orphans: dict[str, Any] = {"persistent": {}, "oneshots": []}
        self._failure_times: dict[str, deque[float]] = {name: deque() for name in self.backends}
        self._recovered = False
        self._core_restore: dict[str, Any] = {
            "attempted": False,
            "ready": False,
            "resources": {},
            "errors": {},
            "updated_at": None,
        }

    def _config(self, name: str) -> BackendConfig:
        try:
            return self.backends[name]
        except KeyError as exc:
            raise ResourceError(f"unknown resource: {name}") from exc

    def _snapshot(self) -> MemorySnapshot:
        snapshot = self.telemetry.snapshot()
        if isinstance(snapshot, MemorySnapshot):
            return snapshot
        # Lightweight compatibility for injected dict/object telemetry.
        if isinstance(snapshot, Mapping):
            return MemorySnapshot(
                float(snapshot.get("available_gib", snapshot.get("available", 0.0))),
                float(snapshot.get("swap_used_gib", snapshot.get("swap_used", 0.0))),
                float(snapshot.get("process_rss_gib", snapshot.get("process_rss", 0.0))),
                str(snapshot.get("pressure", "unknown")),
                dict(snapshot.get("vm_stat", {})) if isinstance(snapshot.get("vm_stat", {}), Mapping) else {},
            )
        return MemorySnapshot(
            float(getattr(snapshot, "available_gib", 0.0)),
            float(getattr(snapshot, "swap_used_gib", 0.0)),
            float(getattr(snapshot, "process_rss_gib", 0.0)),
            str(getattr(snapshot, "pressure", "unknown")),
            dict(getattr(snapshot, "vm_stat", {})) if isinstance(getattr(snapshot, "vm_stat", {}), Mapping) else {},
        )

    def set_queued_resources(self, counts: Mapping[str, int]) -> None:
        """Publish pending/active scheduler demand for safe eviction ranking."""
        self._queued_resources = {
            str(name): max(0, int(count)) for name, count in counts.items() if int(count) > 0
        }

    def _eviction_score(self, name: str) -> tuple[float, float, float]:
        """Prefer freeing larger, older, cheap-to-reload resources."""
        config = self.backends[name]
        idle_seconds = max(0.0, time.monotonic() - self._last_used.get(name, 0.0))
        footprint_benefit = config.estimate(name)
        reload_cost = self._reload_seconds.get(name, 0.0)
        # Higher score is the better eviction candidate; queue-relevant
        # resources are excluded before scoring.
        return (footprint_benefit - reload_cost * 0.1, idle_seconds, -reload_cost)

    @staticmethod
    def _pressure_is_high(pressure: str) -> bool:
        value = pressure.strip().lower()
        return (
            "critical" in value
            or value in {"warn", "warning", "high", "red", "yellow"}
            or "memory pressure: warning" in value
            or "memory pressure: high" in value
        )

    async def _health_ready(self, url: str, headers: Mapping[str, str] | None = None) -> bool:
        if not url:
            return True

        def probe() -> bool:
            try:
                request = urllib.request.Request(url, headers=dict(headers or {}))
                with urllib.request.urlopen(request, timeout=2) as response:
                    return 200 <= response.status < 300
            except Exception:
                return False

        return await asyncio.to_thread(probe)

    async def _admit_locked(self, name: str, *, incremental_gib: float | None = None) -> list[str]:
        config = self._config(name)
        # Available memory already accounts for resident server RSS. Evicting
        # idle manager-owned servers is the only adjustment made here: models
        # stay loaded after use and are unloaded only when another request
        # needs their memory.
        additional = config.estimate(name) if incremental_gib is None else float(incremental_gib)
        needed = additional + self.reserve_gib
        snapshot = self._snapshot()
        pressure_bad = self._pressure_is_high(snapshot.pressure)
        if pressure_bad:
            self._metrics["pressure_events"] += 1
        evicted: list[str] = []
        while snapshot.available_gib < needed or pressure_bad:
            candidates = [
                key for key, proc in self._processes.items()
                if key != name and self._leases.get(key, 0) == 0
                and proc.process.returncode is None
                and self._queued_resources.get(key, 0) == 0
            ]
            if not candidates:
                self._metrics["failed_admissions"] += 1
                detail = f"available={snapshot.available_gib:.1f}GiB needed={needed:.1f}GiB"
                if pressure_bad:
                    detail += f" pressure={snapshot.pressure[:80]}"
                raise ResourceAdmissionError(f"cannot admit {name}: {detail}", evicted=evicted)
            candidate = max(candidates, key=self._eviction_score)
            await self._stop_locked(candidate)
            self._metrics["evictions"] += 1
            evicted.append(candidate)
            await asyncio.sleep(0.15)
            snapshot = self._snapshot()
            pressure_bad = self._pressure_is_high(snapshot.pressure)
        return evicted

    def _write_journal(self) -> None:
        if self.ownership_path is None:
            return
        persistent_records: dict[str, dict[str, Any]] = dict(self._unverified_orphans.get("persistent", {}))
        oneshot_records: list[dict[str, Any]] = list(self._unverified_orphans.get("oneshots", []))
        try:
            import psutil  # type: ignore[import-not-found]

            for name, owned in self._processes.items():
                if owned.process.returncode is None:
                    try:
                        created = psutil.Process(owned.process.pid).create_time()
                    except (psutil.Error, OSError):
                        continue
                    persistent_records[name] = {
                        "pid": owned.process.pid,
                        "create_time": created,
                        "command": list(self.backends[name].command),
                        "token": owned.token,
                    }
            for name, record in self._unverified_orphans.get("persistent", {}).items():
                persistent_records.setdefault(name, record)
            for pid, owned in self._oneshots.items():
                if owned.process.returncode is not None:
                    continue
                try:
                    created = psutil.Process(pid).create_time()
                except (psutil.Error, OSError):
                    continue
                oneshot_records.append({
                    "name": owned.name,
                    "pid": pid,
                    "create_time": created,
                    "argv_sha256": owned.argv_sha256,
                    "token": owned.token,
                })
        except ImportError:
            return
        path = self.ownership_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"version": 2, "persistent": persistent_records, "oneshots": oneshot_records},
                    handle,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            os.chmod(path, 0o600)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)

    async def _recover_orphans(self) -> None:
        if self._recovered:
            return
        self._recovered = True
        path = self.ownership_path
        if path is None or not path.exists():
            return
        try:
            journal = json.loads(path.read_text(encoding="utf-8"))
            import psutil  # type: ignore[import-not-found]
        except (OSError, ValueError, ImportError):
            return
        if isinstance(journal, dict) and journal.get("version") == 2:
            records = journal.get("persistent", {})
            one_shots = journal.get("oneshots", [])
        else:
            # Read the earlier persistent-only format written by the initial
            # manager implementation.
            records = journal if isinstance(journal, dict) else {}
            one_shots = []
        unverified_persistent: dict[str, Any] = {}
        unverified_oneshots: list[dict[str, Any]] = []
        for name, record in records.items():
            # Require every independent ownership marker before signaling. A
            # PID alone is never enough to claim a process.
            if name not in self.backends or not isinstance(record, dict):
                continue
            try:
                pid = int(record["pid"])
                token = str(record["token"])
                expected_cmd = list(record["command"])
                expected_created = float(record["create_time"])
                if expected_cmd != list(self.backends[name].command):
                    continue
                proc = psutil.Process(pid)
                if abs(proc.create_time() - expected_created) > 0.05:
                    continue
                if proc.cmdline() != expected_cmd:
                    continue
                if proc.environ().get("AI_STACK_OWNERSHIP_TOKEN") != token:
                    continue
                if os.getpgid(pid) != pid:
                    continue
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5.0
                while proc.is_running() and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                if proc.is_running():
                    os.killpg(pid, signal.SIGKILL)
            except (KeyError, ValueError, OSError, psutil.Error, ProcessLookupError):
                unverified_persistent[name] = record
                continue
        for record in one_shots:
            if not isinstance(record, dict):
                continue
            try:
                name = str(record["name"])
                if name not in self.backends:
                    unverified_oneshots.append(record)
                    continue
                pid = int(record["pid"])
                token = str(record["token"])
                expected_created = float(record["create_time"])
                expected_hash = str(record["argv_sha256"])
                proc = psutil.Process(pid)
                actual_hash = self._argv_sha256(proc.cmdline())
                if abs(proc.create_time() - expected_created) > 0.05:
                    continue
                if actual_hash != expected_hash:
                    unverified_oneshots.append(record)
                    continue
                if proc.environ().get("AI_STACK_OWNERSHIP_TOKEN") != token:
                    unverified_oneshots.append(record)
                    continue
                if os.getpgid(pid) != pid:
                    unverified_oneshots.append(record)
                    continue
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5.0
                while proc.is_running() and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                if proc.is_running():
                    os.killpg(pid, signal.SIGKILL)
            except (KeyError, ValueError, OSError, psutil.Error, ProcessLookupError):
                unverified_oneshots.append(record)
        # Verified stale ownership records must not be reused after recovery.
        self._unverified_orphans = {"persistent": unverified_persistent, "oneshots": unverified_oneshots}
        self._write_journal()

    @staticmethod
    def _argv_sha256(argv: Sequence[str]) -> str:
        encoded = json.dumps(list(map(str, argv)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _ensure_locked(self, name: str, *, extra_gib: float = 0.0) -> list[str]:
        await self._recover_orphans()
        config = self._config(name)
        existing = self._processes.get(name)
        if existing is not None and existing.process.returncode is None:
            self._last_used[name] = time.monotonic()
            return []
        if not config.persistent:
            return await self._admit_locked(name, incremental_gib=config.estimate(name) + extra_gib)
        if not config.command:
            raise ResourceError(f"persistent resource {name} has no command")
        group = [
            other for other, other_config in self.backends.items()
            if config.exclusive_group and other != name and other_config.exclusive_group == config.exclusive_group
            and (owned := self._processes.get(other)) is not None and owned.process.returncode is None
        ]
        busy = [other for other in group if self._leases.get(other, 0)]
        if busy:
            raise ResourceAdmissionError(f"cannot admit {name}: exclusive group member {busy[0]} is in use")
        for other in group:
            await self._stop_locked(other)
            self._metrics["evictions"] += 1
        if group:
            # Unified memory returns to the OS shortly after the process exits;
            # wait for it rather than refusing admission on a stale reading.
            needed = config.estimate(name) + extra_gib + self.reserve_gib
            deadline = time.monotonic() + RECLAIM_WAIT_SECONDS
            while self._snapshot().available_gib < needed and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
        failures = self._failure_times.setdefault(name, deque())
        now = time.monotonic()
        while failures and now - failures[0] >= 60.0:
            failures.popleft()
        if len(failures) >= 3:
            raise ResourceError(f"{name} startup is backed off after three failures in 60 seconds")
        evicted = await self._admit_locked(name, incremental_gib=config.estimate(name) + extra_gib)
        env = os.environ.copy()
        if config.env:
            env.update({str(key): str(value) for key, value in config.env.items()})
        token = secrets.token_hex(32)
        env["AI_STACK_OWNERSHIP_TOKEN"] = token
        startup_started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *config.command,
                cwd=str(config.cwd) if config.cwd is not None else None,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            failures.append(time.monotonic())
            raise
        self._processes[name] = _OwnedProcess(process, token)
        self._write_journal()
        deadline = time.monotonic() + max(0.0, min(config.health_timeout, 300.0))
        try:
            while True:
                if process.returncode is not None:
                    self._processes.pop(name, None)
                    failures.append(time.monotonic())
                    self._write_journal()
                    raise ResourceError(f"{name} exited before becoming healthy (exit {process.returncode})")
                if await self._health_ready(config.health_url, config.health_headers):
                    self._last_used[name] = time.monotonic()
                    self._reload_seconds[name] = max(0.0, time.monotonic() - startup_started)
                    self._metrics["loads"] += 1
                    return evicted
                if time.monotonic() >= deadline:
                    await self._stop_locked(name)
                    failures.append(time.monotonic())
                    raise ResourceError(f"{name} did not become healthy within {config.health_timeout:.0f}s")
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            await self._stop_locked(name)
            raise

    def is_loaded(self, name: str) -> bool:
        owned = self._processes.get(name)
        return owned is not None and owned.process.returncode is None

    async def ensure(self, name: str) -> None:
        async with self._lock:
            await self._ensure_locked(name)

    async def recover_owned_orphans(self) -> None:
        """Clean verified journaled process groups without loading resources."""
        async with self._lock:
            await self._recover_orphans()

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        # The process group was created by this manager with start_new_session.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def _stop_locked(self, name: str) -> None:
        owned = self._processes.get(name)
        if owned is None:
            return
        if self._leases.get(name, 0):
            raise ResourceBusyError(f"resource {name} has an active lease")
        await self._terminate(owned.process)
        self._processes.pop(name, None)
        self._metrics["unloads"] += 1
        self._write_journal()

    async def stop(self, name: str) -> None:
        self._config(name)
        async with self._lock:
            await self._stop_locked(name)

    async def restore_core(self) -> None:
        """Best-effort restore core servers and report each final readiness state."""
        async with self._lock:
            names = [
                name for name, config in self.backends.items()
                if config.core and config.persistent and config.restore
            ]
            running = {
                name for name in names
                if (owned := self._processes.get(name)) is not None and owned.process.returncode is None
            }
            missing = [name for name in names if name not in running]
            estimated_sum = sum(self.backends[name].estimate(name) for name in missing)
            memory = self._snapshot()
            errors: dict[str, str] = {}
            deferred: dict[str, str] = {}
            self._core_restore = {
                "attempted": True,
                "ready": False,
                "resources": {name: name in running for name in names},
                "errors": errors,
                "deferred": deferred,
                "available_gib": memory.available_gib,
                "estimated_missing_peak_sum_gib": estimated_sum,
                "reserve_gib": self.reserve_gib,
                "updated_at": time.time(),
            }
            protected: list[str] = []
            try:
                # Restore one profile at a time against actual available memory.
                # Hold temporary leases on already-ready cores so a later load
                # cannot evict them when its own peak does not fit.
                # Warm on-demand cores are protected too: restoring the daily
                # driver must not evict a model the user is actively using.
                on_demand = {
                    name for name, config in self.backends.items()
                    if config.core and not config.restore
                    and (owned := self._processes.get(name)) is not None
                    and owned.process.returncode is None
                }
                for name in running | on_demand:
                    self._leases[name] = self._leases.get(name, 0) + 1
                    protected.append(name)
                for name in names:
                    if name in running:
                        continue
                    try:
                        await self._ensure_locked(name)
                    except ResourceAdmissionError as exc:
                        # An on-demand core that is still warm holds the memory;
                        # this resource loads on its next request instead.
                        on_demand_warm = any(
                            not config.restore
                            and (owned := self._processes.get(other)) is not None
                            and owned.process.returncode is None
                            for other, config in self.backends.items() if config.core
                        )
                        if on_demand_warm:
                            deferred[name] = str(exc)
                        else:
                            errors[name] = f"{type(exc).__name__}: {exc}"
                        continue
                    except Exception as exc:
                        errors[name] = f"{type(exc).__name__}: {exc}"
                        continue
                    owned = self._processes.get(name)
                    if owned is not None and owned.process.returncode is None:
                        self._leases[name] = self._leases.get(name, 0) + 1
                        protected.append(name)
            finally:
                for name in protected:
                    self._leases[name] = max(0, self._leases.get(name, 1) - 1)
            final_running = {
                name: bool(
                    (owned := self._processes.get(name)) is not None
                    and owned.process.returncode is None
                )
                for name in names
            }
            for name, ready in final_running.items():
                if not ready and name not in errors and name not in deferred:
                    errors[name] = "not running after restore attempt"
            self._core_restore = {
                "attempted": True,
                "ready": not errors,
                "resources": final_running,
                "errors": errors,
                "deferred": deferred,
                "available_gib": self._snapshot().available_gib,
                "estimated_missing_peak_sum_gib": estimated_sum,
                "reserve_gib": self.reserve_gib,
                "updated_at": time.time(),
            }
            if errors:
                raise ResourceRestoreError("one or more core resources remain unavailable", self._core_restore)

    async def status(self) -> dict[str, Any]:
        memory = self._snapshot()
        resources: dict[str, Any] = {}
        for name, config in self.backends.items():
            owned = self._processes.get(name)
            running = bool(owned and owned.process.returncode is None)
            rss = None
            if running and owned is not None:
                try:
                    import psutil  # type: ignore[import-not-found]

                    rss = psutil.Process(owned.process.pid).memory_info().rss / GIB
                except Exception:
                    pass
            paths = [str(path) for path in config.model_paths]
            resources[name] = {
                "running": running,
                "pid": owned.process.pid if running and owned else None,
                "leased": self._leases.get(name, 0),
                "estimate_gib": config.estimate(name),
                "incremental_gib": config.incremental(name),
                "reload_seconds": self._reload_seconds.get(name),
                "process_rss_gib": rss,
                "core": config.core,
                "persistent": config.persistent,
                "model_paths": paths,
                "missing_model_paths": [path for path in paths if not Path(path).exists()],
            }
        return {
            "memory": {
                "available_gib": memory.available_gib,
                "swap_used_gib": memory.swap_used_gib,
                "process_rss_gib": memory.process_rss_gib,
                "pressure": memory.pressure,
                "vm_stat": dict(memory.vm_stat),
                "reserve_gib": self.reserve_gib,
            },
            "resources": resources,
            "core_restore": dict(self._core_restore),
            "metrics": dict(self._metrics),
            "ownership": {
                "unverified_persistent_orphans": len(self._unverified_orphans.get("persistent", {})),
                "unverified_oneshot_orphans": len(self._unverified_orphans.get("oneshots", [])),
                "owned_oneshots": sum(1 for item in self._oneshots.values() if item.process.returncode is None),
            },
        }

    @contextlib.asynccontextmanager
    async def lease(self, name: str, *, incremental_gib: float | None = None):
        evicted: list[str] = []
        config = self._config(name)
        if config.exclusive_group:
            # Wait (without holding the manager lock) while another member of
            # the exclusive group is serving a request.
            deadline = time.monotonic() + EXCLUSIVE_WAIT_SECONDS
            while True:
                async with self._lock:
                    busy = any(
                        self._leases.get(other, 0)
                        for other, other_config in self.backends.items()
                        if other != name and other_config.exclusive_group == config.exclusive_group
                    )
                if not busy or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.25)
        admission_deadline = time.monotonic() + ADMISSION_WAIT_SECONDS
        while True:
            try:
                await self._lease_admit_once(name, config, incremental_gib)
                break
            except ResourceAdmissionError as admission_error:
                evicted.extend(getattr(admission_error, "evicted", ()))
                if time.monotonic() < admission_deadline:
                    self._metrics["admission_waits"] = self._metrics.get("admission_waits", 0) + 1
                    await asyncio.sleep(ADMISSION_RETRY_SECONDS)
                    continue
                if evicted:
                    await self._restore_after_eviction()
                raise
        try:
            yield self
        except asyncio.CancelledError:
            raise
        finally:
            async with self._lock:
                self._leases[name] = max(0, self._leases.get(name, 1) - 1)
                self._last_used[name] = time.monotonic()
                if asyncio.current_task() and asyncio.current_task().cancelling() and config.persistent:
                    await self._stop_locked(name)
            # The resource stays loaded for the next request of the same kind;
            # evicted cores reload on demand rather than being restored here,
            # which used to bounce chat models back and forth.

    async def _lease_admit_once(self, name: str, config: BackendConfig, incremental_gib: float | None) -> None:
        async with self._lock:
            if incremental_gib is not None and (
                not math.isfinite(float(incremental_gib))
                or float(incremental_gib) < 0
                or float(incremental_gib) > MAX_REQUEST_INCREMENTAL_GIB
            ):
                raise ValueError(f"incremental_gib must be between 0 and {MAX_REQUEST_INCREMENTAL_GIB:g}")
            requested = config.incremental(name) if incremental_gib is None else float(incremental_gib)
            incremental = max(config.incremental(name), requested)
            if not config.core and any(
                self._leases.get(other, 0) for other, item in self.backends.items() if not item.core
            ):
                raise ResourceBusyError("another generator resource is active")
            existing = self._processes.get(name)
            if existing is not None and existing.process.returncode is None:
                await self._admit_locked(name, incremental_gib=incremental)
            else:
                extra_gib = max(0.0, incremental - config.incremental(name))
                await self._ensure_locked(name, extra_gib=extra_gib)
            self._leases[name] = self._leases.get(name, 0) + 1
            self._last_used[name] = time.monotonic()

    async def _restore_after_eviction(self) -> None:
        try:
            await self.restore_core()
        except Exception as exc:
            # Preserve the original admission/inference outcome and publish
            # restoration failure separately for the operator.
            if not isinstance(exc, ResourceRestoreError):
                state = dict(self._core_restore)
                errors = dict(state.get("errors", {}))
                errors["restore_core"] = f"{type(exc).__name__}: {exc}"
                state.update({"attempted": True, "ready": False, "errors": errors, "updated_at": time.time()})
                self._core_restore = state

    async def run_owned(
        self,
        name: str,
        args: Sequence[str],
        *,
        cwd: str | Path | None = None,
        input_data: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a manager-owned one-shot process and return captured stdout."""
        config = self._config(name)
        command = [*config.command, *map(str, args)] if config.command else list(map(str, args))
        if not command:
            raise ResourceError(f"no command supplied for {name}")
        guard = self._runtime_guard_settings(config, name)
        baseline = self._runtime_guard_snapshot(name) if guard is not None else None
        if guard is not None and baseline is not None:
            violation = self._runtime_guard_violation(name, baseline, baseline, *guard)
            if violation:
                raise ResourceAdmissionError(violation)
        env = os.environ.copy()
        if config.env:
            env.update({str(key): str(value) for key, value in config.env.items()})
        token = secrets.token_hex(32)
        env["AI_STACK_OWNERSHIP_TOKEN"] = token
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd if cwd is not None else config.cwd) if (cwd is not None or config.cwd is not None) else None,
            env=env,
            stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._oneshots[process.pid] = _OwnedOneShot(
            process=process,
            name=name,
            token=token,
            argv_sha256=self._argv_sha256(command),
        )
        self._write_journal()
        try:
            if guard is not None and baseline is not None:
                communication = asyncio.create_task(process.communicate(input_data))
                await self._communicate_with_runtime_guard(
                    name, process, communication, baseline, *guard
                )
                out, err = await communication
            else:
                out, err = await process.communicate(input_data)
            return subprocess.CompletedProcess(command, int(process.returncode or 0), out, err or b"")
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        finally:
            self._oneshots.pop(process.pid, None)
            self._write_journal()

    def _runtime_guard_settings(self, config: BackendConfig, name: str) -> tuple[float | None, float | None] | None:
        margin = config.runtime_memory_margin_gib
        swap_growth = config.runtime_swap_growth_gib
        if margin is None and swap_growth is None:
            return None
        for field_name, value in (
            ("runtime_memory_margin_gib", margin),
            ("runtime_swap_growth_gib", swap_growth),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) <= 0
            ):
                raise ValueError(f"{field_name} for {name} must be finite and positive")
        return (float(margin) if margin is not None else None,
                float(swap_growth) if swap_growth is not None else None)

    def _runtime_guard_snapshot(self, name: str) -> MemorySnapshot:
        try:
            snapshot = self._snapshot()
            if (not math.isfinite(snapshot.available_gib) or snapshot.available_gib < 0
                    or not math.isfinite(snapshot.swap_used_gib) or snapshot.swap_used_gib < 0):
                raise ValueError("telemetry returned invalid memory counters")
            return snapshot
        except Exception as exc:
            raise ResourceAdmissionError(
                f"runtime memory guard telemetry failed for {name}: {type(exc).__name__}"
            ) from exc

    async def _watch_runtime_memory(
        self,
        name: str,
        process: asyncio.subprocess.Process,
        baseline: MemorySnapshot,
        margin_gib: float | None,
        swap_growth_gib: float | None,
    ) -> None:
        while process.returncode is None:
            await asyncio.sleep(RUNTIME_GUARD_POLL_SECONDS)
            if process.returncode is not None:
                return
            snapshot = await asyncio.to_thread(self._runtime_guard_snapshot, name)
            if process.returncode is not None:
                return
            violation = self._runtime_guard_violation(
                name, snapshot, baseline, margin_gib, swap_growth_gib
            )
            if violation:
                raise ResourceAdmissionError(violation)

    def _runtime_guard_violation(
        self,
        name: str,
        snapshot: MemorySnapshot,
        baseline: MemorySnapshot,
        margin_gib: float | None,
        swap_growth_gib: float | None,
    ) -> str | None:
        floor = self.reserve_gib + (margin_gib or 0.0)
        if margin_gib is not None and snapshot.available_gib < floor:
            return (f"runtime memory guard stopped {name}: available={snapshot.available_gib:.1f}GiB "
                    f"below reserve+margin={floor:.1f}GiB")
        if self._pressure_is_high(snapshot.pressure):
            return f"runtime memory guard stopped {name}: memory pressure warning/high"
        if (swap_growth_gib is not None
                and snapshot.swap_used_gib - baseline.swap_used_gib > swap_growth_gib):
            return f"runtime memory guard stopped {name}: swap growth exceeded {swap_growth_gib:.1f}GiB"
        return None

    async def _communicate_with_runtime_guard(
        self,
        name: str,
        process: asyncio.subprocess.Process,
        communication: asyncio.Task[tuple[bytes, bytes]],
        baseline: MemorySnapshot,
        margin_gib: float | None,
        swap_growth_gib: float | None,
    ) -> None:
        monitor = asyncio.create_task(self._watch_runtime_memory(
            name, process, baseline, margin_gib, swap_growth_gib
        ))
        try:
            done, _ = await asyncio.wait(
                (communication, monitor), return_when=asyncio.FIRST_COMPLETED
            )
            if monitor in done:
                try:
                    monitor.result()
                except ResourceAdmissionError:
                    await self._terminate(process)
                    await communication
                    raise
                except asyncio.CancelledError as exc:
                    await self._terminate(process)
                    await communication
                    raise ResourceAdmissionError(
                        f"runtime memory guard monitor was cancelled for {name}"
                    ) from exc
                except Exception as exc:
                    await self._terminate(process)
                    await communication
                    raise ResourceAdmissionError(
                        f"runtime memory guard monitor failed for {name}: {type(exc).__name__}"
                    ) from exc
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
            await communication
        except asyncio.CancelledError:
            monitor.cancel()
            await self._terminate(process)
            with contextlib.suppress(BaseException):
                await asyncio.shield(communication)
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
            raise

    async def stop_owned_oneshots(self) -> None:
        """Stop only one-shot subprocess groups this manager currently owns."""
        for owned in list(self._oneshots.values()):
            await self._terminate(owned.process)
