"""Offline, manager-owned image-to-3D jobs using the local Idol MPS worker."""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .resources import BackendConfig


class IdolLocalError(RuntimeError):
    """The local Idol worker is not ready or could not produce a usable PLY."""


class IdolLocalBackend:
    """Run one local Idol worker under Bubble Router memory admission."""

    RESOURCE = "idol-local"
    MODEL_ID = "speridlabs/idol"
    PROFILE_NAME = "idol-local.json"
    CHECK_TIMEOUT_SECONDS = 30

    def __init__(self, root: str | Path, manager: Any):
        self.root = Path(root).expanduser().resolve()
        self.manager = manager
        self.profile_path = self.root / "config" / self.PROFILE_NAME
        self.output_dir = self.root / "outputs" / "3d"
        self.state_dir = self.root / "state" / "idol-local"

    @classmethod
    def _read_profile(cls, root: str | Path) -> dict[str, Any] | None:
        path = Path(root).expanduser() / "config" / cls.PROFILE_NAME
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(profile, dict):
            return None
        required_paths = ("python", "worker", "cwd", "model_root")
        if any(not isinstance(profile.get(key), str) or not profile[key].strip() for key in required_paths):
            return None
        if profile.get("offline") is not True:
            return None
        timeout = profile.get("timeout_seconds", 1800)
        estimate = profile.get("estimate_gib", 20.0)
        if (isinstance(timeout, bool) or not isinstance(timeout, int) or not 30 <= timeout <= 3600
                or isinstance(estimate, bool) or not isinstance(estimate, (int, float))
                or not math.isfinite(float(estimate)) or float(estimate) != 20.0):
            return None
        return {
            **profile,
            # Preserve the venv entrypoint symlink: resolving it to Homebrew's
            # base interpreter would drop the installed Idol packages.
            "python": os.path.abspath(os.path.expanduser(profile["python"])),
            "worker": str(Path(profile["worker"]).expanduser().resolve()),
            "cwd": str(Path(profile["cwd"]).expanduser().resolve()),
            "model_root": str(Path(profile["model_root"]).expanduser().resolve()),
            "timeout_seconds": timeout,
            "estimate_gib": 20.0,
        }

    @classmethod
    def resource_config(cls, root: str | Path) -> BackendConfig | None:
        """Create the internal, nonpersistent 20 GiB resource from local config."""
        profile = cls._read_profile(root)
        if profile is None:
            return None
        return BackendConfig(
            command=(),
            cwd=profile["cwd"],
            model_paths=(profile["model_root"],),
            estimate_gib=20.0,
            persistent=False,
        )

    def _profile(self) -> dict[str, Any] | None:
        return self._read_profile(self.root)

    async def check_ready(self) -> dict[str, Any]:
        """Check local files and MPS readiness without loading the inference model."""
        profile = self._profile()
        if profile is None:
            return {"ready": False, "backend": "local-mps", "missing": [
                "Local Idol Router profile is missing or invalid (config/idol-local.json)."]}
        profiles = getattr(self.manager, "backends", {})
        if not isinstance(profiles, Mapping) or self.RESOURCE not in profiles:
            return {"ready": False, "backend": "local-mps", "missing": [
                "Local Idol resource is not registered with the Router."]}
        python = Path(profile["python"])
        worker = Path(profile["worker"])
        if not python.is_file() or not os.access(python, os.X_OK):
            return {"ready": False, "backend": "local-mps", "missing": [
                "Configured local Idol Python executable is unavailable."]}
        if not worker.is_file():
            return {"ready": False, "backend": "local-mps", "missing": [
                "Configured local Idol worker is unavailable."]}
        try:
            result = await asyncio.wait_for(
                self.manager.run_owned(self.RESOURCE, [str(python), str(worker), "--check"], cwd=profile["cwd"]),
                timeout=self.CHECK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return {"ready": False, "backend": "local-mps", "missing": [
                "Local Idol readiness check timed out."]}
        except Exception as exc:
            return {"ready": False, "backend": "local-mps", "missing": [
                f"Local Idol readiness check failed ({type(exc).__name__})."]}
        try:
            status = json.loads(result.stdout.decode("utf-8", errors="replace"))
        except (ValueError, AttributeError, TypeError):
            status = {"ready": False, "missing": ["Local Idol worker returned an invalid readiness response."]}
        if not isinstance(status, dict):
            status = {"ready": False, "missing": ["Local Idol worker returned an invalid readiness response."]}
        status["backend"] = "local-mps"
        status["ready"] = bool(status.get("ready")) and result.returncode == 0
        status.setdefault("missing", [])
        if result.returncode and status["ready"]:
            status["ready"] = False
            status["missing"] = ["Local Idol worker readiness check failed."]
        return status

    @staticmethod
    def _checked_artifact(value: Any, output_dir: Path, expected: Path, *, kind: str) -> Path:
        if not isinstance(value, str):
            raise IdolLocalError(f"Local Idol worker returned no {kind} artifact.")
        try:
            resolved = Path(value).resolve(strict=True)
            root = output_dir.resolve(strict=True)
            expected_path = expected.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise IdolLocalError(f"Local Idol worker returned a missing {kind} artifact.") from None
        if (resolved != expected_path or not resolved.is_relative_to(root)
                or not resolved.is_file() or resolved.stat().st_size == 0):
            raise IdolLocalError(f"Local Idol worker returned an invalid {kind} artifact.")
        if kind == "PLY":
            with resolved.open("rb") as handle:
                header = handle.read(64 * 1024)
            if not header.startswith(b"ply\n") or b"end_header\n" not in header:
                raise IdolLocalError("Local Idol output is not a valid PLY file.")
        os.chmod(resolved, 0o600)
        return resolved

    @staticmethod
    def _safe_diagnostic_text(value: Any, limit: int = 2000) -> str:
        if not isinstance(value, str):
            return ""
        detail = " ".join(value.split())
        detail = re.sub(r"https?://\S+", "[url]", detail)
        detail = re.sub(r"(?i)\b(?:hf_[A-Za-z0-9]{10,}|Bearer\s+\S+)\b", "[credential]", detail)
        detail = re.sub(r"(?i)(?:token|api[_-]?key|authorization)\s*[:=]\s*\S+", "[credential]", detail)
        return detail[:limit]

    @staticmethod
    def _safe_encoder_sources(value: Any) -> dict[str, dict[str, str]]:
        if not isinstance(value, Mapping):
            return {}
        safe: dict[str, dict[str, str]] = {}
        for name in ("dinov3", "flux2_vae"):
            entry = value.get(name)
            if not isinstance(entry, Mapping):
                continue
            keep = {
                key: str(entry[key])[:300]
                for key in ("repo", "revision", "loader")
                if isinstance(entry.get(key), str)
            }
            if keep:
                safe[name] = keep
        return safe

    def _save_diagnostic(self, job_id: str, process: Any, worker_result: Mapping[str, Any], error: str) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        path = self.state_dir / f"{job_id}-diagnostic.json"
        temporary = self.state_dir / f".{job_id}-diagnostic.tmp"
        payload = {
            "job_id": job_id,
            "backend": "local-mps",
            "returncode": int(getattr(process, "returncode", 1)),
            "error_type": self._safe_diagnostic_text(worker_result.get("error_type"), 80),
            "error": self._safe_diagnostic_text(worker_result.get("error") or error),
            "stderr": self._safe_diagnostic_text(
                getattr(process, "stderr", b"").decode("utf-8", errors="replace")
                if isinstance(getattr(process, "stderr", b""), bytes) else ""),
        }
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        return path

    async def generate(
        self,
        job_id: str,
        image: str | Path,
        *,
        mode: str = "synthetic",
        steps: int = 50,
        cfg_strength: float = 3.0,
        rescale_t: float = 3.0,
        seed: int = 42,
        num_gaussians: int = 262144,
    ) -> dict[str, Any]:
        profile = self._profile()
        if profile is None:
            raise IdolLocalError("Local Idol Router profile is missing or invalid.")
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise IdolLocalError("Router job identifier is invalid.")
        python = Path(profile["python"])
        worker = Path(profile["worker"])
        if not python.is_file() or not worker.is_file():
            raise IdolLocalError("Configured local Idol worker is unavailable.")

        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        request_path = self.state_dir / f".{job_id}-request.json"
        response_path = self.state_dir / f".{job_id}-response.json"
        output = self.output_dir / f"{job_id}.ply"
        prepared = self.output_dir / f"{job_id}_input.png"
        camera = self.output_dir / f"{job_id}_camera.json"
        payload = {
            "image": str(Path(image).resolve(strict=True)),
            "output": str(output),
            "mode": mode,
            "steps": steps,
            "cfg_strength": cfg_strength,
            "rescale_t": rescale_t,
            "seed": seed,
            "num_gaussians": num_gaussians,
        }
        descriptor = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        descriptor = os.open(response_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        process = subprocess.CompletedProcess([], 1, b"", b"")
        worker_result: dict[str, Any] = {}
        succeeded = False
        try:
            try:
                process = await asyncio.wait_for(
                    self.manager.run_owned(
                        self.RESOURCE,
                        [str(python), str(worker), "--request", str(request_path), "--response", str(response_path)],
                        cwd=profile["cwd"],
                    ),
                    timeout=profile["timeout_seconds"],
                )
            except asyncio.TimeoutError as exc:
                raise IdolLocalError(f"Local Idol worker timed out after {profile['timeout_seconds']} seconds.") from exc
            try:
                decoded = json.loads(response_path.read_text(encoding="utf-8"))
                if isinstance(decoded, dict):
                    worker_result = decoded
            except (OSError, ValueError):
                worker_result = {}
            if process.returncode != 0 or worker_result.get("error"):
                reason = self._safe_diagnostic_text(worker_result.get("error"))
                error_type = worker_result.get("error_type")
                error_name = error_type if isinstance(error_type, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", error_type) else "WorkerError"
                message = f"Local Idol worker failed ({error_name})"
                if reason:
                    message += f": {reason}"
                self._save_diagnostic(job_id, process, worker_result, message)
                raise IdolLocalError(message + ".")
            try:
                artifact = self._checked_artifact(worker_result.get("artifact"), self.output_dir, output, kind="PLY")
                input_artifact = self._checked_artifact(
                    worker_result.get("input_artifact"), self.output_dir, prepared, kind="processed input")
                camera_artifact = self._checked_artifact(
                    worker_result.get("camera_artifact"), self.output_dir, camera, kind="camera")
            except IdolLocalError as exc:
                self._save_diagnostic(job_id, process, worker_result, str(exc))
                raise
            returned: dict[str, Any] = {
                "artifact": str(artifact),
                "content_type": "application/octet-stream",
                "filename": artifact.name,
                "backend": "local-mps",
                "model": self.MODEL_ID,
                "device": "mps",
                "offline": True,
                "mode": mode,
                "steps": steps,
                "cfg_strength": cfg_strength,
                "rescale_t": rescale_t,
                "seed": seed,
                "num_gaussians": num_gaussians,
                "input_artifact": str(input_artifact),
                "camera_artifact": str(camera_artifact),
                "encoder_sources": self._safe_encoder_sources(worker_result.get("encoder_sources")),
            }
            for key in ("load_seconds", "generation_seconds", "total_seconds", "metal_allocated_gib_at_finish"):
                value = worker_result.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
                    returned[key] = float(value)
            succeeded = True
            return returned
        finally:
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
            if not succeeded:
                output.unlink(missing_ok=True)
                prepared.unlink(missing_ok=True)
                camera.unlink(missing_ok=True)
