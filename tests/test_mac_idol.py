import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from test_mac_app import AUTH, running
from model_router.mac.idol import IdolLocalBackend


def _wait_for_job(client, job_id, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError("Idol job did not reach a terminal state")


def _configure_local_worker(tmp_path, manager, *, ready=True, block=False, fail=False):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    worker = tmp_path / "local_idol_worker.py"
    worker.write_text("# fake worker used only by the Router contract test\n")
    model_root = tmp_path / "models" / "idol-local"
    model_root.mkdir(parents=True, exist_ok=True)
    profile = {
        "python": sys.executable,
        "worker": str(worker),
        "cwd": str(tmp_path),
        "model_root": str(model_root),
        "offline": True,
        "timeout_seconds": 1800,
        "estimate_gib": 20.0,
    }
    (config_dir / "idol-local.json").write_text(json.dumps(profile))
    manager.backends[IdolLocalBackend.RESOURCE] = IdolLocalBackend.resource_config(tmp_path)
    assert manager.backends[IdolLocalBackend.RESOURCE].estimate_gib == 20.0
    assert manager.backends[IdolLocalBackend.RESOURCE].persistent is False
    calls = []
    started = threading.Event()
    cancelled = threading.Event()

    async def run_owned(name, args, *, cwd=None):
        calls.append((name, list(args), cwd))
        if args[-1] == "--check":
            response = {"ready": ready, "backend": "local-mps", "missing": [] if ready else ["encoder unavailable"]}
            return subprocess.CompletedProcess(args, 0 if ready else 3, json.dumps(response).encode(), b"")
        request_path = Path(args[args.index("--request") + 1])
        response_path = Path(args[args.index("--response") + 1])
        request = json.loads(request_path.read_text())
        started.set()
        if block:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        output = Path(request["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if fail:
            response_path.write_text(json.dumps({
                "error_type": "MPSAllocatorError",
                "error": "MPS SDPA allocation failed; Authorization: Bearer hf_1234567890secret",
                "backend": "local-mps",
            }))
            return subprocess.CompletedProcess(args, 1, b"", b"MPS failure\n")
        prepared = output.with_name(output.stem + "_input.png")
        camera = output.with_name(output.stem + "_camera.json")
        output.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        prepared.write_bytes(b"processed input")
        camera.write_text("{\"camera\":true}")
        response_path.write_text(json.dumps({
            "artifact": str(output), "input_artifact": str(prepared), "camera_artifact": str(camera),
            "encoder_sources": {
                "dinov3": {"repo": "timm/dinov3-hplus", "revision": "a" * 40, "loader": "timm", "snapshot": "/private/path"},
                "flux2_vae": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "b" * 40, "loader": "diffusers", "snapshot": "/private/path"},
            },
            "backend": "local-mps", "device": "mps", "offline": True,
            "load_seconds": 12.5, "generation_seconds": 82.0, "total_seconds": 94.5,
            "metal_allocated_gib_at_finish": 16.25,
        }))
        return subprocess.CompletedProcess(args, 0, b"", b"")

    manager.run_owned = run_owned
    return calls, started, cancelled



def test_idol_profile_preserves_venv_python_symlink(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    venv_bin = tmp_path / ".venv-idol" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(sys.executable)
    worker = tmp_path / "worker.py"
    worker.write_text("# worker\n")
    model_root = tmp_path / "models"
    model_root.mkdir()
    (config / "idol-local.json").write_text(json.dumps({
        "python": str(python), "worker": str(worker), "cwd": str(tmp_path),
        "model_root": str(model_root), "offline": True,
        "timeout_seconds": 1800, "estimate_gib": 20.0,
    }))

    profile = IdolLocalBackend._read_profile(tmp_path)
    assert profile["python"] == str(python)
    assert Path(profile["python"]).is_file()


def test_idol_is_advertised_as_local_mps(running):
    client, *_ = running
    response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    idol = next(item for item in response.json()["data"] if item["id"] == "speridlabs/idol")
    assert idol["owned_by"] == "local-mps"


def test_idol_rejects_missing_and_unapproved_images_before_preflight(running, tmp_path):
    client, app, manager, *_ = running
    missing = client.post("/v1/3d/generations", headers=AUTH, json={})
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    unapproved = client.post("/v1/3d/generations", headers=AUTH, json={"image": str(outside)})
    assert missing.status_code == 422
    assert "image is required" in missing.json()["error"]
    assert unapproved.status_code == 422
    assert manager.lease_calls == []
    assert not hasattr(manager, "run_owned")


def test_idol_preflight_returns_503_without_queue_when_encoders_are_missing(running, tmp_path):
    client, app, manager, *_ = running
    image = tmp_path / "test-assets" / "stick.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    calls, *_ = _configure_local_worker(tmp_path, manager, ready=False)

    response = client.post("/v1/3d/generations", headers=AUTH, json={"image": str(image)})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "local Idol is not ready"
    assert response.json()["detail"]["missing"] == ["encoder unavailable"]
    assert [call[1][-1] for call in calls] == ["--check"]
    assert manager.lease_calls == []
    status = client.get("/router/status", headers=AUTH).json()
    assert "idol-local" not in status["queues"]


def test_idol_job_is_local_owned_and_serves_ply_artifact(running, tmp_path):
    client, app, manager, *_ = running
    image = tmp_path / "test-assets" / "stick.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    calls, *_ = _configure_local_worker(tmp_path, manager)

    response = client.post("/v1/3d/generations", headers=AUTH, json={
        "model": "speridlabs/idol", "image": str(image), "mode": "synthetic", "steps": 50,
        "cfg_strength": 3.0, "rescale_t": 3.0, "seed": 42, "num_gaussians": 1048576,
    })
    assert response.status_code == 202
    job = _wait_for_job(client, response.json()["id"])
    assert job["state"] == "succeeded"
    assert job["resource"] == "idol-local"
    result = job["result"]
    assert result["backend"] == "local-mps" and result["device"] == "mps"
    assert result["offline"] is True and result["model"] == "speridlabs/idol"
    assert result["mode"] == "synthetic" and result["steps"] == 50
    assert result["cfg_strength"] == 3.0 and result["rescale_t"] == 3.0
    assert result["seed"] == 42 and result["num_gaussians"] == 1048576
    assert result["load_seconds"] == 12.5 and result["generation_seconds"] == 82.0
    assert result["total_seconds"] == 94.5 and result["metal_allocated_gib_at_finish"] == 16.25
    for key in ("input_artifact", "camera_artifact"):
        sidecar = Path(result[key])
        assert sidecar.is_file() and sidecar.parent == tmp_path / "outputs" / "3d"
    assert result["encoder_sources"] == {
        "dinov3": {"repo": "timm/dinov3-hplus", "revision": "a" * 40, "loader": "timm"},
        "flux2_vae": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "b" * 40, "loader": "diffusers"},
    }
    restored = app.state.scheduler.instance._restored_result(job["metadata"])
    assert restored["encoder_sources"] == result["encoder_sources"]
    assert "snapshot" not in json.dumps(restored["encoder_sources"])
    assert calls[0][1][-1] == "--check"
    assert "--request" in calls[1][1] and "--response" in calls[1][1]
    assert all(call[0] == "idol-local" for call in calls)
    assert [name for name, _ in manager.lease_calls] == ["idol-local"]
    artifact = client.get(f"/v1/jobs/{response.json()['id']}/artifact", headers=AUTH)
    assert artifact.status_code == 200 and artifact.content.startswith(b"ply\n")
    assert client.delete(f"/v1/jobs/{response.json()['id']}", headers={"Authorization": "Bearer admin-only"}).json()["media_files_deleted"] == 3
    assert not Path(result["input_artifact"]).exists() and not Path(result["camera_artifact"]).exists()


def test_idol_auth_and_parameter_errors_do_not_start_worker(running, tmp_path):
    client, app, manager, *_ = running
    image = tmp_path / "test-assets" / "stick.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    calls, *_ = _configure_local_worker(tmp_path, manager)
    assert client.post("/v1/3d/generations", json={"image": str(image)}).status_code == 401
    invalid = client.post("/v1/3d/generations", headers=AUTH, json={"image": str(image), "num_gaussians": 300000})
    assert invalid.status_code == 422
    assert calls == [] and manager.lease_calls == []



def test_idol_worker_failure_preserves_sanitized_private_diagnostic(running, tmp_path):
    client, app, manager, *_ = running
    image = tmp_path / "test-assets" / "stick.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    calls, *_ = _configure_local_worker(tmp_path, manager, fail=True)

    response = client.post("/v1/3d/generations", headers=AUTH, json={"image": str(image)})
    assert response.status_code == 202
    job = _wait_for_job(client, response.json()["id"])
    assert job["state"] == "failed"
    assert "MPS SDPA allocation failed" in job["error"]
    assert "1234567890secret" not in job["error"]
    diagnostic = tmp_path / "state" / "idol-local" / f"{job['id']}-diagnostic.json"
    saved = json.loads(diagnostic.read_text())
    assert saved["error_type"] == "MPSAllocatorError"
    assert "MPS SDPA allocation failed" in saved["error"]
    assert "1234567890secret" not in diagnostic.read_text()
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    assert diagnostic.parent.stat().st_mode & 0o777 == 0o700
    assert not list(diagnostic.parent.glob(f".{job['id']}-*request.json"))
    assert [call[1][-1] for call in calls if call[1][-1] == "--check"] == ["--check"]


def test_idol_cancellation_uses_manager_owned_worker_and_releases_lease(running, tmp_path):
    client, app, manager, *_ = running
    image = tmp_path / "test-assets" / "stick.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    calls, started, cancelled = _configure_local_worker(tmp_path, manager, block=True)
    response = client.post("/v1/3d/generations", headers=AUTH, json={"image": str(image)})
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert started.wait(2)
    current = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
    assert current["state"] == "running", current.get("error")
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).json()["cancelled"] is True
    job = _wait_for_job(client, job_id)
    assert job["state"] == "cancelled"
    assert cancelled.wait(2)
    assert [call[0] for call in calls] == ["idol-local", "idol-local"]
    assert [name for name, _ in manager.lease_calls] == ["idol-local"]
    assert manager.active == 0


def test_live_idol_adapter_has_no_hosted_client_or_space_endpoint():
    source = (Path(__file__).parents[1] / "model_router" / "mac" / "idol.py").read_text()
    assert "gradio_client" not in source
    assert "huggingface.co" not in source
    assert "client.submit" not in source
