from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_router.mac.media import MediaBackend, video_frame_plan


def media_paths(tmp_path: Path) -> dict[str, dict[str, str]]:
    model_root = tmp_path / "models" / "uncensored"
    model_root.mkdir(parents=True)
    files = (
        "qwen-image-2.1-UC-BF16.gguf",
        "qwen3vl_8b_bf16.safetensors",
        "qwen_image_2.1_vae_bf16.safetensors",
    )
    for name in files:
        (model_root / name).touch()
    return {"abenzerps/Qwen-Image-2.1-Uncensored-GGUF": {"path": str(model_root)}}


def test_image_options_bound_dimensions_steps_and_seed(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a red square", "width": 512, "height": 768, "steps": 20, "seed": 7})
    assert options["width"] == 512
    assert options["height"] == 768
    assert options["steps"] == 20
    assert options["seed"] == 7
    for payload in (
        {"prompt": "x", "width": 63},
        {"prompt": "x", "width": 513},
        {"prompt": "x", "steps": 51},
        {"prompt": "x", "seed": -1},
    ):
        with pytest.raises(ValueError):
            backend._image_options(payload)


def test_model_path_resolves_and_rejects_missing_components(tmp_path):
    paths = media_paths(tmp_path)
    assert MediaBackend._model_file(paths, "qwen-image-2.1-UC-BF16.gguf") == "qwen-image-2.1-UC-BF16.gguf"
    with pytest.raises(RuntimeError, match="missing"):
        MediaBackend._model_file({}, "qwen-image-2.1-UC-BF16.gguf")


def test_qwen21_workflow_uses_native_64_channel_conditioning_and_edit_reference(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a red square", "width": 512, "height": 512})
    graph = backend._workflow(options, "job", "transformer.gguf", "text.safetensors", "vae.safetensors")
    assert graph["5"]["class_type"] == "TextEncodeQwenImage21"
    assert graph["6"]["inputs"]["latent_image"] == ["5", 2]
    assert graph["4"]["class_type"] == "QwenImage21Cache"
    edited = backend._workflow(options, "job", "transformer.gguf", "text.safetensors", "vae.safetensors", "red.png")
    assert edited["5"]["inputs"]["images"] == {"image_1": ["9", 0]}
    assert "EmptySD3LatentImage" not in {node["class_type"] for node in edited.values()}


def test_asset_only_accepts_approved_roots_and_regular_files(tmp_path):
    approved = tmp_path / "test-assets"
    approved.mkdir()
    image = approved / "red.png"
    image.write_bytes(b"image")
    outside = tmp_path.parent / "outside-media.png"
    outside.write_bytes(b"image")
    backend = MediaBackend(None, {}, tmp_path)
    assert backend._asset(str(image)) == image
    with pytest.raises(ValueError, match="approved"):
        backend._asset(str(outside))
    outside.unlink()


@pytest.mark.parametrize(("duration", "native_frames", "effective_frames"), [
    (3.0, 73, 72), (5.0, 124, 120), (15.0, 362, 360),
])
def test_video_duration_plans_align_native_chunk_then_trim(duration, native_frames, effective_frames):
    plan = video_frame_plan({"duration": duration})
    assert plan["requested_seconds"] == duration
    assert plan["native_frames"] == native_frames
    assert plan["effective_frames"] == effective_frames
    assert plan["effective_duration_seconds"] == effective_frames / 24


def test_video_duration_defaults_to_three_seconds_and_accepts_seconds_alias():
    assert video_frame_plan({})["effective_frames"] == 72
    assert video_frame_plan({"seconds": 5})["native_frames"] == 124
    assert video_frame_plan({"duration": 5, "seconds": 5})["effective_frames"] == 120


@pytest.mark.parametrize("payload", [
    {"duration": 0}, {"duration": -1}, {"duration": float("nan")},
    {"duration": float("inf")}, {"duration": 15.001}, {"duration": "5"},
    {"duration": 3, "seconds": 5}, {"duration": 5, "frames": 120},
    {"frames": 361}, {"frames": 4},
])
def test_video_length_rejects_invalid_or_conflicting_values(payload):
    with pytest.raises(ValueError):
        video_frame_plan(payload)


@pytest.mark.asyncio
async def test_video_invokes_native_h3_then_trims_and_measures_requested_duration(tmp_path, monkeypatch):
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / tool).touch()
    monkeypatch.setattr("model_router.mac.media.shutil.which", lambda name: str(tmp_path / name))
    root = tmp_path
    model = root / "models" / "h3"
    model.mkdir(parents=True)
    h3_dir = root / "repos" / "h3.c"
    h3_dir.mkdir(parents=True)
    exe = h3_dir / "h3"
    exe.touch()
    paths = {"minimax-h3": {"path": str(model)}}

    class Manager:
        def __init__(self):
            self.name = None
            self.args = None
            self.cwd = None
            self.calls = []

        async def run_owned(self, name, args, *, cwd=None):
            self.name, self.args, self.cwd = name, list(args), Path(cwd)
            self.calls.append((list(args), Path(cwd)))
            if args[0].endswith("h3"):
                Path(args[args.index("-o") + 1]).write_bytes(b"native-mp4")
                return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
            if args[0].endswith("ffmpeg"):
                Path(args[-1]).write_bytes(b"trimmed-mp4")
                return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
            return SimpleNamespace(returncode=0, stderr=b"", stdout=(
                '{"format":{"duration":"5.000000"},"streams":[{"codec_type":"video",'
                '"nb_frames":"120","r_frame_rate":"24/1"}]}'
            ).encode())

    manager = Manager()
    backend = MediaBackend(manager, paths, root)
    result = await backend.generate_video({"prompt": "slow camera pan", "width": 256, "height": 256,
                                           "duration": 5, "steps": 4, "seed": 9}, "job-a")
    assert manager.name == "minimax-h3"
    h3_args, h3_cwd = manager.calls[0]
    trim_args, _ = manager.calls[1]
    assert h3_cwd == h3_dir
    assert h3_args[0] == str(exe)
    assert h3_args[h3_args.index("--frames") + 1] == "124"
    assert h3_args[h3_args.index("--steps") + 1] == "4"
    assert trim_args[trim_args.index("-frames:v") + 1] == "120"
    assert result["artifact"] == str(root / "outputs" / "video" / "job-a.mp4")
    assert result["content_type"] == "video/mp4"
    assert result["requested_seconds"] == 5
    assert result["native_frames"] == 124
    assert result["effective_frames"] == result["actual_frames"] == 120
    assert result["actual_duration_seconds"] == 5
    assert result["actual_fps"] == 24


@pytest.mark.asyncio
async def test_video_rejects_unapproved_model_and_oversized_settings(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    with pytest.raises(RuntimeError, match="model path"):
        await backend.generate_video({"prompt": "x"}, "job")
