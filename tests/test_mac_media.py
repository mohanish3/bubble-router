from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_router.mac.media import MediaBackend


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


@pytest.mark.asyncio
async def test_video_invokes_only_bounded_manager_owned_command(tmp_path):
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

        async def run_owned(self, name, args, *, cwd=None):
            self.name, self.args, self.cwd = name, list(args), Path(cwd)
            Path(args[args.index("-o") + 1]).write_bytes(b"mp4")
            return SimpleNamespace(returncode=0)

    manager = Manager()
    backend = MediaBackend(manager, paths, root)
    result = await backend.generate_video({"prompt": "slow camera pan", "width": 256, "height": 256,
                                           "frames": 8, "steps": 4, "seed": 9}, "job-a")
    assert manager.name == "minimax-h3"
    assert manager.cwd == h3_dir
    assert manager.args[0] == str(exe)
    assert manager.args[manager.args.index("--frames") + 1] == "8"
    assert manager.args[manager.args.index("--steps") + 1] == "4"
    assert result == {"artifact": str(root / "outputs" / "video" / "job-a.mp4"), "content_type": "video/mp4"}


@pytest.mark.asyncio
async def test_video_rejects_unapproved_model_and_oversized_settings(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    with pytest.raises(RuntimeError, match="model path"):
        await backend.generate_video({"prompt": "x"}, "job")
