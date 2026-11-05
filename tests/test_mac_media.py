from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_router.mac.media import MediaBackend, edit_resolution, image_kind, video_frame_plan

def png_fixture(width=256, height=256):
    import struct, zlib
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\x00' + b'\x00\x80\x80' * width) * height)) + chunk(b'IEND', b''))

PNG = png_fixture()
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


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
    options = backend._image_options({"prompt": "a red square", "width": 512, "height": 768, "steps": 20, "seed": 7}, edit=True)
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

@pytest.mark.asyncio
async def test_default_generation_uses_original_bf16_components_and_40_steps(tmp_path, monkeypatch):
    import httpx
    from unittest.mock import AsyncMock
    backend=MediaBackend(None,media_paths(tmp_path),tmp_path)
    output=tmp_path/'repos/ComfyUI/output';output.mkdir(parents=True)
    (output/'fixture.png').write_bytes(PNG)
    seen=[]
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,path,json):
            seen.append(json['prompt'])
            return httpx.Response(200,json={'prompt_id':'fixture'},request=httpx.Request('POST','http://fixture/prompt'))
        async def get(self,path):
            return httpx.Response(200,json={'fixture':{'outputs':{'8':{'images':[{'filename':'fixture.png','subfolder':''}]}}}},request=httpx.Request('GET','http://fixture/history'))
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:Client())
    monkeypatch.setattr('model_router.mac.media.asyncio.sleep',AsyncMock())
    await backend.generate_image({'prompt':'A red square.','image_model':'qwen-image-uc'},'fixture')
    graph=seen[0]
    assert graph['1']['inputs']['unet_name']=='qwen-image-2.1-UC-BF16.gguf'
    assert graph['2']['inputs']['clip_name']=='qwen3vl_8b_bf16.safetensors'
    assert graph['3']['inputs']['vae_name']=='qwen_image_2.1_vae_bf16.safetensors'
    assert graph['6']['class_type']=='KSampler' and graph['6']['inputs']['steps']==40
    assert graph['4']['inputs']['device']=='off'
    assert graph['4']['inputs']['dtype']=='default'
    assert '13' not in graph


def test_qwen21_workflow_uses_native_64_channel_conditioning_and_edit_reference(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a red square", "width": 512, "height": 512})
    graph = backend._workflow(options, "job", "transformer.gguf", "text.safetensors", "vae.safetensors")
    assert graph["5"]["class_type"] == "TextEncodeQwenImage21"
    assert graph["6"]["inputs"]["latent_image"] == ["5", 2]
    assert graph["4"]["class_type"] == "QwenImage21Cache"
    assert graph["4"]["inputs"]["device"] == "off"
    edited = backend._workflow(options, "job", "transformer.gguf", "text.safetensors", "vae.safetensors", "red.png")
    assert edited["5"]["inputs"]["images.image_1"] == ["9", 0]
    assert "images" not in edited["5"]["inputs"]
    assert "EmptySD3LatentImage" not in {node["class_type"] for node in edited.values()}
    assert edited["4"]["inputs"] == graph["4"]["inputs"]


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
    monkeypatch.setattr("model_router.mac.media.reference_video_size", lambda path, *size: (256, 256))
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

        async def run_owned(self, name, args, *, cwd=None, env=None):
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
                '"nb_frames":"120","r_frame_rate":"24/1","width":256,"height":256}]}'
            ).encode())

    assets = root / "test-assets"
    assets.mkdir()
    (assets / "start.png").write_bytes(PNG)
    (assets / "end.jpg").write_bytes(JPEG)
    manager = Manager()
    backend = MediaBackend(manager, paths, root)
    result = await backend.generate_video({"prompt": "slow camera pan", "width": 256, "height": 256,
                                           "duration": 5, "steps": 4, "seed": 9,
                                           "first_frame": str(assets / "start.png"),
                                           "last_frame": str(assets / "end.jpg")}, "job-a")
    assert manager.name == "minimax-h3"
    h3_args, h3_cwd = manager.calls[2]
    trim_args, _ = manager.calls[3]
    assert h3_cwd == h3_dir
    assert h3_args[0] == str(exe)
    # Without h3-metal installed, every clip streams on h3.c.
    assert "--ssd-streaming" in h3_args and "--use-int8-row-fc2" not in h3_args
    assert h3_args[h3_args.index("--frames") + 1] == "124"
    assert h3_args[h3_args.index("--steps") + 1] == "4"
    assert manager.calls[0][0][manager.calls[0][0].index("-i") + 1] == str(assets / "start.png")
    assert manager.calls[1][0][manager.calls[1][0].index("-i") + 1] == str(assets / "end.jpg")
    assert 'crop=256:256:floor((iw-ow)/2):floor((ih-oh)/2):exact=1' in manager.calls[0][0][manager.calls[0][0].index("-vf") + 1]
    assert not Path(h3_args[h3_args.index("--first-frame") + 1]).exists()
    assert trim_args[trim_args.index("-frames:v") + 1] == "120"
    assert result["artifact"] == str(root / "outputs" / "video" / "job-a.mp4")
    assert result["content_type"] == "video/mp4"
    assert result["requested_seconds"] == 5
    assert result["native_frames"] == 124
    assert result["effective_frames"] == result["actual_frames"] == 120
    assert result["actual_duration_seconds"] == 5
    assert result["actual_fps"] == 24
    assert "scale=" not in trim_args[trim_args.index("-vf") + 1]
    assert (result["width"], result["height"], result["upscale"]) == (256, 256, None)

    manager.calls.clear()
    with pytest.raises(RuntimeError, match="requested size"):
        await backend.generate_video({"prompt": "x", "width": 256, "height": 256, "duration": 5, "upscale": 4}, "job-b")
    assert manager.calls[1][0][manager.calls[1][0].index("-vf") + 1] == "fps=24,scale=1024:1024:flags=lanczos,cas=0.6"

@pytest.mark.parametrize(("payload", "message"), [
    ({"width": 1376}, "between 64 and 1344"),
    ({"width": 1024, "height": 1024}, "768 x 1344"),
    ({"width": 500}, "multiples of 32"),
    ({"upscale": 5}, "2, 3 or 4"),
    ({"upscale": 0}, "2, 3 or 4"),
    ({"upscale": 2.5}, "2, 3 or 4"),
    ({"upscale": 2.0}, "2, 3 or 4"),
    ({"upscale": "2x"}, "2, 3 or 4"),
    ({"upscale": True}, "2, 3 or 4"),
])
def test_video_size_and_upscale_limits(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        MediaBackend(None, {}, tmp_path)._video_options({"prompt": "x", **payload})


def test_video_accepts_h3_768p_canvases(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    for width, height in ((768, 768), (1344, 768), (768, 1344), (1024, 768), (992, 992)):
        assert backend._video_options({"prompt": "x", "width": width, "height": height, "upscale": 3})[:2] == (width, height)


@pytest.mark.asyncio
async def test_video_rejects_unapproved_model_and_oversized_settings(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    with pytest.raises(RuntimeError, match="model path"):
        await backend.generate_video({"prompt": "x"}, "job")


def test_image_kind_uses_magic_bytes_not_names():
    assert image_kind(PNG) == "png"
    assert image_kind(JPEG) == "jpg"
    assert image_kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert image_kind(b"GIF89a") is None
    assert image_kind(b"<svg") is None


def test_edit_resolution_keeps_requested_pixel_area():
    assert edit_resolution(1024, 1024) == 1024
    assert edit_resolution(768, 512) == 640
    assert edit_resolution(64, 64) == 256


def test_non_square_edit_workflow_uses_area_resolution(tmp_path):
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "make it blue", "width": 768, "height": 512}, edit=True)
    workflow = backend._workflow(options, "job", "t.gguf", "c.safetensors", "v.safetensors", "mug.png")
    assert workflow["5"]["inputs"]["resolution"] == 640
    assert workflow["9"]["inputs"]["image"] == "mug.png"


def test_preflight_validates_before_queueing(tmp_path):
    assets = tmp_path / "test-assets"
    assets.mkdir()
    (assets / "mug.png").write_bytes(PNG)
    (assets / "fake.png").write_bytes(b"not an image")
    backend = MediaBackend(None, {}, tmp_path)
    backend.preflight("edit", {"prompt": "x", "image": str(assets / "mug.png"), "width": 768, "height": 512})
    backend.preflight("video", {"prompt": "x", "first_frame": str(assets / "mug.png"), "width": 256, "height": 256})
    for operation, payload, message in (
        ("image", {"prompt": "x", "width": 768, "height": 512}, "square"),
        ("edit", {"prompt": "x"}, "image is required"),
        ("edit", {"prompt": "x", "image": str(assets / "fake.png")}, "PNG, JPEG or WebP"),
        ("edit", {"prompt": "x", "image": "/etc/hosts"}, "approved local image"),
        ("edit", {"prompt": "x", "image": str(assets / "mug.png"), "negative_prompt": 5}, "negative_prompt"),
        ("video", {"prompt": "x", "last_frame": str(assets / "missing.png")}, "last_frame"),
        ("video", {"prompt": "x", "first_frame": "https://example.com/a.png"}, "first_frame"),
    ):
        with pytest.raises(ValueError, match=message):
            backend.preflight(operation, payload)


def test_atomic_default_schedule_and_legacy_edit_default(tmp_path):
    from model_router.mac.media import ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER, TURBO_SIGMAS
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a red mug", "image_model": "qwen-image-uc", "width": 512, "height": 512, "seed": 314})
    assert options["steps"] == 40
    default_graph = backend._workflow(options, "default", ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER, "vae.safetensors")
    assert default_graph["6"]["class_type"] == "KSampler"
    assert default_graph["6"]["inputs"]["steps"] == 40
    assert default_graph["6"]["inputs"]["sampler_name"] == "euler"
    assert default_graph["6"]["inputs"]["scheduler"] == "simple"
    options["steps"] = 8
    graph = backend._workflow(options, "job", ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER, "vae.safetensors")
    assert graph["2"]["class_type"] == "CLIPLoaderGGUF"
    assert graph["6"]["class_type"] == "SamplerCustomAdvanced"
    assert graph["10"]["inputs"]["noise_seed"] == 314
    assert len(graph["13"]["inputs"]["sigmas"].split(",")) - 1 == options["steps"]
    assert graph["13"]["inputs"]["sigmas"] == TURBO_SIGMAS
    assert graph["11"]["inputs"]["conditioning"] == ["5", 0]
    edit = backend._image_options({"prompt": "make it blue", "image_model": "qwen-image-uc", "width": 512, "height": 768}, edit=True)
    assert edit["steps"] == 40
    edited = backend._workflow(edit, "edit", "qwen-image-2.1-UC-BF16.gguf", "qwen3vl_8b_bf16.safetensors", "vae.safetensors", "input.png")
    assert edited["2"]["class_type"] == "CLIPLoader"
    assert edited["6"]["class_type"] == "KSampler"
    assert edited["6"]["inputs"]["steps"] == 40
    assert edited["5"]["inputs"]["images.image_1"] == ["9", 0]


def test_atomic_explicit_step_count_is_not_silently_ignored(tmp_path):
    from model_router.mac.media import ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a red mug", "steps": 20})
    graph = backend._workflow(options, "job", ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER, "vae.safetensors")
    assert graph["6"]["class_type"] == "KSampler"
    assert graph["6"]["inputs"]["steps"] == 20
    assert graph["6"]["inputs"]["scheduler"] == "simple"


def test_atomic_components_resolve_from_their_separate_repositories(tmp_path):
    from model_router.mac.media import ATOMIC_IMAGE_MODEL, ATOMIC_TEXT_ENCODER, ATOMIC_IMAGE_REPO, ATOMIC_ENCODER_REPO
    paths = media_paths(tmp_path)
    for repository, name, key in ((ATOMIC_IMAGE_REPO, ATOMIC_IMAGE_MODEL, "transformer"), (ATOMIC_ENCODER_REPO, ATOMIC_TEXT_ENCODER, "text_encoder")):
        folder = tmp_path / repository
        folder.mkdir(parents=True)
        (folder / name).touch()
        paths[repository] = {key: str(folder / name)}
        assert MediaBackend._model_file(paths, name) == name
    assert MediaBackend._model_file(paths, "qwen_image_2.1_vae_bf16.safetensors") == "qwen_image_2.1_vae_bf16.safetensors"
    del paths[ATOMIC_IMAGE_REPO]
    with pytest.raises(RuntimeError, match="missing"):
        MediaBackend._model_file(paths, ATOMIC_IMAGE_MODEL)


def test_image_model_selection_defaults_and_rejects_unknown(tmp_path):
    from model_router.mac.media import DEFAULT_IMAGE_MODEL
    backend = MediaBackend(None, {}, tmp_path)
    assert backend._image_options({"prompt": "x", "width": 512, "height": 512})["model"] == DEFAULT_IMAGE_MODEL
    assert backend._image_options({"prompt": "x", "model": "qwen-image", "width": 512, "height": 512})["model"] == DEFAULT_IMAGE_MODEL
    assert backend._image_options({"prompt": "x", "model": "noct-q-v4"})["model"] == "noct-q-v4"
    atomic = backend._image_options({"prompt": "x", "image_model": "atomic-turbo"})
    assert atomic["steps"] == 8
    with pytest.raises(ValueError, match="image_model"):
        backend._image_options({"prompt": "x", "image_model": "sdxl"})


def test_noct_allows_portrait_and_uses_its_own_latent_and_cfg(tmp_path):
    from model_router.mac.media import NOCT_IMAGE_MODEL, NOCT_NEGATIVE
    backend = MediaBackend(None, {}, tmp_path)
    options = backend._image_options({"prompt": "a lake", "image_model": "noct-q-v4", "width": 1024, "height": 1536})
    assert (options["steps"], options["cfg"], options["negative_prompt"]) == (25, 3.0, NOCT_NEGATIVE)
    graph = backend._workflow(options, "job", NOCT_IMAGE_MODEL, "qwen3vl_8b_bf16.safetensors", "vae.safetensors")
    assert graph["1"] == {"class_type": "UNETLoader", "inputs": {"unet_name": NOCT_IMAGE_MODEL, "weight_dtype": "default"}}
    assert graph["14"]["inputs"] == {"width": 1024, "height": 1536, "batch_size": 1}
    assert graph["6"]["inputs"]["latent_image"] == ["14", 0]
    assert graph["6"]["inputs"]["cfg"] == 3.0
    assert backend._image_options({"prompt": "x", "image_model": "noct-q-v4", "negative_prompt": ""})["negative_prompt"] == ""
    for size in ({"width": 1000, "height": 1024}, {"width": 2080, "height": 1024}):
        with pytest.raises(ValueError):
            backend._image_options({"prompt": "x", "image_model": "noct-q-v4", **size})
    with pytest.raises(ValueError, match="square"):
        backend._image_options({"prompt": "x", "width": 512, "height": 768})


def test_noct_transformer_resolves_from_its_repository(tmp_path):
    from model_router.mac.media import NOCT_IMAGE_MODEL, NOCT_REPO
    paths = media_paths(tmp_path)
    (tmp_path / NOCT_IMAGE_MODEL).touch()
    paths[NOCT_REPO] = {"transformer": str(tmp_path / NOCT_IMAGE_MODEL)}
    assert MediaBackend._model_file(paths, NOCT_IMAGE_MODEL) == NOCT_IMAGE_MODEL


@pytest.mark.asyncio
async def test_small_video_runs_resident_on_h3_metal_and_large_streams_on_h3c(tmp_path, monkeypatch):
    for tool in ("ffmpeg", "ffprobe"):
        (tmp_path / tool).touch()
    monkeypatch.setattr("model_router.mac.media.shutil.which", lambda name: str(tmp_path / name))
    model = tmp_path / "models" / "h3"; model.mkdir(parents=True)
    for engine in ("h3.c", "h3-metal"):
        (tmp_path / "repos" / engine).mkdir(parents=True); (tmp_path / "repos" / engine / "h3").touch()

    class Manager:
        def __init__(self): self.h3 = []
        async def run_owned(self, name, args, *, cwd=None, env=None):
            if args[0].endswith("h3"):
                self.h3.append((list(args), Path(cwd), env))
                Path(args[args.index("-o") + 1]).write_bytes(b"native-mp4")
                return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
            if args[0].endswith("ffmpeg"):
                self.trimmed = int(args[args.index("-frames:v") + 1])
                Path(args[-1]).write_bytes(b"trimmed-mp4")
                return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
            frames = self.trimmed
            return SimpleNamespace(returncode=0, stderr=b"", stdout=json.dumps({"format": {"duration": str(frames / 24)},
                "streams": [{"codec_type": "video", "nb_frames": str(frames), "r_frame_rate": "24/1", "width": 512, "height": 512}]}).encode())

    manager = Manager()
    backend = MediaBackend(manager, {"minimax-h3": {"path": str(model)}}, tmp_path)
    await backend.generate_video({"prompt": "fox", "width": 512, "height": 512, "duration": 3, "steps": 4}, "small")
    await backend.generate_video({"prompt": "fox", "width": 512, "height": 512, "duration": 10, "steps": 4}, "long")
    (small, small_cwd, small_env), (large, large_cwd, large_env) = manager.h3
    assert small_cwd == tmp_path / "repos" / "h3-metal" and small_env == {"H3_VAE_BF16": "1"}
    assert "--use-int8-row-fc2" in small and "--ssd-streaming" not in small
    assert large_cwd == tmp_path / "repos" / "h3.c" and large_env is None and "--ssd-streaming" in large
