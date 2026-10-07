"""Local image and video generation backends for the Mac control plane."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import httpx

logger = logging.getLogger(__name__)
H3_FPS = 24
H3_MIN_GENERATION_FRAMES = 22
H3_MAX_FRAMES = 362
H3_MAX_DURATION_SECONDS = 15.0


def video_frame_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the requested output length and H3's trained temporal chunk.

    H3 generates at 24 fps and its decoder only accepts 5 + 17k frames. The
    generated native clip is later trimmed to the requested 24 fps frame count.
    """
    has_duration = "duration" in payload
    has_seconds = "seconds" in payload
    has_frames = "frames" in payload
    if has_duration and has_seconds:
        duration = payload["duration"]
        seconds = payload["seconds"]
        if (isinstance(duration, bool) or isinstance(seconds, bool)
                or not isinstance(duration, (int, float)) or not isinstance(seconds, (int, float))):
            raise ValueError("duration and seconds must be finite positive numbers")
        try:
            duration_value, seconds_value = float(duration), float(seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("duration and seconds must be finite positive numbers") from exc
        if duration_value != seconds_value:
            raise ValueError("duration and seconds conflict")
    elif has_duration or has_seconds:
        value = payload["duration"] if has_duration else payload["seconds"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("duration must be a finite positive number")
        try:
            duration_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("duration must be a finite positive number") from exc
    else:
        duration_value = 3.0

    if (has_duration or has_seconds) and has_frames:
        raise ValueError("provide duration/seconds or frames, not both")
    if not math.isfinite(duration_value) or duration_value <= 0 or duration_value > H3_MAX_DURATION_SECONDS:
        raise ValueError("duration must be finite and greater than 0, up to 15 seconds")

    if has_frames:
        requested_frames = payload["frames"]
        if isinstance(requested_frames, bool) or not isinstance(requested_frames, int):
            raise ValueError("frames must be an integer")
        if not 5 <= requested_frames <= round(H3_MAX_DURATION_SECONDS * H3_FPS):
            raise ValueError("frames must be between 5 and 360 (15 seconds at 24 fps)")
        requested_seconds = requested_frames / H3_FPS
        target_frames = requested_frames
    else:
        requested_seconds = duration_value
        target_frames = max(1, round(requested_seconds * H3_FPS))
    native_frames = max(H3_MIN_GENERATION_FRAMES, target_frames)
    native_frames = 5 + ((native_frames - 5 + 16) // 17) * 17
    if native_frames > H3_MAX_FRAMES:
        raise ValueError("requested duration exceeds H3's 362-frame generation limit")
    return {
        "requested_seconds": requested_seconds,
        "requested_frames": target_frames,
        "native_frames": native_frames,
        "effective_frames": target_frames,
        "fps": H3_FPS,
        "effective_duration_seconds": target_frames / H3_FPS,
    }


class MediaBackend:
    """Run ComfyUI Qwen-Image and h3.c through manager-owned resources."""

    def __init__(self, manager: Any, paths: Mapping[str, Any], root: str | Path,
                 *, comfy_url: str = "http://127.0.0.1:8188", timeout: float = 900.0) -> None:
        self.manager = manager
        self.paths = paths
        self.root = Path(root).expanduser().resolve()
        self.comfy_url = comfy_url.rstrip("/")
        self.timeout = timeout

    def _current_paths(self) -> Mapping[str, Any]:
        """Refresh the manifest so a model downloaded after app startup is visible."""
        manifest = self.root / "config" / "model-paths.json"
        if manifest.is_file():
            try:
                loaded = json.loads(manifest.read_text())
                if isinstance(loaded, Mapping):
                    self.paths = loaded
            except (OSError, json.JSONDecodeError):
                pass
        return self.paths

    def _asset(self, value: Any) -> Path:
        if not isinstance(value, str) or not value or value.startswith(("http:", "https:", "data:")):
            raise ValueError("Media input must be a local approved asset path")
        path = Path(value.removeprefix("file://")).expanduser().resolve(strict=True)
        roots = [self.root / "test-assets", self.root / "outputs", self.root / "state" / "uploads"]
        if not any(path.is_relative_to(candidate.resolve()) for candidate in roots):
            raise ValueError("Media input must be in approved AI asset directories")
        if not path.is_file() or path.stat().st_size > 100 * 1024**2:
            raise ValueError("Invalid or oversized media asset")
        return path

    @staticmethod
    def _prompt(payload: Mapping[str, Any]) -> str:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16_000:
            raise ValueError("prompt must contain 1..16000 characters")
        return prompt.strip()

    @staticmethod
    def _int(payload: Mapping[str, Any], key: str, default: int, low: int, high: int) -> int:
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    def _image_options(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        width = self._int(payload, "width", 1024, 64, 1024)
        height = self._int(payload, "height", 1024, 64, 1024)
        if width % 8 or height % 8:
            raise ValueError("width and height must be multiples of 8")
        return {
            "prompt": self._prompt(payload),
            "width": width,
            "height": height,
            "steps": self._int(payload, "steps", 40, 1, 50),
            "seed": self._int(payload, "seed", 42, 0, 2**63 - 1),
            "negative_prompt": payload.get("negative_prompt", ""),
        }

    @staticmethod
    def _model_file(paths: Mapping[str, Any], filename: str) -> str:
        entry = paths.get("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", {})
        if not isinstance(entry, Mapping):
            raise RuntimeError("Qwen Image model paths are not configured")
        key = {
            "qwen-image-2.1-UC-BF16.gguf": "transformer",
            "qwen3vl_8b_bf16.safetensors": "text_encoder",
            "qwen_image_2.1_vae_bf16.safetensors": "vae",
        }[filename]
        direct = entry.get(key)
        if direct:
            candidate = Path(str(direct)).expanduser()
            if candidate.is_dir():
                candidate = candidate / filename
        else:
            candidate = Path(str(entry.get("path", ""))).expanduser() / filename
        if not candidate.is_file():
            # Hub snapshots retain their repository subdirectories.
            snapshot = Path(str(entry.get("path", ""))).expanduser()
            folder = {"transformer": "", "text_encoder": "text_encoders", "vae": "vae"}[key]
            candidate = snapshot / folder / filename
        if not candidate.is_file():
            files = entry.get("files", {})
            candidate = Path(str(files.get(filename, ""))).expanduser() if isinstance(files, Mapping) else Path()
        if not candidate.is_file():
            raise RuntimeError(f"Required Qwen Image model file is missing: {filename}")
        return candidate.name

    def _workflow(self, options: Mapping[str, Any], job_id: str, model: str, clip: str, vae: str,
                  uploaded_image: str | None = None) -> dict[str, Any]:
        if uploaded_image is None and options["width"] != options["height"]:
            raise ValueError("Qwen Image 2.1 text-to-image currently requires a square output")
        workflow: dict[str, Any] = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
            "4": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
            "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": options["prompt"],
                  "negative_prompt": options["negative_prompt"], "vae": ["3", 0],
                  "resolution": options["width"] if uploaded_image is None else max(options["width"], options["height"]),
                  "images": {}}},
            "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1],
                  "latent_image": ["5", 2], "seed": options["seed"], "steps": options["steps"], "cfg": 1.0,
                  "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"bubblerouter/{job_id}"}},
        }
        if uploaded_image is not None:
            workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": uploaded_image}}
            workflow["5"]["inputs"]["images"] = {"image_1": ["9", 0]}
        return workflow

    async def _upload(self, client: httpx.AsyncClient, path: Path) -> str:
        with path.open("rb") as stream:
            response = await client.post("/upload/image", files={"image": (path.name, stream, "application/octet-stream")},
                                         data={"type": "input", "overwrite": "true"})
        response.raise_for_status()
        result = response.json()
        name = result.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("ComfyUI did not return an uploaded image name")
        return str(Path(result.get("subfolder", "")) / name)

    async def _generate_comfy(self, options: dict[str, Any], job_id: str,
                              input_image: Path | None = None) -> dict[str, str]:
        model = self._model_file(self.paths, "qwen-image-2.1-UC-BF16.gguf")
        clip = self._model_file(self.paths, "qwen3vl_8b_bf16.safetensors")
        vae = self._model_file(self.paths, "qwen_image_2.1_vae_bf16.safetensors")
        output_dir = self.root / "outputs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        workflow = self._workflow(options, job_id, model, clip, vae)
        async with httpx.AsyncClient(base_url=self.comfy_url, timeout=httpx.Timeout(self.timeout, connect=10), trust_env=False) as client:
            if input_image is not None:
                uploaded = await self._upload(client, input_image)
                workflow = self._workflow(options, job_id, model, clip, vae, uploaded)
            response = await client.post("/prompt", json={"prompt": workflow})
            response.raise_for_status()
            prompt_id = response.json().get("prompt_id")
            if not isinstance(prompt_id, str):
                raise RuntimeError("ComfyUI did not return a prompt id")
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                response = await client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                history = response.json().get(prompt_id)
                if not history:
                    continue
                status = history.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False and status.get("messages"):
                    raise RuntimeError("ComfyUI generation failed")
                outputs = history.get("outputs", {})
                images = outputs.get("8", {}).get("images", [])
                if images:
                    image = images[0]
                    saved = (self.root / "repos" / "ComfyUI" / "output" / image.get("subfolder", "") / image["filename"]).resolve(strict=True)
                    comfy_output = (self.root / "repos" / "ComfyUI" / "output").resolve()
                    if not saved.is_relative_to(comfy_output) or not saved.is_file():
                        raise RuntimeError("ComfyUI returned an invalid output path")
                    destination = output_dir / f"{job_id}.png"
                    shutil.copyfile(saved, destination)
                    return {"artifact": str(destination), "content_type": "image/png"}
            raise TimeoutError("ComfyUI generation timed out")

    async def generate_image(self, payload: Mapping[str, Any], job_id: str) -> dict[str, str]:
        self._current_paths()
        return await self._generate_comfy(self._image_options(payload), job_id)

    async def edit_image(self, payload: Mapping[str, Any], job_id: str) -> dict[str, str]:
        self._current_paths()
        options = self._image_options(payload)
        image = self._asset(payload.get("image", payload.get("image_path")))
        return await self._generate_comfy(options, job_id, image)

    async def generate_video(self, payload: Mapping[str, Any], job_id: str) -> dict[str, Any]:
        self._current_paths()
        prompt = self._prompt(payload)
        entry = self.paths.get("MiniMaxAI/MiniMax-H3", self.paths.get("minimax-h3", {}))
        configured_model_path = entry.get("path") if isinstance(entry, Mapping) else None
        model_dir = Path(str(configured_model_path)).expanduser() if configured_model_path else None
        if model_dir is None or not model_dir.is_dir():
            raise RuntimeError("MiniMax H3 model path is not configured")
        executable = self.root / "repos" / "h3.c" / "h3"
        if not executable.is_file():
            raise RuntimeError("h3.c executable is missing")
        width = self._int(payload, "width", 512, 64, 720)
        height = self._int(payload, "height", 512, 64, 720)
        if width % 32 or height % 32:
            raise ValueError("H3 width and height must be multiples of 32")
        plan = video_frame_plan(payload)
        steps = self._int(payload, "steps", 4, 1, 20)
        seed = self._int(payload, "seed", 42, 0, 2**63 - 1)
        output_dir = self.root / "outputs" / "video"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{job_id}.mp4"
        args = [str(executable), "-d", str(model_dir), "-p", prompt, "-o", str(output), "--width", str(width),
                "--height", str(height), "--frames", str(plan["native_frames"]), "--steps", str(steps), "--seed", str(seed),
                "--ssd-streaming"]
        result = await self.manager.run_owned("minimax-h3", args, cwd=self.root / "repos" / "h3.c")
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            stderr_tail = (result.stderr or b"")[-1500:].decode(errors="replace")
            logger.warning("h3.c exited %s; stderr tail:\n%s", result.returncode, stderr_tail)
            raise RuntimeError(f"h3.c video generation failed (exit {result.returncode})")
        ffmpeg = shutil.which("ffmpeg") or str(Path("/opt/homebrew/bin/ffmpeg"))
        ffprobe = shutil.which("ffprobe") or str(Path("/opt/homebrew/bin/ffprobe"))
        ffmpeg = ffmpeg if Path(ffmpeg).is_file() else None
        ffprobe = ffprobe if Path(ffprobe).is_file() else None
        if not ffmpeg or not ffprobe:
            raise RuntimeError("ffmpeg and ffprobe are required to trim and verify H3 output")
        trimmed = output.with_name(f".{output.stem}.trimmed.mp4")
        trim_args = [ffmpeg, "-y", "-v", "error", "-i", str(output), "-map", "0:v:0", "-map", "0:a?",
                     "-vf", f"fps={plan['fps']}", "-frames:v", str(plan["effective_frames"]),
                     "-t", f"{plan['effective_duration_seconds']:.9f}", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                     "-movflags", "+faststart", str(trimmed)]
        trim_result = await self.manager.run_owned("minimax-h3", trim_args, cwd=output_dir)
        if trim_result.returncode != 0 or not trimmed.is_file() or trimmed.stat().st_size == 0:
            stderr_tail = (trim_result.stderr or b"")[-1500:].decode(errors="replace")
            logger.warning("ffmpeg trim exited %s; stderr tail:\n%s", trim_result.returncode, stderr_tail)
            raise RuntimeError("could not trim H3 output to the requested duration")
        probe = await self.manager.run_owned("minimax-h3", [ffprobe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,nb_frames,r_frame_rate", "-of", "json", str(trimmed)], cwd=output_dir)
        try:
            probe_data = json.loads(probe.stdout) if probe.returncode == 0 else {}
            actual_duration = float(probe_data["format"]["duration"])
            video_stream = next(s for s in probe_data.get("streams", []) if s.get("codec_type") == "video")
            actual_frames = int(video_stream["nb_frames"])
            actual_fps = float(Fraction(video_stream["r_frame_rate"]))
        except (ValueError, TypeError, KeyError, StopIteration, ZeroDivisionError, json.JSONDecodeError) as exc:
            trimmed.unlink(missing_ok=True)
            raise RuntimeError("ffprobe could not verify the trimmed H3 output") from exc
        duration_tolerance = 1 / H3_FPS
        if (not math.isfinite(actual_duration) or actual_duration <= 0
                or actual_duration > H3_MAX_DURATION_SECONDS + duration_tolerance
                or abs(actual_duration - plan["effective_duration_seconds"]) > duration_tolerance
                or actual_frames != plan["effective_frames"]
                or not math.isclose(actual_fps, H3_FPS, rel_tol=0, abs_tol=1e-6)):
            trimmed.unlink(missing_ok=True)
            raise RuntimeError("trimmed H3 output does not match the requested duration, frame count, or 24 fps")
        os.replace(trimmed, output)
        return {"artifact": str(output), "content_type": "video/mp4", **plan,
                "actual_duration_seconds": actual_duration, "actual_frames": actual_frames, "actual_fps": actual_fps}
