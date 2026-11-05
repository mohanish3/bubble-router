# Mac control plane

Bubble Router is the single authenticated local entry point for the Mac model stack. OpenClaw skills and other inference clients call `http://127.0.0.1:8090/v1`; model servers bind to loopback on their own ports and are not client-facing. The macOS app is `model_router.mac.app:app`; the existing Debian entry point remains `model_router.main`.

## Processes and runtimes

Each runtime has its own Python environment so MLX, PyTorch/ComfyUI, and the Router dependencies can be upgraded independently.

| Capability | Runtime | Local endpoint / process | Current model |
| --- | --- | --- | --- |
| Text, vision, video understanding | MLX-VLM in `~/AI/venvs/vlm` | `127.0.0.1:8080` | `mlx-community/Qwen3.8-27B-4bit` |
| Speech recognition and speech generation | MLX-Audio in `~/AI/venvs/vlm` | `127.0.0.1:8765` | Qwen3-ASR 1.7B BF16; Qwen3-TTS 1.7B CustomVoice 8-bit |
| Image generation and editing | ComfyUI in `~/AI/venvs/comfy`; `leejet/ComfyUI-GGUF` custom node | `127.0.0.1:8188`, started on demand | `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`, BF16 transformer, BF16 Qwen3-VL encoder and BF16 VAE |
| Video generation | Native `h3.c` Metal binary in `~/AI/repos/h3.c` | Manager-owned one-shot process | MiniMax H3 FL2VA checkpoint; `-d` points at the checkpoint root containing `FL2VA/` |
| Classification and prompt screening | Laya on CPU/PyTorch in the Router environment | In process | `convaiinnovations/laya`; ONNX Runtime installed but unused because no exported graph is supplied |

Model servers are launched and stopped by `model_router/mac/resources.py`; callers do not invoke ComfyUI or `h3` directly. `model_router/mac/media.py` submits the native Qwen Image 2.1 conditioning workflow to ComfyUI and runs H3 as a manager-owned subprocess. The workflow uses ComfyUI's `TextEncodeQwenImage21` latent/conditioning node and the GGUF loader; the Qwen Image edit workflow supplies reference images through that node's image inputs. BF16 GGUF values are reconstructed per layer and cast to BF16 for inference. The selected image files are linked from the Hugging Face snapshot into ComfyUI's model directories instead of copied.

`model_router/mac/resources.py` owns process groups, checks process health, journals ownership markers, and admits work against host memory. Current configured resource estimates are 21 GiB for Qwen-VLM, 10 GiB for audio, 34 GiB for image and 28 GiB for H3 clips with at most124 native frames, with a 12 GiB system reserve. Longer H3 clips use a 40 GiB estimate and request the additional12 GiB through the same resource lease. The optional H3 runtime guard polls once per second and terminates only the owned process group if available memory falls below16 GiB, pressure becomes warning/high, swap grows by more than1 GiB from that process baseline, or telemetry fails. This is a best-effort guard, not an OS hard limit. A non-core generator gets an exclusive lease; an idle core process may be evicted to satisfy admission, then restored after the lease. A request lease may also evict an idle model that has its own queued work (models without queued work go first). If memory is still short while another model holds a lease, the request stays queued until that model finishes and can be unloaded (up to 3 h), instead of failing after the 120 s memory wait. These admission estimates are separate from measured process RSS and total GPU memory; longer H3 clips still need measurement.

H3 video requests default to 3 seconds and accept a positive `duration` (or `seconds`) up to 15 seconds. H3 generates 24 fps clips in native 5+17k-frame chunks (22 through 362 frames); the Router trims both video and audio to the nearest requested 24 fps frame count, then verifies the output with ffprobe. The native 15-second path and memory peak have not been benchmarked; resource admission remains responsible for deciding whether a job can run.

`model_router/mac/scheduler.py` prioritizes work and stores job identifiers, resource, priority, sanitized caller metadata, state, timestamps, and error class in `~/AI/state/jobs.sqlite3`. It does not persist prompts or generated content. Image/video jobs return a job ID and artifact URL; audio and text calls use their normal response paths. Cancelling a running job cancels its task and terminates only the process group owned for that job.

### Manual requests shelve background work

Chat requests get a lane from their `x-source` header (`shelf` in `control-plane.json`). Untagged and `manual` requests are **manual**. `background`, `cron` and `loop` requests are **background**. Any other source gets no lane. While manual work is queued or running, and for `grace_seconds` (120) after it, the scheduler holds background jobs in the queue. The agent or cron run behind a held call is simply waiting on its HTTP request, so it continues on its own once the hold ends. A background job held for `max_hold_seconds` (900) runs anyway, because OpenClaw's provider timeout is 1800 s. Manual jobs go ahead of aged jobs. A running call is never interrupted. Draining releases held work.

A manual `auto` request stays on whichever loaded chat model is listed in `prefer_loaded` (swift, arex, qwen) instead of waiting for a reload. Explicit model names, long coding runs and vision requests the loaded model cannot handle keep their routing.

OpenClaw tags scheduled work through the `local-bg` provider, which is the same endpoint with a fixed `x-source: background` header. Cron agent turns use `--model local-bg/auto --fallbacks local-bg/swift`. `modelctl shelve [seconds]` and `modelctl unshelve` hold or release background work by hand. `/router/status` and `/router/queues` report the `shelf` state and mark held jobs. Taste Gate loop turns use the `local-loop` provider (`x-source: loop`).

Each job also records who sent it, for display only: `origin` (`loop`, `cron`, `manual`, `background`, `experiment` or `other`) and `origin_label`, such as `Loop · Daily AI & business trends` or `Cron · donna-daily-brief`. Provider headers are fixed per OpenClaw provider, so the loop or cron identity comes from the user message that opens the turn. The loop engine starts every turn prompt with `TASTE GATE LOOP turn for loop "<name>" (<loop id>), run <run id>.`, giving `loop_name`, `loop_id` and `loop_run_id`. OpenClaw starts cron turns with `[cron:<job id> <job name>]`, giving `cron_id` and `cron_name`. These markers are read only on OpenClaw's own sources (untagged, `manual`, `background`, `cron`, `loop`). The lane still comes from `x-source` alone, so an untagged cron turn shows as `Cron · <name>` but stays in the manual lane.

## Authentication and trust

All client requests to port 8090 except `/health` require a bearer credential. The Router distinguishes three principals:

- **Inference** is used by the seven OpenClaw skills and ordinary inference clients.
- **OpenClaw provider** is held only by the configured OpenClaw provider. It marks provider-authenticated system messages as trusted for the security scanner.
- **Administrator** is used by `~/AI/bin/modelctl` for local resource, queue, security, and lifecycle controls under `/router/control/*`. Those controls are loopback-only.

The credentials are distinct and stored in private local configuration. Do not put their values in this document, shell history, skills, source control, or diagnostics. Skills should call port 8090 only; the model ports are internal backends.

The Router scans prompts before dispatch. Audit records contain security scores, provenance, timing, and caller metadata, not request bodies. Media inputs must resolve to approved local asset directories, and generated artifacts are served from `~/AI/outputs` through the job artifact route.

## Scheduler and operations

Use `~/AI/bin/modelctl` for operator actions. It sends its administrator credential to the Router rather than starting or stopping models itself. The `ai.bubble.router` per-user LaunchAgent is configured to run `~/AI/services/router-supervisor.py`, which supervises Uvicorn on `127.0.0.1:8090` with access logging disabled. The plist and supervisor are present; login restart and long-running recovery still require acceptance checks.

Useful local endpoints include `/health`, `/router/status`, `/router/resources`, `/router/queues`, `/router/security`, and `/metrics`. `/v1/models` reports `speridlabs/idol` as `local-mps`. `POST /v1/3d/generations` queues an offline local image-to-3D request in the shared scheduler. It accepts an image path under the approved asset directories or a multipart PNG/JPEG/WebP upload (up to 20 MiB). Before it accepts the job, the Router runs the configured local worker with `--check`; missing local weights, encoders, or MPS readiness return HTTP 503 without creating a job. The worker runs under the nonpersistent `idol-local` resource with a 20 GiB estimate plus the configured system reserve, using a manager lease and manager-owned one-shot process so cancellation terminates only that worker. It writes the Gaussian-splat `.ply`, processed input image, and camera JSON under `~/AI/outputs/3d/`. Poll `/v1/jobs/{id}` and fetch the PLY from `/v1/jobs/{id}/artifact`; job results include local backend/device/offline status, settings, timings, sidecar paths, and sanitized encoder-source provenance. No hosted inference fallback is used. Worker failure diagnostics are sanitized and kept privately under `~/AI/state/idol-local/`.

## OpenClaw skills and local models

The installed skills are `qwen-vision`, `qwen-video-understanding`, `qwen-asr`, `qwen-tts`, `qwen-image`, `minimax-h3`, and `bubble-status`. They route through the authenticated 8090 endpoint. Their inference credential does not grant lifecycle administration or the trusted OpenClaw provider role.

The installed model revisions and full package-version snapshots are recorded in `~/AI/config/runtime-versions.json`. The current model revisions are:

| Model | Revision | Runtime | State |
| --- | --- | --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | MLX-VLM 0.7.6 | Installed; text and image/video understanding smoke checks passed |
| `mlx-community/Qwen3-ASR-1.7B-bf16` | `e1f6c266914abc5a46e8756e02580f834a6cf8a7` | MLX-Audio 0.5.8 | Installed; local transcription smoke check passed |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | `41d3337e8b7f2843a75841595fc14e4b9a7a4b96` | MLX-Audio 0.5.8 | Installed; local speech-generation smoke check passed |
| `convaiinnovations/laya` | `7b928d828b7b0e022f929d9bd2e44165aa270148` | CPU/PyTorch 2.14.1 | Installed; fixture-tuned policy exercised; raw confidence remains uncalibrated |
| `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` | `6b34e59458d3eb7ba6a6f86a116aed5253dc02c3` | ComfyUI 0.39.0, PyTorch 2.14.1, GGUF node `373048b8403a7820620065210a691263d4da0a61` | BF16 transformer and companions installed; workflow schema validated, full generation not yet successful |
| `MiniMaxAI/MiniMax-H3` | `42ed227ee7df40d41602854ae760620d6eb651fe` | `h3.c` Metal runtime, commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | Installed; real5s OpenClaw generation passed (512×512,120frames/24fps,audio);15s remains unbenchmarked |

## Verification limits

The earlier `/compress`/Telegram queue-clear operation is historical. Current configuration has `warm_core=false` and `start_drained=false`; live Router admission was accepting jobs during the October7 tests. Production Telegram is enabled, but no messages were sent by this acceptance run. Production Gateway health responds while its status/agent runtime is stalled after transcript teardown; a restart decision is pending because older interrupted main inputs may resume. A separate loopback test Gateway with all channels disabled completed the actual video test.

ASR accepts ordered multi-file batches at `/v1/audio/transcriptions/batch`, with at most eight files and 100 MiB combined input. This uses one worker and lease, rather than simultaneous tensor inference. The 15 OpenClaw agents use the same local model and seven capability skills. API credentials separate trusted framework messages and administration; processes running as the same macOS user can read that user's private files, so this is not an OS sandbox between agents.

This is an implementation and current-state note, not an acceptance sign-off. OpenClaw is configured with a 131,072-token context window, but 128K context use has not been stress-tested. No full Qwen Image generation has completed yet: the first BF16 job was rejected at memory admission before ComfyUI loaded the model. H3 generation passed in an isolated OpenClaw Gateway:5.000s/120frames/24fps/512×512/audio,91.030s Router job,243.286s complete agent turn. The attachment path and authenticated artifact download were verified. The15s native path and total memory peak remain unbenchmarked; longer requests remain subject to resource admission. Native Telegram delivery, own-user Telegram checks, and the another-device remote check are still pending. Basic local text, streaming, vision, video-understanding, ASR, TTS, security, and workflow-schema checks do not establish those remaining end-to-end results.

## Source files

- `model_router/mac/app.py`: authenticated API, security dispatch, media job and artifact routes.
- `model_router/mac/idol.py`: offline local MPS worker adapter, readiness preflight, memory admission integration, cancellation, artifact validation, sidecar/provenance reporting, and private diagnostics.
- `model_router/mac/resources.py`: memory admission, exclusive leases, owned subprocesses, process recovery journal.
- `model_router/mac/scheduler.py`: durable metadata queue, priorities, job cancellation.
- `model_router/mac/security.py`: prompt scanning and policy.
- `model_router/mac/media.py`: ComfyUI and H3 adapters.
- `~/AI/services/router-supervisor.py` and `~/AI/services/ai.bubble.router.plist`: supervised local Router process.
- `~/AI/bin/modelctl`: administrator CLI for Router controls.
