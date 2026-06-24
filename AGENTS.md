# AGENTS.md

Quick-start for AI agents working in this repo.

## What this is

`bubble-router` is an OpenAI-compatible HTTP proxy that classifies incoming prompts using a 16ms NLI classifier and forwards them to the appropriate specialist model backend. It runs on a single GPU and manages model switching automatically.

## Module map

| File | Purpose | Key symbols |
|------|---------|-------------|
| `model_router/classifier.py` | NLI prompt → model label | `TaskClassifier.classify(payload, metadata) -> Classification` |
| `model_router/scheduler.py` | FIFO queue + loaded-model preference | `Scheduler.enqueue(job)`, `Scheduler.next_job() -> Job` |
| `model_router/service.py` | Orchestrate classify → ensure → forward | `RouterService.submit(payload, metadata) -> Job` |
| `model_router/config.py` | Load + validate `model-router.json` | `load_config(path?) -> RouterConfig`, `ModelConfig` |
| `model_router/app.py` | FastAPI endpoints | `/v1/chat/completions`, `/health`, `/router/status` |
| `model_router/main.py` | Uvicorn entry point | `app` |
| `model_router/backends/base.py` | Backend contract + HTTP forwarding helper | `Backend` ABC, `_http_forward()`, `RoutedResponse` |
| `model_router/backends/llamacpp.py` | llama-server process lifecycle | `LlamaCppBackend.ensure(model_key, model_config)` |
| `model_router/backends/ollama.py` | Ollama HTTP + auto-pull | `OllamaBackend` |
| `model_router/backends/vllm.py` | vLLM HTTP proxy | `VLLMBackend` |
| `model_router/backends/lmstudio.py` | LM Studio HTTP proxy | `LMStudioBackend` |
| `model_router/backends/api.py` | Stateless cloud API calls | `APIBackend(provider, model_config)` |
| `model_router/backends/__init__.py` | Registry + factory | `create_backends(config) -> dict[str, Backend]` |

## Data flow

```
request → app.py (auth + parse)
        → service.classify()      # NLI classifier picks model
        → scheduler.enqueue()     # FIFO queue
        → service._worker()       # picks next job, calls ensure + forward
        → backend.ensure()        # load/switch model if needed
        → backend.forward()       # proxy to upstream, return RoutedResponse
        → app.py (stream or body) # return to client
```

## How to add a backend (4 steps)

1. Create `model_router/backends/<name>.py` subclassing `HTTPProxyBackend` (for HTTP servers) or `Backend` (for anything else)
2. Implement `ensure(model_key, model_config)` — load/switch the model; no-op if already active
3. Implement `forward(payload, wants_stream) -> RoutedResponse` — or inherit from `HTTPProxyBackend` which handles this
4. Register in `model_router/backends/__init__.py`: add to `create_backends()` with `elif backend_type == "<name>":`

## Key invariants

- Only one model is active at a time on the GPU (single-process constraint)
- `Scheduler.choose()` prefers the currently-loaded model for up to `max_loaded_jobs` requests or `max_loaded_seconds` before switching
- `LlamaCppBackend` shares a single instance across all `llamacpp`-backend models (they compete for one process)
- `OllamaBackend` instances are shared per `base_url`
- API backends (`claude`, `openai`, `gemini`) are stateless — `ensure()` is a no-op

## Test commands

```bash
# Run all tests
pytest tests/ -v

# Run specific suite
pytest tests/test_classifier.py -v
pytest tests/test_scheduler.py  -v
pytest tests/test_service.py    -v
pytest tests/test_app.py        -v
pytest tests/test_config.py     -v

# Run routing accuracy eval (requires sentence-transformers + internet for model download)
python evals/routing_accuracy.py

# Run classifier benchmark
python benchmarks/classifier_device_benchmark.py
```

## Test fixture

Tests use `tests/fixtures/model-router.test.json` (no real model files needed). The `FakeBackend` class in `test_service.py` implements `Backend` with a mock httpx client — import it for new tests:

```python
from test_service import FakeBackend
```

## Known edge cases

- Classifier is CPU-only. A CUDA version exists but uses 171 MiB VRAM — too much when a 12B model is loaded
- Short follow-ups like "improve it" lose coding context; see `evals/routing_multiturn_eval.py`
- Calendar events with technical titles (Python, Terraform) can misroute to the coding model
- Queue is non-durable — restart drops in-flight jobs
- `llm_classify` reclassification is disabled by default (`llm_classify: false` in config)

## Config structure

`model-router.json` at repo root (gitignored — copy from `config/model-router.example.json`).
`RouterConfig.models` is `dict[str, ModelConfig]`. `ModelConfig.backend` is the key field:

```python
backend: str          # "llamacpp" | "ollama" | "vllm" | "lmstudio" | "claude" | "openai" | "gemini"
public_id: str        # shown in /v1/models, sent to upstream
model_path: Path|None # llamacpp only
model_id: str|None    # ollama/vllm/lmstudio/api — model name on server
base_url: str|None    # server URL override
provider: str|None    # api backend only
api_key_env: str|None # env var name for API key
```
