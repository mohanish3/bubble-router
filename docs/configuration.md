# Configuration reference

Configuration lives in `model-router.json` at the repo root (gitignored).
Copy `config/model-router.example.json` to start.

## `server`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Address the router listens on |
| `port` | int | `8090` | Port the router listens on |
| `upstream_base_url` | string | `"http://127.0.0.1:8080"` | Base URL for llamacpp backend (llama-server) |
| `api_key_env` | string | `"MODEL_ROUTER_API_KEY"` | Env var name for bearer token auth. Empty = no auth. |
| `default_api_key` | string | `""` | Fallback key if env var is unset |
| `warm_model` | string | first model | Model to load on startup |
| `ready_timeout_seconds` | int | `180` | How long to wait for llama-server to be ready |
| `stop_timeout_seconds` | int | `10` | How long to wait for graceful shutdown before force-kill |
| `queue_warning_depth` | int | `8` | Log a warning when a model's queue reaches this depth |
| `max_loaded_jobs` | int | `4` | Max requests to serve on one model before considering a switch |
| `max_loaded_seconds` | float | `30` | Max seconds to stay on one model before considering a switch |
| `llm_classify` | bool | `false` | Enable LLM reclassification of ambiguous follow-ups |
| `llm_classify_timeout` | float | `2.0` | Timeout in seconds for LLM reclassification |

## `classifier`

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | HuggingFace model ID for the NLI cross-encoder |
| `confidence_threshold` | float | Below this, route to the first/default model |
| `labels` | object | `{model_key: "description of what this model handles"}` |

The label descriptions are the hypothesis side of NLI inference. Write them to describe what tasks belong to that model, not the model's capabilities.

## `binary_candidates` (llamacpp only)

List of paths or names to search for the `llama-server` binary. Checked in order; first match wins. Relative paths are resolved from the repo root. Bare names (like `"llama-server"`) are searched on `PATH`.

## `aliases`

Maps friendly names to model keys or `"auto"`:

```json
{
  "aliases": {
    "auto":    "auto",
    "fast":    "gemma",
    "smart":   "qwen-opus"
  }
}
```

`"auto"` triggers the NLI classifier. Any other alias value must be a key in `models`.

## `models`

Each entry is a model key → config object. Fields depend on the backend.

### All backends

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `backend` | string | yes | One of: `llamacpp`, `ollama`, `vllm`, `lmstudio`, `claude`, `openai`, `gemini` |
| `public_id` | string | yes | Model ID exposed via `/v1/models` and sent to upstream |

### llamacpp

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | yes | Path to the `.gguf` file (absolute, or relative to repo root) |
| `context_size` | int | yes | Context window in tokens |
| `max_predict` | int | yes | Max tokens to generate per request |
| `chat_template` | string | no | Path to a `.jinja` chat template file |
| `extra_args` | array | no | Extra args appended after `common_args` for this model only |

### ollama / vllm / lmstudio

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Model name on the server (e.g., `"qwen2.5:7b"`) |
| `base_url` | string | no | Server URL (default per backend if omitted) |

### claude / openai / gemini

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Model name at the provider (e.g., `"claude-sonnet-4-6"`) |
| `provider` | string | yes | Same as backend type |
| `api_key_env` | string | no | Env var name for API key (sensible defaults per provider) |
| `base_url` | string | no | Override provider endpoint |

## `common_args` (llamacpp only)

Array of extra flags appended to every `llama-server` launch. See `config/model-router.example.json` for recommended GPU settings.
