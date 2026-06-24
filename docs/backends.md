# Backends

## llama.cpp (`"llamacpp"`)

Manages a `llama-server` child process. Supports hot-swapping between GGUF models.

**Prerequisites:**
- Download `llama-server` binary (from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases))
- Set `binary_candidates` in config to its path
- Download GGUF model files
- Set `model_path` in each model's config

**Config:**
```json
{
  "binary_candidates": ["/path/to/llama-server"],
  "models": {
    "my-model": {
      "backend": "llamacpp",
      "public_id": "org/model:Q5_K_M",
      "model_path": "/path/to/model.gguf",
      "context_size": 131072,
      "max_predict": 8192
    }
  }
}
```

**Notes:**
- `common_args` in config are appended to every llama-server launch
- All llamacpp models share one process (one GPU slot)
- Model switching takes 5–6s for 9–12B models on an RTX 5070 Ti

---

## Ollama (`"ollama"`)

Routes to a running [Ollama](https://ollama.com) server. Auto-pulls models not yet downloaded.

**Prerequisites:**
- Install and start Ollama: `ollama serve`
- Optionally set `OLLAMA_MAX_LOADED_MODELS=1` to enforce single-model pinning

**Config:**
```json
{
  "models": {
    "my-model": {
      "backend":  "ollama",
      "public_id": "qwen2.5:7b",
      "model_id": "qwen2.5:7b",
      "base_url": "http://127.0.0.1:11434"
    }
  }
}
```

**Notes:**
- `model_id` is the Ollama model tag (same as `ollama pull <model_id>`)
- On first request, if the model is not local, the router pulls it automatically (first-request latency = download time)
- Previous model is asked to unload via `keep_alive=0` before switching

---

## vLLM (`"vllm"`)

Connects to a running [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server.

**Prerequisites:**
- Start vLLM: `python -m vllm.entrypoints.openai.api_server --model <model>`

**Config:**
```json
{
  "models": {
    "my-model": {
      "backend":  "vllm",
      "public_id": "Qwen/Qwen2.5-7B-Instruct",
      "model_id": "Qwen/Qwen2.5-7B-Instruct",
      "base_url": "http://127.0.0.1:8000"
    }
  }
}
```

**Notes:**
- vLLM starts with a fixed model; the router does not switch models
- Use separate vLLM instances for different models (different `base_url` values)

---

## LM Studio (`"lmstudio"`)

Connects to [LM Studio](https://lmstudio.ai)'s local server.

**Prerequisites:**
- Open LM Studio → load a model → enable "Local Server"

**Config:**
```json
{
  "models": {
    "my-model": {
      "backend":  "lmstudio",
      "public_id": "my-model",
      "model_id": "my-model",
      "base_url": "http://127.0.0.1:1234"
    }
  }
}
```

**Notes:**
- The router warns (but does not fail) if the expected `model_id` is not active in LM Studio
- Model switching must be done manually in the LM Studio UI

---

## Cloud APIs (`"claude"`, `"openai"`, `"gemini"`)

Stateless HTTP calls to cloud providers via their OpenAI-compatible endpoints.

**Prerequisites:**
- Set the API key env var: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`

**Config:**
```json
{
  "models": {
    "claude": {
      "backend":     "claude",
      "public_id":   "claude-sonnet-4-6",
      "model_id":    "claude-sonnet-4-6",
      "provider":    "claude",
      "api_key_env": "ANTHROPIC_API_KEY"
    },
    "gpt4o": {
      "backend":     "openai",
      "public_id":   "gpt-4o",
      "model_id":    "gpt-4o",
      "provider":    "openai"
    }
  }
}
```

**Provider endpoints:**
| Provider | Base URL |
|----------|---------|
| `claude` | `https://api.anthropic.com/v1` |
| `openai` | `https://api.openai.com/v1` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` |

Override with `"base_url"` for custom endpoints or proxies.

**Notes:**
- `ensure()` is a no-op — no process management
- `is_alive()` always returns `True`
- Streaming works via SSE passthrough
