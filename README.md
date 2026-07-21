# bubble-router

Routes prompts to specialist local models automatically, based on classified intent, instead of picking a model per request by hand.

99.2% routing accuracy (496/500) on a held-out set of 500 prompts (see [methodology and known failure modes](docs/accuracy.md)). Classifier runs on CPU in ~16ms with zero VRAM overhead, so it doesn't compete with inference for GPU memory.

One OpenAI-compatible endpoint. Multiple specialist backends. No manual model selection.

## How it works

```
Prompt
  |
  v
1. NLI classifier (~16ms, CPU)   -> general / reasoning / code
2. Scheduler                     -> prefers the currently loaded model, to minimize swaps
3. Backend                       -> llama.cpp / Ollama / vLLM / LM Studio / API
  |
  v
Response (OpenAI-compatible)
```

### Example routing (default 3-model setup)

| Prompt | Routed to |
|---|---|
| "What should I cook tonight?" | Gemma 4 12B (general) |
| "Prove this theorem using a multi-step derivation" | Qwen 3.5 Opus Distill 9B (reasoning) |
| "Debug this Python function and add regression tests" | OmniCoder 9B (coding) |
| `{"model": "qwen-opus", ...}` | Qwen 3.5 directly (explicit override) |

## Quickstart

```bash
git clone https://github.com/mohanish3/bubble-router
cd bubble-router
pip install -r requirements.txt

cp config/model-router.example.json model-router.json
# edit model-router.json: set model_path for each model

python -m uvicorn model_router.main:app --host 127.0.0.1 --port 8090
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="")

response = client.chat.completions.create(
    model="auto",   # let the router decide
    messages=[{"role": "user", "content": "Debug this segfault..."}]
)
```

Use `model="auto"` to route automatically, or pass a model/alias name (e.g. `"coding"`, `"qwen-opus"`) to override.

## Supported backends

| Backend | Config value | Notes |
|---|---|---|
| llama.cpp | `llamacpp` | Manages the llama-server process; hot-swaps GGUF files |
| Ollama | `ollama` | Auto-pulls models; pins one at a time |
| vLLM | `vllm` | Connects to a running vLLM server |
| LM Studio | `lmstudio` | Connects to the LM Studio local server |
| Claude API | `claude` | Requires `ANTHROPIC_API_KEY` |
| OpenAI API | `openai` | Requires `OPENAI_API_KEY` |
| Gemini API | `gemini` | Via an OpenAI-compatible endpoint; requires `GEMINI_API_KEY` |

Backends can be mixed: route routine prompts to a local model and hard ones to an API. Example: [config/model-router.mixed.json](config/model-router.mixed.json).

## Accuracy

99.2% (496/500) on a diverse holdout set: 400/400 on normal prompts, 96/100 on curveballs (ambiguous phrasing, context switches, negations). Full methodology and failure modes: [docs/accuracy.md](docs/accuracy.md).

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Router and upstream status |
| GET | `/v1/models` | List configured models and aliases |
| POST | `/v1/chat/completions` | Route and forward (OpenAI-compatible) |
| GET | `/router/status` | Queue depths, load counts, latency stats |

## Configuration

Full reference: [docs/configuration.md](docs/configuration.md)

```json
{
  "classifier": {
    "labels": { "my-model": "description of what this model is best at" }
  },
  "models": {
    "my-model": {
      "backend": "llamacpp",
      "public_id": "org/model:quant",
      "model_path": "/path/to/model.gguf",
      "context_size": 131072,
      "max_predict": 8192
    }
  },
  "aliases": { "auto": "auto", "fast": "my-model" }
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

For AI agents, see [AGENTS.md](AGENTS.md).

## License

MIT
