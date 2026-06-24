# bubble-router

> Call expert models when experts are required.

Route prompts to specialist models on your personal hardware — automatically.
**99.2% routing accuracy** across 500 diverse prompts. Zero VRAM overhead for the router itself.

One OpenAI-compatible endpoint. Multiple specialist models. Zero manual model selection.

---

## How it works

```
Prompt
  │
  ▼
┌────────────────────────────────────┐
│           bubble-router            │
│   POST /v1/chat/completions        │
│                                    │
│   1. NLI Classifier  (~16ms CPU)   │
│      → general / reasoning / code  │
│                                    │
│   2. Scheduler                     │
│      → prefers currently-loaded    │
│         model (fewer swaps)        │
│                                    │
│   3. Backend                       │
│      → llama.cpp / Ollama /        │
│         vLLM / LM Studio / API     │
└────────────────────────────────────┘
  │
  ▼
Response (OpenAI-compatible)
```

The classifier runs on CPU in ~16ms and uses zero VRAM — your GPU stays fully available for inference. The scheduler minimises model swaps by batching requests to the same model before switching.

### Example routing (default 3-model setup)

| Prompt | Routed to |
|--------|-----------|
| "What should I cook tonight?" | Gemma 4 12B (general) |
| "Prove this theorem using a multi-step derivation" | Qwen 3.5 Opus Distill 9B (reasoning) |
| "Debug this Python function and add regression tests" | OmniCoder 9B (coding) |
| `{"model": "qwen-opus", ...}` | Qwen 3.5 directly (explicit override) |

---

## Quickstart

```bash
git clone https://github.com/mohanish3/bubble-router
cd bubble-router
pip install -r requirements.txt

# Copy the example config and fill in your model paths
cp config/model-router.example.json model-router.json
# Edit model-router.json: set model_path for each model

# Start the router
python -m uvicorn model_router.main:app --host 127.0.0.1 --port 8090
```

Point any OpenAI-compatible client at `http://127.0.0.1:8090/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="")

response = client.chat.completions.create(
    model="auto",   # let the router decide
    messages=[{"role": "user", "content": "Debug this segfault..."}]
)
```

Use `model="auto"` to let the classifier route, or pass `model="coding"` / `model="qwen-opus"` to override.

---

## Supported backends

| Backend | Value in config | Notes |
|---------|----------------|-------|
| llama.cpp | `"llamacpp"` | Manages llama-server process. Hot-swaps GGUF files. |
| Ollama | `"ollama"` | Auto-pulls models. Pins one at a time. |
| vLLM | `"vllm"` | Connects to a running vLLM server. |
| LM Studio | `"lmstudio"` | Connects to LM Studio local server. |
| Claude API | `"claude"` | Calls Anthropic API directly. Set `ANTHROPIC_API_KEY`. |
| OpenAI API | `"openai"` | Calls OpenAI API directly. Set `OPENAI_API_KEY`. |
| Gemini API | `"gemini"` | Calls Gemini via OpenAI-compat endpoint. Set `GEMINI_API_KEY`. |

Mix backends freely — route easy questions to a local model and hard ones to an API.
See [`config/model-router.mixed.json`](config/model-router.mixed.json) for an example.

---

## Accuracy

The built-in NLI classifier achieves **99.2% routing accuracy** (496/500) on a diverse holdout set:

- 400/400 on normal prompts (100%)
- 96/100 on curveballs (ambiguous phrasing, context switches, negations)
- Classifier latency: ~16ms on CPU, zero VRAM consumed

See [`docs/accuracy.md`](docs/accuracy.md) for the full methodology and known failure modes.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Router + upstream status |
| GET | `/v1/models` | List configured models and aliases |
| POST | `/v1/chat/completions` | Route and forward (OpenAI-compatible) |
| GET | `/router/status` | Queue depths, load counts, latency stats |

---

## Configuration

Full reference: [`docs/configuration.md`](docs/configuration.md)

Key fields in `model-router.json`:

```json
{
  "classifier": {
    "labels": {
      "my-model": "description of what this model is best at"
    }
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
  "aliases": {
    "auto": "auto",
    "fast": "my-model"
  }
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick version:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

For AI agents, see [AGENTS.md](AGENTS.md).

---

## License

MIT
