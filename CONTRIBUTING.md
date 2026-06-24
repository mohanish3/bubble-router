# Contributing to bubble-router

## Setup

```bash
git clone https://github.com/skysavv/bubble-router
cd bubble-router
pip install -e ".[dev]"

# Create your local config (gitignored)
cp config/model-router.example.json model-router.json
# Edit model-router.json: set model_path for each model

# Start the router
python -m uvicorn model_router.main:app --host 127.0.0.1 --port 8090
```

## Run tests

```bash
pytest tests/ -v
```

Tests use a fixture config (`tests/fixtures/model-router.test.json`) with placeholder paths. No real model files or running servers needed.

## Module map

| File | Purpose |
|------|---------|
| `model_router/classifier.py` | NLI classifier — maps prompts to model labels |
| `model_router/scheduler.py` | FIFO queue with loaded-model preference |
| `model_router/service.py` | Orchestration: classify → ensure → forward |
| `model_router/config.py` | Config loading and validation |
| `model_router/app.py` | FastAPI app and endpoints |
| `model_router/backends/` | Per-backend implementations |

See [AGENTS.md](AGENTS.md) for the full data flow and invariants.

## Adding a backend

1. Create `model_router/backends/<name>.py` — subclass `HTTPProxyBackend` for HTTP servers, or `Backend` directly
2. Implement `ensure(model_key, model_config)` — load/switch model; no-op if already active
3. `forward(payload, wants_stream)` — inherit from `HTTPProxyBackend` or implement directly
4. Register in `create_backends()` in `model_router/backends/__init__.py`
5. Add the backend type string to `config/model-router.schema.json` enum

## PR checklist

- [ ] `pytest tests/ -v` passes
- [ ] New backend: subclasses `Backend`, registered in `__init__.py`, enum entry in schema
- [ ] Config change: update `model-router.schema.json` + both example configs
- [ ] Classifier change: run `python evals/routing_accuracy.py` — accuracy must stay ≥ 99%
- [ ] New test for non-trivial logic

## Code style

- No comments unless the WHY is non-obvious
- No docstrings on self-explanatory functions
- Type hints on all public functions
- Tests use `FakeBackend` from `test_service.py` — import it, don't duplicate

## Questions

Open an issue or start a discussion on GitHub.
