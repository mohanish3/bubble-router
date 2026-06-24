# How it works

## Classifier pipeline

```
Incoming request (messages[])
         │
         ▼
  ┌──────────────────────────────────┐
  │        Text assembly             │
  │  priority: metadata > latest     │
  │  user message > prior turns >    │
  │  system prompt (truncated last)  │
  └──────────┬───────────────────────┘
             │  max 2000 chars
             ▼
  ┌──────────────────────────────────┐
  │   NLI cross-encoder              │
  │   cross-encoder/nli-MiniLM2      │
  │   -L6-H768  (~16ms CPU)          │
  │                                  │
  │  Pairs: (text, "This task is X") │
  │  Output: entailment score 0–1    │
  └──────────┬───────────────────────┘
             │
             ▼
  ┌──────────────────────────────────┐
  │   Intent priors                  │
  │   (hand-crafted regex boosts)    │
  │                                  │
  │  "implement", "debug" → +coding  │
  │  "prove", "derive" → +reasoning  │
  │  "calendar tool" → +general      │
  │  "explain to a child" → +general │
  └──────────┬───────────────────────┘
             │
             ▼
  ┌──────────────────────────────────┐
  │   Softmax over combined logits   │
  │   → selected model + confidence  │
  │                                  │
  │  score < threshold → "gemma"     │
  │  (safe default for low conf)     │
  └──────────┬───────────────────────┘
             │
             ▼
         Classification
```

## Why NLI?

NLI (Natural Language Inference) directly models "does this request *entail* this task label?" — a better fit than cosine similarity over embeddings because it reasons about the relationship, not just topic overlap.

The cross-encoder model runs on CPU, uses zero VRAM, and takes ~16ms per request. A CUDA version is ~4× faster but requires 171 MiB of VRAM — too costly when a 12B model is already loaded on a 12 GB card.

## Scheduler

The scheduler minimises model swaps (which take 5–6 seconds for GGUF hot-swap) by batching requests to the currently-loaded model:

- Keeps serving the loaded model for up to `max_loaded_jobs` (default 4) requests
- OR up to `max_loaded_seconds` (default 30) seconds
- After either limit, picks the job with the globally oldest enqueue time
- Jobs are per-model FIFO queues — ordering within a model is always preserved

## Model switching (llamacpp)

When the scheduler picks a model that isn't currently loaded:

1. `SIGTERM` (Linux/Mac) or `CTRL_BREAK` (Windows) sent to llama-server
2. Wait up to `stop_timeout_seconds` (default 10s); force-kill if exceeded
3. Start new llama-server process with the target model's `model_path`
4. Poll `/v1/models` until ready, up to `ready_timeout_seconds` (default 180s)
5. Log cold-load latency; forward the waiting job

Typical swap time: **5–6 seconds** on an RTX 5070 Ti with 9–12B GGUF models.
