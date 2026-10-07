"""Local text embeddings for OpenClaw memory search (OpenAI /v1/embeddings shape).

EmbeddingGemma 2 (text-only Q8_0 GGUF) runs in a loopback llama.cpp server
(launchd ai.bubble.embeddings), about 0.6 GB resident with no Python/torch.
llama.cpp does not apply the model's task prompts, so they are added here.
"""

from __future__ import annotations

from typing import Any

EMBED_URL = "http://127.0.0.1:8766/v1/embeddings"
MODEL_ID = "embeddinggemma-2"
MAX_INPUTS = 256
MAX_INPUT_CHARS = 32_000
PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}


def parse_inputs(payload: dict[str, Any]) -> list[str]:
    value = payload.get("input")
    texts = [value] if isinstance(value, str) else value
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("input must be a non-empty string or list of strings")
    if len(texts) > MAX_INPUTS:
        raise ValueError(f"at most {MAX_INPUTS} inputs per request")
    if any(len(t) > MAX_INPUT_CHARS for t in texts):
        raise ValueError(f"each input must be at most {MAX_INPUT_CHARS} characters")
    return texts


def prompted(payload: dict[str, Any], texts: list[str]) -> list[str]:
    prefix = PROMPTS["query" if payload.get("input_type") == "query" else "document"]
    return [prefix + text for text in texts]
