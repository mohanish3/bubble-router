from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .api import APIBackend
from .base import Backend, RoutedResponse
from .llamacpp import LlamaCppBackend
from .lmstudio import LMStudioBackend
from .ollama import OllamaBackend
from .vllm import VLLMBackend

if TYPE_CHECKING:
    pass

__all__ = [
    "Backend",
    "RoutedResponse",
    "LlamaCppBackend",
    "OllamaBackend",
    "VLLMBackend",
    "LMStudioBackend",
    "APIBackend",
    "create_backends",
]


def create_backends(config: Any) -> dict[str, Backend]:
    """Instantiate one backend per model. Backends of the same type + base_url are shared."""
    instances: dict[str, Backend] = {}
    shared: dict[tuple[str, ...], Backend] = {}

    llamacpp_instance: LlamaCppBackend | None = None

    for model_key, model_config in config.models.items():
        backend_type = model_config.backend

        if backend_type == "llamacpp":
            if llamacpp_instance is None:
                llamacpp_instance = LlamaCppBackend(config)
            instances[model_key] = llamacpp_instance

        elif backend_type == "ollama":
            base_url = model_config.base_url or "http://127.0.0.1:11434"
            key = ("ollama", base_url)
            if key not in shared:
                shared[key] = OllamaBackend(base_url)
            instances[model_key] = shared[key]

        elif backend_type == "vllm":
            base_url = model_config.base_url or "http://127.0.0.1:8000"
            key = ("vllm", base_url)
            if key not in shared:
                shared[key] = VLLMBackend(base_url)
            instances[model_key] = shared[key]

        elif backend_type == "lmstudio":
            base_url = model_config.base_url or "http://127.0.0.1:1234"
            key = ("lmstudio", base_url)
            if key not in shared:
                shared[key] = LMStudioBackend(base_url)
            instances[model_key] = shared[key]

        elif backend_type in ("claude", "openai", "gemini"):
            instances[model_key] = APIBackend(backend_type, model_config)

        else:
            raise ValueError(
                f"Unknown backend type {backend_type!r} for model {model_key!r}. "
                f"Valid: llamacpp, ollama, vllm, lmstudio, claude, openai, gemini"
            )

    return instances
