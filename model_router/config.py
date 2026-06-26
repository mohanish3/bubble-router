from __future__ import annotations

import json
import yaml
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardPinConfig:
    labels: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    backend: str
    public_id: str
    # llamacpp-specific
    model_path: Path | None = None
    context_size: int = 8192
    max_predict: int = 4096
    chat_template: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    # ollama / vllm / lmstudio / api
    model_id: str | None = None    # model name on the server
    base_url: str | None = None    # server base URL
    provider: str | None = None    # "claude" | "openai" | "gemini"
    api_key_env: str | None = None # env var name holding the API key


@dataclass(frozen=True)
class RouterConfig:
    root: Path
    raw: dict[str, Any]
    models: dict[str, ModelConfig]
    aliases: dict[str, str]
    hard_pin: HardPinConfig | None = None

    @property
    def server(self) -> dict[str, Any]:
        return self.raw["server"]

    @property
    def classifier(self) -> dict[str, Any]:
        return self.raw["classifier"]

    @property
    def pinning(self) -> dict[str, Any]:
        return self.raw.get("pinning", {})

    def resolve_model(self, requested: str) -> str:
        # First check hard-pin aliases (forces specific model)
        pinned = self.pinning.get("aliases", {})
        if requested in pinned:
            target = pinned[requested]
            if target == "auto":
                return "auto"
            if target in self.pinning.get("models", {}):
                return target
            if target in self.models:
                return target
        value = self.aliases.get(requested, requested)
        if value == "auto":
            return value
        if value in self.models:
            return value
        for key, model in self.models.items():
            if requested == model.public_id:
                return key
        raise KeyError(requested)

    def find_binary(self) -> str:
        for candidate in self.raw.get("binary_candidates", ["llama-server"]):
            path = Path(candidate)
            if not path.is_absolute():
                path = self.root / path
            if path.exists():
                return str(path.resolve())
            found = shutil.which(candidate)
            if found:
                return found
        raise FileNotFoundError("llama-server not found; set binary_candidates in config")

    def command(self, model_key: str, context_size: int | None = None) -> list[str]:
        model = self.models[model_key]
        if model.model_path is None:
            raise ValueError(f"Model {model_key!r} has no model_path (not a llamacpp model?)")
        if not model.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model.model_path}")
        args = [
            self.find_binary(),
            "-m", str(model.model_path),
            "--host", "127.0.0.1",
            "--port", "8080",
            "-c", str(context_size or model.context_size),
            "-n", str(model.max_predict),
            "-a", model.public_id,
        ]
        if model.chat_template:
            if not model.chat_template.exists():
                raise FileNotFoundError(f"Chat template not found: {model.chat_template}")
            args += ["--chat-template-file", str(model.chat_template)]
        return args + model.extra_args + [str(v) for v in self.raw.get("common_args", [])]


def load_config(path: Path | None = None) -> RouterConfig:
    config_path = path or Path(__file__).parents[1] / "model-router.json"
    if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.resolve()
    hard_pin = None
    if "pinning" in raw:
        hard_pin = HardPinConfig(
            labels=raw["pinning"].get("labels", {}),
            aliases=raw["pinning"].get("aliases", {}),
            models=raw["pinning"].get("models", {}),
        )
    models: dict[str, ModelConfig] = {}
    for key, value in raw["models"].items():
        backend = value.get("backend", "llamacpp")
        model_path: Path | None = None
        chat_template: Path | None = None
        if backend == "llamacpp":
            mp = value.get("model_path")
            if mp:
                model_path = (root / mp).resolve()
            t = value.get("chat_template")
            if t:
                chat_template = (root / t).resolve()
        models[key] = ModelConfig(
            key=key,
            backend=backend,
            public_id=value["public_id"],
            model_path=model_path,
            context_size=int(value.get("context_size", 8192)),
            max_predict=int(value.get("max_predict", 4096)),
            chat_template=chat_template,
            extra_args=list(value.get("extra_args", [])),
            model_id=value.get("model_id"),
            base_url=value.get("base_url"),
            provider=value.get("provider"),
            api_key_env=value.get("api_key_env"),
        )
    return RouterConfig(root=root, raw=raw, models=models, aliases=raw["aliases"], hard_pin=hard_pin)
