from pathlib import Path

from model_router.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "model-router.test.json"


def test_manifest_owns_expected_models_and_omnicoder_quant():
    config = load_config(FIXTURE)
    assert set(config.models) == {"gemma", "qwen-opus", "omnicoder"}
    assert config.models["omnicoder"].public_id.endswith(":Q6_K")
    assert config.models["omnicoder"].model_path.name == "omnicoder-9b-q6_k.gguf"
    assert config.resolve_model("coding") == "omnicoder"


def test_backend_field_parsed():
    config = load_config(FIXTURE)
    assert all(m.backend == "llamacpp" for m in config.models.values())


def test_aliases_resolve():
    config = load_config(FIXTURE)
    assert config.resolve_model("general") == "gemma"
    assert config.resolve_model("complex-reasoning") == "qwen-opus"
    assert config.resolve_model("auto") == "auto"
