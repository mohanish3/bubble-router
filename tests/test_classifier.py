import pytest

from model_router.classifier import TaskClassifier


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Fix this Python function and add regression tests", "omnicoder"),
        ("Prove this theorem using a multi-step derivation", "qwen-opus"),
        ("What should I cook tonight?", "gemma"),
        ("Use available tools to check my calendar", "gemma"),
    ],
)
async def test_classifier_fixtures(prompt, expected):
    classifier = TaskClassifier(
        "unused",
        {
            "gemma": "general reasoning or tool use",
            "qwen-opus": "complex reasoning",
            "omnicoder": "coding",
        },
        threshold=0.55,
        predictor=TaskClassifier._heuristic_predict,
    )
    result = await classifier.classify(
        {"messages": [{"role": "user", "content": prompt}]}, {}
    )
    assert result.model == expected


@pytest.mark.asyncio
async def test_metadata_hint_can_route_coding():
    classifier = TaskClassifier(
        "unused",
        {
            "gemma": "general reasoning or tool use",
            "qwen-opus": "complex reasoning",
            "omnicoder": "coding",
        },
        predictor=TaskClassifier._heuristic_predict,
    )
    result = await classifier.classify(
        {"messages": [{"role": "user", "content": "continue"}]},
        {"agent_id": "coding-debug-agent"},
    )
    assert result.model == "omnicoder"


@pytest.mark.asyncio
async def test_low_confidence_defaults_to_gemma():
    classifier = TaskClassifier(
        "unused",
        {"gemma": "general", "qwen-opus": "complex", "omnicoder": "coding"},
        threshold=0.9,
        predictor=lambda pairs: [0.1, 0.2, 0.3],
    )
    result = await classifier.classify(
        {"messages": [{"role": "user", "content": "ambiguous"}]}, {}
    )
    assert result.model == "gemma"
    assert result.low_confidence

