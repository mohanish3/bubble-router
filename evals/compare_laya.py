"""Compare local NLI, hosted JEV, and local Laya on the shared routing dataset."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.compare_jev_nli import load_cases, load_jev_key

from model_router.classifier import Classification, TaskClassifier
from model_router.config import load_config
from model_router.jev_classifier import JevClassifier

METHODS = ("nli", "jev", "laya")

# Laya wants role-shaped keys and plain task vocabulary, not the router's model nicknames
# and NLI-tuned label text. Ablation on golden_300 (evals/laya_ablation_300_report.json),
# typed-decisions checkpoint: nicknames + config descriptions 83.3%, role keys + config
# descriptions 79.7%, role keys + plain descriptions 87.0%.
#
# Every error left was deliberative_work leaking into the other two, so deliberative_work
# spells out the case the model kept missing: reasoning *about* code is not writing code.
# That clause alone is +1.3 on golden_300 and +3.6 on the sealed 500-case holdout
# (evals/laya_optimization_report.json). Fitted readouts on top of it — per-class log-bias,
# logistic regression, extra sub-questions in the same pass — all won on one holdout and lost
# on the other, so the readout stays plain argmax.
LAYA_ROLE_CRITERIA = {
    "direct_assistance": "ordinary questions, explanations, advice, planning, casual requests",
    "deliberative_work": (
        "hard reasoning and analysis, including about code or systems, without writing code: "
        "proofs, comparisons, tradeoffs, root-cause analysis"
    ),
    "engineering_work": "code, programming, debugging, tests, refactoring, repository work",
}


class LayaClassifier:
    """Laya answering the same routing decision as JevClassifier, from local weights.

    Laya reads a dict state whose field names the instructions refer to, so the request goes
    in as {"request": ...} rather than the flat NLI text blob: same ablation, 72.3% -> 76.3%
    on the english checkpoint for that change alone.
    """

    def __init__(self, labels, threshold, *, default_model, model, device=None, subfolder=None,
                 criteria=None, key_to_model=None):
        import laya

        self.labels = labels
        self.threshold = threshold
        self.default_model = default_model or next(iter(labels))
        self._key_to_model = key_to_model or {key: key for key in labels}
        self._agent = laya.load(model, device=device, subfolder=subfolder)
        self._questions = {
            "route": {
                "type": "choice",
                "instructions": "Which kind of work does `request` require?",
                "criteria": criteria or labels,
            }
        }

    @staticmethod
    def _state(payload: dict, metadata: dict) -> dict:
        """Laya reads named fields. Passing TaskClassifier._text()'s flattened blob as one
        `request` string costs 9 points on golden_300 (87.0% -> 78.0%): the "metadata:" and
        "latest user:" scaffolding is noise inside the field the question asks about."""
        messages = payload.get("messages") or []
        user_messages = [m for m in messages if m.get("role") == "user"]
        state = {
            "request": str(user_messages[-1].get("content", "")) if user_messages else "",
        }
        prior = [str(m.get("content", "")) for m in user_messages[-4:-1]]
        if prior:
            state["earlier_turns"] = " | ".join(prior)[:1000]
        system = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        if system:
            state["system_prompt"] = system[:1000]
        body_metadata = payload.get("metadata")
        hints = dict(metadata)
        if isinstance(body_metadata, dict):
            hints.update(body_metadata)
        if hints:
            state["metadata"] = " ".join(f"{key}={value}" for key, value in hints.items())
        return state

    async def classify(self, payload, metadata=None) -> Classification:
        state = self._state(payload, metadata or {})
        answer = self._agent.predict(state, self._questions)["answers"]["route"]
        raw = answer["probabilities"]
        scores = {}
        for key, probability in raw.items():
            model = self._key_to_model.get(str(key))
            if model is None:
                raise ValueError(f"Laya returned an unknown routing key: {key}")
            scores[model] = float(probability)
        assert all(0.0 <= v <= 1.0 for v in scores.values()), "Laya probabilities out of range"
        selected = max(scores, key=scores.get)
        # ponytail: threshold on the selected probability, never answer["confidence"] — that
        # field is normalized Shannon entropy on a clamped-temperature checkpoint. Laya
        # spreads probability mass far flatter than NLI/JEV, so the router's shared 0.55
        # cutoff costs it 32 points (87.0% -> 55.3%); every cutoff tested loses accuracy,
        # hence the 0.0 default. Raise it only against a Laya-specific sweep.
        low_confidence = scores[selected] < self.threshold
        return Classification(
            self.default_model if low_confidence else selected, scores, low_confidence
        )


async def run_one(classifier, payload: dict, metadata: dict, expected: str) -> dict:
    """One timed classification. Retries once: a transient DNS/socket failure on the
    hosted classifier would otherwise be scored as a wrong answer."""
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            result = await classifier.classify(payload, metadata)
            return {
                "predicted": result.model,
                "correct": result.model == expected,
                "low_confidence": result.low_confidence,
                "scores": result.scores,
                "seconds": round(time.perf_counter() - started, 4),
            }
        except Exception as exc:
            failure = {
                "predicted": None,
                "correct": False,
                "seconds": round(time.perf_counter() - started, 4),
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
            if attempt == 2:
                return failure
            await asyncio.sleep(1.0)


def summarize(results: list[dict], role_to_model: dict[str, str]) -> dict:
    summary = {}
    for name in METHODS:
        scored = [row for row in results if "scores" in row[name]]
        correct = sum(row[name]["correct"] for row in results)
        times = sorted(row[name]["seconds"] for row in results)
        argmax_correct = sum(
            max(row[name]["scores"], key=row[name]["scores"].get) == row["expected"]
            for row in scored
        )
        summary[name] = {
            "correct": correct,
            "total": len(results),
            "accuracy": round(correct / len(results), 4),
            # Routing decision before the low-confidence fallback. The shared 0.55 threshold
            # is calibrated for NLI/JEV probability scales, so raw argmax is the comparable
            # accuracy across models that spread probability mass differently.
            "argmax_accuracy": round(argmax_correct / len(scored), 4) if scored else None,
            "argmax_correct": argmax_correct,
            "argmax_total": len(scored),
            "median_seconds": round(statistics.median(times), 4),
            "p95_seconds": round(times[min(len(times) - 1, int(0.95 * len(times)))], 4),
            "median_max_probability": round(
                statistics.median(max(row[name]["scores"].values()) for row in scored), 4
            ) if scored else None,
            "low_confidence_fallbacks": sum(row[name].get("low_confidence", False) for row in results),
            "errors": dict(Counter(row[name]["error"] for row in results if "error" in row[name])),
            "by_role": {
                role: {
                    "correct": sum(row[name]["correct"] for row in results if row.get("expected_role") == role),
                    "total": sum(row.get("expected_role") == role for row in results),
                }
                for role in role_to_model
            },
            "failures": [
                {"index": row["index"], "id": row.get("id", row["index"]), "prompt": row["prompt"][:120],
                 "expected": row["expected"], "predicted": row[name]["predicted"]}
                for row in results if not row[name]["correct"]
            ],
        }
    return summary


async def compare(config_path: Path, rounds: int, dataset: Path | None, laya_model: str,
                  device: str | None, output: Path | None, subfolder: str | None = None,
                  laya_threshold: float = 0.0) -> dict:
    config = load_config(config_path)
    labels = config.classifier["labels"]
    threshold = float(config.classifier["confidence_threshold"])
    default = config.aliases.get("general")

    nli_options = {}
    if "default_model" in inspect.signature(TaskClassifier).parameters:
        nli_options["default_model"] = default
    nli = TaskClassifier(config.classifier["model"], labels, threshold, **nli_options)
    await nli._load()
    if nli._predictor is TaskClassifier._heuristic_predict:
        raise RuntimeError("NLI model did not load; refusing to compare its heuristic fallback")
    jev = JevClassifier(labels, threshold, default_model=default)

    role_to_model = {
        "direct_assistance": config.aliases["general"],
        "deliberative_work": config.aliases["complex-reasoning"],
        "engineering_work": config.aliases["coding"],
    }
    laya = LayaClassifier(labels, laya_threshold, default_model=default, model=laya_model,
                          device=device, subfolder=subfolder,
                          criteria=LAYA_ROLE_CRITERIA, key_to_model=role_to_model)
    raw_rows, suite, dataset_sha256 = load_cases(dataset)
    rows = [
        {**case, "expected": role_to_model[case["expected_role"]]}
        if "expected_role" in case else
        {**case, "expected": {"gemma": role_to_model["direct_assistance"],
                              "qwen-opus": role_to_model["deliberative_work"],
                              "omnicoder": role_to_model["engineering_work"]}[case["expected"]]}
        for case in raw_rows
    ]

    # Warm up the local models so first-call CUDA/graph setup stays out of the medians.
    warmup = {"messages": [{"role": "user", "content": "warm up the local classifiers"}]}
    await nli.classify(warmup, {})
    await laya.classify(warmup, {})

    results = []
    for round_number in range(1, rounds + 1):
        for index, case in enumerate(rows, start=1):
            payload = {"messages": case.get("messages", [{"role": "user", "content": case["prompt"]}])}
            metadata = case.get("metadata", {})
            predictions = {}
            for name, classifier in (("nli", nli), ("jev", jev), ("laya", laya)):
                predictions[name] = await run_one(classifier, payload, metadata, case["expected"])
            results.append({"round": round_number, "index": index, **case, **predictions})
            if index % 25 == 0:
                print(f"Round {round_number}/{rounds}: {index}/{len(rows)}", flush=True)

    summary = summarize(results, role_to_model)
    report = {
        "suite": suite,
        "dataset_sha256": dataset_sha256,
        "unique_cases": len(rows),
        "rounds": rounds,
        "laya_model": laya_model,
        "laya_device": device or "auto",
        "laya_subfolder": subfolder,
        "laya_threshold": laya_threshold,
        "laya_criteria": LAYA_ROLE_CRITERIA,
        "role_to_model": role_to_model,
        "summary": summary,
        "agreement": {
            f"{a}_vs_{b}": sum(row[a]["predicted"] == row[b]["predicted"] for row in results)
            for a, b in (("nli", "jev"), ("nli", "laya"), ("jev", "laya"))
        },
        "all_wrong": sum(not any(row[m]["correct"] for m in METHODS) for row in results),
        "results": results,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--laya-model", default="convaiinnovations/laya")
    parser.add_argument("--device", default=None)
    parser.add_argument("--subfolder", default="typed-decisions",
                        help="Laya checkpoint subfolder: multilingual or typed-decisions")
    parser.add_argument("--laya-threshold", type=float, default=0.0,
                        help="Laya-specific confidence cutoff; 0 routes on argmax (see ablation)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_jev_key(args.env_file)
    report = asyncio.run(compare(args.config, args.rounds, args.dataset,
                                 args.laya_model, args.device, args.output, args.subfolder,
                                 args.laya_threshold))
    print(json.dumps({name: {k: report["summary"][name][k] for k in
                             ("accuracy", "correct", "total", "median_seconds", "p95_seconds", "errors")}
                      for name in METHODS}, indent=2))


if __name__ == "__main__":
    main()
