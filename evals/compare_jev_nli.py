"""Compare only NLI and JEV on a shared labeled routing dataset."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.routing_accuracy import CURVEBALL_CASES, NORMAL_CASES

from model_router.classifier import TaskClassifier
from model_router.config import load_config
from model_router.jev_classifier import JevClassifier


def load_jev_key(path: Path | None) -> None:
    if path is None or os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY"):
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name in {"TYPESAFE_API_KEY", "JEV_API_KEY"}:
            os.environ[name] = value.strip().strip('"').strip("'")


def regression_cases() -> list[dict]:
    rows = [
        {"kind": "normal", "expected": expected, "prompt": prompt}
        for expected, prompt in NORMAL_CASES
    ]
    rows += [{"kind": "curveball", **case} for case in CURVEBALL_CASES]
    assert len(rows) == 50
    return rows


def load_cases(dataset: Path | None) -> tuple[list[dict], str, str | None]:
    if dataset is None:
        return regression_cases(), "regression_50", None
    content = dataset.read_bytes()
    rows = json.loads(content)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON list")
    roles = {"direct_assistance", "deliberative_work", "engineering_work"}
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("expected_role") not in roles:
            raise ValueError("Dataset contains an invalid expected role")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Dataset contains an empty prompt")
        if row.get("id") in ids:
            raise ValueError("Dataset contains duplicate IDs")
        ids.add(row.get("id"))
    return rows, dataset.name, hashlib.sha256(content).hexdigest()


async def compare(
    config_path: Path, rounds: int, output: Path | None, dataset: Path | None = None
) -> dict:
    config = load_config(config_path)
    labels = config.classifier["labels"]
    threshold = float(config.classifier["confidence_threshold"])
    default = config.aliases.get("general")
    nli_options = {}
    if "default_model" in inspect.signature(TaskClassifier).parameters:
        nli_options["default_model"] = default
    nli = TaskClassifier(config.classifier["model"], labels, threshold, **nli_options)
    jev = JevClassifier(labels, threshold, default_model=default)
    await nli._load()
    if nli._predictor is TaskClassifier._heuristic_predict:
        raise RuntimeError("NLI model did not load; refusing to compare its heuristic fallback")

    role_keys = {
        "gemma": config.aliases["general"],
        "qwen-opus": config.aliases["complex-reasoning"],
        "omnicoder": config.aliases["coding"],
    }
    raw_rows, suite, dataset_sha256 = load_cases(dataset)
    role_to_model = {
        "direct_assistance": role_keys["gemma"],
        "deliberative_work": role_keys["qwen-opus"],
        "engineering_work": role_keys["omnicoder"],
    }
    rows = [
        {**case, "expected": role_to_model[case["expected_role"]]}
        if "expected_role" in case else
        {**case, "expected": role_keys[case["expected"]]}
        for case in raw_rows
    ]
    results = []
    for round_number in range(1, rounds + 1):
        for index, case in enumerate(rows, start=1):
            payload = {"messages": case.get("messages", [{"role": "user", "content": case["prompt"]}])}
            metadata = case.get("metadata", {})
            predictions = {}
            for name, classifier in (("nli", nli), ("jev", jev)):
                started = time.perf_counter()
                try:
                    result = await classifier.classify(payload, metadata)
                    predictions[name] = {
                        "predicted": result.model,
                        "correct": result.model == case["expected"],
                        "low_confidence": result.low_confidence,
                        "scores": result.scores,
                        "seconds": round(time.perf_counter() - started, 4),
                    }
                except Exception as exc:
                    predictions[name] = {
                        "predicted": None,
                        "correct": False,
                        "seconds": round(time.perf_counter() - started, 4),
                        "error": type(exc).__name__,
                    }
            results.append({"round": round_number, "index": index, **case, **predictions})
            if index % 25 == 0:
                print(f"Round {round_number}/{rounds}: {index}/{len(rows)}", flush=True)
        print(f"Completed round {round_number}/{rounds}", flush=True)

    summary = {}
    for name in ("nli", "jev"):
        correct = sum(row[name]["correct"] for row in results)
        errors = Counter(row[name].get("error") for row in results if "error" in row[name])
        summary[name] = {
            "correct": correct,
            "total": len(results),
            "accuracy": round(correct / len(results), 4),
            "accuracy_by_round": [
                round(sum(row[name]["correct"] for row in results if row["round"] == run) / len(rows), 4)
                for run in range(1, rounds + 1)
            ],
            "median_seconds": round(statistics.median(row[name]["seconds"] for row in results), 4),
            "errors": dict(errors),
            "by_role": {
                role: {
                    "correct": sum(row[name]["correct"] for row in results if row.get("expected_role") == role),
                    "total": sum(row.get("expected_role") == role for row in results),
                }
                for role in role_to_model
            },
            "failures": [
                {"round": row["round"], "index": row["index"],
                 "expected": row["expected"], "predicted": row[name]["predicted"]}
                for row in results if not row[name]["correct"]
            ],
        }
    report = {
        "suite": suite,
        "dataset_sha256": dataset_sha256,
        "unique_cases": len(rows),
        "config": str(config_path),
        "role_to_model": role_to_model,
        "rounds": rounds,
        "summary": summary,
        "paired": {
            "nli_only_correct": sum(row["nli"]["correct"] and not row["jev"]["correct"] for row in results),
            "jev_only_correct": sum(row["jev"]["correct"] and not row["nli"]["correct"] for row in results),
            "both_wrong": sum(not row["jev"]["correct"] and not row["nli"]["correct"] for row in results),
        },
        "results": results,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_probabilities_csv(report: dict, output: Path) -> None:
    if report["rounds"] != 1:
        raise ValueError("Probability CSV requires exactly one round")
    roles = ("direct_assistance", "deliberative_work", "engineering_work")
    model_to_role = {model: role for role, model in report["role_to_model"].items()}
    fieldnames = ["id", "source", "prompt", "metadata_json", "expected_class"]
    fieldnames += [f"nli_p_{role}" for role in roles]
    fieldnames += [f"jev_p_{role}" for role in roles]
    fieldnames += ["nli_choice", "jev_choice", "nli_low_confidence", "jev_low_confidence"]
    rows = []
    for case in report["results"]:
        if "scores" not in case["nli"] or "scores" not in case["jev"]:
            raise RuntimeError(f"Missing classifier probabilities for {case.get('id', case['index'])}")
        row = {
            "id": case.get("id", case["index"]),
            "source": case.get("source", case.get("kind", "")),
            "prompt": case["prompt"],
            "metadata_json": json.dumps(case.get("metadata", {}), sort_keys=True),
            "expected_class": case.get("expected_role", model_to_role[case["expected"]]),
            "nli_choice": model_to_role[case["nli"]["predicted"]],
            "jev_choice": model_to_role[case["jev"]["predicted"]],
            "nli_low_confidence": case["nli"]["low_confidence"],
            "jev_low_confidence": case["jev"]["low_confidence"],
        }
        for method in ("nli", "jev"):
            for role in roles:
                row[f"{method}_p_{role}"] = case[method]["scores"][report["role_to_model"][role]]
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probabilities-csv", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.probabilities_csv and args.rounds != 1:
        parser.error("--probabilities-csv requires --rounds 1")
    load_jev_key(args.env_file)
    report = asyncio.run(compare(args.config, args.rounds, args.output, args.dataset))
    if args.probabilities_csv:
        write_probabilities_csv(report, args.probabilities_csv)
    print(json.dumps({
        "cases": report["unique_cases"],
        "nli_correct": report["summary"]["nli"]["correct"],
        "jev_correct": report["summary"]["jev"]["correct"],
        "errors": {name: report["summary"][name]["errors"] for name in ("nli", "jev")},
        "probabilities_csv": str(args.probabilities_csv) if args.probabilities_csv else None,
    }, indent=2))


if __name__ == "__main__":
    main()
