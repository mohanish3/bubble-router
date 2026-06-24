"""
Tests Method 2 LLM classification with each model loaded.

For each model: triggers a load via router, then runs LLM classify eval
directly against llama-server with that model active.

Usage:
  python -m evals.routing_multiturn_allmodels
  python -m evals.routing_multiturn_allmodels --output result.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parents[1]))

from model_router.config import load_config
from evals.routing_multiturn_eval import CASES, accuracy, print_report

ROUTER_URL = "http://127.0.0.1:8090"
UPSTREAM_URL = "http://127.0.0.1:8080"
LOAD_TIMEOUT = 180  # seconds to wait for model to load


async def trigger_model_load(model_key: str, config: Any) -> bool:
    """Send a minimal request through the router to force model load."""
    public_id = config.models[model_key].public_id
    print(f"  Triggering load of {model_key} ({public_id}) ...")
    try:
        async with httpx.AsyncClient(base_url=ROUTER_URL, timeout=LOAD_TIMEOUT) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": public_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                print(f"  Load complete.")
                return True
            print(f"  Load request returned {resp.status_code}")
            return False
    except Exception as exc:
        print(f"  Load failed: {exc}")
        return False


async def verify_active_model(expected_public_id: str) -> bool:
    """Confirm llama-server has the expected model loaded."""
    try:
        async with httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=10.0) as client:
            resp = await client.get("/v1/models")
            data = resp.json().get("data", [])
            if data and data[0]["id"] == expected_public_id:
                return True
            actual = data[0]["id"] if data else "none"
            print(f"  Warning: expected {expected_public_id}, got {actual}")
            return False
    except Exception as exc:
        print(f"  verify_active_model failed: {exc}")
        return False


async def run_llm_classify_with_model(
    cases: list[dict],
    labels: dict[str, str],
    model_public_id: str,
) -> list[dict]:
    label_list = ", ".join(labels)
    rubric = "\n".join(f"- {k}: {v}" for k, v in labels.items())
    grammar = "root ::= (" + " | ".join(f'"{k}"' for k in labels) + ")"
    system_prompt = (
        "You are a request router. Classify the conversation to exactly one label.\n"
        f"{rubric}\nOutput the label name only. No explanation."
    )

    results = []
    async with httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=60.0) as client:
        for case in cases:
            classify_payload = {
                "model": model_public_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *case["messages"],
                    {"role": "user", "content": f"Route to one of: {label_list}"},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
                "stream": False,
                "grammar": grammar,
            }
            try:
                response = await client.post("/v1/chat/completions", json=classify_payload)
                response.raise_for_status()
                msg = response.json()["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning_content") or "").strip().lower()
                predicted = next((k for k in labels if k in content), None)
            except Exception as exc:
                predicted = None
                content = f"ERROR: {exc}"

            results.append({
                "label": case["label"],
                "kind": case["kind"],
                "expected": case["expected"],
                "predicted": predicted,
                "correct": predicted == case["expected"],
                "raw_output": content,
            })
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config()
    labels: dict[str, str] = config.classifier["labels"]
    model_keys = list(config.models.keys())

    print(f"Testing Method 2 LLM classify with each model: {model_keys}")
    print(f"Cases: {len(CASES)}\n")

    all_results: dict[str, Any] = {}

    for model_key in model_keys:
        public_id = config.models[model_key].public_id
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key}  ({public_id})")
        print("="*60)

        ok = await trigger_model_load(model_key, config)
        if not ok:
            print(f"  Skipping {model_key} - could not load")
            continue

        confirmed = await verify_active_model(public_id)
        if not confirmed:
            print(f"  Skipping {model_key} - wrong model active after load")
            continue

        results = await run_llm_classify_with_model(CASES, labels, public_id)
        print_report(f"Method 2 LLM - {model_key}", results)

        all_results[model_key] = {
            "public_id": public_id,
            "accuracy": accuracy(results),
            "results": results,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY - Method 2 LLM accuracy by model")
    print(f"{'Model':<20} {'Overall':>10} {'follow_up':>12} {'truncation':>12} {'context_sw':>12}")
    print("-"*68)
    for key, data in all_results.items():
        acc = data["accuracy"]
        o = acc["overall"]
        bk = acc["by_kind"]
        fu = bk.get("follow_up", {}).get("accuracy", 0)
        tr = bk.get("truncation", {}).get("accuracy", 0)
        cs = bk.get("context_switch", {}).get("accuracy", 0)
        print(f"{key:<20} {o['accuracy']*100:>9.1f}% {fu*100:>11.1f}% {tr*100:>11.1f}% {cs*100:>11.1f}%")
    print("="*60)
    print("(Method 1 NLI baseline:     88.2%       100.0%       100.0%       33.3%)")

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
