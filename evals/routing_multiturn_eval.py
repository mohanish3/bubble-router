"""
Multi-turn routing eval: compares NLI classifier (Method 1) vs LLM reclassify (Method 2).

Method 1 runs on CPU — always available.
Method 2 requires llama-server running on port 8080 with any model loaded.

Usage:
  python -m evals.routing_multiturn_eval                      # Method 1 only
  python -m evals.routing_multiturn_eval --llm               # both methods
  python -m evals.routing_multiturn_eval --output result.json
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

from model_router.classifier import TaskClassifier
from model_router.config import load_config


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # --- follow_up: ambiguous short message that only makes sense with prior context ---
    {
        "kind": "follow_up",
        "expected": "omnicoder",
        "label": "coding follow-up: improve it",
        "messages": [
            {"role": "user", "content": "Implement a Python LRU cache with O(1) operations."},
            {"role": "assistant", "content": "Here's an implementation using OrderedDict…"},
            {"role": "user", "content": "improve it"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "omnicoder",
        "label": "coding follow-up: add tests",
        "messages": [
            {"role": "user", "content": "Write a FastAPI endpoint for user authentication."},
            {"role": "assistant", "content": "Here's the endpoint with JWT handling…"},
            {"role": "user", "content": "add tests for that"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "omnicoder",
        "label": "coding follow-up: edge case",
        "messages": [
            {"role": "user", "content": "Fix the race condition in this async queue."},
            {"role": "assistant", "content": "The bug is in the lock ordering…"},
            {"role": "user", "content": "can you also handle the edge case?"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "omnicoder",
        "label": "coding follow-up: same for others",
        "messages": [
            {"role": "user", "content": "Refactor this React component to use hooks."},
            {"role": "assistant", "content": "Here's the refactored component…"},
            {"role": "user", "content": "now do the same for the other components"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "omnicoder",
        "label": "coding follow-up: make it faster",
        "messages": [
            {"role": "user", "content": "Write a PostgreSQL query for monthly active users."},
            {"role": "assistant", "content": "SELECT date_trunc('month', created_at)…"},
            {"role": "user", "content": "make it faster"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "qwen-opus",
        "label": "reasoning follow-up: adversarial case",
        "messages": [
            {"role": "user", "content": "Prove that this greedy algorithm is optimal."},
            {"role": "assistant", "content": "By exchange argument, suppose there exists…"},
            {"role": "user", "content": "what about the adversarial case?"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "qwen-opus",
        "label": "reasoning follow-up: go deeper",
        "messages": [
            {"role": "user", "content": "Analyze three competing explanations for the Fermi paradox."},
            {"role": "assistant", "content": "The three main theories are…"},
            {"role": "user", "content": "go deeper on the second one"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "qwen-opus",
        "label": "reasoning follow-up: how so",
        "messages": [
            {"role": "user", "content": "Determine whether eventual consistency can satisfy these invariants."},
            {"role": "assistant", "content": "Under network partition, invariant 2 cannot hold…"},
            {"role": "user", "content": "how so?"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "gemma",
        "label": "general follow-up: what else",
        "messages": [
            {"role": "user", "content": "Help me plan a trip to Japan."},
            {"role": "assistant", "content": "Here's a 7-day itinerary…"},
            {"role": "user", "content": "what else should I pack?"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "gemma",
        "label": "general follow-up: make it shorter",
        "messages": [
            {"role": "user", "content": "Draft an email to reschedule tomorrow's meeting."},
            {"role": "assistant", "content": "Subject: Meeting Reschedule…"},
            {"role": "user", "content": "make it shorter"},
        ],
    },
    {
        "kind": "follow_up",
        "expected": "gemma",
        "label": "general follow-up: suggest more",
        "messages": [
            {"role": "user", "content": "Give me a 5-day vegetarian meal plan."},
            {"role": "assistant", "content": "Day 1: Oatmeal for breakfast…"},
            {"role": "user", "content": "suggest more variety for dinners"},
        ],
    },

    # --- truncation: long system prompt that eats the 512-token budget ---
    {
        "kind": "truncation",
        "expected": "omnicoder",
        "label": "long system + coding user",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant with the following tools: read_file, write_file, "
                    "bash, grep, glob, edit, web_search. Always read files before editing. "
                    "Run tests after every change. Write descriptive commit messages. "
                    "Follow the existing code style. Never break backward compatibility. "
                    "Prefer small incremental changes over large rewrites. "
                ) * 20,
            },
            {"role": "user", "content": "fix the authentication bug in auth.py"},
        ],
    },
    {
        "kind": "truncation",
        "expected": "qwen-opus",
        "label": "long system + reasoning user",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Always cite sources. "
                    "Maintain academic rigor. Use hedged language when uncertain. "
                    "Provide multiple perspectives. "
                ) * 30,
            },
            {"role": "user", "content": "prove whether this scheduling algorithm is optimal for all inputs"},
        ],
    },
    {
        "kind": "truncation",
        "expected": "gemma",
        "label": "long system + general user",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful personal assistant. You have access to calendar, email, "
                    "notes, and weather tools. Always be polite and concise. "
                    "Confirm actions before executing them. "
                ) * 25,
            },
            {"role": "user", "content": "find my dentist appointment next Tuesday"},
        ],
    },

    # --- context_switch: final message reverses intent of conversation ---
    {
        "kind": "context_switch",
        "expected": "gemma",
        "label": "switch: coding → explain to child",
        "messages": [
            {"role": "user", "content": "Implement a quicksort algorithm."},
            {"role": "assistant", "content": "Here's quicksort in Python…"},
            {
                "role": "user",
                "content": "Actually forget the code. Explain sorting to a 10-year-old without any code.",
            },
        ],
    },
    {
        "kind": "context_switch",
        "expected": "omnicoder",
        "label": "switch: general → write docker compose",
        "messages": [
            {"role": "user", "content": "What are the pros and cons of microservices?"},
            {"role": "assistant", "content": "Microservices offer several advantages…"},
            {"role": "user", "content": "OK now write the Docker Compose for a 3-service setup"},
        ],
    },
    {
        "kind": "context_switch",
        "expected": "qwen-opus",
        "label": "switch: coding → formal proof",
        "messages": [
            {"role": "user", "content": "Here's my sorting implementation."},
            {"role": "assistant", "content": "The implementation looks correct…"},
            {
                "role": "user",
                "content": "No code changes needed. Formally prove whether it's correct for all inputs.",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# NLI classifier runner (Method 1)
# ---------------------------------------------------------------------------

async def run_nli(cases: list[dict], classifier: TaskClassifier) -> list[dict]:
    results = []
    for case in cases:
        payload = {"messages": case["messages"]}
        result = await classifier.classify(payload, {})
        results.append({
            "label": case["label"],
            "kind": case["kind"],
            "expected": case["expected"],
            "predicted": result.model,
            "correct": result.model == case["expected"],
            "low_confidence": result.low_confidence,
            "scores": {k: round(v, 4) for k, v in result.scores.items()},
        })
    return results


# ---------------------------------------------------------------------------
# LLM classifier runner (Method 2)
# ---------------------------------------------------------------------------

LLM_CLASSIFY_SYSTEM = (
    "You are a request router. Classify the conversation to exactly one label.\n"
    "{rubric}\n"
    "Output the label name only. No explanation."
)


async def run_llm(cases: list[dict], config: Any, base_url: str) -> list[dict] | None:
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            health = await client.get("/health")
            if health.status_code != 200:
                print(f"llama-server not ready at {base_url} - skipping Method 2")
                return None
            # Discover the currently loaded model from /v1/models
            models_resp = await client.get("/v1/models")
            models_data = models_resp.json().get("data", [])
            if not models_data:
                print("No model loaded in llama-server - skipping Method 2")
                return None
            model_public_id: str = models_data[0]["id"]
            print(f"Method 2 using active model: {model_public_id}")
    except Exception as exc:
        print(f"llama-server unreachable ({exc}) - skipping Method 2")
        return None

    labels: dict[str, str] = config.classifier["labels"]
    label_list = ", ".join(labels)
    rubric = "\n".join(f"- {k}: {v}" for k, v in labels.items())
    grammar = "root ::= (" + " | ".join(f'"{k}"' for k in labels) + ")"
    system_prompt = LLM_CLASSIFY_SYSTEM.format(rubric=rubric)

    results = []
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def accuracy(results: list[dict]) -> dict:
    by_kind: dict[str, dict] = {}
    for row in results:
        k = row["kind"]
        by_kind.setdefault(k, {"correct": 0, "total": 0})
        by_kind[k]["total"] += 1
        if row["correct"]:
            by_kind[k]["correct"] += 1

    total = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "overall": {"correct": correct, "total": total, "accuracy": round(correct / total, 4) if total else 0},
        "by_kind": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
            for k, v in by_kind.items()
        },
        "failures": [r for r in results if not r["correct"]],
    }


def print_report(method: str, results: list[dict]) -> None:
    acc = accuracy(results)
    o = acc["overall"]
    print(f"\n{'='*60}")
    print(f"  {method}")
    print(f"  Overall: {o['correct']}/{o['total']} ({o['accuracy']*100:.1f}%)")
    for kind, v in acc["by_kind"].items():
        print(f"  {kind:20s}: {v['correct']}/{v['total']} ({v['accuracy']*100:.1f}%)")
    if acc["failures"]:
        print(f"\n  Failures ({len(acc['failures'])}):")
        for f in acc["failures"]:
            predicted = f.get("predicted", "?")
            label = f['label'].encode("ascii", errors="replace").decode()
            print(f"    [{f['kind']}] {label}")
            print(f"      expected={f['expected']}  got={predicted}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also run Method 2 (requires llama-server)")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config()
    classifier = TaskClassifier(
        config.classifier["model"],
        config.classifier["labels"],
        float(config.classifier["confidence_threshold"]),
    )

    print(f"Running multi-turn eval on {len(CASES)} cases …")

    nli_results = await run_nli(CASES, classifier)
    print_report("Method 1 - NLI (cross-encoder, improved context window)", nli_results)

    llm_results: list[dict] | None = None
    if args.llm:
        llm_results = await run_llm(CASES, config, args.upstream)
        if llm_results:
            print_report("Method 2 - LLM reclassify (active GPU model, full context)", llm_results)

    report = {
        "cases": len(CASES),
        "method1_nli": accuracy(nli_results),
        "method1_nli_results": nli_results,
    }
    if llm_results:
        report["method2_llm"] = accuracy(llm_results)
        report["method2_llm_results"] = llm_results

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
