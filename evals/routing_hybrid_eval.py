"""
Hybrid routing eval: NLI classify first, LLM reclassify on low_confidence.

Better than 25-char threshold because context_switch cases (LLM's strength)
all have long messages that a char threshold would wrongly send to NLI.

Logic:
  1. NLI classify
  2. If high confidence -> done
  3. If low confidence -> LLM reclassify with gemma (warm model, best performer)
  4. LLM result wins if it returns a valid label; else keep NLI result

Usage:
  python -m evals.routing_hybrid_eval
  python -m evals.routing_hybrid_eval --output result.json
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
from evals.routing_multiturn_eval import CASES, accuracy, print_report

ROUTER_URL  = "http://127.0.0.1:8090"
UPSTREAM_URL = "http://127.0.0.1:8080"

NEUTRAL_GRAMMAR = 'root ::= ("general" | "reasoning" | "coding")'

LABEL_TO_MODEL = {
    "general":   "gemma",
    "reasoning": "qwen-opus",
    "coding":    "omnicoder",
}

# Optimized tiebreaker prompt for gemma.
# Role: called ONLY when NLI is uncertain — mixed signals, context switches,
# ambiguous follow-ups. Needs to handle:
#   - "explain to a child" after coding discussion   -> general
#   - "write docker compose" after architecture Q&A  -> coding
#   - "no code, prove it" after algorithm exchange   -> reasoning
#   - short ambiguous: "improve it", "how so?"       -> inherit prior context
#
# Key optimizations vs earlier prompts:
#   1. Neutral output tokens (no model-name priming)
#   2. Explicit "latest message overrides" rule for topic switches
#   3. Explicit "inherit prior context" rule for short follow-ups
#   4. Few-shot examples covering every failure mode observed
TIEBREAKER_PROMPT = """\
You decide the routing for a conversation where automatic classification is uncertain.

Output exactly one word — nothing else:
  coding    = writing, debugging, reviewing, or modifying code
  reasoning = formal proof, mathematical derivation, logical analysis (no code)
  general   = questions, planning, tool use, writing, explanations, calendar, email

Decision rules (apply in order):
1. OVERRIDE: If the latest user message explicitly changes topic or intent \
("forget the code", "no code needed", "explain to a child", "now write the X"), \
classify by the NEW intent in that latest message — ignore prior context.
2. INHERIT: If the latest user message is short and context-dependent \
("improve it", "add tests", "how so?", "go deeper", "make it faster"), \
classify by what the PRIOR conversation was about.
3. DEFAULT: Classify by the clearest signal in the full conversation.

Few-shot examples:
[Prior: implementing LRU cache] [Latest: "Actually, explain this to a 10-year-old without any code"] -> general
[Prior: microservices Q&A] [Latest: "OK now write the Docker Compose for a 3-service setup"] -> coding
[Prior: sorting algorithm] [Latest: "No code changes. Formally prove this is correct for all inputs."] -> reasoning
[Prior: LRU cache code] [Latest: "improve it"] -> coding
[Prior: algorithm proof] [Latest: "what about the adversarial case?"] -> reasoning
[Prior: trip planning] [Latest: "what else should I pack?"] -> general
[Prior: email draft] [Latest: "make it shorter"] -> general
[Prior: FastAPI endpoint] [Latest: "add tests for that"] -> coding

Output one word: coding, reasoning, or general"""


async def ensure_gemma_loaded(config: Any) -> bool:
    """Trigger a warm-up request to make sure gemma is loaded."""
    public_id = config.models["gemma"].public_id
    try:
        async with httpx.AsyncClient(base_url=ROUTER_URL, timeout=180.0) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": public_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "stream": False,
            })
            return r.status_code == 200
    except Exception as exc:
        print(f"  gemma load failed: {exc}")
        return False


async def llm_tiebreak(messages: list[dict], model_public_id: str) -> str | None:
    label_list = "coding, reasoning, or general"
    payload = {
        "model": model_public_id,
        "messages": [
            {"role": "system", "content": TIEBREAKER_PROMPT},
            *messages,
            {"role": "user", "content": f"Output one word: {label_list}"},
        ],
        "max_tokens": 15,
        "temperature": 0.0,
        "stream": False,
        "grammar": NEUTRAL_GRAMMAR,
    }
    async with httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=30.0) as c:
        r = await c.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip().lower()
        neutral = next((t for t in LABEL_TO_MODEL if t in raw), None)
        return LABEL_TO_MODEL.get(neutral) if neutral else None


async def run_hybrid(
    cases: list[dict],
    classifier: TaskClassifier,
    gemma_public_id: str,
) -> list[dict]:
    results = []
    nli_used = 0
    llm_used = 0
    llm_overrides = 0

    for case in cases:
        payload = {"messages": case["messages"]}
        nli_result = await classifier.classify(payload, {})

        method = "nli"
        predicted = nli_result.model

        if nli_result.low_confidence:
            try:
                llm_prediction = await llm_tiebreak(case["messages"], gemma_public_id)
                if llm_prediction:
                    if llm_prediction != nli_result.model:
                        llm_overrides += 1
                    predicted = llm_prediction
                    method = "llm"
            except Exception as exc:
                pass  # keep NLI result on LLM error

        if method == "nli":
            nli_used += 1
        else:
            llm_used += 1

        results.append({
            "label": case["label"],
            "kind": case["kind"],
            "expected": case["expected"],
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "method_used": method,
            "nli_model": nli_result.model,
            "nli_low_confidence": nli_result.low_confidence,
            "nli_scores": {k: round(v, 4) for k, v in nli_result.scores.items()},
        })

    print(f"  NLI handled: {nli_used}/{len(cases)}, LLM tiebreak: {llm_used}/{len(cases)}, LLM overrides: {llm_overrides}")
    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config()
    classifier = TaskClassifier(
        config.classifier["model"],
        config.classifier["labels"],
        float(config.classifier["confidence_threshold"]),
    )
    gemma_public_id = config.models["gemma"].public_id

    print("Hybrid eval: NLI -> low_confidence -> LLM tiebreak (gemma)")
    print(f"Cases: {len(CASES)}")
    print("Ensuring gemma loaded ...")
    if not await ensure_gemma_loaded(config):
        print("ERROR: could not load gemma")
        return

    results = await run_hybrid(CASES, classifier, gemma_public_id)
    print_report("Hybrid (NLI + LLM tiebreak on low_confidence)", results)

    # Per-method breakdown
    nli_results = [r for r in results if r["method_used"] == "nli"]
    llm_results = [r for r in results if r["method_used"] == "llm"]
    if nli_results:
        nli_acc = sum(r["correct"] for r in nli_results) / len(nli_results)
        print(f"  NLI-only cases ({len(nli_results)}): {nli_acc*100:.1f}% accuracy")
    if llm_results:
        llm_acc = sum(r["correct"] for r in llm_results) / len(llm_results)
        print(f"  LLM tiebreak cases ({len(llm_results)}): {llm_acc*100:.1f}% accuracy")

    # Compare against baselines
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n{'='*55}")
    print(f"  Hybrid overall:      {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  NLI baseline:        15/17 (88.2%)")
    print(f"  Best LLM (gemma):    13/17 (76.5%)")
    print(f"{'='*55}")

    # Show which NLI decisions LLM changed and whether it helped
    overrides = [r for r in results if r["method_used"] == "llm" and r["nli_model"] != r["predicted"]]
    if overrides:
        print(f"\nLLM overrode NLI on {len(overrides)} case(s):")
        for r in overrides:
            outcome = "CORRECT" if r["correct"] else "WRONG"
            label = r["label"].encode("ascii", errors="replace").decode()
            print(f"  [{outcome}] NLI={r['nli_model']} -> LLM={r['predicted']} | {label}")

    if args.output:
        args.output.write_text(json.dumps({
            "total": total, "correct": correct,
            "accuracy": round(correct / total, 4),
            "accuracy_by_kind": accuracy(results)["by_kind"],
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
