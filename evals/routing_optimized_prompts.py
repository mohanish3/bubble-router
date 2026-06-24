"""
Tests optimized per-model routing prompts.

Key insight from baseline: model-name tokens cause identity bias.
Fix: use neutral tokens (general/reasoning/coding) + per-model anti-bias instructions.

Token→model mapping:
  general   -> gemma
  reasoning -> qwen-opus
  coding    -> omnicoder

Usage:
  python -m evals.routing_optimized_prompts
  python -m evals.routing_optimized_prompts --output result.json
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

# Neutral token grammar — no model names in output, eliminates identity priming
NEUTRAL_GRAMMAR = 'root ::= ("general" | "reasoning" | "coding")'

# Maps neutral output token back to router model key
LABEL_TO_MODEL = {
    "general": "gemma",
    "reasoning": "qwen-opus",
    "coding": "omnicoder",
}


# ---------------------------------------------------------------------------
# Per-model optimized prompts
# ---------------------------------------------------------------------------

# gemma: general-purpose model, understands context well.
# Failure mode: routes ambiguous short follow-ups ("improve it", "edge case")
# to itself as "general" even when prior context is coding.
# Fix: explicit rule to check prior context for follow-up resolution.
PROMPT_GEMMA = """\
Classify a conversation to exactly one task type. Output one word only.

Task types:
- coding: writing, debugging, refactoring, reviewing, or testing code; any request that produces code output
- reasoning: formal proofs, mathematical derivation, logical analysis, multi-step deduction with no code needed
- general: questions, explanations, planning, tool use, writing, calendar, email — anything not requiring code

Critical rule for short follow-up messages:
If the latest user message is short and context-dependent ("improve it", "add tests", \
"fix that", "make it faster", "edge case", "same for others"), look at what the \
PRIOR messages were about to determine the task type. Do not default to general.

Output one word: coding, reasoning, or general"""


# qwen-opus: reasoning-distilled model. Thinks via reasoning_content.
# Failure mode: routes almost everything to "general" (gemma) — over-defers.
# Fix: explicit signal that coding follow-ups stay coding, reasoning follow-ups
# stay reasoning. Only use general for truly general requests.
PROMPT_QWEN_OPUS = """\
Classify the conversation below. Output one word only — no explanation.

Definitions:
- coding: the FINAL user request needs code written, modified, debugged, reviewed, or tested
  (includes short follow-ups like "improve it", "add tests", "fix that" when prior context is coding)
- reasoning: the FINAL user request needs formal proof, theorem, mathematical analysis, or \
multi-step logical deduction WITHOUT code output
  (includes short follow-ups like "go deeper", "adversarial case", "how so" when prior context is reasoning)
- general: questions, planning, tool use, writing, email, calendar, explanations — no code, no formal proof

Decision rules:
1. Read the full conversation history to understand what "it", "that", "same" refer to.
2. A short follow-up inherits the task type of the prior exchange.
3. If the latest message explicitly switches topic (e.g. "forget the code, explain to a child"), \
reclassify based on the NEW intent.
4. Use general ONLY if neither coding nor reasoning applies.

Output one word: coding, reasoning, or general"""


# omnicoder: coding-specialized model.
# Failure mode: routes nearly everything to "coding" — max self-bias.
# Fix: strong explicit rules against over-routing to coding.
# Especially: "explain it", "what else", "make it shorter" are NOT coding.
PROMPT_OMNICODER = """\
Classify the conversation to exactly one task type. Output one word only.

- coding: explicitly requires writing code, modifying source files, debugging errors, \
reviewing diffs, or generating code output. Short coding follow-ups like "improve it", \
"add tests", "make it faster" count as coding when the prior conversation was about code.

- reasoning: requires formal mathematical proof, theorem derivation, logical analysis, \
or multi-step deduction. No code output needed.

- general: questions, explanations, planning, writing, calendar, email, tool use — \
anything that does NOT require producing or modifying code.

IMPORTANT rules:
- "explain it", "what else", "make it shorter", "suggest more" are GENERAL, not coding, \
  even if earlier messages mentioned technical topics.
- "go deeper", "adversarial case", "how so" after a reasoning discussion = reasoning.
- Only choose coding if writing or modifying actual code is the clear next step.

Output one word: coding, reasoning, or general"""


MODEL_PROMPTS = {
    "gemma": PROMPT_GEMMA,
    "qwen-opus": PROMPT_QWEN_OPUS,
    "omnicoder": PROMPT_OMNICODER,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def trigger_model_load(model_key: str, config: Any) -> bool:
    public_id = config.models[model_key].public_id
    print(f"  Loading {model_key} ...")
    try:
        async with httpx.AsyncClient(base_url=ROUTER_URL, timeout=180.0) as c:
            r = await c.post("/v1/chat/completions", json={
                "model": public_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": False,
            })
            return r.status_code == 200
    except Exception as exc:
        print(f"  Load failed: {exc}")
        return False


async def run_with_prompt(
    cases: list[dict],
    system_prompt: str,
    model_public_id: str,
) -> list[dict]:
    label_list = "coding, reasoning, or general"
    results = []
    async with httpx.AsyncClient(base_url=UPSTREAM_URL, timeout=60.0) as client:
        for case in cases:
            payload = {
                "model": model_public_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *case["messages"],
                    {"role": "user", "content": f"Output one word: {label_list}"},
                ],
                "max_tokens": 15,
                "temperature": 0.0,
                "stream": False,
                "grammar": NEUTRAL_GRAMMAR,
            }
            try:
                r = await client.post("/v1/chat/completions", json=payload)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                raw = (msg.get("content") or msg.get("reasoning_content") or "").strip().lower()
                # Extract neutral token then map to model key
                neutral = next((t for t in LABEL_TO_MODEL if t in raw), None)
                predicted = LABEL_TO_MODEL.get(neutral) if neutral else None
            except Exception as exc:
                raw = f"ERROR: {exc}"
                predicted = None

            results.append({
                "label": case["label"],
                "kind": case["kind"],
                "expected": case["expected"],
                "predicted": predicted,
                "correct": predicted == case["expected"],
                "raw_output": raw,
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config()

    print(f"Testing optimized per-model prompts. Cases: {len(CASES)}")
    print("Output tokens: general/reasoning/coding (neutral, no model-name bias)\n")

    all_results: dict[str, Any] = {}

    for model_key, prompt in MODEL_PROMPTS.items():
        public_id = config.models[model_key].public_id
        print(f"\n{'='*60}")
        print(f"MODEL: {model_key}")

        ok = await trigger_model_load(model_key, config)
        if not ok:
            print(f"  Skipping {model_key} - could not load")
            continue

        results = await run_with_prompt(CASES, prompt, public_id)
        print_report(f"Optimized prompt - {model_key}", results)
        all_results[model_key] = {"public_id": public_id, "accuracy": accuracy(results), "results": results}

    # Summary vs baselines
    print(f"\n{'='*68}")
    print("SUMMARY — optimized prompts vs baselines")
    print(f"{'Model':<20} {'Optimized':>10} {'Baseline LLM':>13} {'NLI M1':>8}")
    print("-"*55)
    baselines_llm = {"gemma": 0.706, "qwen-opus": 0.294, "omnicoder": 0.353}
    for key, data in all_results.items():
        opt = data["accuracy"]["overall"]["accuracy"]
        base = baselines_llm.get(key, 0)
        delta = opt - base
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {opt*100:>9.1f}%  {base*100:>10.1f}%  ({sign}{delta*100:.1f}pp)")
    print(f"\n  NLI Method 1 baseline: 88.2% overall")
    print("="*68)

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
