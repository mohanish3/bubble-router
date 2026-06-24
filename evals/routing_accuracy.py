from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

from model_router.classifier import TaskClassifier
from model_router.config import load_config


NORMAL_CASES = [
    ("gemma", "Explain why the sky appears blue during the day."),
    ("gemma", "Summarize the pros and cons of renting versus buying a home."),
    ("gemma", "Draft a polite email rescheduling tomorrow's meeting."),
    ("gemma", "Give me a five-day vegetarian meal plan."),
    ("gemma", "What questions should I ask before adopting a dog?"),
    ("gemma", "Compare solar and wind energy in plain language."),
    ("gemma", "Help me plan a three-day itinerary for Jaipur."),
    ("gemma", "Turn these notes into a concise project update."),
    ("gemma", "Use my calendar tool to find an open hour next Tuesday."),
    ("gemma", "Search my notes for the latest budget decision."),
    ("gemma", "Check the weather tool and suggest what clothing to pack."),
    ("gemma", "Create a checklist for moving into a new apartment."),
    ("gemma", "Explain compound interest to a high-school student."),
    ("gemma", "Rewrite this paragraph to sound more professional."),
    ("qwen-opus", "Prove that the square root of 2 is irrational."),
    ("qwen-opus", "Derive the time complexity of this recursive recurrence: T(n)=2T(n/2)+n log n."),
    ("qwen-opus", "Analyze three competing explanations for the Fermi paradox and identify their weakest assumptions."),
    ("qwen-opus", "Design a rigorous experiment to distinguish correlation from causation in this observational study."),
    ("qwen-opus", "Solve this multi-step logic puzzle and show every inference."),
    ("qwen-opus", "Find the root cause of a distributed-system outage from these conflicting timelines."),
    ("qwen-opus", "Compare two database architectures under latency, consistency, and failure-domain constraints."),
    ("qwen-opus", "Derive the optimal strategy for this constrained game-theory problem."),
    ("qwen-opus", "Evaluate this acquisition using scenario analysis, sensitivity analysis, and downside risk."),
    ("qwen-opus", "Construct a formal argument for and against this constitutional interpretation."),
    ("qwen-opus", "Explain why this proposed perpetual-motion machine cannot work using thermodynamics."),
    ("qwen-opus", "Develop a causal model for declining customer retention across five interacting factors."),
    ("qwen-opus", "Optimize this supply chain under uncertain demand and capacity constraints."),
    ("omnicoder", "Implement a Python LRU cache with O(1) get and put operations."),
    ("omnicoder", "Debug this TypeScript error: Type 'undefined' is not assignable to type 'User'."),
    ("omnicoder", "Review this pull request for concurrency bugs and missing tests."),
    ("omnicoder", "Write a SQL migration that adds a unique partial index safely."),
    ("omnicoder", "Refactor this React component to remove duplicated state."),
    ("omnicoder", "Add integration tests for the FastAPI authentication middleware."),
    ("omnicoder", "Explain why this Rust function fails the borrow checker and fix it."),
    ("omnicoder", "Design a REST API and provide an OpenAPI schema for it."),
    ("omnicoder", "Investigate this stack trace and identify the failing code path."),
    ("omnicoder", "Convert this Bash deployment script to PowerShell."),
    ("omnicoder", "Optimize this PostgreSQL query and explain the execution plan."),
    ("omnicoder", "Create a GitHub Actions workflow for lint, tests, and release builds."),
    ("omnicoder", "Fix the race condition in this asynchronous queue implementation."),
]


CURVEBALL_CASES = [
    {
        "expected": "omnicoder",
        "prompt": "Don't write code. Diagnose why our Python API deadlocks under load and describe the exact code-level fix.",
    },
    {
        "expected": "gemma",
        "prompt": "My grocery list contains the words Python, Rust, and Java. Group the groceries by supermarket aisle.",
    },
    {
        "expected": "qwen-opus",
        "prompt": "No code is needed: reason carefully about whether this lock-free algorithm can violate linearizability.",
    },
    {
        "expected": "gemma",
        "prompt": "Use the email tool to send the approved release announcement. Do not redesign or debug anything.",
    },
    {
        "expected": "omnicoder",
        "prompt": "Write the smallest patch for this bug; the business explanation and architecture discussion are already complete.",
    },
    {
        "expected": "qwen-opus",
        "prompt": "A program prints the correct answer, but prove whether its underlying greedy algorithm is correct for every input.",
    },
    {
        "expected": "gemma",
        "prompt": "Explain what a software API is to a restaurant owner without discussing implementation.",
    },
    {
        "expected": "omnicoder",
        "prompt": "The task sounds simple: rename one field across the schema, migrations, serializers, clients, and tests without breaking compatibility.",
        "metadata": {"agent_id": "repository-maintainer"},
    },
    {
        "expected": "qwen-opus",
        "prompt": "First ignore the obvious coding solution. Determine whether the requirements are mutually consistent, then derive the feasible cases.",
    },
    {
        "expected": "gemma",
        "prompt": "The meeting title is 'Debug API architecture'. Find it on my calendar and tell me its start time.",
    },
]

HOLDOUT_NORMAL_CASES = [
    ("gemma", "Explain how inflation affects household savings."),
    ("gemma", "Draft a short apology for arriving late to an appointment."),
    ("gemma", "Make a packing list for a week-long beach holiday."),
    ("gemma", "Use my contacts and email tools to send Priya the agenda."),
    ("gemma", "Find tomorrow's dentist appointment on my calendar."),
    ("gemma", "Summarize this interview transcript into five bullets."),
    ("gemma", "Suggest indoor activities for a rainy weekend with children."),
    ("qwen-opus", "Determine whether this voting system satisfies independence of irrelevant alternatives."),
    ("qwen-opus", "Show that the proposed invariant is sufficient to establish termination."),
    ("qwen-opus", "Reconcile these contradictory witness accounts and rank the plausible timelines."),
    ("qwen-opus", "Assess whether this monetary-policy proposal remains stable under three adverse scenarios."),
    ("qwen-opus", "Work through this probability puzzle and justify each conditional dependency."),
    ("qwen-opus", "Identify hidden assumptions in this argument and test whether its conclusion follows."),
    ("omnicoder", "Fix an npm dependency conflict that appears only in the CI build."),
    ("omnicoder", "Add pagination and validation to this Express endpoint."),
    ("omnicoder", "Review this Terraform module for unsafe defaults."),
    ("omnicoder", "Write unit tests for a Go HTTP handler."),
    ("omnicoder", "Investigate why this Docker container exits with code 137."),
    ("omnicoder", "Replace unsafe type assertions in these test fixtures."),
    ("omnicoder", "Create a backward-compatible database column rename."),
]


HOLDOUT_CURVEBALL_CASES = [
    {
        "expected": "gemma",
        "prompt": "The book club meeting is named 'Python Debugging'. Locate it in my calendar.",
    },
    {
        "expected": "qwen-opus",
        "prompt": "Do not implement anything. Decide whether eventual consistency can satisfy these four business invariants simultaneously.",
    },
    {
        "expected": "omnicoder",
        "prompt": "No architecture essay: identify why the production deployment rolls back and supply the configuration change.",
        "metadata": {"agent_id": "devops-agent"},
    },
    {
        "expected": "gemma",
        "prompt": "Explain the phrase 'race condition' using a supermarket checkout analogy.",
    },
    {
        "expected": "qwen-opus",
        "prompt": "Although this is about source code, establish whether the transformation preserves semantics for all possible inputs.",
    },
]


def build_large_suite() -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    general_actions = [
        "summarize", "explain", "compare", "outline", "rewrite", "brainstorm",
        "plan", "list", "describe", "organize",
    ]
    general_topics = [
        "household budgeting", "healthy meal planning", "travel packing",
        "renewable energy", "public speaking", "home gardening",
        "time management", "sleep habits", "career interviews",
        "team communication", "language learning", "museum visits",
        "personal fitness", "pet adoption",
    ]
    general_formats = [
        "for a beginner", "in five bullets", "using plain language",
        "with a short example", "as a practical checklist",
    ]

    complex_actions = [
        "prove", "derive", "determine whether", "evaluate rigorously",
        "construct a formal argument about", "identify hidden assumptions in",
        "solve step by step", "analyze competing explanations for",
        "establish necessary and sufficient conditions for",
        "test the logical consistency of",
    ]
    complex_topics = [
        "a graph-theory claim", "a probability paradox",
        "a causal inference proposal", "a voting-system property",
        "a constrained optimization problem", "a game-theory equilibrium",
        "a thermodynamics argument", "a constitutional interpretation",
        "a monetary-policy scenario", "a distributed-consensus invariant",
        "an experimental design", "a supply-chain decision",
        "a statistical identification strategy", "a formal logic puzzle",
    ]
    complex_constraints = [
        "and justify every inference", "under three adverse scenarios",
        "while ranking alternative hypotheses", "for all possible cases",
        "without assuming the conclusion", "with sensitivity analysis",
    ]

    coding_actions = [
        "implement", "debug", "refactor", "add tests for", "review",
        "optimize", "migrate", "fix", "instrument", "document",
    ]
    coding_topics = [
        "a Python caching service", "a TypeScript authentication middleware",
        "a React state-management component", "a PostgreSQL migration",
        "a Rust async worker", "a Go HTTP handler",
        "a Docker deployment", "a GitHub Actions workflow",
        "a Terraform module", "a FastAPI endpoint",
        "a Redis-backed job queue", "a Bash release script",
        "an OpenAPI client", "an npm monorepo build",
    ]
    coding_constraints = [
        "with regression tests", "without breaking compatibility",
        "and explain the failing code path", "for production use",
        "with safe rollback behavior", "and include validation",
    ]

    def expand(
        expected: str,
        actions: list[str],
        topics: list[str],
        suffixes: list[str],
        count: int,
    ) -> list[tuple[str, str]]:
        rows = []
        for action in actions:
            for topic in topics:
                for suffix in suffixes:
                    rows.append((expected, f"{action.capitalize()} {topic} {suffix}."))
                    if len(rows) == count:
                        return rows
        raise ValueError(f"Not enough templates for {expected}")

    normal = (
        expand("gemma", general_actions, general_topics, general_formats, 134)
        + expand("qwen-opus", complex_actions, complex_topics, complex_constraints, 133)
        + expand("omnicoder", coding_actions, coding_topics, coding_constraints, 133)
    )

    curveball = []
    calendar_titles = [
        "Python API Review", "Debugging Workshop", "Rust Migration",
        "Database Architecture", "Race Condition Retrospective",
        "Terraform Planning", "CI Failure Review", "OpenAPI Discussion",
        "Docker Networking", "Code Quality Meeting", "Algorithm Seminar",
        "Production Incident",
    ]
    tool_verbs = ["find", "locate", "check", "tell me the time of"]
    for title, verb in [
        (title, verb) for title in calendar_titles for verb in tool_verbs
    ][:34]:
        curveball.append({
            "expected": "gemma",
            "prompt": (
                f"{verb.capitalize()} the calendar event titled '{title}'. "
                "This is a calendar lookup only; do not analyze its technical words."
            ),
        })

    proof_subjects = [
        "this greedy algorithm", "this schema transformation",
        "this lock-free queue", "this database consistency model",
        "this compiler optimization", "this distributed protocol",
        "this recursive program", "this caching policy",
        "this authorization rule", "this migration strategy",
        "this scheduling algorithm",
    ]
    proof_verbs = [
        "prove whether", "determine whether", "establish whether",
        "reason carefully about whether",
    ]
    for subject, verb in [
        (subject, verb) for subject in proof_subjects for verb in proof_verbs
    ][:33]:
        curveball.append({
            "expected": "qwen-opus",
            "prompt": (
                f"Do not implement or patch anything. {verb.capitalize()} {subject} "
                "is correct for every valid input, and justify all assumptions."
            ),
        })

    repair_subjects = [
        "Python API deadlock", "TypeScript build failure",
        "Docker rollback loop", "PostgreSQL migration bug",
        "React hydration error", "Rust ownership failure",
        "Terraform state drift", "CI release failure",
        "FastAPI authentication regression", "Redis queue race",
        "npm dependency conflict",
    ]
    repair_outputs = [
        "supply the exact patch", "provide the configuration change",
        "write the regression test", "show the corrected function",
    ]
    for subject, output in [
        (subject, output) for subject in repair_subjects for output in repair_outputs
    ][:33]:
        curveball.append({
            "expected": "omnicoder",
            "prompt": (
                f"Skip the broad architecture essay. Diagnose the {subject} and "
                f"{output}; keep the change backward-compatible."
            ),
            "metadata": {"agent_id": "repository-maintainer"},
        })

    return normal, curveball


async def evaluate(output: Path | None = None, suite: str = "regression") -> dict[str, Any]:
    config = load_config()
    classifier = TaskClassifier(
        config.classifier["model"],
        config.classifier["labels"],
        float(config.classifier["confidence_threshold"]),
    )
    if suite == "regression":
        normal_cases, curveball_cases = NORMAL_CASES, CURVEBALL_CASES
    elif suite == "holdout":
        normal_cases, curveball_cases = HOLDOUT_NORMAL_CASES, HOLDOUT_CURVEBALL_CASES
    else:
        normal_cases, curveball_cases = build_large_suite()
    cases = [
        {"kind": "normal", "expected": expected, "prompt": prompt}
        for expected, prompt in normal_cases
    ] + [{"kind": "curveball", **case} for case in curveball_cases]

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    results = []
    for index, case in enumerate(cases, start=1):
        result = await classifier.classify(
            {"messages": [{"role": "user", "content": case["prompt"]}]},
            case.get("metadata", {}),
        )
        correct = result.model == case["expected"]
        confusion[case["expected"]][result.model] += 1
        results.append(
            {
                "index": index,
                **case,
                "predicted": result.model,
                "correct": correct,
                "low_confidence": result.low_confidence,
                "scores": {key: round(value, 4) for key, value in result.scores.items()},
            }
        )

    def score(kind: str) -> dict[str, Any]:
        subset = [row for row in results if row["kind"] == kind]
        passed = sum(row["correct"] for row in subset)
        return {
            "correct": passed,
            "total": len(subset),
            "accuracy": round(passed / len(subset), 4),
        }

    correct = sum(row["correct"] for row in results)
    report = {
        "classifier": config.classifier["model"],
        "suite": suite,
        "threshold": config.classifier["confidence_threshold"],
        "overall": {
            "correct": correct,
            "total": len(results),
            "accuracy": round(correct / len(results), 4),
        },
        "normal": score("normal"),
        "curveball": score("curveball"),
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "failures": [row for row in results if not row["correct"]],
        "results": results,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite", choices=("regression", "holdout", "large"), default="regression"
    )
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.output, args.suite))
    print(json.dumps({key: report[key] for key in (
        "classifier", "suite", "threshold", "overall", "normal", "curveball",
        "confusion", "failures"
    )}, indent=2))


if __name__ == "__main__":
    main()
