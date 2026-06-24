from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("model_router.classifier")


@dataclass(frozen=True)
class Classification:
    model: str
    scores: dict[str, float]
    low_confidence: bool


class TaskClassifier:
    def __init__(
        self,
        model_name: str,
        labels: dict[str, str],
        threshold: float = 0.55,
        predictor: Callable[[list[tuple[str, str]]], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.labels = labels
        self.threshold = threshold
        self._predictor = predictor
        self._load_lock = asyncio.Lock()
        # Infer which config key maps to each semantic role from label descriptions.
        # Falls back gracefully when a role is absent (e.g. 2-model setups).
        self._general_key = self._infer_role_key(labels, r"\b(general|tool|plan|ordinary|question)\b")
        self._reasoning_key = self._infer_role_key(labels, r"\b(reasoning|proof|analys|research|theorem)\b")
        self._coding_key = self._infer_role_key(labels, r"\b(coding|code|software|debug|engineer)\b")

    @staticmethod
    def _infer_role_key(labels: dict[str, str], pattern: str) -> str | None:
        scored = [(len(re.findall(pattern, desc, re.I)), key) for key, desc in labels.items()]
        best_score, best_key = max(scored)
        return best_key if best_score > 0 else None

    async def _load(self) -> None:
        if self._predictor is not None:
            return
        async with self._load_lock:
            if self._predictor is not None:
                return
            try:
                from sentence_transformers import CrossEncoder

                model = await asyncio.to_thread(
                    CrossEncoder, self.model_name, device="cpu", activation_fn=None
                )
                label_map = {
                    str(label).lower(): int(index)
                    for index, label in model.model.config.id2label.items()
                }
                entailment_index = next(
                    (
                        index for label, index in label_map.items()
                        if "entail" in label
                    ),
                    1,
                )

                def predict_entailment(pairs: list[tuple[str, str]]) -> list[float]:
                    rows = model.predict(pairs, apply_softmax=True)
                    return [
                        float(row[entailment_index]) if hasattr(row, "__len__") else float(row)
                        for row in rows
                    ]

                self._predictor = predict_entailment
                logger.info("classifier_loaded model=%s device=cpu", self.model_name)
            except Exception:
                logger.exception("classifier_load_failed; using deterministic fallback")
                self._predictor = self._heuristic_predict

    @staticmethod
    def _heuristic_predict(pairs: list[tuple[str, str]]) -> list[float]:
        """Deterministic fallback when sentence-transformers is unavailable.
        Infers scores from hypothesis text so it works with any model names.
        """
        if not pairs:
            return []
        text = pairs[0][0].lower()
        coding_count = len(re.findall(
            r"\b(code|coding|bug|debug|function|class|api|sql|python|typescript|"
            r"javascript|repository|refactor|test|compile|stack trace|git)\b", text
        ))
        reasoning_count = len(re.findall(
            r"\b(prove|proof|derive|theorem|deep analysis|multi-step|optimi[sz]e|"
            r"research|trade-?off|architecture|root cause)\b", text
        ))
        scores = []
        for _, hypothesis in pairs:
            h = hypothesis.lower()
            if re.search(r"\b(coding|code|software|debug|engineer|repository)\b", h):
                scores.append(min(0.99, 0.2 + 0.3 * coding_count))
            elif re.search(r"\b(reasoning|proof|analys|research|theorem)\b", h):
                scores.append(min(0.99, 0.2 + 0.3 * reasoning_count))
            else:
                scores.append(0.6)
        return scores

    def _intent_priors(self, text: str, latest_user: str = "") -> dict[str, float]:
        lowered = text.lower()
        latest = latest_user.lower()
        priors = {key: 0.0 for key in self.labels}
        gk = self._general_key    # key for general/tool-use role (may be None)
        rk = self._reasoning_key  # key for reasoning role (may be None)
        ck = self._coding_key     # key for coding role (may be None)

        # --- General tool-use (full text) ---
        general_tool = re.search(
            r"\b(calendar|email tool|weather tool|search my notes|grocery list)\b",
            lowered,
        ) and re.search(r"\b(find|send|check|search|tell|group)\b", lowered)
        if general_tool:
            if gk: priors[gk] += 5.0

        # --- Explanatory general ---
        # Check latest_user first (higher weight) — a latest-message switch to
        # plain-language explanation overrides prior coding/reasoning context.
        _expl_pattern = (
            r"\b(explain|define)\b.*\b(analogy|plain language|high.school|"
            r"restaurant owner|without discussing implementation|to a child|"
            r"for a (beginner|friend)|in simple terms)\b"
        )
        explanatory_general_latest = bool(latest) and bool(re.search(_expl_pattern, latest))
        explanatory_general_full = bool(re.search(_expl_pattern, lowered))
        if explanatory_general_latest:
            if gk: priors[gk] += 6.0
            if ck: priors[ck] -= 2.0  # override prior coding inertia
        elif explanatory_general_full:
            if gk: priors[gk] += 4.0
        explanatory_general = explanatory_general_latest or explanatory_general_full

        # Explicit "forget the code / explain simply" switch in latest message
        if bool(latest) and re.search(
            r"\b(forget the code|never mind the code|"
            r"explain.{0,35}(child|10.year|beginner|friend|plain)|"
            r"without any code|no code.{0,15}(please|needed|just)|"
            r"don't write code|no more code|just (summarize|explain|describe))\b",
            latest,
        ):
            if gk: priors[gk] += 4.0
            if ck: priors[ck] -= 2.0

        # --- suppress_coding: scope to LATEST USER MESSAGE ONLY ---
        # A prior turn saying "no code" must not suppress coding intent in the
        # latest message. Single-turn requests are unaffected (latest == full msg).
        negated_coding_latest = bool(latest) and bool(re.search(
            r"\b(no code|no coding|do not implement|don't implement|"
            r"do not write code|don't write code|although this is about source code|"
            r"no more code|without implementing|forget the code|never mind the code)\b",
            latest,
        ))
        requests_concrete_fix = bool(latest) and bool(re.search(
            r"\b(code-level fix|exact fix|smallest patch|configuration change)\b",
            latest,
        ))
        suppress_coding = negated_coding_latest and not requests_concrete_fix

        # --- Reasoning priors ---
        _reasoning_pattern = (
            r"\b(prove|proof|derive|formally?|formal argument|reason carefully|"
            r"every inference|linearizability|linearizable|mutually consistent|"
            r"feasible cases|causal model|scenario analysis|sensitivity analysis|"
            r"game.theory|thermodynamics|fermi paradox|correlation from causation|"
            r"logic puzzle|determine whether|assess whether|establish whether|"
            r"show that|probability puzzle|conditional dependency|hidden assumptions|"
            r"conclusion follows|preserves semantics|all possible inputs|"
            r"independence of irrelevant alternatives)\b"
        )
        # Negation prefix in latest message — "forget the formal proof", "skip the math" —
        # means user is ABANDONING reasoning mode, not requesting it.
        _reasoning_negated = bool(latest) and bool(re.search(
            r"\b(forget|drop|skip|ignore|abandon|no more|never mind).{0,25}"
            r"(proof|theorem|math|formal|analysis|derivation)\b",
            latest,
        ))
        # Latest-message reasoning signal gets higher weight to override prior context.
        if bool(latest) and not _reasoning_negated and re.search(_reasoning_pattern, latest):
            if rk: priors[rk] += 5.0
            if ck: priors[ck] -= 2.0  # suppress prior coding inertia
        elif re.search(_reasoning_pattern, lowered):
            if rk: priors[rk] += 3.5

        if re.search(
            r"\b(conflicting timelines|failure-domain constraints|uncertain demand|"
            r"capacity constraints|weakest assumptions|optimal strategy|"
            r"contradictory witness accounts|plausible timelines|business invariants|"
            r"eventual consistency|adverse scenarios|establish termination)\b",
            lowered,
        ):
            if rk: priors[rk] += 3.0
        if suppress_coding:
            if rk: priors[rk] += 2.0

        # --- Coding priors ---
        _coding_pattern = (
            r"\b(implement|debug|refactor|pull request|code review|sql migration|"
            r"integration tests?|borrow checker|openapi|stack trace|code path|"
            r"bash deployment|powershell|postgresql query|execution plan|"
            r"github actions|race condition|code-level fix|smallest patch|"
            r"repository.maintainer|serializers?|compatibility|npm|ci build|"
            r"express endpoint|terraform|unit tests?|go http handler|"
            r"docker.?(compose|container|file)|dockerfile|rest api|graphql|"
            r"type assertions?|test fixtures?|database column|production deployment|"
            r"configuration change|devops.agent|"
            r"now (implement|write|build|create|add))\b"
        )
        coding_signal_latest = bool(latest) and bool(re.search(_coding_pattern, latest))
        coding_signal_full   = bool(re.search(_coding_pattern, lowered))

        # Latest-message explicit coding switch gets stronger boost and suppresses
        # general inertia from prior turns (e.g. "explain microservices" → "write compose").
        # Use explanatory_general_LATEST only — a prior "explain to restaurant owner"
        # must not block a later "now build a REST API".
        if coding_signal_latest and not suppress_coding and not explanatory_general_latest:
            if ck: priors[ck] += 4.5
            if gk: priors[gk] -= 1.5
        elif coding_signal_full and not suppress_coding and not explanatory_general:
            if ck: priors[ck] += 3.5

        # In latest message: explicit coding verb + language = strong coding switch signal.
        # Covers "Write a Python simulation", "Build a TypeScript module", etc.
        # "write" alone is too generic; require a programming language alongside it.
        _lang_pattern = r"\b(python|typescript|javascript|rust|react|fastapi|go|java|kotlin)\b"
        _action_verb   = r"\b(write|build|create|generate|develop)\b"
        if (bool(latest) and re.search(_action_verb, latest)
                and re.search(_lang_pattern, latest)
                and not suppress_coding and not explanatory_general_latest):
            if ck: priors[ck] += 3.5
            if rk: priors[rk] -= 1.5  # suppress prior reasoning inertia

        # lang + action in full text (covers single-turn cases)
        if re.search(_lang_pattern, lowered) and re.search(
            r"\b(function|error|fix|tests?|component|middleware|deadlock|api|"
            r"simulation|script|program|module|class|method)\b",
            lowered,
        ) and not suppress_coding and not explanatory_general_latest:
            if ck: priors[ck] += 2.0

        return priors

    @staticmethod
    def _text(payload: dict[str, Any], metadata: dict[str, str], max_chars: int = 2000) -> str:
        messages = payload.get("messages") or []
        system = "\n".join(
            str(m.get("content", ""))
            for m in messages
            if m.get("role") == "system"
        )
        user_messages = [m for m in messages if m.get("role") == "user"]
        latest_user = str(user_messages[-1].get("content", "")) if user_messages else ""
        # Prior user turns (up to 3) provide follow-up resolution context ("improve it", etc.)
        prior_user = [str(m.get("content", "")) for m in user_messages[-4:-1]]

        body_metadata = payload.get("metadata")
        hints = " ".join(f"{key}={value}" for key, value in metadata.items())
        if isinstance(body_metadata, dict):
            hints += " " + " ".join(f"{key}={value}" for key, value in body_metadata.items())

        # BERT truncates from the end — put critical content first so long system
        # prompts never displace the latest user message from the 512-token window.
        # Priority: metadata + latest user > prior turns > system (truncate last).
        header = f"metadata: {hints}\nlatest user: {latest_user}"
        used = len(header)
        extra: list[str] = []

        if prior_user and used < max_chars:
            prior_text = " | ".join(reversed(prior_user))
            budget = min(max_chars // 3, max_chars - used - 1)
            if budget > 20:
                extra.append(f"prior: {prior_text[:budget]}")
                used += len(extra[-1]) + 1

        if system and used < max_chars:
            budget = max_chars - used - 1
            if budget > 20:
                extra.append(f"system: {system[:budget]}")

        return (header + "\n" + "\n".join(extra)) if extra else header

    async def classify(
        self, payload: dict[str, Any], metadata: dict[str, str] | None = None
    ) -> Classification:
        await self._load()
        text = self._text(payload, metadata or {})
        messages = payload.get("messages") or []
        latest_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        keys = list(self.labels)
        pairs = [(text, f"This task is {self.labels[key]}.") for key in keys]
        raw = await asyncio.to_thread(self._predictor, pairs)
        def normalize_entailment(score: Any) -> float:
            value = float(score)
            return value if 0.0 <= value <= 1.0 else 1.0 / (1.0 + math.exp(-value))

        entailment = {
            key: normalize_entailment(score)
            for key, score in zip(keys, raw, strict=True)
        }
        total_entailment = sum(entailment.values()) or 1.0
        relative = {key: value / total_entailment for key, value in entailment.items()}
        priors = self._intent_priors(text, latest_user)  # instance method; uses self role keys
        logits = {
            key: relative.get(key, 0.0) + priors.get(key, 0.0)
            for key in keys
        }
        maximum = max(logits.values())
        exponentials = {key: math.exp(value - maximum) for key, value in logits.items()}
        denominator = sum(exponentials.values())
        scores = {key: value / denominator for key, value in exponentials.items()}
        selected = max(scores, key=scores.get)
        low_confidence = scores[selected] < self.threshold
        if low_confidence:
            selected = self._general_key or next(iter(self.labels))
        logger.info(
            "classification selected=%s low_confidence=%s scores=%s",
            selected,
            low_confidence,
            {key: round(value, 4) for key, value in scores.items()},
        )
        return Classification(selected, scores, low_confidence)
