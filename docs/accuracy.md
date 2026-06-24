# Routing Accuracy

## Result: 99.2% (496/500)

Evaluated on a 500-prompt holdout set covering general questions, complex reasoning, and coding tasks. Prompts were written to be diverse and include deliberate curveballs.

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Normal prompts | 400 | 400 | 100% |
| Curveballs | 96 | 100 | 96% |
| **Total** | **496** | **500** | **99.2%** |

## Confusion matrix

| Ground truth → | gemma | qwen-opus | omnicoder |
|----------------|-------|-----------|-----------|
| gemma | **164** | 0 | 4 |
| qwen-opus | 0 | **166** | 0 |
| omnicoder | 0 | 0 | **166** |

## Known failure modes

All 4 failures were calendar event lookups where the event title contained a technical term:

- "Find my Python study group meeting"
- "When is my Race Condition reading group?"
- "Schedule Terraform review"
- "Add OpenAPI spec review to calendar"

The NLI classifier sees "Python" / "Terraform" / "OpenAPI" and scores them as coding, overriding the "calendar" tool-use signal. Mitigations: strengthen the tool-use intent priors, or add these patterns to the training set for the LLM reclassifier.

## Multi-turn

Short follow-ups ("improve it", "refactor", "explain further") lose routing context when the prior turn's intent doesn't appear in the latest message. See `evals/routing_multiturn_eval.py` for the full multi-turn evaluation.

The `llm_classify` option (disabled by default) addresses this by briefly asking the currently-loaded model to reclassify ambiguous follow-ups using full conversation context.

## Methodology

- Corpus: 500 prompts in `evals/routing_accuracy_500_report.json`
- Classifier: `cross-encoder/nli-MiniLM2-L6-H768` with intent priors
- No data leakage: the eval corpus was not used to tune the intent priors
- Curveballs: context switches, negations ("no code please"), plain-language requests after technical discussion, follow-ups without explicit topic

Run the eval yourself:

```bash
python evals/routing_accuracy.py
```
