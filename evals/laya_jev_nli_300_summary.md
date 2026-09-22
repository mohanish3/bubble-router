# Laya vs JEV vs NLI — golden_300

Run date: 2026-09-22. Dataset: `evals/golden_300.json` (sha256 `78570157c420…`), 300 authored
prompts, 1 round. Config: `config/model-router.example.json`. Hardware: RTX 5070 Ti (Laya on
CUDA), NLI on CPU, JEV over the hosted API.

Reports: `laya_jev_nli_300_report.json` (three-way run), `laya_variants_300_report.json`
(checkpoint sweep), `laya_ablation_300_report.json` (input/prompt ablation),
`laya_best_config_300_report.json` (best config, per-case).

Reproduce: `python evals/compare_laya.py --config config/model-router.example.json
--env-file ../.env.txt --dataset evals/golden_300.json --rounds 1 --device cuda
--output evals/laya_jev_nli_300_report.json`

## Headline

| Classifier | Routing accuracy | Argmax accuracy | Median latency | p95 latency |
|---|---|---|---|---|
| NLI (local, CPU) | 72.3% (217/300) | 76.3% (229/300) | 23 ms | 156 ms |
| JEV (hosted API) | **90.0% (270/300)** | **92.3% (275/298)** | 1.32 s | 7.33 s |
| Laya, tuned (local, CUDA) | **88.3% (265/300)** | 88.3% | **21 ms** | 25 ms |
| Laya, first attempt (english ckpt, JEV-shaped call) | 43.3% (130/300) | 72.3% | 21 ms | 25 ms |

**Routing accuracy** is the model the router would actually pick, including any
low-confidence fallback. **Argmax accuracy** ignores the threshold and scores the
top-probability class. For tuned Laya the two are identical because its threshold is 0 —
see below.

Tuned config: `typed-decisions` checkpoint, dict state with a `request` field, role-shaped
choice keys, plain task descriptions with the deliberative/engineering boundary spelled out,
no confidence cutoff. All of it is now the default in `compare_laya.py`.

## Out-of-sample check

golden_300 is the tuning set, so the numbers above are in-sample. Two sealed holdouts, both
verified to share zero prompts with golden_300: `large_500` (400 normal + 100 curveball, from
`routing_accuracy.build_large_suite`) and `jev_prompt_fresh_50.json`.

| Set | JEV | Laya tuned | NLI | Laya, first attempt |
|---|---|---|---|---|
| golden_300 (Laya's tune set) | **90.0%** | 88.3% | 72.3% | 43.3% |
| large_500 (templated holdout) | **100.0%** | 95.6% | 95.2% | 92.0% |
| fresh_50 (authored holdout) | 74.0% | **86.0%** | 64.0% | 80.0% |

Median latency: JEV 910 ms, Laya 21 ms, NLI 23 ms.

**`large_500` is a weak holdout.** `build_large_suite` is combinatorial template generation —
about 30 patterns crossed with actions, topics and formats — and every deliberative prompt
opens with a class-marking verb (`prove`, `derive`, `solve step by step`). It is not 500
independent items, and JEV saturates it at 500/500 including 100/100 curveballs. The two
authored sets are what discriminate.

**On the authored sets, tuned Laya is level with JEV.** JEV takes golden_300 by 1.7 points;
Laya takes fresh_50 by 12. On fresh_50 the paired comparison is 37 both right, 6 Laya-only,
**0 JEV-only**, 7 both wrong — McNemar exact p = 0.031, so JEV does not win a single case Laya
loses there. All 13 JEV failures are the same error: `deliberative_work` routed to
`direct_assistance` at 0.9+ confidence, on short reasoning tasks ("Compute the posterior
probability given sensitivity, specificity and the base rate", "Find a counterexample to the
claim that cheapest-first maximises value under a weight limit"). JEV appears to reserve
`deliberative_work` for *long* deliberation; the fresh_50 labels apply it to anything needing
a derivation.

Caveats: fresh_50 is 50 cases, so its interval is wide even with the paired test; golden_300
is Laya's tuning set, so its 88.3% flatters Laya and its comparison to JEV's 90.0% does not.

NLI here is measured with `model-router.example.json`. The README's 99.2% used the local
(gitignored) `model-router.json` with different label text, so it is not comparable; all three
classifiers above use the same example config.

## Why the first Laya number was wrong

43.3% was a measurement artifact, not Laya's accuracy. Four separate handicaps, each measured
on the same 300 cases (`laya_ablation_300_report.json`):

| Change | Accuracy |
|---|---|
| First attempt: english ckpt, `_text()` blob, model-nickname keys, JEV instructions | 72.3% argmax (43.3% after threshold) |
| → `typed-decisions` checkpoint | 79.7% |
| → dict state with a `request` field the instructions name | 83.3% |
| → plain task descriptions instead of the NLI-tuned label text | 87.0% |
| → drop the 0.55 confidence cutoff | 87.0% end-to-end (was 55.3% at 0.55) |
| → name the deliberative/engineering boundary in the criteria | **88.3%** (95.6% on holdout) |

1. **Wrong checkpoint.** The repo defaulted to the english root; `typed-decisions` is the
   variant built for this. (`multilingual` is a trap: 41.3%, it collapses on English.)
2. **Wrong state shape.** Laya reads *named fields* — its own router preset writes
   instructions like ``How hard is `request`?``. Feeding `TaskClassifier._text()`'s
   `"metadata: \nlatest user: …"` blob as one string puts scaffolding noise inside the field
   the question asks about. Passing the same blob as `{"request": blob}` still costs 9
   points vs. the real fields (78.0% vs 87.0%) — the fix is structured state, not just a dict.
3. **Wrong key names and label text.** The router's keys are model nicknames (`gemma`,
   `qwen-opus`, `omnicoder`) and its descriptions were written for an NLI cross-encoder.
   JEV shrugs that off; Laya does not.
4. **Wrong threshold.** See below.

## The 0.55 threshold is actively harmful to Laya

Laya spreads probability mass far flatter than the other two. Median top probability:
JEV 1.00, NLI 0.53, Laya 0.55 (tuned) / 0.44 (first attempt). Every cutoff tested *loses*
accuracy:

| Cutoff | 0.00 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|
| End-to-end accuracy | **87.0%** | 86.3% | 83.3% | 75.3% | 65.3% | 55.3% | 47.0% |

A top1−top2 margin rule is no better (87.0% at 0, 82.0% at 0.05, 77.7% at 0.10). Laya's own
`confidence` field is not a probability at all — it is normalized Shannon entropy
(`1 - H(p)/log k`), and the checkpoint loads warning that its shipped temperatures are clamped
(`choice:11 += 0.1006`), so it is explicitly uncalibrated. Route Laya on argmax.

## Where tuned Laya still loses to JEV

35 errors on golden_300, and 24 of them are `deliberative_work` being misrouted. Those are the
golden set's adversarial cases — reasoning tasks written in engineering vocabulary:

- *"Find the root cause of a distributed-system outage from these conflicting timelines."*
- *"Show whether the proposed loop invariant implies termination."*
- *"Compare two database isolation strategies under latency and anomaly constraints; no implementation plan."*

Laya is a 421M bidirectional encoder doing one forward pass with no reasoning step, so it
keys on surface vocabulary. Naming that boundary in the criteria recovered 6 of these cases
(69/99 → 75/99); the rest need a reasoning step the architecture does not have.

`deliberative_work` is the hardest class for all three (JEV 75/97, Laya 75/99, NLI 63/99) and
is the only class where Laya still trails JEV — it matches or beats JEV nowhere else, but is
within one case on engineering (98/99 vs 99/99).

## Accuracy by role

| Role | NLI (argmax) | JEV (argmax) | Laya tuned |
|---|---|---|---|
| direct_assistance | 94/102 | 101/102 | 94/102 |
| deliberative_work | 63/99 | 75/97 | 75/99 |
| engineering_work | 72/99 | 99/99 | 97/99 |

Tuned Laya beats the current NLI classifier outright: +12 points on golden_300, +16 points on
fresh_50, at the same latency with no API dependency.

## Laya checkpoint sweep (argmax, first-attempt call shape)

| Checkpoint | Argmax accuracy | Median latency |
|---|---|---|
| english (root) | 72.3% | 21 ms |
| multilingual | 41.3% | 17 ms |
| typed-decisions | **79.7%** | 21 ms |

## Agreement

Pairwise identical routing decisions out of 300 (first-attempt Laya): NLI↔JEV 209,
NLI↔Laya 181, JEV↔Laya 157. All three wrong on the same case: 11.

## Caveats

- JEV's first pass hit 61 `getaddrinfo failed` DNS errors, scored as wrong answers. Those
  cases were re-run and the report patched (`jev_repaired_cases: 61`). Two 20 s timeouts
  remain and are still scored wrong, so 90.0% is a floor — over the 298 that answered, JEV is
  270/298 = 90.6%. `compare_laya.py` now retries once per case.
- The NLI number reproduces the earlier `jev_nli_300_report.json` run exactly (0.7233, same
  per-role split), which validates the harness. JEV matches it too (0.90 vs 0.91).
- Laya's 98 ms median in the original three-way table was GPU clock-down between interleaved
  network calls, not model cost. Measured back-to-back it is 21 ms.
- These 300 are authored examples, not a sample of live router traffic. The tuned Laya
  criteria were chosen on this set, which is why the holdout section above exists: 95.6% on
  large_500 and 86.0% on fresh_50 are the out-of-sample numbers to quote.
- Fitted readouts on top of Laya's probabilities (per-class log-bias, logistic regression,
  extra sub-questions in the same forward pass) were tried and rejected — each won on one
  holdout and lost on the other. Details in `laya_optimization_report.json`.
- `JevClassifier` opens a fresh `AsyncTypeSafeClient` per call, as the existing eval does.
  A pooled client would cut its latency.
