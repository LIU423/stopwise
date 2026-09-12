# Evaluation

StopWise has two separate evaluation questions. They must not be collapsed into one score.

## Policy regression evaluation

The cases in [`eval/policy/cases.jsonl`](../eval/policy/cases.jsonl) check whether an analyzer follows the intended action policy. They include canonical sanity checks and contrastive minimal pairs around relevant evidence, robustness checks, numeric uncertainty, contingencies, exploration, goal refinement, stakes, FOCUS/COMMIT boundaries, and DEFER with useful main-decision analysis remaining.

These handcrafted cases are regression fixtures, not a conclusive benchmark. They are small, authored with known labels, and do not measure effects on real conversations.

The evaluator reports action accuracy; false intervention, premature COMMIT, unsafe DEFER, and missed intervention rates; `NO_INTERVENTION` precision and recall; and micro signal precision, recall, and F1.

Rates use eligible-case denominators. In particular, FOCUS is not premature just because action-changing information remains. Premature COMMIT is counted only when `COMMIT` occurs while such information remains, and unsafe DEFER is evaluated only on cases explicitly marked unsafe to defer.

Run a live policy pass with an OpenAI-compatible provider:

```bash
python eval/policy/evaluate.py --model gpt-4o-mini
```

Or score saved predictions without a model call:

```bash
python eval/policy/evaluate.py --predictions predictions.json
```

## Exploratory model matrix

The stored 12-case Codex matrix is retained under [`eval/legacy/codex_matrix`](../eval/legacy/codex_matrix) as an exploratory policy-classification artifact. It contains one run per model/effort configuration, nearly identical results, obvious cases, and an ambiguous reference label. It measures label agreement only and is not evidence that StopWise improves conversations or model rankings.

## Controlled end-to-end effectiveness evaluation

The scaffold in [`eval/controlled`](../eval/controlled) is designed to compare:

```text
Baseline assistant
vs
same base assistant + StopWise
```

The base model, decoding settings, task order, information environment, and user simulator should be identical across conditions. Structured tasks reveal attributes incrementally and record which queries were required, action-changing, secondary, or irrelevant.

End-to-end outcomes include optimal-choice accuracy, constraint satisfaction, utility, regret, information-seeking turns, token use, redundant queries, premature commits, and Action-Changing Rate (ACR): action-changing queries divided by all additional queries.

The desired empirical pattern is not merely fewer turns:

```text
redundant search decreases
decision quality is approximately preserved or improves
premature commit remains low
```

The scaffold can validate and score paired logs, but no controlled LLM experiment is included as a reported result. Until such an experiment is run with adequate replication and analysis, claims that StopWise reduces redundant turns or preserves decision quality remain unmeasured.
