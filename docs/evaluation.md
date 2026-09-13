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

The framework in [`eval/controlled`](../eval/controlled) is designed to compare:

```text
Baseline assistant
vs
same base assistant + StopWise
```

The base model and exact version, decoding settings, tool permissions, context and turn budgets, task/replicate seeds, information environment, and user simulator must be identical across paired conditions. The primary treatment appends the direct-chat StopWise prompt; middleware is a separate experimental condition because it adds calls, cost, latency, and failure modes.

Structured tasks reveal attributes incrementally. The dynamic oracle compares pre/post feasible and possible-optimal sets, constraint status, recommendation, ranking, utility-margin state, and information sufficiency; a query is not action-changing merely because it concerns a primary criterion. End-to-end outcomes include optimal-choice accuracy, constraint satisfaction, utility, regret, information-seeking and total turns, token use, redundant queries, premature commits, unsafe defers, false/missed interventions, FOCUS continuation, silence precision/recall, and Action-Changing Rate (ACR): dynamically action-changing new queries divided by all new information queries.

The paired report includes condition means, medians, distributions, paired differences, bootstrap intervals, paired binary comparisons, effect sizes, and task-domain strata. Raw JSONL and derived summaries are separate, and scoring replays raw query IDs against the versioned task file.

The desired empirical pattern is not merely fewer turns:

```text
redundant search decreases
decision quality is approximately preserved or improves
premature commit remains low
```

The repository includes deterministic smoke validation only. It can validate and score paired logs, but no controlled LLM experiment is included as a reported result. Until such an experiment is run with adequate replication and analysis, claims that StopWise reduces redundant turns or preserves decision quality remain unmeasured. See the full [controlled protocol](../eval/controlled/PROTOCOL.md).

**尚未形成 StopWise 端到端有效性证据。**
