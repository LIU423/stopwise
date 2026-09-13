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
baseline: no StopWise
current_prompt: frozen pre-optimization strategy
minimal_prompt: new canonical strategy
```

The base model and exact version, decoding settings, tool permissions, context and turn budgets, task/replicate seeds, information environment, user simulator, and request limits must be identical across grouped paired conditions. Only the system prompt differs. The report gives `current_prompt - baseline`, `minimal_prompt - baseline`, and `minimal_prompt - current_prompt` contrasts; direction is explicit because lower is favorable for some search/cost metrics while higher is favorable for utility and quality metrics. Middleware is a separate experimental condition because it adds calls, cost, latency, and failure modes.

Structured tasks reveal attributes incrementally. The dynamic oracle compares pre/post feasible and possible-optimal sets, constraint status, recommendation, ranking, utility-margin state, and information sufficiency; a query is not action-changing merely because it concerns a primary criterion. End-to-end outcomes include optimal-choice accuracy, constraint satisfaction, utility, regret, information-seeking and total turns, token use, redundant queries, premature commits, unsafe defers, false/missed interventions, FOCUS continuation, silence precision/recall, and Action-Changing Rate (ACR): dynamically action-changing new queries divided by all new information queries.

The grouped paired report includes condition means, medians, distributions, explicitly directed paired differences, bootstrap intervals, paired binary comparisons, effect sizes, and task-domain strata. Raw JSONL and derived summaries are separate, and scoring replays raw query IDs against the versioned task file.

The desired empirical pattern is not merely fewer turns:

```text
redundant search decreases
decision quality is approximately preserved or improves
premature commit remains low
```

The repository includes implemented infrastructure, deterministic smoke validation, and an explicitly gated OpenAI Responses API live CLI. The smoke assistant uses hard-coded rules, and its user simulator reads action labels and oracle state; this validates mechanisms, not prompt effectiveness. The live path excludes hidden answers and oracle/evaluation labels from the assistant, and gives the user simulator only condition-blind visible natural language. It records provider-reported token usage separately from conservative cost estimates, performs no automatic retries, and refuses a plan whose guarded worst-case cost exceeds the requested hard cap. A minimal one-replicate pilot, even when run, is descriptive pilot-model output rather than controlled effectiveness evidence. See the full [controlled protocol](../eval/controlled/PROTOCOL.md).

**尚未形成 StopWise 端到端有效性证据。**
