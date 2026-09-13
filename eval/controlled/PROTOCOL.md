# StopWise controlled experiment protocol

## 1. Research question and evidence layers

The experiment asks whether adding StopWise to an otherwise identical assistant reduces low-value, redundant information search while preserving or improving decision quality and keeping premature `COMMIT` rare.

The target pattern is joint, not a turn-count objective:

```text
redundant search decreases
decision quality is preserved or improves
premature COMMIT remains low
action-changing information density increases
```

Two evidence layers remain separate:

1. `eval/policy/` is policy-label regression testing.
2. `eval/controlled/` observes complete query/reveal/response/choice trajectories.

Passing infrastructure tests, reproducing a scripted fixture, or matching labels does not establish effectiveness.

## 2. Paired design and fair comparison

The pairing unit is `(experiment_id, task_id, replicate, seed)`. Each unit must contain exactly one `baseline` and one `stopwise` episode. The pair validator fails on missing or duplicate conditions.

The conditions must match on:

- provider, base model, and exact model/version identifier;
- temperature, top-p, output cap, context budget, extra parameters, and tool permissions;
- task version, hidden values, initial information, termination cap, simulator identity/version, replicate, and seed;
- base assistant prompt and callback implementation.

The only planned difference is that the StopWise condition appends the canonical `prompts/custom_instruction.md`. Condition order is deterministically randomized within each pair. Prompt and policy SHA-256 hashes are logged.

The primary experiment uses direct-chat StopWise because visible assistant behavior is the treatment. Middleware using `StopWise(generator=...)` remains supported by the core package but must be a separately named experiment condition; its extra model calls, errors, tokens, and latency may not be silently mixed into the primary comparison.

## 3. Tasks and hidden information

`tasks.json` includes laptop, hotel, subscription, course, travel, and device-procurement decisions. Each task declares alternatives, initially visible values, hidden queryable attributes, primary and secondary criteria, hard constraints, normalized utility weights, a satisfactory-utility threshold, low-value attributes, branch timing, and a deterministic simulator plan.

The controls span action-changing evidence, irrelevant but superficially novel evidence, independent robustness checks, repeated confirmation, measured uncertainty, pseudo-precision, a risk required now, a future deferable branch, continued valid search after `FOCUS`, and branch-local `DEFER`.

Task wording describes the decision but never tells the assistant when to stop. Full values and oracle calculations stay inside the environment/scorer.

## 4. Dynamic information-value oracle

Static `action_changing: true/false` labels are not used by the v1 dynamic oracle. Every criterion maps a full value to `[0, 1]`; declared weights yield utility. Hard constraints determine feasibility.

For unrevealed decision attributes, the environment uses the criterion's full `[0, 1]` utility range. For every alternative it computes:

- lower and upper utility bounds;
- expected utility using the midpoint only for current recommendation/ranking;
- constraint state: unknown, satisfied, or violated;
- possible feasible, confirmed feasible, recommended, and possible-optimal sets.

An alternative is possibly optimal when its upper bound is at least the highest lower bound among alternatives not ruled out. Information is sufficient only when there is one possible optimum, it is confirmed feasible, and its worst-case lead over every remaining competitor is at least the task's declared decision margin.

For each query, the oracle compares the pre/post state. A query is action-changing only if the newly revealed value changes at least one of:

- recommendation;
- possible-optimal set;
- feasible set;
- constraint status;
- utility ranking;
- whether the expected utility gap crosses the action threshold;
- information-sufficiency state.

Repeated observations are never action-changing. A primary label alone is not enough. The raw event records the concrete effects that triggered the dynamic judgment.

`oracle_annotation` records source, version, review status, and notes. The current set declares `source: dynamic` and contains no manual action-changing labels. If a future task uses `manual` or `hybrid`, its query-level assumptions must be populated, independently reviewed, versioned, and described as oracle assumptions open to correction.

## 5. Episode state machine

```text
task initialization + partial information
  -> user information request
  -> environment reveals exactly one requested attribute
  -> assistant gives the normal answer
  -> optional StopWise action/nudge
  -> simulator queries again or chooses
  -> final choice or environment turn cap
  -> replay scoring
```

The log distinguishes user, assistant, environment, query, and policy events. `NO_INTERVENTION` has no visible nudge. `FOCUS` narrows search but does not terminate; the simulator may continue with primary or robustness queries. `DEFER` targets one declared branch, and the main decision continues. `COMMIT` is premature whenever the dynamic state is not sufficient at the action event. The environment cap has a distinct termination reason.

## 6. User simulators

The deterministic simulator is seeded, has a fixed query plan, and applies branch-local responses to visible nudges. It is for unit tests, replay checks, and dry runs only. Its scripted compliance makes it unsuitable for an effectiveness claim.

The text-only LLM callback is the intended pilot path. A pilot simulator must be separate from the assistant, identically configured across conditions, and blind to condition, action label, oracle state, hidden values, and policy logs. It receives only task wording and the visible transcript. Its prompt should describe a decision-maker's goals and natural continuation/choice behavior, not instruct it to obey `COMMIT` or expose a StopWise taxonomy.

## 7. Metrics

### Decision quality

- `optimal_choice`: final feasible choice has maximum full utility.
- `satisfactory_choice`: constraints pass and utility meets the task threshold.
- `constraint_satisfied`: all hard constraints pass.
- `utility`: weighted normalized utility of the final choice.
- `regret`: best feasible utility minus chosen utility, floored at zero; an infeasible choice receives zero constrained utility so a high raw score cannot hide a hard-constraint violation.

### Search and model cost

- information-seeking turns: number of attribute queries;
- total conversation turns: visible user plus assistant turns;
- prompt, completion, and total tokens reported separately;
- latency only when its source is labeled `provider` or `client`; no local/provider time mixing.

### Information value

- redundant query: a repeated query; a no-effect query after sufficiency; a no-effect irrelevant/secondary query; or a no-effect, currently deferable branch query. A legitimate robustness or primary query is not called redundant merely because its realized value did not change the action.
- redundant query rate: redundant queries divided by all queries.
- Action-Changing Rate (ACR): dynamically action-changing new queries divided by all non-repeated information queries. Repeated confirmations remain search cost and may be redundant, but do not pretend to add a new observation to this denominator.
- queries before sufficiency: includes the query that first establishes sufficiency.
- queries after sufficiency: queries strictly after that event.

### StopWise safety

- premature `COMMIT`: `COMMIT` while dynamic sufficiency is false; rate denominator is all `COMMIT` events.
- unsafe `DEFER`: wrong/missing target, a non-deferable or required-now branch, or a branch unrelated to the current query; denominator is all `DEFER` events.
- false intervention: `FOCUS`, `COMMIT`, or `DEFER` when the independent state rule expects silence; denominator is silence-eligible events.
- missed intervention: `NO_INTERVENTION` when the rule expects an intervention; denominator is intervention-eligible events.
- `FOCUS` continuation: `FOCUS` events followed by a later dynamically action-changing query divided by all `FOCUS` events.
- `NO_INTERVENTION` precision/recall: event-level silence agreement with the independent state rule.

The policy oracle is an authored evaluation rule, not ground truth about human cognition. Disagreements must be reviewed at the event level using the logged state transition.

## 8. Statistical analysis

All comparisons are paired on task, seed, and replicate. Reports include each condition's mean, median, and raw distribution; StopWise-minus-baseline paired differences; paired bootstrap confidence intervals for mean differences; standardized paired effect size `d_z` when the paired-difference variance is nonzero; and per-domain strata.

Binary decision outcomes report both rates, paired risk difference, discordant pair counts, and an exact two-sided McNemar test. Bootstrap resampling is over complete pairs, never individual condition rows. Replicates and bootstrap seeds are fixed and logged.

Small samples are reported descriptively with intervals. No p-value, smoke result, or favorable point estimate may be promoted into a universal or causal claim outside the preregistered design. Multiple primary outcomes and any multiplicity correction should be fixed before a paid pilot.

## 9. Logging and replay

Each raw episode records experiment/task/version and a canonical task-definition hash, condition, replicate, paired seed, base and exact model IDs, all model parameters and permissions, prompt/policy hashes, simulator and callback versions, complete turns, query IDs and revealed attributes, dynamic effects, policy actions and visible nudges, final choice, token usage, termination reason, errors/retries, and source-labeled optional latency.

Credential-like keys are rejected from model configuration. The JSONL writer also redacts credential-bearing keys and common bearer/API-token strings. Raw `episodes.jsonl` and derived `summary.json` are separate. `score.py` rebuilds query effects and all metrics from raw query IDs plus the versioned task file, and rejects revealed values that disagree with the task definition.

## 10. Result labels and claim boundary

Reports must identify one of four levels:

1. implemented infrastructure;
2. deterministic smoke validation;
3. pilot model results;
4. controlled effectiveness evidence.

The repository currently reaches levels 1 and 2 only. It has not measured live assistant behavior, live LLM-user behavior, provider latency, paid token use, or generalization beyond the authored tasks.

**尚未形成 StopWise 端到端有效性证据。**
