# StopWise

> Know when more information stops helping.

StopWise is a lightweight metacognitive stopping and intervention policy for LLM conversations.

LLMs are good at answering the next question. **StopWise asks whether the next question is still worth asking.**

StopWise is a policy, not a single deployment. Use it as a zero-code custom instruction, structured Python middleware, or a reusable assistant Skill.

```text
                         StopWise Policy
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     Custom Instruction    Middleware        Skill
              │               │               │
        direct chat       structured API   host assistant
        no coding         JSON analysis    capability
```

Two principles anchor the project:

> **Novel information is not necessarily decision-relevant information.**

> **Silence is a first-class action.**

StopWise is not a psychological diagnosis, autonomous agent, productivity coach, recommendation engine, or “overthinking detector.” It does not decide for the user. It asks whether more information is likely to change the action and whether a short metacognitive nudge would help.

## Why StopWise?

Suppose laptop A fits the budget, runs the required software, has adequate battery life, and wins on the user's primary criteria. The user then asks:

> Give me 20 more alternatives just in case there is a globally optimal one.

A normal assistant will usually keep expanding the list. StopWise asks:

> Is this new search likely to change the user's action?

If yes—or if the evidence is unclear—the assistant keeps helping without commentary. If not, it may add a brief, non-judgmental stopping rule. Conversation length alone is never evidence of diminishing returns.

## Research inspiration

StopWise is a **research-inspired engineering synthesis** with two primary influences.

### Cognitive stopping rules

[Browne, Pitts, and Wetherbe (2007)](https://doi.org/10.2307/25148782) study how information search is terminated in online tasks. Their work motivates reasoning about stopping criteria, informational sufficiency, and task-dependent search. StopWise does not infer a user's literal cognitive rule; it operationalizes observable conversational proxies for diminishing-return search.

### Adaptive metacognitive intervention

[Liu, Baraniuk, and Sonkar's MetaCLASS (2026)](https://arxiv.org/abs/2602.02457) frames metacognitive coaching as trajectory-aware action selection and makes non-intervention explicit. StopWise applies the general design lesson—not MetaCLASS's educational taxonomy—that a metacognitive system must decide not only what to say, but whether intervention is warranted at all.

| Research idea | StopWise design consequence |
|---|---|
| Information search requires stopping criteria | Estimate whether material action-changing information remains |
| Search can reach informational sufficiency | `COMMIT` represents low expected decision value from further search |
| Stopping depends on task context | Stakes and reversibility adjust intervention thresholds |
| Metacognitive support depends on trajectory | Analyze conversation history, not only the latest turn |
| Intervention is not mandatory | Treat `NO_INTERVENTION` as a first-class action |
| Models may over-intervene | Use conservative thresholds and evaluate false intervention explicitly |

StopWise is not a reproduction of either work. It applies stopping-rule and adaptive-intervention ideas to general LLM-assisted decision conversations. See [theory and design rationale](docs/theory.md) and the verified [BibTeX references](references.bib).

## Three ways to use StopWise

### 1. Custom Instruction — zero code

Copy [`prompts/custom_instruction.md`](prompts/custom_instruction.md) into the system/custom-instruction field of ChatGPT, Claude, Gemini, or another conversational assistant. A shorter version is available at [`prompts/custom_instruction_compact.md`](prompts/custom_instruction_compact.md).

```text
user asks question
        ↓
assistant answers normally
        ↓
StopWise checks for diminishing returns
        ↓
optional 1–3 sentence nudge
```

This mode does not expose JSON. When no intervention is warranted, the user sees no StopWise meta-comment.

### 2. Middleware — structured API

The Python package analyzes conversation history and returns a validated policy result. The host application decides whether and how to render it.

```python
from openai import OpenAI
from stopwise import StopWise

stopwise = StopWise(client=OpenAI(), model="gpt-4o-mini")

result = stopwise.analyze([
    {"role": "user", "content": "A meets my main needs. Should I compare again?"}
])

print(result.action)
print(result.message)
```

The core is provider-light: OpenAI-compatible transports are optional, and arbitrary SDKs or local models can use `generator=`. Provider details belong to this deployment and are documented under [integrations](docs/integrations.md).

### 3. Skill — reusable assistant capability

[`skills/stopwise/SKILL.md`](skills/stopwise/SKILL.md) packages the same policy as a host-assistant capability. It tracks the goal, criteria, stakes, reversibility, and trajectory; answers normally first; then adds a short nudge only when useful.

The Skill is not an autonomous agent. With `NO_INTERVENTION`, it remains silent about StopWise.

## How it works

```text
conversation trajectory
        ↓
goal + current decision
        ↓
primary / unresolved criteria
        ↓
diminishing-return signals
        ↓
stakes × reversibility gate
        ↓
NO_INTERVENTION / FOCUS / COMMIT / DEFER
```

The central question is:

> **Is additional information still likely to materially change the user's action?**

The answer is contextual. High-impact or hard-to-reverse decisions warrant more inquiry. Low-impact, reversible choices can often use a satisfactory stopping rule once primary criteria are resolved.

## Actions

| Action | Meaning | Important boundary |
|---|---|---|
| `NO_INTERVENTION` | No metacognitive nudge is useful | Continue helping normally; usually render nothing |
| `FOCUS` | Continue analysis, but narrow to decision-relevant variables | **FOCUS means continue, but narrow** |
| `COMMIT` | Primary criteria are resolved and further search is unlikely to change the action | Strongest stopping recommendation; never valid while material information remains |
| `DEFER` | One future branch does not need to be solved now | Main-decision analysis may continue |

`NO_INTERVENTION` is a real action, not a generic fallback. It is appropriate for useful follow-ups, weak evidence of diminishing returns, high-stakes uncertainty, and situations where a nudge would add little value.

`FOCUS` can coexist with `unresolved_action_changing_information = true`. It redirects useful inquiry; it is not premature stopping.

`DEFER` applies to a branch, not necessarily the whole decision. A distant contingency can wait while a current primary criterion is still investigated.

## Signals

- `search_space_expansion`: viable choices exist, but alternatives keep being added without a decision-relevant reason.
- `redundant_verification`: the same conclusion is repeatedly rechecked without new evidence. Legitimate robustness checks do not count.
- `pseudo_precision`: requested numerical precision exceeds the measurement basis.
- `secondary_optimization`: low-impact differences dominate after primary criteria are satisfied.
- `contingency_branching`: a low-probability, low-cost, distant, or reversible branch consumes attention too early.
- `goal_drift`: the subproblem no longer materially serves the original objective.

The policy distinguishes primary versus secondary criteria, robustness versus reassurance, measurable uncertainty versus pseudo-precision, relevant versus merely novel evidence, and productive exploration versus diminishing-return search.

## Examples

### COMMIT

> A meets price, performance, and battery requirements, while B fails battery. The same specifications have been checked three times without change.

Possible nudge: “A already satisfies the primary criteria, and the repeated checks have not changed the choice. Unless a new primary requirement appears, the decision is stable enough to make.”

### FOCUS

> Six laptops run the required software, but final price and measured battery life remain unresolved among pages of secondary comparisons.

Possible nudge: “There is still useful uncertainty here, but only price and battery life can change the decision. Focus on those two.”

### NO_INTERVENTION

> The preferred laptop's compatibility with required CAD software has not been verified.

The assistant answers the compatibility question normally and adds no StopWise comment.

### DEFER

> A current subscription fits, can be changed free, and an optional add-on might matter next year but can be enabled instantly later.

Possible nudge: “That future add-on is easier to evaluate when the need exists and costs nothing to enable later. It does not need to block today's plan choice.”

## Middleware installation

StopWise requires Python 3.11 or newer.

```bash
pip install -e '.[openai,dev]'
```

Only Pydantic is required by the core package. The `openai` extra supports compatible clients; no orchestration framework is required.

## Evaluation

Evaluation is deliberately split into two questions.

### Policy regression evaluation

[`eval/policy/cases.jsonl`](eval/policy/cases.jsonl) contains 24 handcrafted sanity and contrastive cases. They test canonical behavior, schema constraints, and difficult policy boundaries. They are **regression cases, not proof of practical effectiveness**.

The evaluator reports action accuracy, false-intervention rate, premature-`COMMIT` rate, unsafe-`DEFER` rate, missed-intervention rate, `NO_INTERVENTION` precision/recall, and signal precision/recall/F1.

```bash
python eval/policy/evaluate.py --predictions path/to/predictions.json
```

The older model/effort matrix remains as [legacy exploratory policy-classification results](eval/legacy/codex_matrix/README.md). Its small single-run design and ambiguous label cannot establish model rankings or conversation outcomes.

### End-to-end effectiveness evaluation

[`eval/controlled/`](eval/controlled/) now provides six interactive task domains, a dynamic information-value oracle, deterministic and provider-neutral simulator/callback interfaces, a paired runner, replayable raw logs, safety metrics, and bootstrap summaries for baseline assistant versus the same assistant plus StopWise. The checked-in offline run is a deterministic smoke fixture, not a model study.

A live controlled model experiment has **not** been run here. The desired result is not simply fewer turns:

```text
redundant search ↓
decision quality ≈ or ↑
premature commit remains low
```

See [evaluation design](docs/evaluation.md) for definitions and claim boundaries.

## Integrations

The middleware supports:

- compatible Responses APIs;
- compatible Chat Completions APIs, including common Qwen, Ollama, and vLLM deployments;
- custom provider or local-model callbacks.

These are implementation details inside the middleware mode, not separate StopWise identities. See [integration examples](docs/integrations.md), [`examples/basic_usage.py`](examples/basic_usage.py), and [`examples/qwen_usage.py`](examples/qwen_usage.py).

## Project status

Currently implemented:

- the four-action StopWise policy and six conversational signals;
- full and compact direct-chat instructions;
- provider-light Python analyzer middleware with Pydantic validation;
- a reusable Skill representation;
- contrastive policy regression cases and action-specific metrics;
- controlled-evaluation tasks, dynamic oracle, paired runner, replay scoring, and deterministic smoke infrastructure.

Preliminary evidence is limited to policy classification on handcrafted cases. No claim is made that StopWise currently reduces redundant turns, preserves decision quality, or improves user outcomes. Those are hypotheses for controlled end-to-end evaluation.

## Limitations

- StopWise inherits the limitations and biases of its underlying language model.
- It may infer the user's goal, utilities, criteria, stakes, reversibility, or missing evidence incorrectly.
- Medical, legal, financial, immigration, safety, major career, and other hard-to-reverse decisions require especially conservative behavior.
- A low-value nudge can itself be a false intervention; silence may be better.
- Policy labels are contextual and may admit reasonable disagreement.
- StopWise is not a mental-health tool and must not diagnose the user.

## Citation and license

StopWise is software, not a published paper. Project citation metadata is in [`CITATION.cff`](CITATION.cff); research inspirations are in [`references.bib`](references.bib).

MIT licensed.
