# StopWise

> Know when more information stops helping.

StopWise is a lightweight metacognitive layer for LLM conversations.

LLMs are good at answering the next question. StopWise asks whether the next question is still worth asking.

```text
Conversation
    -> decision-state analysis
    -> diminishing-return signals
    -> intervention policy
    -> optional metacognitive nudge
```

**Silence is a first-class action.** StopWise defaults to `NO_INTERVENTION` when more information could still change the user's action, the evidence is ambiguous, or a nudge would add little value. It is deliberately more concerned about stopping useful inquiry too early than about missing an opportunity to intervene.

## Actions

- `NO_INTERVENTION`: continue normally; often emits no message.
- `FOCUS`: keep analyzing, but narrow the search to decision-relevant variables.
- `COMMIT`: primary criteria are resolved and more search is unlikely to change the action.
- `DEFER`: postpone a low-value future contingency until it becomes relevant.

StopWise looks for six conversational signals: search-space expansion, redundant verification, pseudo-precision, secondary optimization, contingency branching, and goal drift. These are operational conversation signals inspired by stopping-rule research—not claims about a user's mental state.

## Installation

This prototype requires Python 3.11 or newer.

```bash
conda create -n stopwise python=3.11
conda activate stopwise
pip install -e '.[openai,dev]'
```

## Usage

The core package is provider-agnostic. It supports:

- clients exposing an OpenAI-compatible Responses API;
- clients exposing an OpenAI-compatible Chat Completions API, including Qwen, Ollama, vLLM, and many hosted providers;
- arbitrary SDKs and local models through a small `generator=` callback.

### OpenAI

```python
from openai import OpenAI
from stopwise import StopWise

client = OpenAI()
stopwise = StopWise(client=client, model="gpt-4o-mini")

messages = [
    {"role": "user", "content": "A and B both fit my budget, but I still need to check whether A runs the software required for my course."}
]

result = stopwise.analyze(messages)
print(result.action)
print(result.message)
```

Or use the functional API:

```python
from stopwise import analyze_conversation

result = analyze_conversation(messages=messages, client=client)
```

### Qwen / Alibaba Cloud Model Studio

Qwen exposes OpenAI-compatible interfaces, so only the endpoint, API key, model, and transport change. Keep the endpoint in an environment variable because it varies by region and workspace. See the [official compatibility documentation](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope).

```python
import os
from openai import OpenAI
from stopwise import StopWise

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

stopwise = StopWise(
    client=client,
    model="qwen-plus",
    transport="chat_completions",
    request_options={"extra_body": {"enable_thinking": False}},
)

result = stopwise.analyze(messages)
```

The explicit transport avoids assuming that every compatible provider implements the Responses API identically. JSON mode is used for broad model compatibility; the Pydantic schema still validates the returned object.

### Any other model or SDK

Pass a callback that translates StopWise's neutral request into the provider's API. It may return a JSON string, a mapping, or a `StopWiseResult`.

```python
from stopwise import StopWise

def generate(*, model, system_prompt, messages, json_schema, request_options):
    response = my_provider.generate(
        model=model,
        system=system_prompt,
        messages=messages,
        schema=json_schema,
        **request_options,
    )
    return response.text

stopwise = StopWise(generator=generate, model="my-model")
result = stopwise.analyze(messages)
```

`request_options=` is passed to compatible clients or exposed to the custom callback, allowing provider-specific controls without adding provider dependencies to StopWise.

The output is a validated Pydantic model. A custom prompt can be supplied through `system_prompt=`; the canonical prompt is in [`prompts/stopwise_system.md`](prompts/stopwise_system.md).

## Evaluation

The repository includes handcrafted cases targeting both intervention opportunities and false-positive traps:

```bash
export OPENAI_API_KEY=...
conda run -n stopwise python eval/evaluate.py --model gpt-4o-mini
```

For Qwen:

```bash
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=https://YOUR_REGION_AND_WORKSPACE/compatible-mode/v1
conda run -n stopwise python eval/evaluate.py \
  --model qwen-plus \
  --base-url "$DASHSCOPE_BASE_URL" \
  --api-key-env DASHSCOPE_API_KEY \
  --transport chat_completions
```

The evaluator reports action accuracy, premature intervention rate, and missed intervention rate. The objective is not simply fewer turns:

```text
Redundant turns decrease
Decision quality does not decrease
Premature intervention remains low
```

For the reproducible Codex model/effort matrix runner and its single-run baseline, see [`eval/run_codex_matrix.py`](eval/run_codex_matrix.py) and [`eval/results/codex_matrix/README.md`](eval/results/codex_matrix/README.md).

## Design boundaries

StopWise is a small prompt-driven policy layer, not a recommendation engine, autonomous agent, psychological diagnostic tool, or browser automation system. It does not decide on the user's behalf; it only judges the likely decision value of further information seeking.

## Limitations

- StopWise is heuristic and inherits the limitations of its underlying language model.
- It may infer the user's true utility, constraints, or priorities incorrectly.
- It should be especially conservative in high-stakes or hard-to-reverse decisions.
- It must not discourage necessary medical, legal, financial, immigration, or safety-related information seeking.
- Novel information can be interesting without being decision-relevant, but the distinction is context-dependent.

See [`docs/theory.md`](docs/theory.md) for the theoretical framing and policy rationale.

## License

MIT
