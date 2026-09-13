# Middleware integrations

Provider integrations belong to the optional **Python analyzer adapter**. The default StopWise entry point is the direct system/custom prompt. The analyzer is a transport-neutral way to obtain a Pydantic-validated policy result; validation checks schema and internal invariants, not the objective correctness of the model's judgment.

## Responses API

```python
from openai import OpenAI
from stopwise import StopWise

stopwise = StopWise(client=OpenAI(), model="gpt-4o-mini")
result = stopwise.analyze(messages)
```

When the client exposes `responses.create`, the default `auto` transport uses strict structured output.

## Chat Completions-compatible APIs

Qwen, Ollama, vLLM, and many hosted providers expose compatible Chat Completions surfaces. Supply the provider's endpoint, credentials, and model, then select the transport explicitly when needed:

```python
from openai import OpenAI
from stopwise import StopWise

client = OpenAI(api_key="...", base_url="https://provider.example/v1")
stopwise = StopWise(
    client=client,
    model="provider-model",
    transport="chat_completions",
    request_options={"extra_body": {"provider_option": True}},
)
result = stopwise.analyze(messages)
```

Compatibility varies by server. The middleware requests a JSON object and validates it locally with the `StopWiseResult` Pydantic model.

## Custom SDKs and local models

Use `generator=` for any backend that does not expose one of those interfaces:

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

The callback may return a JSON string, mapping, or validated `StopWiseResult`. Provider-specific options pass through `request_options`.

## Prompt experiments

`StopWise(system_prompt=...)` accepts custom analyzer instructions. The packaged prompt loader also exposes:

```python
from stopwise import load_prompt

analyzer_prompt = load_prompt("analyzer")
direct_chat_prompt = load_prompt("custom_instruction")
compact_prompt = load_prompt("custom_instruction_compact")
pre_optimization_snapshot = load_prompt("custom_instruction_current_snapshot")
```

`custom_instruction_compact` remains a compatibility alias synchronized to the canonical direct-chat prompt. The frozen snapshot exists for the controlled `current_prompt` condition and should not replace the default prompt. Direct-chat prompts are not analyzer prompts: they answer normally and may add a human-readable nudge, without JSON.

If the analyzer participates in an effectiveness experiment, use a separate `middleware` condition and record its additional calls, tokens, latency, failures, and estimated/provider cost. Do not mix those costs into the three primary prompt-only conditions.
