# Controlled end-to-end evaluation

This directory implements a paired, interactive comparison of a baseline assistant and the same assistant with the direct-chat StopWise policy. It is separate from `eval/policy/`, which tests policy-label agreement but not conversational effects.

## What is implemented

- six versioned decision domains with hidden, queryable attributes;
- Pydantic task, model, turn, query, policy-event, and episode-log schemas;
- a stateful decision environment and dynamic action-changing oracle;
- deterministic assistant/user fixtures for offline validation;
- provider-neutral assistant and text-only user-simulator callback protocols;
- fair baseline/StopWise condition construction using the canonical direct-chat prompt;
- paired episode running, secret-redacted raw JSONL logs, replay scoring, bootstrap intervals, paired binary comparisons, effect sizes, and domain strata;
- automated tests for the safety and metric boundaries.

The deterministic fixture is deliberately simple and policy-responsive. It validates mechanics and reproducibility; it does not estimate how people or LLM users respond to StopWise.

## Files

- `PROTOCOL.md` — design, oracle, metrics, statistics, logging, and claim boundaries.
- `schemas.py` — authoritative Pydantic schemas.
- `export_schemas.py` — optional machine-readable JSON Schema export.
- `tasks.json` — structured task set and oracle provenance.
- `environment.py` — partial-information state and dynamic query effects.
- `conditions.py` — fair condition prompts and provider-neutral assistant callback.
- `simulators.py` — deterministic simulator and blinded text-only LLM hook.
- `runner.py` — episode state machine and JSONL logging.
- `score.py` — replay scoring from raw logs.
- `statistics.py` — paired summaries and intervals.
- `example_config.json` — full deterministic example.
- `fixtures/smoke_config.json` — two-task, one-replicate offline smoke fixture.

## Offline smoke run

From an editable install, run:

```bash
python -m eval.controlled.run_experiment \
  --config eval/controlled/fixtures/smoke_config.json \
  --output-dir /tmp/stopwise-smoke
```

From a source checkout without installation, prefix the command with `PYTHONPATH=src`.

The runner writes separate artifacts:

- `episodes.jsonl`: raw episode logs;
- `summary.json`: rebuilt run scores, condition summaries, paired differences, intervals, and domain strata.

The summary also reports callback invocation counts. `external_requests_executed` is always zero for the deterministic runner; the remote-request estimate is a planning upper bound when both the assistant and LLM user simulator use one request per turn.

Re-score the raw log independently:

```bash
PYTHONPATH=src python -m eval.controlled.score \
  /tmp/stopwise-smoke/episodes.jsonl \
  --tasks eval/controlled/tasks.json \
  --output /tmp/stopwise-smoke/rescored.json
```

No command in this directory automatically invokes an external model.

Export task, episode-log, and experiment-config JSON Schemas when another harness needs a language-neutral contract:

```bash
PYTHONPATH=src python -m eval.controlled.export_schemas --output-dir /tmp/stopwise-schemas
```

## Adding a real provider

Implement `AssistantModelCallback` so it sends `AssistantRequest.system_prompt`, messages, visible attributes, model identifier, and decoding parameters to the provider and returns `AssistantResponse`. The request intentionally contains no hidden values or information-sufficiency oracle. Query relevance fields support the deterministic fixture and evaluation adapter; do not serialize them into a live model prompt unless the same user-stated criteria are already visible in the transcript. In the main comparison, the StopWise condition is the canonical direct-chat/Skill policy; an analyzer-middleware experiment must be named and run as a separate condition because it adds calls, tokens, latency, and possible failure modes.

For an LLM user simulator, implement the text-only callback contract. It must receive the same simulator model/version, prompt, parameters, seed policy, and transcript in both conditions; it must not receive the condition name, StopWise labels, hidden values, dynamic oracle fields, or policy-event records.

Before any paid pilot, record the provider and exact model version, task/condition/replicate count, expected requests, token/cost bounds, and output paths, then obtain explicit authorization.

## Evidence boundary

The checked-in code and offline fixture are implemented infrastructure plus deterministic smoke validation. They are not pilot model results and not controlled effectiveness evidence.

**尚未形成 StopWise 端到端有效性证据。**
