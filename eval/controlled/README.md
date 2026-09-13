# Controlled end-to-end evaluation

This directory implements a grouped paired, interactive comparison of three prompt-only conditions: `baseline`, a frozen pre-optimization `current_prompt`, and the new `minimal_prompt`. It is separate from `eval/policy/`, which tests policy-label agreement but not conversational effects.

## What is implemented

- six versioned decision domains with hidden, queryable attributes;
- Pydantic task, model, turn, query, policy-event, and episode-log schemas;
- a stateful decision environment and dynamic action-changing oracle;
- deterministic assistant/user fixtures for offline validation, including intentionally hard-coded policy response and oracle-aware scripted choice;
- a provider-neutral assistant request that contains visible values but no hidden answers, oracle state, relevance labels, branch annotations, repeat flags, or condition name;
- a text-only live user-simulator adapter that receives visible natural language and an unlabeled action menu, but no condition, StopWise action, hidden value, oracle state, or policy event;
- fair three-condition construction using a frozen current-policy snapshot and the canonical minimal prompt;
- paired episode running, secret-redacted raw JSONL logs, replay scoring, bootstrap intervals, paired binary comparisons, effect sizes, and domain strata;
- automated tests for the safety and metric boundaries.

The deterministic fixture is deliberately simple and policy-responsive: its assistant uses hard-coded rules, while its user reads structured action labels and oracle state. It validates mechanics and reproducibility only; it cannot estimate prompt effectiveness or how people or blinded LLM users respond to StopWise.

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
- `openai_live.py` — Responses API adapters plus request/token-price budget guard.
- `run_live_experiment.py` — explicit paid-run CLI with dry-run preflight and checkpoints.
- `fixtures/pilot_openai_gpt-5.5-2026-04-23.json` — reviewed minimal-pilot configuration.

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

The summary also reports callback invocation counts. `external_requests_executed` is always zero for the deterministic runner; the remote-request estimate is a planning upper bound when both the assistant and LLM user simulator use one request per turn. Fixture word counts are explicitly labeled `estimated_fixture`; they are not provider token usage and must not be used for cost or effectiveness claims.

Re-score the raw log independently:

```bash
PYTHONPATH=src python -m eval.controlled.score \
  /tmp/stopwise-smoke/episodes.jsonl \
  --tasks eval/controlled/tasks.json \
  --output /tmp/stopwise-smoke/rescored.json
```

The deterministic command never invokes an external model.

Export task, episode-log, and experiment-config JSON Schemas when another harness needs a language-neutral contract:

```bash
PYTHONPATH=src python -m eval.controlled.export_schemas --output-dir /tmp/stopwise-schemas
```

## Adding a real provider

Implement `AssistantModelCallback` so it sends `AssistantRequest.system_prompt`, messages, visible attributes, latest visible observation, model identifier, and decoding parameters to the provider and returns `AssistantResponse`. The request schema makes hidden values, oracle fields, evaluation labels, branch metadata, repeat flags, and condition names unavailable to the callback. The three primary conditions differ only in their system prompt.

For an LLM user simulator, use `TextOnlyUserSimulator` with the text-only callback contract and pass a condition-independent factory to `run_grouped_experiment`. It must use the same simulator model/version, prompt, parameters, seed policy, and request limit in all three conditions. Its callback sees the visible transcript and natural-language query/choice menu only. It never receives a structured StopWise action, so a COMMIT-like sentence remains advice that the simulated user may accept or reject rather than a mechanical stop command.

An analyzer-middleware experiment must be a separate `middleware` condition because it adds model calls, tokens, latency, cost, and failure modes. Middleware logs must record those extra requests and token usage instead of silently attributing them to a prompt-only condition.

Before any paid pilot, record the provider and exact model version, task/condition/replicate count, expected requests, token/cost bounds, and output paths, then obtain explicit authorization. The live CLI requires either `--dry-run` or the explicit `--confirm-paid-run` flag, refuses to overwrite a non-empty output directory, disables SDK retries, checkpoints completed episodes, and reads the API key only from an environment variable.

Preflight the checked-in minimal pilot without an API call:

```bash
PYTHONPATH=src python -m eval.controlled.run_live_experiment \
  --config eval/controlled/fixtures/pilot_openai_gpt-5.5-2026-04-23.json \
  --output-dir /tmp/stopwise-pilot-preflight \
  --max-cost-usd 5 \
  --dry-run
```

After reviewing that manifest and explicitly authorizing the cost, replace the output directory and `--dry-run` with `--confirm-paid-run`. This configuration uses OpenAI Responses API model snapshot `gpt-5.5-2026-04-23`, two tasks, three conditions, one replicate, eight information turns, no tools, no retries, and at most 96 requests. At the prices reviewed on 2026-09-13, its conservative worst-case estimate is USD 4.73088 including a 10% billing uplift. The output directory contains `config.json`, checkpointed `episodes.jsonl`, and `summary.json`; failures additionally write `failure.json`.

For an ephemeral zsh credential, first run the hidden-input prompt below in the same terminal that will launch the pilot. Paste the key only after the `OpenAI API Key:` prompt appears; never paste a key at the ordinary shell prompt or into chat.

```zsh
read -s "OPENAI_API_KEY?OpenAI API Key: "; export OPENAI_API_KEY; echo
```

Assistant and LLM-user usage comes from provider-reported response fields. Cost is a conservative token-price estimate that charges cached input at the full input rate and includes the configured uplift; it is not an exact invoice. The response schema is fixed across conditions, and the model-facing assistant payload differs only in system instructions. The user simulator sees only visible natural language and an unlabeled action menu, never the condition or StopWise action.

## Evidence boundary

The checked-in code is implemented infrastructure. A successful offline run is deterministic smoke validation. Neither is a pilot model result or controlled effectiveness evidence; those labels require actual model runs and appropriately powered interpretation, respectively.

**尚未形成 StopWise 端到端有效性证据。**
