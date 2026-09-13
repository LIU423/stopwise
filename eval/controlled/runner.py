"""Episode state machine and paired controlled-experiment runner."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable

from .conditions import (
    AssistantModelCallback,
    AssistantRequest,
    ConditionHarness,
    assert_fair_conditions,
    build_conditions,
)
from .environment import DecisionEnvironment
from .schemas import (
    ControlledTask,
    ConversationTurn,
    EpisodeLog,
    ExperimentConfig,
    PolicyEvent,
    StopWiseAction,
    TokenUsage,
    UsageSource,
)
from .simulators import DeterministicUserSimulator, TextOnlyUserSimulator


def load_tasks(path: Path) -> list[ControlledTask]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = [ControlledTask.model_validate(item) for item in raw]
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    return tasks


def paired_seed(base_seed: int, task_id: str, replicate: int) -> int:
    payload = f"{base_seed}:{task_id}:{replicate}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def task_definition_hash(task: ControlledTask) -> str:
    canonical = json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def run_episode(
    *,
    experiment_id: str,
    task: ControlledTask,
    condition: ConditionHarness,
    replicate: int,
    seed: int,
    max_information_turns: int,
    assistant: AssistantModelCallback,
    simulator: DeterministicUserSimulator | TextOnlyUserSimulator,
    user_simulator_id: str,
    user_simulator_version: str,
    assistant_callback_id: str = "provider-neutral-callback",
) -> EpisodeLog:
    """Run initialization -> query/reveal/response -> choice -> scoring log."""

    environment = DecisionEnvironment(task)
    turns: list[ConversationTurn] = []
    query_events = []
    policy_events: list[PolicyEvent] = []
    token_usage = TokenUsage()
    user_simulator_token_usage = TokenUsage()
    user_simulator_token_usage_source = UsageSource.NOT_RECORDED
    token_usage_source = UsageSource.NOT_RECORDED
    assistant_request_count = 0
    user_simulator_request_count = 0
    middleware_request_count = 0
    middleware_token_usage = TokenUsage()
    errors: list[str] = []
    estimated_cost_usd = 0.0
    latency_ms = 0.0

    def add_turn(role: str, content: str) -> None:
        turns.append(ConversationTurn(index=len(turns), role=role, content=content))

    add_turn("system", condition.system_prompt)
    initial = f"{task.scenario}\nInitially visible: {json.dumps(environment.visible_values(), sort_keys=True)}"
    add_turn("user", initial)
    transcript = [{"role": "user", "content": initial}]
    final_choice: str | None = None
    termination_reason = "user_choice"

    while len(query_events) < max_information_turns:
        if isinstance(simulator, DeterministicUserSimulator):
            user_action = simulator.next_action(environment.state())
        else:
            user_action = simulator.next_action(transcript)
            user_simulator_request_count += 1
            user_simulator_token_usage = TokenUsage(
                prompt_tokens=user_simulator_token_usage.prompt_tokens + user_action.prompt_tokens,
                completion_tokens=user_simulator_token_usage.completion_tokens + user_action.completion_tokens,
            )
            if user_action.usage_source != UsageSource.NOT_RECORDED:
                if user_simulator_token_usage_source not in {
                    UsageSource.NOT_RECORDED,
                    user_action.usage_source,
                }:
                    raise ValueError("user simulator mixed incompatible token usage sources")
                user_simulator_token_usage_source = user_action.usage_source
            estimated_cost_usd += user_action.estimated_cost_usd
            latency_ms += user_action.latency_ms
        if user_action.kind == "choose":
            final_choice = user_action.choice
            add_turn("user", f"Final choice: {final_choice}")
            break
        if user_action.kind != "query" or not user_action.query_id:
            errors.append("simulator returned an invalid action")
            termination_reason = "error"
            break

        query_id = user_action.query_id
        query = task.queries[query_id]
        add_turn("user", f"Information query {query_id}: {query.rationale}")
        transcript.append({"role": "user", "content": turns[-1].content})

        outcome = environment.query(query_id)
        query_events.append(outcome.event)
        observation = (
            f"{query.alternative_id}.{query.attribute_id} = "
            f"{json.dumps(outcome.event.revealed_value)}"
        )
        add_turn("environment", observation)

        request = AssistantRequest(
            system_prompt=condition.system_prompt,
            model=condition.model,
            messages=transcript,
            visible_attributes=environment.visible_values(),
            latest_observation=observation,
        )
        response = assistant(request)
        assistant_request_count += 1
        token_usage = TokenUsage(
            prompt_tokens=token_usage.prompt_tokens + response.prompt_tokens,
            completion_tokens=token_usage.completion_tokens + response.completion_tokens,
        )
        if response.usage_source != UsageSource.NOT_RECORDED:
            if token_usage_source not in {UsageSource.NOT_RECORDED, response.usage_source}:
                raise ValueError("assistant callback mixed incompatible token usage sources")
            token_usage_source = response.usage_source
        middleware_request_count += response.middleware_calls
        middleware_token_usage = TokenUsage(
            prompt_tokens=middleware_token_usage.prompt_tokens + response.middleware_prompt_tokens,
            completion_tokens=middleware_token_usage.completion_tokens + response.middleware_completion_tokens,
        )
        estimated_cost_usd += response.estimated_cost_usd
        latency_ms += response.latency_ms
        rendered = response.content
        if response.visible_nudge:
            rendered = f"{rendered}\n\n{response.visible_nudge}"
        add_turn("assistant", rendered)
        transcript.append({"role": "assistant", "content": rendered})

        if condition.name == "baseline":
            if response.action is not None or response.visible_nudge:
                raise ValueError("baseline callback must not emit StopWise actions or nudges")
        else:
            if response.action is None:
                raise ValueError("StopWise callback must emit an action, including NO_INTERVENTION")
            expected, expected_branch = environment.expected_policy_action(outcome)
            policy_events.append(
                PolicyEvent(
                    turn_index=turns[-1].index,
                    action=response.action,
                    expected_action=expected,
                    visible_nudge=response.visible_nudge,
                    target_branch_id=response.target_branch_id,
                    sufficient_at_action=outcome.after.sufficient,
                )
            )
            if response.action == StopWiseAction.DEFER and response.target_branch_id != expected_branch:
                # Keep the raw mismatch scoreable instead of silently repairing it.
                errors.append(
                    f"DEFER targeted {response.target_branch_id!r}; oracle branch is {expected_branch!r}"
                )
        if isinstance(simulator, DeterministicUserSimulator):
            simulator.observe(response)
    else:
        termination_reason = "max_information_turns"
        candidates = environment.state().recommended or environment.state().possible_feasible
        final_choice = candidates[0] if candidates else sorted(task.alternatives)[0]
        add_turn("user", f"Environment-forced final choice: {final_choice}")

    if final_choice is None:
        state = environment.state()
        candidates = state.recommended or state.possible_feasible
        final_choice = candidates[0] if candidates else sorted(task.alternatives)[0]

    return EpisodeLog(
        experiment_id=experiment_id,
        task_id=task.task_id,
        task_version=task.task_version,
        task_definition_hash=task_definition_hash(task),
        domain=task.domain,
        condition=condition.name,
        replicate=replicate,
        seed=seed,
        base_model=condition.model.base_model,
        exact_model_version=condition.model.exact_model_version,
        assistant_model_config=condition.model,
        base_prompt_hash=condition.base_prompt_hash,
        prompt_version_hash=condition.prompt_version_hash,
        policy_version_hash=condition.policy_version_hash,
        assistant_callback_id=assistant_callback_id,
        user_simulator_id=user_simulator_id,
        user_simulator_version=user_simulator_version,
        max_information_turns=max_information_turns,
        conversation_turns=turns,
        query_ids=[event.query_id for event in query_events],
        revealed_attributes=[f"{event.alternative_id}.{event.attribute_id}" for event in query_events],
        query_events=query_events,
        stopwise_actions=[event.action for event in policy_events],
        policy_events=policy_events,
        visible_nudges=[event.visible_nudge for event in policy_events if event.visible_nudge],
        final_choice=final_choice,
        token_usage=token_usage,
        token_usage_source=token_usage_source,
        user_simulator_token_usage=user_simulator_token_usage,
        user_simulator_token_usage_source=user_simulator_token_usage_source,
        assistant_request_count=assistant_request_count,
        user_simulator_request_count=user_simulator_request_count,
        middleware_request_count=middleware_request_count,
        middleware_token_usage=middleware_token_usage,
        estimated_cost_usd=estimated_cost_usd,
        termination_reason=termination_reason,
        errors=errors,
        retries=0,
        latency_ms=latency_ms,
        latency_source="client" if latency_ms else "not_recorded",
    )


def run_paired_experiment(
    config: ExperimentConfig,
    tasks: Iterable[ControlledTask],
    assistant: AssistantModelCallback,
) -> list[EpisodeLog]:
    """Backwards-compatible deterministic grouped smoke runner."""

    return run_grouped_experiment(
        config,
        tasks,
        assistant,
        simulator_factory=lambda task, seed: DeterministicUserSimulator(task, seed),
    )


def run_grouped_experiment(
    config: ExperimentConfig,
    tasks: Iterable[ControlledTask],
    assistant: AssistantModelCallback,
    *,
    simulator_factory: Callable[
        [ControlledTask, int], DeterministicUserSimulator | TextOnlyUserSimulator
    ],
    on_episode: Callable[[list[EpisodeLog]], None] | None = None,
) -> list[EpisodeLog]:
    """Run all three primary conditions with an identical simulator factory."""

    conditions = build_conditions(config.base_system_prompt, config.model)
    assert_fair_conditions(list(conditions.values()))
    logs: list[EpisodeLog] = []
    for task in tasks:
        for replicate in range(config.replicates):
            seed = paired_seed(config.base_seed, task.task_id, replicate)
            order = ["baseline", "current_prompt", "minimal_prompt"]
            Random(seed).shuffle(order)
            for name in order:
                simulator = simulator_factory(task, seed)
                logs.append(
                    run_episode(
                        experiment_id=config.experiment_id,
                        task=task,
                        condition=conditions[name],
                        replicate=replicate,
                        seed=seed,
                        max_information_turns=config.max_information_turns,
                        assistant=assistant,
                        simulator=simulator,
                        user_simulator_id=config.user_simulator_id,
                        user_simulator_version=config.user_simulator_version,
                        assistant_callback_id=config.assistant_callback_id,
                    )
                )
                if on_episode is not None:
                    on_episode(list(logs))
    return logs


_TOKEN_PATTERN = re.compile(r"(?i)(?:bearer\s+|sk-)[A-Za-z0-9._-]{8,}")
_SECRET_PARTS = ("api_key", "apikey", "access_token", "password", "secret", "authorization", "cookie")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            result[key] = (
                "[REDACTED]"
                if any(part in normalized for part in _SECRET_PARTS)
                else redact_secrets(item)
            )
        return result
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _TOKEN_PATTERN.sub("[REDACTED]", value)
    return value


def write_jsonl(logs: Iterable[EpisodeLog], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(redact_secrets(log.model_dump(mode="json")), sort_keys=True, ensure_ascii=False)
        for log in logs
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[EpisodeLog]:
    return [
        EpisodeLog.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
