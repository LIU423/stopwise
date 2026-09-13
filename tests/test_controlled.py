import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from eval.controlled.conditions import (
    AssistantRequest,
    AssistantResponse,
    DeterministicAssistantCallback,
    build_conditions,
)
from eval.controlled.environment import DecisionEnvironment
from eval.controlled.export_schemas import export_schemas
from eval.controlled.runner import (
    load_tasks,
    read_jsonl,
    run_episode,
    run_grouped_experiment,
    run_paired_experiment,
    write_jsonl,
)
from eval.controlled.openai_live import (
    OpenAIAssistantCallback,
    OpenAIResponsesTransport,
    OpenAITextOnlyUserCallback,
    PaidRunBudget,
)
from eval.controlled.run_live_experiment import (
    build_run_manifest,
    load_config_and_tasks,
    make_budget,
    request_limit,
)
from eval.controlled.schemas import (
    EpisodeLog,
    ExperimentConfig,
    ModelConfig,
    StopWiseAction,
    TokenUsage,
    UsageSource,
)
from eval.controlled.score import score_episode, score_logs
from eval.controlled.simulators import (
    DeterministicUserSimulator,
    TextOnlyUserSimulator,
    UserAction,
    UserModelRequest,
)
from eval.controlled.statistics import paired_summary, validate_paired_logs


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "eval/controlled/tasks.json"
CONFIG_PATH = ROOT / "eval/controlled/example_config.json"
LIVE_CONFIG_PATH = ROOT / "eval/controlled/fixtures/pilot_openai_gpt-5.5-2026-04-23.json"


def config(replicates=1):
    parsed = ExperimentConfig.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))
    return parsed.model_copy(update={"replicates": replicates, "bootstrap_samples": 200})


def tasks():
    return load_tasks(TASK_PATH)


def paired_logs(task_count=1, replicates=1):
    return run_paired_experiment(
        config(replicates), tasks()[:task_count], DeterministicAssistantCallback()
    )


def test_task_schema_validates_six_domains_and_dynamic_oracle_annotations():
    loaded = tasks()
    assert len(loaded) == 6
    assert {task.domain for task in loaded} == {
        "laptop selection",
        "hotel selection",
        "subscription plan",
        "course selection",
        "travel plan",
        "device procurement",
    }
    assert all(task.oracle_annotation.source == "dynamic" for task in loaded)
    assert all(not task.oracle_annotation.manual_action_changing for task in loaded)
    assert all(len(task.simulator_query_plan) >= 10 for task in loaded)


def test_machine_readable_task_episode_and_config_schemas_export(tmp_path):
    paths = export_schemas(tmp_path)
    assert {path.name for path in paths} == {
        "task.schema.json", "episode.schema.json", "experiment.schema.json"
    }
    episode = json.loads((tmp_path / "episode.schema.json").read_text(encoding="utf-8"))
    assert {"experiment_id", "task_id", "query_events", "policy_events", "final_choice"} <= set(
        episode["required"]
    )


def test_all_tasks_reach_an_explainable_information_sufficient_state():
    for task in tasks():
        environment = DecisionEnvironment(task)
        assert environment.state().sufficient is False
        for query_id in task.simulator_query_plan:
            environment.query(query_id)
        state = environment.state()
        assert state.sufficient is True, task.task_id
        assert len(state.possible_optimal) == 1
        assert state.possible_optimal[0] in state.confirmed_feasible


def test_dynamic_action_changing_is_not_static_relevance():
    task = tasks()[0]
    environment = DecisionEnvironment(task)
    irrelevant = environment.query("lap-box-color")
    assert irrelevant.event.action_changing is False
    assert irrelevant.event.redundant is True
    first = environment.query("lap-a-cad")
    assert first.event.action_changing is True
    repeated = environment.query("lap-a-cad")
    assert repeated.event.action_changing is False
    assert repeated.event.repeated is True
    assert repeated.event.redundant is True


def test_same_seed_and_fixture_are_reproducible():
    first = [item.model_dump(mode="json") for item in paired_logs(task_count=2)]
    second = [item.model_dump(mode="json") for item in paired_logs(task_count=2)]
    assert first == second


def test_conditions_hold_model_parameters_and_seed_constant():
    logs = paired_logs(task_count=2)
    pairs = validate_paired_logs(logs)
    for pair in pairs.values():
        assert set(pair) == {"baseline", "current_prompt", "minimal_prompt"}
        for condition in ("current_prompt", "minimal_prompt"):
            assert pair["baseline"].assistant_model_config == pair[condition].assistant_model_config
            assert pair["baseline"].seed == pair[condition].seed
            assert pair["baseline"].base_prompt_hash == pair[condition].base_prompt_hash
            assert pair["baseline"].prompt_version_hash != pair[condition].prompt_version_hash
        assert pair["current_prompt"].prompt_version_hash != pair["minimal_prompt"].prompt_version_hash


def test_redundancy_acr_utility_regret_and_constraint_scoring():
    task = tasks()[0]
    logs = paired_logs()
    baseline = next(item for item in logs if item.condition == "baseline")
    score = score_episode(task, baseline)
    assert score.redundant_query_count > 0
    assert score.redundant_query_rate == pytest.approx(
        score.redundant_query_count / score.information_turns
    )
    assert score.action_changing_rate == pytest.approx(
        score.action_changing_count / score.new_information_query_count
    )
    assert score.optimal_choice is True
    assert score.constraint_satisfied is True
    assert score.regret == 0

    suboptimal = baseline.model_copy(update={"final_choice": "C"})
    suboptimal_score = score_episode(task, suboptimal)
    assert suboptimal_score.constraint_satisfied is True
    assert suboptimal_score.optimal_choice is False
    assert suboptimal_score.regret > 0

    violating = baseline.model_copy(update={"final_choice": "B"})
    assert score_episode(task, violating).constraint_satisfied is False


class AlwaysCommitAssistant:
    def __call__(self, request):
        return AssistantResponse(
            content="Here is the requested value.",
            action=StopWiseAction.COMMIT,
            visible_nudge="Commit now.",
            prompt_tokens=10,
            completion_tokens=5,
        )


def test_premature_commit_is_detected():
    task = tasks()[0]
    conditions = build_conditions(config().base_system_prompt, config().model)
    log = run_episode(
        experiment_id="premature-test",
        task=task,
        condition=conditions["minimal_prompt"],
        replicate=0,
        seed=11,
        max_information_turns=10,
        assistant=AlwaysCommitAssistant(),
        simulator=DeterministicUserSimulator(task, 11),
        user_simulator_id="deterministic-v1",
        user_simulator_version="1.0.0",
    )
    result = score_episode(task, log)
    assert result.commit_count == 1
    assert result.premature_commit_count == 1


def test_focus_does_not_terminate_and_valid_search_continues():
    task = tasks()[0]
    log = next(item for item in paired_logs() if item.condition == "minimal_prompt")
    assert log.policy_events[0].action == StopWiseAction.FOCUS
    assert len(log.query_events) > 1
    result = score_episode(task, log)
    assert result.focus_count > 0
    assert result.focus_followed_by_valid_query > 0
    assert result.focus_followed_by_valid_query <= result.focus_count


def test_defer_only_skips_the_target_branch_and_main_search_continues():
    log = next(item for item in paired_logs() if item.condition == "minimal_prompt")
    assert "lap-future-dock" in log.query_ids
    assert "lap-future-countries" not in log.query_ids
    assert "lap-b-cad" in log.query_ids
    defer = next(item for item in log.policy_events if item.action == StopWiseAction.DEFER)
    assert defer.target_branch_id == "future-dock"


def test_no_intervention_has_no_visible_nudge():
    log = next(item for item in paired_logs() if item.condition == "minimal_prompt")
    no_interventions = [
        item for item in log.policy_events if item.action == StopWiseAction.NO_INTERVENTION
    ]
    assert no_interventions
    assert all(item.visible_nudge == "" for item in no_interventions)
    with pytest.raises(ValidationError, match="NO_INTERVENTION"):
        AssistantResponse(
            content="answer",
            action=StopWiseAction.NO_INTERVENTION,
            visible_nudge="visible",
        )


class UnsafeDeferAssistant:
    def __init__(self):
        self.default = DeterministicAssistantCallback()

    def __call__(self, request):
        if "hotel-a-access" in request.messages[-1]["content"]:
            return AssistantResponse(
                content="Here is the access result.",
                action=StopWiseAction.DEFER,
                visible_nudge="Defer accessibility.",
                target_branch_id="access-now",
            )
        return self.default(request)


def test_unsafe_defer_is_detected_for_a_required_now_branch():
    task = tasks()[1]
    harness = build_conditions(config().base_system_prompt, config().model)["minimal_prompt"]
    log = run_episode(
        experiment_id="unsafe-defer-test",
        task=task,
        condition=harness,
        replicate=0,
        seed=17,
        max_information_turns=16,
        assistant=UnsafeDeferAssistant(),
        simulator=DeterministicUserSimulator(task, 17),
        user_simulator_id="deterministic-v1",
        user_simulator_version="1.0.0",
    )
    assert score_episode(task, log).unsafe_defer_count == 1


def test_logs_redact_credentials_and_model_config_rejects_secret_keys(tmp_path):
    log = paired_logs()[0]
    turns = list(log.conversation_turns)
    turns[1] = turns[1].model_copy(
        update={"content": "Authorization: Bearer secret-token-123456"}
    )
    tainted = log.model_copy(update={"conversation_turns": turns})
    path = tmp_path / "episodes.jsonl"
    write_jsonl([tainted], path)
    rendered = path.read_text(encoding="utf-8")
    assert "secret-token-123456" not in rendered
    assert "[REDACTED]" in rendered

    with pytest.raises(ValidationError, match="credentials"):
        ModelConfig(
            provider="x",
            base_model="m",
            exact_model_version="m-1",
            parameters={"api_key": "do-not-log"},
        )


def test_missing_pair_fails_clearly():
    one = [next(item for item in paired_logs() if item.condition == "baseline")]
    with pytest.raises(ValueError, match="paired conditions missing"):
        validate_paired_logs(one)


def test_scorer_rebuilds_from_raw_logs(tmp_path):
    logs = paired_logs(task_count=2)
    original_score = score_episode(tasks()[0], next(log for log in logs if log.task_id == "laptop-engineering"))
    target_index = next(
        index for index, event in enumerate(logs[0].query_events) if event.action_changing
    )
    tampered_events = list(logs[0].query_events)
    tampered_events[target_index] = tampered_events[target_index].model_copy(
        update={"action_changing": False, "effects": []}
    )
    logs[0] = logs[0].model_copy(update={"query_events": tampered_events})
    path = tmp_path / "episodes.jsonl"
    write_jsonl(logs, path)
    loaded = read_jsonl(path)
    assert loaded == logs
    scores, summary = score_logs(TASK_PATH, path)
    assert len(scores) == 6
    assert set(summary) == {"baseline", "current_prompt", "minimal_prompt"}
    assert all(0 <= item.action_changing_rate <= 1 for item in scores)
    rebuilt = next(score for score in scores if score.task_id == "laptop-engineering" and score.condition == logs[0].condition)
    assert rebuilt.action_changing_count == original_score.action_changing_count


def test_paired_bootstrap_binary_comparison_and_domain_strata():
    loaded_tasks = tasks()[:2]
    logs = run_paired_experiment(config(replicates=2), loaded_tasks, DeterministicAssistantCallback())
    task_map = {task.task_id: task for task in loaded_tasks}
    scores = [score_episode(task_map[log.task_id], log) for log in logs]
    summary = paired_summary(scores, bootstrap_samples=200, seed=7)
    assert summary["pair_count"] == 4
    comparisons = summary["comparisons"]
    assert set(comparisons) == {
        "current_prompt_vs_baseline",
        "minimal_prompt_vs_baseline",
        "minimal_prompt_vs_current_prompt",
    }
    minimal_vs_current = comparisons["minimal_prompt_vs_current_prompt"]
    assert minimal_vs_current["difference_direction"] == "minimal_prompt_minus_current_prompt"
    assert len(minimal_vs_current["numeric"]["redundant_query_count"]["bootstrap_ci"]) == 2
    assert "exact_mcnemar_p" in minimal_vs_current["binary"]["optimal_choice"]
    assert set(summary["by_domain"]) == {"laptop selection", "hotel selection"}
    assert "Descriptive" in summary["inference_note"]


def test_deterministic_tokens_are_labeled_as_fixture_estimates():
    assert all(log.token_usage_source == UsageSource.ESTIMATED_FIXTURE for log in paired_logs())


def test_live_requests_exclude_oracle_labels_and_commit_is_not_mechanical():
    task = tasks()[0]
    assistant_requests: list[AssistantRequest] = []

    class CapturingAssistant:
        def __call__(self, request):
            assistant_requests.append(request)
            return AssistantResponse(
                content=f"Visible result: {request.latest_observation}",
                action=StopWiseAction.COMMIT,
                visible_nudge="The choice looks stable enough to make.",
                usage_source=UsageSource.PROVIDER_REPORTED,
                prompt_tokens=12,
                completion_tokens=6,
            )

    user_requests = []

    class ContinuingUser:
        def __call__(self, request):
            user_requests.append(request)
            if len(user_requests) == 1:
                return UserAction(kind="query", query_id="lap-box-color")
            if len(user_requests) == 2:
                return UserAction(kind="query", query_id="lap-a-cad")
            return UserAction(kind="choose", choice="A")

    log = run_episode(
        experiment_id="live-isolation-test",
        task=task,
        condition=build_conditions(config().base_system_prompt, config().model)["minimal_prompt"],
        replicate=0,
        seed=23,
        max_information_turns=4,
        assistant=CapturingAssistant(),
        simulator=TextOnlyUserSimulator(task, 23, ContinuingUser()),
        user_simulator_id="text-only-test",
        user_simulator_version="1",
    )

    assert len(log.query_events) == 2
    assert log.user_simulator_request_count == 3
    assert log.token_usage_source == UsageSource.PROVIDER_REPORTED
    assert set(assistant_requests[0].model_dump()) == {
        "system_prompt", "model", "messages", "visible_attributes", "latest_observation"
    }
    assert "A.cad" not in assistant_requests[0].visible_attributes
    assert "A.cad" in assistant_requests[1].visible_attributes
    request_fields = set(user_requests[1].model_dump())
    assert request_fields == {
        "task_id", "scenario", "transcript", "available_queries", "alternatives", "seed"
    }
    serialized = json.dumps(user_requests[1].model_dump(mode="json"))
    assert "oracle" not in serialized.lower()
    assert "relevance" not in serialized.lower()
    assert "minimal_prompt" not in serialized
    assert "COMMIT" not in serialized


def test_middleware_condition_requires_separate_call_accounting():
    raw = next(log for log in paired_logs() if log.condition == "baseline").model_dump(mode="json")
    raw.update(
        condition="middleware",
        middleware_request_count=2,
        middleware_token_usage=TokenUsage(prompt_tokens=30, completion_tokens=10).model_dump(),
        estimated_cost_usd=0.004,
    )
    middleware = EpisodeLog.model_validate(raw)
    assert middleware.middleware_request_count == 2
    assert middleware.middleware_token_usage.total_tokens == 40

    raw["middleware_request_count"] = 0
    raw["middleware_token_usage"] = TokenUsage().model_dump()
    with pytest.raises(ValidationError, match="extra model calls"):
        EpisodeLog.model_validate(raw)


def test_grouped_runner_supports_condition_blind_text_only_simulators():
    seen_requests = []

    class ShortNaturalUser:
        def __init__(self):
            self.turn = 0

        def __call__(self, request):
            seen_requests.append(request)
            self.turn += 1
            if self.turn == 1:
                return UserAction(kind="query", query_id="lap-a-cad")
            return UserAction(kind="choose", choice="A")

    logs = run_grouped_experiment(
        config(),
        tasks()[:1],
        DeterministicAssistantCallback(),
        simulator_factory=lambda task, seed: TextOnlyUserSimulator(
            task, seed, ShortNaturalUser()
        ),
    )

    assert {log.condition for log in logs} == {
        "baseline", "current_prompt", "minimal_prompt"
    }
    assert len({log.seed for log in logs}) == 1
    assert all("condition" not in request.model_dump() for request in seen_requests)


class _FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return SimpleNamespace(
            output_text=json.dumps(output),
            usage=SimpleNamespace(input_tokens=400, output_tokens=40),
            status="completed",
            model="gpt-5.5-2026-04-23",
        )


class _FakeOpenAIClient:
    def __init__(self, outputs):
        self.responses = _FakeResponses(outputs)


def _live_budget(max_requests=96, max_cost_usd=5.0):
    return PaidRunBudget(
        max_cost_usd=max_cost_usd,
        max_requests=max_requests,
        max_input_tokens_per_request=8192,
        max_output_tokens_per_request=128,
        input_usd_per_million=5.0,
        output_usd_per_million=30.0,
        billing_uplift=1.10,
    )


def test_live_pilot_manifest_has_three_fair_conditions_and_hard_budget():
    live_config, live_tasks = load_config_and_tasks(LIVE_CONFIG_PATH)
    budget = make_budget(
        live_config,
        task_count=len(live_tasks),
        max_cost_usd=5.0,
        billing_uplift=1.10,
    )
    manifest = build_run_manifest(live_config, live_tasks, budget)
    assert manifest["conditions"] == ["baseline", "current_prompt", "minimal_prompt"]
    assert request_limit(live_config, len(live_tasks)) == 96
    assert budget.maximum_run_cost_usd <= 5.0
    assert live_config.model.exact_model_version == "gpt-5.5-2026-04-23"
    assert live_config.model.temperature is None
    assert live_config.model.top_p is None
    assert manifest["request_budget"]["invoice_exact"] is False


def test_paid_budget_rejects_an_unsafe_plan_and_oversized_payload():
    with pytest.raises(ValueError, match="worst-case run cost"):
        _live_budget(max_requests=100, max_cost_usd=1.0)
    budget = _live_budget(max_requests=1)
    with pytest.raises(RuntimeError, match="input bound"):
        budget.before_request({"input": "x" * 9000})
    assert budget.requests == 0


def test_openai_assistant_adapter_records_server_usage_without_leaking_oracle():
    live_config, _ = load_config_and_tasks(LIVE_CONFIG_PATH)
    client = _FakeOpenAIClient(
        [{
            "content": "The visible result is A.cad = true.",
            "action": "DEFER",
            "visible_nudge": "That distant branch can wait while the main choice continues.",
            "target_query_id": "lap-future-dock",
        }]
    )
    callback = OpenAIAssistantCallback(
        OpenAIResponsesTransport(client, live_config.model, _live_budget()),
        query_branch_by_id={"lap-future-dock": "future-dock-secret"},
    )
    condition = build_conditions(live_config.base_system_prompt, live_config.model)["minimal_prompt"]
    response = callback(
        AssistantRequest(
            system_prompt=condition.system_prompt,
            model=live_config.model,
            messages=[{
                "role": "user",
                "content": "Information query lap-future-dock: Investigate a distant dock.",
            }],
            visible_attributes={"A.price": 1000},
            latest_observation="A.future_dock = possible",
        )
    )
    serialized = json.dumps(client.responses.calls[0])
    assert "future-dock-secret" not in serialized
    assert "oracle" not in serialized.lower()
    assert "relevance" not in serialized.lower()
    assert "minimal_prompt" not in serialized
    assert response.target_branch_id == "future-dock-secret"
    assert response.usage_source == UsageSource.PROVIDER_REPORTED
    assert response.prompt_tokens == 400
    assert response.completion_tokens == 40
    assert response.estimated_cost_usd > 0


def test_openai_user_adapter_is_condition_blind_and_commit_is_not_an_input_signal():
    live_config, _ = load_config_and_tasks(LIVE_CONFIG_PATH)
    client = _FakeOpenAIClient(
        [{"kind": "query", "query_id": "lap-a-cad", "choice": None}]
    )
    callback = OpenAITextOnlyUserCallback(
        OpenAIResponsesTransport(client, live_config.model, _live_budget())
    )
    action = callback(
        UserModelRequest(
            task_id="laptop-engineering",
            scenario="Choose a laptop.",
            transcript=[{"role": "assistant", "content": "The choice looks stable enough."}],
            available_queries={"lap-a-cad": "Verify CAD support."},
            alternatives={"A": "Laptop A", "B": "Laptop B"},
            seed=23,
        )
    )
    serialized = json.dumps(client.responses.calls[0])
    assert "COMMIT" not in serialized
    assert "StopWise" not in serialized
    assert "condition" not in serialized
    assert "oracle" not in serialized.lower()
    assert "relevance" not in serialized.lower()
    assert action.kind == "query"
    assert action.usage_source == UsageSource.PROVIDER_REPORTED


def test_live_assistant_conditions_differ_only_in_system_instructions():
    live_config, _ = load_config_and_tasks(LIVE_CONFIG_PATH)
    output = {
        "content": "The requested fact is visible.",
        "action": "NO_INTERVENTION",
        "visible_nudge": "",
        "target_query_id": None,
    }
    client = _FakeOpenAIClient([output, output, output])
    callback = OpenAIAssistantCallback(
        OpenAIResponsesTransport(client, live_config.model, _live_budget()),
        query_branch_by_id={},
    )
    conditions = build_conditions(live_config.base_system_prompt, live_config.model)
    for name in ("baseline", "current_prompt", "minimal_prompt"):
        callback(
            AssistantRequest(
                system_prompt=conditions[name].system_prompt,
                model=live_config.model,
                messages=[{"role": "user", "content": "Information query q: Check q."}],
                visible_attributes={"A.q": True},
                latest_observation="A.q = true",
            )
        )
    calls = client.responses.calls
    assert len({call["instructions"] for call in calls}) == 3
    normalized = [{key: value for key, value in call.items() if key != "instructions"} for call in calls]
    assert normalized[0] == normalized[1] == normalized[2]
