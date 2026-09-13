import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.controlled.conditions import (
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
    run_paired_experiment,
    write_jsonl,
)
from eval.controlled.schemas import ExperimentConfig, ModelConfig, StopWiseAction
from eval.controlled.score import score_episode, score_logs
from eval.controlled.simulators import DeterministicUserSimulator
from eval.controlled.statistics import paired_summary, validate_paired_logs


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "eval/controlled/tasks.json"
CONFIG_PATH = ROOT / "eval/controlled/example_config.json"


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
        assert pair["baseline"].assistant_model_config == pair["stopwise"].assistant_model_config
        assert pair["baseline"].seed == pair["stopwise"].seed
        assert pair["baseline"].base_prompt_hash == pair["stopwise"].base_prompt_hash
        assert pair["baseline"].prompt_version_hash != pair["stopwise"].prompt_version_hash


def test_redundancy_acr_utility_regret_and_constraint_scoring():
    task = tasks()[0]
    logs = paired_logs()[0:2]
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
        condition=conditions["stopwise"],
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
    log = next(item for item in paired_logs() if item.condition == "stopwise")
    assert log.policy_events[0].action == StopWiseAction.FOCUS
    assert len(log.query_events) > 1
    result = score_episode(task, log)
    assert result.focus_count > 0
    assert result.focus_followed_by_valid_query > 0
    assert result.focus_followed_by_valid_query <= result.focus_count


def test_defer_only_skips_the_target_branch_and_main_search_continues():
    log = next(item for item in paired_logs() if item.condition == "stopwise")
    assert "lap-future-dock" in log.query_ids
    assert "lap-future-countries" not in log.query_ids
    assert "lap-b-cad" in log.query_ids
    defer = next(item for item in log.policy_events if item.action == StopWiseAction.DEFER)
    assert defer.target_branch_id == "future-dock"


def test_no_intervention_has_no_visible_nudge():
    log = next(item for item in paired_logs() if item.condition == "stopwise")
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
        if request.last_query_branch_id == "access-now":
            return AssistantResponse(
                content="Here is the access result.",
                action=StopWiseAction.DEFER,
                visible_nudge="Defer accessibility.",
                target_branch_id="access-now",
            )
        return self.default(request)


def test_unsafe_defer_is_detected_for_a_required_now_branch():
    task = tasks()[1]
    harness = build_conditions(config().base_system_prompt, config().model)["stopwise"]
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
    assert len(scores) == 4
    assert set(summary) == {"baseline", "stopwise"}
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
    assert len(summary["numeric"]["redundant_query_count"]["bootstrap_ci"]) == 2
    assert "exact_mcnemar_p" in summary["binary"]["optimal_choice"]
    assert set(summary["by_domain"]) == {"laptop selection", "hotel selection"}
    assert "Descriptive" in summary["inference_note"]
