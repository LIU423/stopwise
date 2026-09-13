import json
from pathlib import Path
import runpy

from eval.controlled.conditions import DeterministicAssistantCallback
from eval.controlled.runner import load_tasks, run_paired_experiment
from eval.controlled.schemas import ExperimentConfig
from eval.controlled.score import score_episode


ROOT = Path(__file__).resolve().parents[1]
POLICY = runpy.run_path(str(ROOT / "eval/policy/evaluate.py"))


def test_contrastive_cases_and_perfect_metrics():
    cases = POLICY["load_cases"](ROOT / "eval/policy/cases.jsonl")
    assert len(cases) >= 20
    assert len({case.get("pair_id") for case in cases if case.get("pair_id")}) >= 8

    predictions = [
        {
            "id": case["id"],
            "action": case["expected_actions"][0],
            "signals": case["expected_signals"],
        }
        for case in cases
    ]
    metrics = POLICY["score"](cases, predictions)

    assert metrics.action_accuracy == 1.0
    assert metrics.false_intervention_rate == 0.0
    assert metrics.premature_commit_rate == 0.0
    assert metrics.unsafe_defer_rate == 0.0
    assert metrics.missed_intervention_rate == 0.0
    assert metrics.no_intervention_precision == 1.0
    assert metrics.no_intervention_recall == 1.0
    assert metrics.signal_f1 == 1.0


def test_focus_with_unresolved_information_is_not_premature_commit():
    cases = [
        {
            "id": "focus",
            "expected_actions": ["FOCUS"],
            "expected_signals": ["secondary_optimization"],
            "action_changing_information_remains": True,
            "unsafe_to_defer": False,
        }
    ]
    predictions = [
        {"id": "focus", "action": "FOCUS", "signals": ["secondary_optimization"]}
    ]

    metrics = POLICY["score"](cases, predictions)
    assert metrics.action_accuracy == 1.0
    assert metrics.premature_commit_rate == 0.0
    assert metrics.false_intervention_rate == 0.0


def test_action_specific_errors_and_silence_metrics():
    cases = [
        {
            "id": "unresolved",
            "expected_actions": ["NO_INTERVENTION"],
            "expected_signals": [],
            "action_changing_information_remains": True,
            "unsafe_to_defer": True,
        },
        {
            "id": "stable",
            "expected_actions": ["COMMIT"],
            "expected_signals": ["redundant_verification"],
            "action_changing_information_remains": False,
            "unsafe_to_defer": False,
        },
    ]
    predictions = [
        {"id": "unresolved", "action": "DEFER", "signals": []},
        {"id": "stable", "action": "NO_INTERVENTION", "signals": []},
    ]

    metrics = POLICY["score"](cases, predictions)
    assert metrics.false_intervention_rate == 1.0
    assert metrics.premature_commit_rate == 0.0
    assert metrics.unsafe_defer_rate == 1.0
    assert metrics.missed_intervention_rate == 1.0
    assert metrics.no_intervention_precision == 0.0
    assert metrics.no_intervention_recall == 0.0


def test_premature_commit_is_distinct_from_focus():
    cases = [
        {
            "id": "unresolved",
            "expected_actions": ["FOCUS"],
            "expected_signals": [],
            "action_changing_information_remains": True,
            "unsafe_to_defer": False,
        }
    ]
    predictions = [{"id": "unresolved", "action": "COMMIT", "signals": []}]
    assert POLICY["score"](cases, predictions).premature_commit_rate == 1.0


def test_controlled_scorer_tracks_quality_cost_and_acr():
    task = load_tasks(ROOT / "eval/controlled/tasks.json")[0]
    config = ExperimentConfig.model_validate_json(
        (ROOT / "eval/controlled/example_config.json").read_text(encoding="utf-8")
    ).model_copy(update={"replicates": 1})
    logs = run_paired_experiment(config, [task], DeterministicAssistantCallback())
    log = next(item for item in logs if item.condition == "stopwise")
    result = score_episode(task, log)
    assert result.optimal_choice is True
    assert result.constraint_satisfied is True
    assert result.premature_commit_count == 0
    assert result.redundant_query_count > 0
    assert 0 < result.action_changing_rate < 1
    assert result.total_tokens > 0


def test_deployment_assets_exist_and_are_nonempty():
    paths = [
        ROOT / "prompts/analyzer_system.md",
        ROOT / "prompts/custom_instruction.md",
        ROOT / "prompts/custom_instruction_compact.md",
        ROOT / "skills/stopwise/SKILL.md",
        ROOT / "references.bib",
    ]
    assert all(path.read_text(encoding="utf-8").strip() for path in paths)
