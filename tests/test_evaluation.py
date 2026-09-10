import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
EVALUATE = runpy.run_path(str(ROOT / "eval/evaluate.py"))


def test_handcrafted_cases_and_metric_definitions():
    cases = json.loads((ROOT / "eval/cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 10

    predictions = [
        {"id": case["id"], "action": case["expected_actions"][0]} for case in cases
    ]
    metrics = EVALUATE["score"](cases, predictions)

    assert metrics.action_accuracy == 1.0
    assert metrics.premature_intervention_rate == 0.0
    assert metrics.missed_intervention_rate == 0.0


def test_premature_and_missed_interventions_are_counted_separately():
    cases = [
        {
            "id": "unresolved",
            "expected_actions": ["NO_INTERVENTION"],
            "action_changing_information_remains": True,
        },
        {
            "id": "stable",
            "expected_actions": ["COMMIT"],
            "action_changing_information_remains": False,
        },
    ]
    predictions = [
        {"id": "unresolved", "action": "FOCUS"},
        {"id": "stable", "action": "NO_INTERVENTION"},
    ]

    metrics = EVALUATE["score"](cases, predictions)
    assert metrics.premature_intervention_rate == 1.0
    assert metrics.missed_intervention_rate == 1.0
