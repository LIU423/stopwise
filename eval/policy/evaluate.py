"""Run and score StopWise policy regression cases."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from stopwise import Action, StopWise


@dataclass(frozen=True)
class Metrics:
    total: int
    action_accuracy: float
    false_intervention_rate: float
    premature_commit_rate: float
    unsafe_defer_rate: float
    missed_intervention_rate: float
    no_intervention_precision: float
    no_intervention_recall: float
    signal_precision: float
    signal_recall: float
    signal_f1: float


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL policy cases, ignoring blank lines."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> Metrics:
    """Score policy decisions using action-specific safety errors."""

    by_id = {prediction["id"]: prediction for prediction in predictions}
    case_ids = {case["id"] for case in cases}
    if len(by_id) != len(predictions) or set(by_id) != case_ids:
        raise ValueError("prediction IDs must be unique and exactly match case IDs")

    correct = false_interventions = false_intervention_opportunities = 0
    premature_commits = premature_commit_opportunities = 0
    unsafe_defers = unsafe_defer_opportunities = 0
    missed = intervention_opportunities = 0
    ni_true_positive = ni_predictions = ni_expected = 0
    signal_true_positive = signal_predictions = signal_expected = 0

    for case in cases:
        prediction = by_id[case["id"]]
        predicted = prediction["action"]
        expected = set(case["expected_actions"])
        correct += predicted in expected

        prefers_silence = expected == {Action.NO_INTERVENTION.value}
        if prefers_silence:
            false_intervention_opportunities += 1
            false_interventions += predicted != Action.NO_INTERVENTION.value

        if case["action_changing_information_remains"]:
            premature_commit_opportunities += 1
            premature_commits += predicted == Action.COMMIT.value

        if case.get("unsafe_to_defer", False):
            unsafe_defer_opportunities += 1
            unsafe_defers += predicted == Action.DEFER.value

        if Action.NO_INTERVENTION.value not in expected:
            intervention_opportunities += 1
            missed += predicted == Action.NO_INTERVENTION.value

        predicted_ni = predicted == Action.NO_INTERVENTION.value
        expected_ni = Action.NO_INTERVENTION.value in expected
        ni_predictions += predicted_ni
        ni_expected += expected_ni
        ni_true_positive += predicted_ni and expected_ni

        predicted_signals = set(prediction.get("signals", []))
        expected_signals = set(case.get("expected_signals", []))
        signal_true_positive += len(predicted_signals & expected_signals)
        signal_predictions += len(predicted_signals)
        signal_expected += len(expected_signals)

    signal_precision = _rate(signal_true_positive, signal_predictions)
    signal_recall = _rate(signal_true_positive, signal_expected)
    signal_f1 = _rate(2 * signal_precision * signal_recall, signal_precision + signal_recall)

    return Metrics(
        total=len(cases),
        action_accuracy=_rate(correct, len(cases)),
        false_intervention_rate=_rate(false_interventions, false_intervention_opportunities),
        premature_commit_rate=_rate(premature_commits, premature_commit_opportunities),
        unsafe_defer_rate=_rate(unsafe_defers, unsafe_defer_opportunities),
        missed_intervention_rate=_rate(missed, intervention_opportunities),
        no_intervention_precision=_rate(ni_true_positive, ni_predictions),
        no_intervention_recall=_rate(ni_true_positive, ni_expected),
        signal_precision=signal_precision,
        signal_recall=signal_recall,
        signal_f1=signal_f1,
    )


def run(
    cases: Iterable[dict[str, Any]],
    model: str,
    *,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    transport: str = "auto",
) -> list[dict[str, Any]]:
    """Collect live predictions from one OpenAI-compatible backend."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the evaluation dependency: pip install -e '.[openai]'") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Set the {api_key_env} environment variable")

    client_options: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    analyzer = StopWise(client=OpenAI(**client_options), model=model, transport=transport)

    predictions = []
    for case in cases:
        result = analyzer.analyze(case["messages"])
        predictions.append(
            {
                "id": case["id"],
                "action": result.action.value,
                "signals": [signal.value for signal in result.signals],
                "result": result.model_dump(mode="json"),
            }
        )
        print(f"{case['id']}: {result.action.value}")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--transport", choices=("auto", "responses", "chat_completions"), default="auto")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--predictions", type=Path, help="Score an existing prediction list without model calls.")
    parser.add_argument("--save", type=Path, help="Save live predictions as JSON.")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    predictions = (
        json.loads(args.predictions.read_text(encoding="utf-8"))
        if args.predictions
        else run(cases, args.model, base_url=args.base_url, api_key_env=args.api_key_env, transport=args.transport)
    )
    if args.save:
        args.save.write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = score(cases, predictions)
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
