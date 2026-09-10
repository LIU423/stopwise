"""Run the handcrafted StopWise intervention-classification evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stopwise import Action, StopWise


@dataclass(frozen=True)
class Metrics:
    total: int
    action_accuracy: float
    premature_intervention_rate: float
    missed_intervention_rate: float


def score(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> Metrics:
    by_id = {prediction["id"]: prediction for prediction in predictions}
    if set(by_id) != {case["id"] for case in cases}:
        raise ValueError("prediction IDs must exactly match case IDs")

    correct = 0
    premature = 0
    premature_opportunities = 0
    missed = 0
    intervention_opportunities = 0

    for case in cases:
        predicted = by_id[case["id"]]["action"]
        expected = case["expected_actions"]
        correct += predicted in expected

        if case["action_changing_information_remains"]:
            premature_opportunities += 1
            premature += predicted != Action.NO_INTERVENTION.value

        if Action.NO_INTERVENTION.value not in expected:
            intervention_opportunities += 1
            missed += predicted == Action.NO_INTERVENTION.value

    return Metrics(
        total=len(cases),
        action_accuracy=correct / len(cases),
        premature_intervention_rate=(
            premature / premature_opportunities if premature_opportunities else 0.0
        ),
        missed_intervention_rate=(
            missed / intervention_opportunities if intervention_opportunities else 0.0
        ),
    )


def run(
    cases: list[dict[str, Any]],
    model: str,
    *,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    transport: str = "auto",
) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the evaluation dependency: pip install -e '.[openai]'") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Set the {api_key_env} environment variable")

    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    analyzer = StopWise(
        client=OpenAI(**client_options),
        model=model,
        transport=transport,
    )
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
    parser.add_argument(
        "--transport",
        choices=("auto", "responses", "chat_completions"),
        default="auto",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("cases.json")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Score an existing JSON prediction list instead of calling a model.",
    )
    parser.add_argument("--save", type=Path, help="Save live predictions as JSON.")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    else:
        predictions = run(
            cases,
            args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            transport=args.transport,
        )

    if args.save:
        args.save.write_text(
            json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    metrics = score(cases, predictions)
    print(f"\nCases: {metrics.total}")
    print(f"Action accuracy: {metrics.action_accuracy:.1%}")
    print(f"Premature intervention rate: {metrics.premature_intervention_rate:.1%}")
    print(f"Missed intervention rate: {metrics.missed_intervention_rate:.1%}")


if __name__ == "__main__":
    main()
