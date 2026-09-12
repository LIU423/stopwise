"""Score controlled StopWise experiment logs against task ground truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class RunScore:
    task_id: str
    condition: str
    optimal_choice: bool
    constraint_satisfied: bool
    utility: float
    regret: float
    information_turns: int
    token_usage: int
    redundant_queries: int
    premature_commit: bool
    action_changing_rate: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    return {task["task_id"]: task for task in tasks}


def score_run(task: dict[str, Any], run: dict[str, Any]) -> RunScore:
    query_ids = run["query_ids"]
    catalog = task["query_catalog"]
    unknown = set(query_ids) - set(catalog)
    if unknown:
        raise ValueError(f"{task['task_id']}: unknown query IDs: {sorted(unknown)}")

    choice = run["chosen_alternative"]
    try:
        selected = task["alternatives"][choice]
    except KeyError as exc:
        raise ValueError(f"{task['task_id']}: unknown alternative {choice!r}") from exc

    required = set(task["required_query_ids"])
    seen: set[str] = set()
    sufficient_at: int | None = None
    for index, query_id in enumerate(query_ids, start=1):
        seen.add(query_id)
        if sufficient_at is None and required <= seen:
            sufficient_at = index

    redundant = 0
    if sufficient_at is not None:
        redundant = sum(
            not catalog[query_id]["action_changing"]
            for query_id in query_ids[sufficient_at:]
        )

    action_changing = sum(catalog[query_id]["action_changing"] for query_id in query_ids)
    best_utility = task["alternatives"][task["optimal_choice"]]["utility"]
    utility = float(selected["utility"])

    return RunScore(
        task_id=task["task_id"],
        condition=run["condition"],
        optimal_choice=choice == task["optimal_choice"],
        constraint_satisfied=bool(selected["satisfies_constraints"]),
        utility=utility,
        regret=max(0.0, best_utility - utility),
        information_turns=len(query_ids),
        token_usage=int(run.get("prompt_tokens", 0)) + int(run.get("completion_tokens", 0)),
        redundant_queries=redundant,
        premature_commit=not required <= set(query_ids),
        action_changing_rate=action_changing / len(query_ids) if query_ids else 0.0,
    )


def aggregate(scores: list[RunScore]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for condition in sorted({score.condition for score in scores}):
        group = [score for score in scores if score.condition == condition]
        result[condition] = {
            "runs": float(len(group)),
            "optimal_choice_accuracy": mean(score.optimal_choice for score in group),
            "constraint_satisfaction_rate": mean(score.constraint_satisfied for score in group),
            "mean_utility": mean(score.utility for score in group),
            "mean_regret": mean(score.regret for score in group),
            "mean_information_turns": mean(score.information_turns for score in group),
            "mean_token_usage": mean(score.token_usage for score in group),
            "mean_redundant_queries": mean(score.redundant_queries for score in group),
            "premature_commit_rate": mean(score.premature_commit for score in group),
            "action_changing_rate": mean(score.action_changing_rate for score in group),
        }
    return result


def score_logs(tasks_path: Path, logs_path: Path) -> tuple[list[RunScore], dict[str, dict[str, float]]]:
    tasks = load_tasks(tasks_path)
    runs = load_jsonl(logs_path)
    scores = []
    for run in runs:
        try:
            task = tasks[run["task_id"]]
        except KeyError as exc:
            raise ValueError(f"unknown task ID: {run.get('task_id')!r}") from exc
        scores.append(score_run(task, run))
    return scores, aggregate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks.json"))
    args = parser.parse_args()
    scores, summary = score_logs(args.tasks, args.logs)
    print(json.dumps({"runs": [asdict(score) for score in scores], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
