"""Rebuild controlled metrics from raw episode logs and task ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .environment import DecisionEnvironment
from .runner import load_tasks, read_jsonl, task_definition_hash
from .schemas import ControlledTask, EpisodeLog, StopWiseAction


@dataclass(frozen=True)
class RunScore:
    experiment_id: str
    task_id: str
    task_version: str
    domain: str
    condition: str
    replicate: int
    seed: int
    optimal_choice: bool
    satisfactory_choice: bool
    constraint_satisfied: bool
    utility: float
    regret: float
    information_turns: int
    new_information_query_count: int
    total_conversation_turns: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    redundant_query_count: int
    redundant_query_rate: float
    action_changing_count: int
    action_changing_rate: float
    reached_information_sufficiency: bool
    queries_before_sufficiency: int
    queries_after_sufficiency: int
    premature_commit_count: int
    commit_count: int
    unsafe_defer_count: int
    defer_count: int
    false_intervention_count: int
    false_intervention_opportunities: int
    missed_intervention_count: int
    intervention_opportunities: int
    focus_count: int
    focus_followed_by_valid_query: int
    no_intervention_true_positive: int
    no_intervention_predictions: int
    no_intervention_expected: int

    @property
    def premature_commit(self) -> bool:
        return self.premature_commit_count > 0


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_episode(task: ControlledTask, log: EpisodeLog) -> RunScore:
    if log.task_id != task.task_id or log.task_version != task.task_version:
        raise ValueError("log task identity/version does not match task definition")
    if log.task_definition_hash != task_definition_hash(task):
        raise ValueError("log task_definition_hash does not match task definition")
    if log.final_choice not in task.alternatives:
        raise ValueError(f"{task.task_id}: unknown final choice {log.final_choice!r}")

    environment = DecisionEnvironment(task)
    initially_sufficient = environment.state().sufficient
    outcomes = [environment.query(query_id) for query_id in log.query_ids]
    for stored, rebuilt in zip(log.query_events, outcomes, strict=True):
        if stored.revealed_value != rebuilt.event.revealed_value:
            raise ValueError(f"{task.task_id}: raw log revealed value does not match task")

    selected_utility = environment.full_utility(log.final_choice)
    feasible_utilities = [
        environment.full_utility(item)
        for item in task.alternatives
        if environment.satisfies_constraints(item)
    ]
    best_utility = max(feasible_utilities) if feasible_utilities else 0.0
    optimal = log.final_choice in environment.optimal_choices()
    constraint_satisfied = environment.satisfies_constraints(log.final_choice)

    first_sufficient = 0 if initially_sufficient else next(
        (index for index, outcome in enumerate(outcomes, start=1) if outcome.after.sufficient), None
    )
    before_sufficient = first_sufficient if first_sufficient is not None else len(outcomes)
    after_sufficient = len(outcomes) - first_sufficient if first_sufficient is not None else 0

    premature_commit_count = commit_count = 0
    unsafe_defer_count = defer_count = 0
    false_count = false_opportunities = 0
    missed_count = intervention_opportunities = 0
    focus_indices: list[int] = []
    ni_tp = ni_predictions = ni_expected = 0

    if log.condition != "baseline":
        if len(log.policy_events) != len(outcomes):
            raise ValueError("every StopWise query response must have one policy event")
        for index, (policy, outcome) in enumerate(zip(log.policy_events, outcomes, strict=True)):
            expected, expected_branch = environment.expected_policy_action(outcome)
            predicted = policy.action
            commit_count += predicted == StopWiseAction.COMMIT
            premature_commit_count += predicted == StopWiseAction.COMMIT and not outcome.after.sufficient
            if predicted == StopWiseAction.DEFER:
                defer_count += 1
                query = task.queries[outcome.event.query_id]
                branch = task.branches.get(policy.target_branch_id or "")
                unsafe_defer_count += (
                    branch is None
                    or branch.required_now
                    or not branch.deferable
                    or policy.target_branch_id != expected_branch
                    or query.branch_id != policy.target_branch_id
                )
            if expected == StopWiseAction.NO_INTERVENTION:
                false_opportunities += 1
                false_count += predicted != StopWiseAction.NO_INTERVENTION
            else:
                intervention_opportunities += 1
                missed_count += predicted == StopWiseAction.NO_INTERVENTION
            predicted_ni = predicted == StopWiseAction.NO_INTERVENTION
            expected_ni = expected == StopWiseAction.NO_INTERVENTION
            ni_predictions += predicted_ni
            ni_expected += expected_ni
            ni_tp += predicted_ni and expected_ni
            if predicted == StopWiseAction.FOCUS:
                focus_indices.append(index)

    focus_followed = sum(
        any(later.event.action_changing for later in outcomes[index + 1 :])
        for index in focus_indices
    )
    redundant = sum(outcome.event.redundant for outcome in outcomes)
    action_changing = sum(outcome.event.action_changing for outcome in outcomes)
    new_information = sum(not outcome.event.repeated for outcome in outcomes)
    conversation_turns = sum(turn.role in {"user", "assistant"} for turn in log.conversation_turns)

    return RunScore(
        experiment_id=log.experiment_id,
        task_id=task.task_id,
        task_version=task.task_version,
        domain=task.domain,
        condition=log.condition,
        replicate=log.replicate,
        seed=log.seed,
        optimal_choice=optimal,
        satisfactory_choice=constraint_satisfied and selected_utility >= task.satisfactory_utility,
        constraint_satisfied=constraint_satisfied,
        utility=selected_utility,
        regret=max(0.0, best_utility - (selected_utility if constraint_satisfied else 0.0)),
        information_turns=len(outcomes),
        new_information_query_count=new_information,
        total_conversation_turns=conversation_turns,
        prompt_tokens=log.token_usage.prompt_tokens,
        completion_tokens=log.token_usage.completion_tokens,
        total_tokens=log.token_usage.total_tokens,
        redundant_query_count=redundant,
        redundant_query_rate=redundant / len(outcomes) if outcomes else 0.0,
        action_changing_count=action_changing,
        action_changing_rate=action_changing / new_information if new_information else 0.0,
        reached_information_sufficiency=first_sufficient is not None,
        queries_before_sufficiency=before_sufficient,
        queries_after_sufficiency=after_sufficient,
        premature_commit_count=premature_commit_count,
        commit_count=commit_count,
        unsafe_defer_count=unsafe_defer_count,
        defer_count=defer_count,
        false_intervention_count=false_count,
        false_intervention_opportunities=false_opportunities,
        missed_intervention_count=missed_count,
        intervention_opportunities=intervention_opportunities,
        focus_count=len(focus_indices),
        focus_followed_by_valid_query=focus_followed,
        no_intervention_true_positive=ni_tp,
        no_intervention_predictions=ni_predictions,
        no_intervention_expected=ni_expected,
    )


_MEAN_METRICS = (
    "optimal_choice", "satisfactory_choice", "constraint_satisfied", "utility", "regret",
    "information_turns", "new_information_query_count", "total_conversation_turns", "prompt_tokens", "completion_tokens",
    "total_tokens", "redundant_query_count", "redundant_query_rate", "action_changing_rate",
    "queries_before_sufficiency", "queries_after_sufficiency",
)


def aggregate(scores: Iterable[RunScore]) -> dict[str, dict[str, Any]]:
    scores = list(scores)
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted({score.condition for score in scores}):
        group = [score for score in scores if score.condition == condition]
        metrics: dict[str, Any] = {"runs": len(group)}
        for name in _MEAN_METRICS:
            values = [float(getattr(score, name)) for score in group]
            metrics[name] = {"mean": mean(values), "median": median(values), "distribution": values}
        sums = lambda field: sum(getattr(score, field) for score in group)
        metrics["premature_commit_rate"] = _rate(sums("premature_commit_count"), sums("commit_count"))
        metrics["unsafe_defer_rate"] = _rate(sums("unsafe_defer_count"), sums("defer_count"))
        metrics["false_intervention_rate"] = _rate(
            sums("false_intervention_count"), sums("false_intervention_opportunities")
        )
        metrics["missed_intervention_rate"] = _rate(
            sums("missed_intervention_count"), sums("intervention_opportunities")
        )
        metrics["focus_valid_query_continuation_rate"] = _rate(
            sums("focus_followed_by_valid_query"), sums("focus_count")
        )
        metrics["no_intervention_precision"] = _rate(
            sums("no_intervention_true_positive"), sums("no_intervention_predictions")
        )
        metrics["no_intervention_recall"] = _rate(
            sums("no_intervention_true_positive"), sums("no_intervention_expected")
        )
        result[condition] = metrics
    return result


def score_logs(tasks_path: Path, logs_path: Path) -> tuple[list[RunScore], dict[str, dict[str, Any]]]:
    tasks = {task.task_id: task for task in load_tasks(tasks_path)}
    logs = read_jsonl(logs_path)
    scores = []
    for log in logs:
        try:
            task = tasks[log.task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task ID: {log.task_id!r}") from exc
        scores.append(score_episode(task, log))
    return scores, aggregate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores, summary = score_logs(args.tasks, args.logs)
    payload = {
        "status": "scored infrastructure; not controlled effectiveness evidence",
        "runs": [asdict(score) for score in scores],
        "summary": summary,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
