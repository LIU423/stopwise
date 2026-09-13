"""Interactive decision environment and dynamic information-value oracle."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .schemas import ControlledTask, QueryEvent, QueryRelevance, StopWiseAction


class OracleState(BaseModel):
    """Decision state computed only from attributes revealed so far."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    possible_feasible: tuple[str, ...]
    confirmed_feasible: tuple[str, ...]
    constraint_status: dict[str, Literal["unknown", "satisfied", "violated"]]
    possible_optimal: tuple[str, ...]
    recommended: tuple[str, ...]
    ranking: tuple[str, ...]
    expected_utility: dict[str, float]
    utility_bounds: dict[str, tuple[float, float]]
    expected_margin: float
    sufficient: bool


@dataclass(frozen=True)
class QueryOutcome:
    event: QueryEvent
    before: OracleState
    after: OracleState


class DecisionEnvironment:
    """Reveal one task attribute at a time and recompute the choice set."""

    def __init__(self, task: ControlledTask) -> None:
        self.task = task
        self.revealed: set[str] = set(task.initially_visible)
        self.query_history: list[str] = []

    def clone_fresh(self) -> "DecisionEnvironment":
        return DecisionEnvironment(self.task)

    def visible_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for ref in sorted(self.revealed):
            alternative_id, attribute_id = ref.split(".", 1)
            values[ref] = self.task.alternatives[alternative_id].attributes[attribute_id]
        return values

    def full_utility(self, alternative_id: str) -> float:
        alternative = self.task.alternatives[alternative_id]
        total_weight = sum(item.weight for item in self.task.criteria.values())
        weighted = sum(
            criterion.weight * criterion.score(alternative.attributes[criterion_id])
            for criterion_id, criterion in self.task.criteria.items()
        )
        return weighted / total_weight

    def satisfies_constraints(self, alternative_id: str) -> bool:
        alternative = self.task.alternatives[alternative_id]
        return all(
            criterion.hard_constraint is None
            or criterion.hard_constraint.accepts(alternative.attributes[criterion_id])
            for criterion_id, criterion in self.task.criteria.items()
        )

    def optimal_choices(self) -> tuple[str, ...]:
        feasible = [item for item in self.task.alternatives if self.satisfies_constraints(item)]
        if not feasible:
            return ()
        best = max(self.full_utility(item) for item in feasible)
        return tuple(
            sorted(item for item in feasible if isclose(self.full_utility(item), best, abs_tol=1e-12))
        )

    def state(self) -> OracleState:
        total_weight = sum(item.weight for item in self.task.criteria.values())
        bounds: dict[str, tuple[float, float]] = {}
        expected: dict[str, float] = {}
        statuses: dict[str, Literal["unknown", "satisfied", "violated"]] = {}

        for alternative_id, alternative in self.task.alternatives.items():
            lower = upper = midpoint = 0.0
            missing_constraint = False
            violated = False
            for criterion_id, criterion in self.task.criteria.items():
                ref = f"{alternative_id}.{criterion_id}"
                if ref in self.revealed:
                    value = alternative.attributes[criterion_id]
                    component = criterion.weight * criterion.score(value)
                    lower += component
                    upper += component
                    midpoint += component
                    if criterion.hard_constraint and not criterion.hard_constraint.accepts(value):
                        violated = True
                else:
                    upper += criterion.weight
                    midpoint += criterion.weight * 0.5
                    if criterion.hard_constraint:
                        missing_constraint = True
            bounds[alternative_id] = (lower / total_weight, upper / total_weight)
            expected[alternative_id] = midpoint / total_weight
            statuses[alternative_id] = (
                "violated" if violated else "unknown" if missing_constraint else "satisfied"
            )

        possible_feasible = tuple(sorted(key for key, value in statuses.items() if value != "violated"))
        confirmed_feasible = tuple(sorted(key for key, value in statuses.items() if value == "satisfied"))
        if possible_feasible:
            highest_lower = max(bounds[item][0] for item in possible_feasible)
            possible_optimal = tuple(
                sorted(
                    item
                    for item in possible_feasible
                    if bounds[item][1] + 1e-12 >= highest_lower
                )
            )
            best_expected = max(expected[item] for item in possible_feasible)
            recommended = tuple(
                sorted(
                    item
                    for item in possible_feasible
                    if isclose(expected[item], best_expected, abs_tol=1e-12)
                )
            )
            ranking = tuple(sorted(possible_feasible, key=lambda item: (-expected[item], item)))
            values = sorted((expected[item] for item in possible_feasible), reverse=True)
            margin = values[0] - values[1] if len(values) > 1 else values[0]
        else:
            possible_optimal = recommended = ranking = ()
            margin = 0.0

        sufficient = False
        if len(possible_optimal) == 1:
            winner = possible_optimal[0]
            competitors = [item for item in possible_feasible if item != winner]
            robust_margin = min(
                (bounds[winner][0] - bounds[item][1] for item in competitors),
                default=bounds[winner][0],
            )
            sufficient = winner in confirmed_feasible and robust_margin >= self.task.decision_margin

        return OracleState(
            possible_feasible=possible_feasible,
            confirmed_feasible=confirmed_feasible,
            constraint_status=statuses,
            possible_optimal=possible_optimal,
            recommended=recommended,
            ranking=ranking,
            expected_utility={key: round(value, 12) for key, value in expected.items()},
            utility_bounds={
                key: (round(value[0], 12), round(value[1], 12)) for key, value in bounds.items()
            },
            expected_margin=round(margin, 12),
            sufficient=sufficient,
        )

    def query(self, query_id: str) -> QueryOutcome:
        try:
            query = self.task.queries[query_id]
        except KeyError as exc:
            raise ValueError(f"{self.task.task_id}: unknown query ID {query_id!r}") from exc

        before = self.state()
        ref = f"{query.alternative_id}.{query.attribute_id}"
        repeated = ref in self.revealed
        self.revealed.add(ref)
        self.query_history.append(query_id)
        after = self.state()

        effects: list[str] = []
        if before.recommended != after.recommended:
            effects.append("recommended_choice")
        if before.possible_optimal != after.possible_optimal:
            effects.append("possible_optimal_set")
        if before.possible_feasible != after.possible_feasible:
            effects.append("feasible_set")
        if before.constraint_status != after.constraint_status:
            effects.append("constraint_status")
        if before.ranking != after.ranking:
            effects.append("utility_ranking")
        before_margin = before.expected_margin >= self.task.decision_margin
        after_margin = after.expected_margin >= self.task.decision_margin
        if before_margin != after_margin:
            effects.append("actionable_utility_margin")
        if not before.sufficient and after.sufficient:
            effects.append("information_sufficiency")

        action_changing = bool(effects) and not repeated
        redundant = self._is_redundant(query.relevance, query.branch_id, repeated, before, action_changing)
        event = QueryEvent(
            sequence=len(self.query_history),
            query_id=query_id,
            alternative_id=query.alternative_id,
            attribute_id=query.attribute_id,
            revealed_value=self.task.alternatives[query.alternative_id].attributes[query.attribute_id],
            relevance=query.relevance,
            branch_id=query.branch_id,
            repeated=repeated,
            action_changing=action_changing,
            effects=effects,
            redundant=redundant,
            sufficient_before=before.sufficient,
            sufficient_after=after.sufficient,
        )
        return QueryOutcome(event=event, before=before, after=after)

    def _is_redundant(
        self,
        relevance: QueryRelevance,
        branch_id: str | None,
        repeated: bool,
        before: OracleState,
        action_changing: bool,
    ) -> bool:
        if repeated:
            return True
        if action_changing:
            return False
        if before.sufficient:
            return True
        if relevance == QueryRelevance.IRRELEVANT:
            return True
        if relevance == QueryRelevance.DEFERABLE and branch_id:
            branch = self.task.branches[branch_id]
            return branch.deferable and not branch.required_now
        if relevance == QueryRelevance.SECONDARY:
            return True
        return False

    def expected_policy_action(self, outcome: QueryOutcome) -> tuple[StopWiseAction, str | None]:
        """Independent task-state oracle for evaluating policy events."""

        query = self.task.queries[outcome.event.query_id]
        if query.branch_id is not None:
            branch = self.task.branches[query.branch_id]
            if branch.required_now or not branch.deferable:
                return StopWiseAction.NO_INTERVENTION, None
            return StopWiseAction.DEFER, query.branch_id
        if query.relevance == QueryRelevance.ROBUSTNESS:
            return StopWiseAction.NO_INTERVENTION, None
        if outcome.event.repeated:
            return (
                StopWiseAction.COMMIT if outcome.after.sufficient else StopWiseAction.FOCUS,
                None,
            )
        if query.relevance in {QueryRelevance.IRRELEVANT, QueryRelevance.SECONDARY}:
            return (
                StopWiseAction.COMMIT if outcome.after.sufficient else StopWiseAction.FOCUS,
                None,
            )
        return StopWiseAction.NO_INTERVENTION, None
