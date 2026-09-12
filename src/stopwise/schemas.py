"""Validated output types for StopWise."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class Action(StrEnum):
    NO_INTERVENTION = "NO_INTERVENTION"
    FOCUS = "FOCUS"
    COMMIT = "COMMIT"
    DEFER = "DEFER"


class Level(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Signal(StrEnum):
    SEARCH_SPACE_EXPANSION = "search_space_expansion"
    REDUNDANT_VERIFICATION = "redundant_verification"
    PSEUDO_PRECISION = "pseudo_precision"
    SECONDARY_OPTIMIZATION = "secondary_optimization"
    CONTINGENCY_BRANCHING = "contingency_branching"
    GOAL_DRIFT = "goal_drift"


class StopWiseResult(BaseModel):
    """A structured policy decision and optional metacognitive intervention."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    goal: str
    current_decision: str
    stakes: Level
    reversibility: Level
    primary_criteria: list[str]
    resolved_primary_criteria: list[str]
    unresolved_action_changing_information: bool
    decision_stability: Level
    signals: list[Signal]
    action: Action
    reason: str
    message: str

    @model_validator(mode="after")
    def enforce_policy_invariants(self) -> "StopWiseResult":
        unknown_resolved = set(self.resolved_primary_criteria) - set(
            self.primary_criteria
        )
        if unknown_resolved:
            raise ValueError(
                "resolved_primary_criteria must be a subset of primary_criteria"
            )
        if self.action == Action.COMMIT and self.unresolved_action_changing_information:
            raise ValueError(
                "COMMIT is invalid while action-changing information remains"
            )
        if self.action != Action.NO_INTERVENTION and not self.message.strip():
            raise ValueError("intervention actions require a message")
        if self.action == Action.NO_INTERVENTION and self.message:
            raise ValueError("NO_INTERVENTION requires an empty message")
        return self
