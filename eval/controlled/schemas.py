"""Validated schemas for controlled end-to-end StopWise experiments."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Scalar = bool | int | float | str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StopWiseAction(StrEnum):
    NO_INTERVENTION = "NO_INTERVENTION"
    FOCUS = "FOCUS"
    COMMIT = "COMMIT"
    DEFER = "DEFER"


class QueryRelevance(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ROBUSTNESS = "robustness"
    IRRELEVANT = "irrelevant"
    DEFERABLE = "deferable"


class ConstraintSpec(StrictModel):
    op: Literal["le", "lt", "ge", "gt", "eq", "in"]
    value: Scalar | list[Scalar]

    def accepts(self, observed: Scalar) -> bool:
        if self.op == "le":
            return float(observed) <= float(self.value)
        if self.op == "lt":
            return float(observed) < float(self.value)
        if self.op == "ge":
            return float(observed) >= float(self.value)
        if self.op == "gt":
            return float(observed) > float(self.value)
        if self.op == "eq":
            return observed == self.value
        allowed = self.value if isinstance(self.value, list) else [self.value]
        return observed in allowed


class CriterionSpec(StrictModel):
    label: str
    weight: float = Field(ge=0)
    primary: bool = True
    value_type: Literal["number", "boolean", "categorical"]
    direction: Literal["min", "max"] | None = None
    min_value: float | None = None
    max_value: float | None = None
    desired_value: Scalar | None = None
    value_scores: dict[str, float] = Field(default_factory=dict)
    hard_constraint: ConstraintSpec | None = None

    @model_validator(mode="after")
    def validate_scoring_rule(self) -> "CriterionSpec":
        if self.value_type == "number":
            if self.direction is None or self.min_value is None or self.max_value is None:
                raise ValueError("numeric criteria require direction, min_value, and max_value")
            if self.max_value <= self.min_value:
                raise ValueError("max_value must exceed min_value")
        elif self.value_type == "boolean" and not isinstance(self.desired_value, bool):
            raise ValueError("boolean criteria require a boolean desired_value")
        elif self.value_type == "categorical" and not self.value_scores:
            raise ValueError("categorical criteria require value_scores")
        return self

    def score(self, value: Scalar) -> float:
        if self.value_type == "number":
            assert self.min_value is not None and self.max_value is not None
            scaled = (float(value) - self.min_value) / (self.max_value - self.min_value)
            scaled = min(1.0, max(0.0, scaled))
            return scaled if self.direction == "max" else 1.0 - scaled
        if self.value_type == "boolean":
            return 1.0 if value == self.desired_value else 0.0
        return min(1.0, max(0.0, float(self.value_scores.get(str(value), 0.0))))


class AlternativeSpec(StrictModel):
    label: str
    attributes: dict[str, Scalar]


class QuerySpec(StrictModel):
    alternative_id: str
    attribute_id: str
    relevance: QueryRelevance
    branch_id: str | None = None
    rationale: str
    measurement_basis: str | None = None


class BranchSpec(StrictModel):
    label: str
    deferable: bool
    required_now: bool = False


class OracleAnnotation(StrictModel):
    source: Literal["dynamic", "manual", "hybrid"] = "dynamic"
    version: str
    reviewed_by: list[str] = Field(default_factory=list)
    notes: str = ""
    manual_action_changing: dict[str, bool] = Field(default_factory=dict)


class ControlledTask(StrictModel):
    task_id: str
    task_version: str
    domain: str
    title: str
    scenario: str
    primary_criteria: list[str]
    secondary_criteria: list[str]
    criteria: dict[str, CriterionSpec]
    alternatives: dict[str, AlternativeSpec]
    initially_visible: list[str]
    queries: dict[str, QuerySpec]
    branches: dict[str, BranchSpec] = Field(default_factory=dict)
    simulator_query_plan: list[str]
    decision_margin: float = Field(default=0.05, ge=0, le=1)
    satisfactory_utility: float = Field(default=0.7, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)
    oracle_annotation: OracleAnnotation

    @model_validator(mode="after")
    def validate_references(self) -> "ControlledTask":
        if len(self.alternatives) < 2:
            raise ValueError("a controlled task needs at least two alternatives")
        if not self.criteria or sum(item.weight for item in self.criteria.values()) <= 0:
            raise ValueError("criteria must have positive total weight")
        if set(self.primary_criteria) | set(self.secondary_criteria) != set(self.criteria):
            raise ValueError("primary_criteria and secondary_criteria must partition criteria")
        if set(self.primary_criteria) & set(self.secondary_criteria):
            raise ValueError("primary and secondary criteria cannot overlap")

        for alternative_id, alternative in self.alternatives.items():
            missing = set(self.criteria) - set(alternative.attributes)
            if missing:
                raise ValueError(f"{alternative_id} is missing criterion values: {sorted(missing)}")

        valid_refs = {
            f"{alternative_id}.{attribute_id}"
            for alternative_id, alternative in self.alternatives.items()
            for attribute_id in alternative.attributes
        }
        unknown_visible = set(self.initially_visible) - valid_refs
        if unknown_visible:
            raise ValueError(f"unknown initially_visible attributes: {sorted(unknown_visible)}")

        for query_id, query in self.queries.items():
            ref = f"{query.alternative_id}.{query.attribute_id}"
            if ref not in valid_refs:
                raise ValueError(f"{query_id} points to unknown attribute {ref}")
            if ref in self.initially_visible:
                raise ValueError(f"{query_id} points to an initially visible attribute")
            if query.branch_id is not None and query.branch_id not in self.branches:
                raise ValueError(f"{query_id} points to unknown branch {query.branch_id}")
        unknown_plan = set(self.simulator_query_plan) - set(self.queries)
        if unknown_plan:
            raise ValueError(f"simulator plan has unknown queries: {sorted(unknown_plan)}")
        unknown_manual = set(self.oracle_annotation.manual_action_changing) - set(self.queries)
        if unknown_manual:
            raise ValueError(f"manual oracle has unknown queries: {sorted(unknown_manual)}")
        return self


_SECRET_KEY_PARTS = ("api_key", "apikey", "access_token", "password", "secret", "authorization", "cookie")


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                return True
            if contains_secret_key(item):
                return True
    elif isinstance(value, list):
        return any(contains_secret_key(item) for item in value)
    return False


class ModelConfig(StrictModel):
    provider: str
    base_model: str
    exact_model_version: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = Field(default=1024, gt=0)
    context_budget: int = Field(default=8192, gt=0)
    tool_permissions: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_credentials(self) -> "ModelConfig":
        if contains_secret_key(self.parameters):
            raise ValueError("model parameters must not contain credentials")
        return self


class ExperimentConfig(StrictModel):
    experiment_id: str
    task_file: str = "tasks.json"
    task_ids: list[str] | None = None
    replicates: int = Field(default=3, gt=0)
    base_seed: int = 20260912
    max_information_turns: int = Field(default=16, gt=0)
    model: ModelConfig
    base_system_prompt: str = "Help the user make the stated decision using only revealed information."
    user_simulator_id: str = "deterministic-v1"
    user_simulator_version: str = "1.0.0"
    assistant_callback_id: str = "deterministic-fixture-v1"
    bootstrap_samples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ConversationTurn(StrictModel):
    index: int = Field(ge=0)
    role: Literal["system", "user", "assistant", "environment"]
    content: str


class QueryEvent(StrictModel):
    sequence: int = Field(ge=1)
    query_id: str
    alternative_id: str
    attribute_id: str
    revealed_value: Scalar
    relevance: QueryRelevance
    branch_id: str | None = None
    repeated: bool
    action_changing: bool
    effects: list[str]
    redundant: bool
    sufficient_before: bool
    sufficient_after: bool


class PolicyEvent(StrictModel):
    turn_index: int = Field(ge=0)
    action: StopWiseAction
    expected_action: StopWiseAction
    visible_nudge: str = ""
    target_branch_id: str | None = None
    sufficient_at_action: bool

    @model_validator(mode="after")
    def validate_visibility(self) -> "PolicyEvent":
        if self.action == StopWiseAction.NO_INTERVENTION and self.visible_nudge:
            raise ValueError("NO_INTERVENTION cannot produce a visible nudge")
        if self.action != StopWiseAction.NO_INTERVENTION and not self.visible_nudge.strip():
            raise ValueError("interventions require a visible nudge")
        return self


class EpisodeLog(StrictModel):
    schema_version: str = "1.0.0"
    experiment_id: str
    task_id: str
    task_version: str
    task_definition_hash: str
    domain: str
    condition: Literal["baseline", "stopwise", "middleware"]
    replicate: int = Field(ge=0)
    seed: int
    base_model: str
    exact_model_version: str
    assistant_model_config: ModelConfig
    base_prompt_hash: str
    prompt_version_hash: str
    policy_version_hash: str | None = None
    assistant_callback_id: str
    user_simulator_id: str
    user_simulator_version: str
    max_information_turns: int = Field(gt=0)
    conversation_turns: list[ConversationTurn]
    query_ids: list[str]
    revealed_attributes: list[str]
    query_events: list[QueryEvent]
    stopwise_actions: list[StopWiseAction]
    policy_events: list[PolicyEvent]
    visible_nudges: list[str]
    final_choice: str
    token_usage: TokenUsage
    termination_reason: Literal["user_choice", "max_information_turns", "error"]
    errors: list[str] = Field(default_factory=list)
    retries: int = Field(default=0, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    latency_source: Literal["provider", "client", "not_recorded"] = "not_recorded"

    @model_validator(mode="after")
    def validate_identity(self) -> "EpisodeLog":
        if self.base_model != self.assistant_model_config.base_model:
            raise ValueError("base_model must match assistant_model_config.base_model")
        if self.exact_model_version != self.assistant_model_config.exact_model_version:
            raise ValueError("exact_model_version must match assistant_model_config")
        if self.condition == "baseline" and self.policy_events:
            raise ValueError("baseline episodes cannot contain StopWise policy events")
        if self.stopwise_actions != [event.action for event in self.policy_events]:
            raise ValueError("stopwise_actions must match policy_events")
        if self.visible_nudges != [event.visible_nudge for event in self.policy_events if event.visible_nudge]:
            raise ValueError("visible_nudges must match policy_events")
        if self.query_ids != [event.query_id for event in self.query_events]:
            raise ValueError("query_ids must match query_events")
        return self
