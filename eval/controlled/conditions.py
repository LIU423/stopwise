"""Fair condition construction and provider-neutral assistant callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stopwise.prompts import load_prompt

from .schemas import ModelConfig, QueryRelevance, StopWiseAction


def prompt_hash(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class ConditionHarness:
    name: str
    system_prompt: str
    model: ModelConfig
    base_prompt_hash: str
    prompt_version_hash: str
    policy_version_hash: str | None


def build_conditions(base_system_prompt: str, model: ModelConfig) -> dict[str, ConditionHarness]:
    """Build the two direct-chat conditions with one shared model config."""

    policy = load_prompt("custom_instruction")
    stopwise_prompt = f"{base_system_prompt.rstrip()}\n\n{policy}"
    return {
        "baseline": ConditionHarness(
            name="baseline",
            system_prompt=base_system_prompt,
            model=model,
            base_prompt_hash=prompt_hash(base_system_prompt),
            prompt_version_hash=prompt_hash(base_system_prompt),
            policy_version_hash=None,
        ),
        "stopwise": ConditionHarness(
            name="stopwise",
            system_prompt=stopwise_prompt,
            model=model,
            base_prompt_hash=prompt_hash(base_system_prompt),
            prompt_version_hash=prompt_hash(stopwise_prompt),
            policy_version_hash=prompt_hash(policy),
        ),
    }


def assert_fair_conditions(conditions: Sequence[ConditionHarness]) -> None:
    if {item.name for item in conditions} != {"baseline", "stopwise"}:
        raise ValueError("paired experiment requires baseline and stopwise conditions")
    configs = [item.model.model_dump(mode="json") for item in conditions]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("baseline and StopWise must use identical model configuration")


class AssistantRequest(BaseModel):
    """Provider-neutral request. Hidden alternative values are never included."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    system_prompt: str
    model: ModelConfig
    messages: list[dict[str, str]]
    visible_attributes: dict[str, Any]
    available_query_ids: list[str]
    last_query_id: str
    last_query_relevance: QueryRelevance
    last_query_branch_id: str | None = None
    last_query_repeated: bool


class AssistantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    action: StopWiseAction | None = None
    visible_nudge: str = ""
    target_branch_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_action_visibility(self) -> "AssistantResponse":
        if self.action in {None, StopWiseAction.NO_INTERVENTION} and self.visible_nudge:
            raise ValueError("no action or NO_INTERVENTION must not render a nudge")
        if self.action not in {None, StopWiseAction.NO_INTERVENTION} and not self.visible_nudge.strip():
            raise ValueError("intervention actions require a visible nudge")
        return self


class AssistantModelCallback(Protocol):
    """Adapter point for an OpenAI, Anthropic, local, or other provider."""

    def __call__(self, request: AssistantRequest) -> AssistantResponse: ...


class DeterministicAssistantCallback:
    """Offline fixture callback; validates plumbing, not conversational efficacy."""

    def __call__(self, request: AssistantRequest) -> AssistantResponse:
        revealed = request.visible_attributes
        ref = next(
            (
                key
                for key in revealed
                if key.endswith(f".{request.last_query_id.split('__')[-1]}")
            ),
            None,
        )
        substantive = (
            f"The requested information is now available ({ref or request.last_query_id})."
        )
        prompt_tokens = sum(len(item["content"].split()) for item in request.messages) + len(
            request.system_prompt.split()
        )

        if request.condition == "baseline":
            return AssistantResponse(
                content=substantive,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(substantive.split()),
            )

        action = StopWiseAction.NO_INTERVENTION
        target_branch_id = None
        if request.last_query_repeated:
            action = StopWiseAction.COMMIT
        elif request.last_query_relevance == QueryRelevance.DEFERABLE:
            action = StopWiseAction.DEFER
            target_branch_id = request.last_query_branch_id
        elif request.last_query_relevance in {
            QueryRelevance.IRRELEVANT,
            QueryRelevance.SECONDARY,
        }:
            action = StopWiseAction.FOCUS

        nudge = ""
        if action == StopWiseAction.FOCUS:
            nudge = "There is still useful uncertainty; focus the next search on a primary criterion."
        elif action == StopWiseAction.COMMIT:
            nudge = "The decision-relevant information is sufficient; another repeat is unlikely to change the choice."
        elif action == StopWiseAction.DEFER:
            nudge = "This future branch can wait; continue resolving the current decision."

        rendered = substantive if not nudge else f"{substantive} {nudge}"
        return AssistantResponse(
            content=substantive,
            action=action,
            visible_nudge=nudge,
            target_branch_id=target_branch_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(rendered.split()),
        )
