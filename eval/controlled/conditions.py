"""Fair condition construction and provider-neutral assistant callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stopwise.prompts import load_prompt

from .schemas import ModelConfig, StopWiseAction, UsageSource


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
    """Build the three primary prompt conditions with one shared model config."""

    current_policy = load_prompt("custom_instruction_current_snapshot")
    minimal_policy = load_prompt("custom_instruction")
    current_prompt = f"{base_system_prompt.rstrip()}\n\n{current_policy}"
    minimal_prompt = f"{base_system_prompt.rstrip()}\n\n{minimal_policy}"
    return {
        "baseline": ConditionHarness(
            name="baseline",
            system_prompt=base_system_prompt,
            model=model,
            base_prompt_hash=prompt_hash(base_system_prompt),
            prompt_version_hash=prompt_hash(base_system_prompt),
            policy_version_hash=None,
        ),
        "current_prompt": ConditionHarness(
            name="current_prompt",
            system_prompt=current_prompt,
            model=model,
            base_prompt_hash=prompt_hash(base_system_prompt),
            prompt_version_hash=prompt_hash(current_prompt),
            policy_version_hash=prompt_hash(current_policy),
        ),
        "minimal_prompt": ConditionHarness(
            name="minimal_prompt",
            system_prompt=minimal_prompt,
            model=model,
            base_prompt_hash=prompt_hash(base_system_prompt),
            prompt_version_hash=prompt_hash(minimal_prompt),
            policy_version_hash=prompt_hash(minimal_policy),
        ),
    }


def assert_fair_conditions(conditions: Sequence[ConditionHarness]) -> None:
    required = {"baseline", "current_prompt", "minimal_prompt"}
    if {item.name for item in conditions} != required:
        raise ValueError(f"controlled experiment requires conditions: {sorted(required)}")
    configs = [item.model.model_dump(mode="json") for item in conditions]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("all prompt conditions must use identical model configuration")


class AssistantRequest(BaseModel):
    """Provider-neutral live request containing visible information only.

    Condition names, oracle state, relevance labels, branch annotations, repeat
    flags, and hidden alternative values are deliberately absent.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    model: ModelConfig
    messages: list[dict[str, str]]
    visible_attributes: dict[str, Any]
    latest_observation: str


class AssistantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    action: StopWiseAction | None = None
    visible_nudge: str = ""
    target_branch_id: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    usage_source: UsageSource = UsageSource.NOT_RECORDED
    middleware_calls: int = Field(default=0, ge=0)
    middleware_prompt_tokens: int = Field(default=0, ge=0)
    middleware_completion_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)

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
        last_user = next(
            item["content"] for item in reversed(request.messages) if item["role"] == "user"
        )
        query_id = last_user.split(":", 1)[0].removeprefix("Information query ").strip()
        query_text = last_user.split(":", 1)[1].strip() if ":" in last_user else last_user
        substantive = f"The requested information is {request.latest_observation}."
        prompt_tokens = sum(len(item["content"].split()) for item in request.messages) + len(
            request.system_prompt.split()
        )

        policy_enabled = all(
            marker in request.system_prompt for marker in ("StopWise", "FOCUS", "COMMIT", "DEFER")
        )
        if not policy_enabled:
            return AssistantResponse(
                content=substantive,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(substantive.split()),
                usage_source=UsageSource.ESTIMATED_FIXTURE,
            )

        action = StopWiseAction.NO_INTERVENTION
        target_branch_id = None
        repeated = sum(query_id in item["content"] for item in request.messages if item["role"] == "user") > 1
        lowered = query_text.lower()
        if repeated:
            action = StopWiseAction.COMMIT
        elif any(word in lowered for word in ("future", "years after", "years later", "hypothetical", "2028", "2029", "2030")):
            action = StopWiseAction.DEFER
            target_branch_id = self._DEFER_BRANCHES.get(query_id)
        elif any(word in lowered for word in self._LOW_VALUE_TERMS):
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
            usage_source=UsageSource.ESTIMATED_FIXTURE,
        )

    _LOW_VALUE_TERMS = (
        "color", "packaging", "typography", "font", "cover art", "logo",
        "invoice", "syllabus", "meal-card", "shipping-crate", "secondary",
        "refundability", "teaching format", "warranty", "carry weight",
        "hundredth-point", "decimal", "precision",
    )
    _DEFER_BRANCHES = {
        "lap-future-dock": "future-dock",
        "lap-future-countries": "future-dock",
        "hotel-spa-future": "future-spa",
        "hotel-spa-menu": "future-spa",
        "sub-ai-future": "future-ai",
        "sub-ai-regions": "future-ai",
        "course-alumni-future": "future-alumni",
        "course-alumni-cities": "future-alumni",
        "travel-side-future": "future-side-trip",
        "travel-side-maps": "future-side-trip",
        "device-factory-future": "future-factory",
        "device-factory-layouts": "future-factory",
    }
