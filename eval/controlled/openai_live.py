"""OpenAI Responses API adapters for paid controlled pilot runs.

The model-facing requests are deliberately built only from the provider-neutral
visible request objects. Evaluation oracle state, hidden values, condition names,
relevance labels, and StopWise policy events never enter these payloads.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from .conditions import AssistantRequest, AssistantResponse
from .schemas import ModelConfig, StopWiseAction, UsageSource
from .simulators import UserAction, UserModelRequest


ASSISTANT_OUTPUT_PROTOCOL = """
Return the conversational answer and evaluation instrumentation through the
provided JSON schema. The `content` and `visible_nudge` fields are the only text
shown to the simulated user. Never put an internal action label in either field.
If the system instructions do not define StopWise, use NO_INTERVENTION and an
empty nudge. Otherwise choose the action prescribed by those instructions.
FOCUS narrows continued inquiry, DEFER postpones only a named side branch, and
COMMIT is advice rather than an instruction that forces the user to stop.
Set target_query_id to the visible query whose branch is being deferred, or null.
""".strip()

USER_SIMULATOR_INSTRUCTIONS = """
Act as the person making the decision in the visible scenario. Use only the
natural-language transcript, revealed facts, query menu, and alternative menu in
the input. Do not assume hidden values. Choose another information query when it
could still change feasibility or a main tradeoff; otherwise choose an
alternative. Treat any assistant suggestion to focus, defer, or decide as advice
that you may accept or reject based on the visible evidence. Return only the
structured action requested by the response schema.
""".strip()

_QUERY_ID_RE = re.compile(r"Information query ([^:]+):")


class _AssistantTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    action: StopWiseAction
    visible_nudge: str
    target_query_id: str | None


class _UserDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["query", "choose"]
    query_id: str | None
    choice: str | None

    @model_validator(mode="after")
    def validate_target(self) -> "_UserDecision":
        if self.kind == "query" and (not self.query_id or self.choice is not None):
            raise ValueError("query decisions require only query_id")
        if self.kind == "choose" and (not self.choice or self.query_id is not None):
            raise ValueError("choice decisions require only choice")
        return self


class ResponsesClient(Protocol):
    responses: Any


@dataclass(frozen=True)
class ProviderResult:
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_ms: float


class PaidRunBudget:
    """Conservative request and token-price guard for one live run.

    Cached inputs are charged at the full input rate and ``billing_uplift`` is
    applied to every token. This is an intentionally conservative run estimate,
    not a promise that it matches the provider invoice exactly.
    """

    def __init__(
        self,
        *,
        max_cost_usd: float,
        max_requests: int,
        max_input_tokens_per_request: int,
        max_output_tokens_per_request: int,
        input_usd_per_million: float,
        output_usd_per_million: float,
        billing_uplift: float = 1.0,
        framing_token_reserve: int = 128,
    ) -> None:
        if max_cost_usd <= 0 or max_requests <= 0:
            raise ValueError("paid-run limits must be positive")
        self.max_cost_usd = max_cost_usd
        self.max_requests = max_requests
        self.max_input_tokens_per_request = max_input_tokens_per_request
        self.max_output_tokens_per_request = max_output_tokens_per_request
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.billing_uplift = billing_uplift
        self.framing_token_reserve = framing_token_reserve
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0

        if self.maximum_run_cost_usd > self.max_cost_usd + 1e-12:
            raise ValueError(
                "configured worst-case run cost exceeds the hard cap: "
                f"{self.maximum_run_cost_usd:.6f} > {self.max_cost_usd:.6f} USD"
            )

    @property
    def maximum_request_cost_usd(self) -> float:
        return self.cost_for_usage(
            self.max_input_tokens_per_request,
            self.max_output_tokens_per_request,
        )

    @property
    def maximum_run_cost_usd(self) -> float:
        return self.max_requests * self.maximum_request_cost_usd

    def cost_for_usage(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self.billing_uplift * (
            prompt_tokens * self.input_usd_per_million
            + completion_tokens * self.output_usd_per_million
        ) / 1_000_000

    def before_request(self, payload: dict[str, Any]) -> None:
        # UTF-8 byte length plus a framing reserve is a deliberately loose upper
        # bound on input tokens for these ASCII-heavy JSON requests.
        payload_bound = len(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ) + self.framing_token_reserve
        if payload_bound > self.max_input_tokens_per_request:
            raise RuntimeError(
                "request exceeds the configured conservative input bound: "
                f"{payload_bound} > {self.max_input_tokens_per_request}"
            )
        if self.requests >= self.max_requests:
            raise RuntimeError("paid-run request cap reached")
        if self.estimated_cost_usd + self.maximum_request_cost_usd > self.max_cost_usd + 1e-12:
            raise RuntimeError("paid-run cost cap would be exceeded by the next request")
        self.requests += 1

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> float:
        if prompt_tokens > self.max_input_tokens_per_request:
            raise RuntimeError("provider-reported input usage exceeded the guarded bound")
        if completion_tokens > self.max_output_tokens_per_request:
            raise RuntimeError("provider-reported output usage exceeded the guarded bound")
        cost = self.cost_for_usage(prompt_tokens, completion_tokens)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.estimated_cost_usd += cost
        if self.estimated_cost_usd > self.max_cost_usd + 1e-12:
            raise RuntimeError("paid-run cost cap exceeded")
        return cost

    def snapshot(self) -> dict[str, Any]:
        return {
            "hard_cap_usd": self.max_cost_usd,
            "conservative_estimated_cost_usd": self.estimated_cost_usd,
            "maximum_run_cost_usd": self.maximum_run_cost_usd,
            "requests_executed": self.requests,
            "max_requests": self.max_requests,
            "provider_reported_prompt_tokens": self.prompt_tokens,
            "provider_reported_completion_tokens": self.completion_tokens,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "billing_uplift": self.billing_uplift,
            "cached_input_discount_assumed": False,
            "invoice_exact": False,
        }


class OpenAIResponsesTransport:
    def __init__(self, client: ResponsesClient, model: ModelConfig, budget: PaidRunBudget) -> None:
        self.client = client
        self.model = model
        self.budget = budget

    def generate(
        self,
        *,
        instructions: str,
        input_messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> ProviderResult:
        text_options = dict(self.model.parameters.get("text", {}))
        text_options["format"] = {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema,
        }
        payload: dict[str, Any] = {
            "model": self.model.exact_model_version,
            "instructions": instructions,
            "input": input_messages,
            "max_output_tokens": self.model.max_output_tokens,
            "text": text_options,
            "tools": [],
            "store": bool(self.model.parameters.get("store", False)),
        }
        if self.model.temperature is not None:
            payload["temperature"] = self.model.temperature
        if self.model.top_p is not None:
            payload["top_p"] = self.model.top_p
        if "reasoning" in self.model.parameters:
            payload["reasoning"] = self.model.parameters["reasoning"]
        if "service_tier" in self.model.parameters:
            payload["service_tier"] = self.model.parameters["service_tier"]

        self.budget.before_request(payload)
        started = perf_counter()
        response = self.client.responses.create(**payload)
        latency_ms = (perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        if usage is None:
            raise RuntimeError("provider response omitted token usage")
        prompt_tokens = int(getattr(usage, "input_tokens"))
        completion_tokens = int(getattr(usage, "output_tokens"))
        cost = self.budget.record_usage(prompt_tokens, completion_tokens)
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise RuntimeError(f"provider response was not completed: {status}")
        returned_model = getattr(response, "model", self.model.exact_model_version)
        if returned_model != self.model.exact_model_version:
            raise RuntimeError(
                "provider returned an unexpected model version: "
                f"{returned_model!r} != {self.model.exact_model_version!r}"
            )
        return ProviderResult(
            output_text=str(getattr(response, "output_text")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
        )


class OpenAIAssistantCallback:
    def __init__(
        self,
        transport: OpenAIResponsesTransport,
        *,
        query_branch_by_id: dict[str, str | None],
    ) -> None:
        self.transport = transport
        self.query_branch_by_id = query_branch_by_id

    def __call__(self, request: AssistantRequest) -> AssistantResponse:
        visible_context = json.dumps(request.visible_attributes, sort_keys=True, ensure_ascii=False)
        input_messages = [dict(item) for item in request.messages]
        input_messages.append(
            {
                "role": "user",
                "content": (
                    f"Latest controlled result: {request.latest_observation}\n"
                    f"All currently visible attributes: {visible_context}"
                ),
            }
        )
        result = self.transport.generate(
            instructions=f"{request.system_prompt.rstrip()}\n\n{ASSISTANT_OUTPUT_PROTOCOL}",
            input_messages=input_messages,
            schema_name="stopwise_assistant_turn",
            schema=_AssistantTurn.model_json_schema(),
        )
        parsed = _AssistantTurn.model_validate_json(result.output_text)
        policy_enabled = all(
            marker in request.system_prompt for marker in ("StopWise", "FOCUS", "COMMIT", "DEFER")
        )
        if not policy_enabled:
            action: StopWiseAction | None = None
            visible_nudge = ""
            target_branch_id = None
        else:
            action = parsed.action
            visible_nudge = parsed.visible_nudge.strip()
            if action == StopWiseAction.NO_INTERVENTION:
                visible_nudge = ""
            elif not visible_nudge:
                raise RuntimeError("provider intervention omitted its visible nudge")
            asked_query_ids = [
                match.group(1)
                for item in request.messages
                if item["role"] == "user"
                for match in [_QUERY_ID_RE.search(item["content"])]
                if match
            ]
            target_query_id = parsed.target_query_id
            if target_query_id not in asked_query_ids:
                target_query_id = asked_query_ids[-1] if asked_query_ids else None
            target_branch_id = (
                self.query_branch_by_id.get(target_query_id) if action == StopWiseAction.DEFER else None
            )
        return AssistantResponse(
            content=parsed.content.strip(),
            action=action,
            visible_nudge=visible_nudge,
            target_branch_id=target_branch_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            usage_source=UsageSource.PROVIDER_REPORTED,
            estimated_cost_usd=result.estimated_cost_usd,
            latency_ms=result.latency_ms,
        )


class OpenAITextOnlyUserCallback:
    def __init__(self, transport: OpenAIResponsesTransport) -> None:
        self.transport = transport

    def __call__(self, request: UserModelRequest) -> UserAction:
        # This serialization contains only fields allowed by UserModelRequest.
        visible_input = json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        result = self.transport.generate(
            instructions=USER_SIMULATOR_INSTRUCTIONS,
            input_messages=[{"role": "user", "content": visible_input}],
            schema_name="stopwise_user_decision",
            schema=_UserDecision.model_json_schema(),
        )
        parsed = _UserDecision.model_validate_json(result.output_text)
        return UserAction(
            kind=parsed.kind,
            query_id=parsed.query_id,
            choice=parsed.choice,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            usage_source=UsageSource.PROVIDER_REPORTED,
            estimated_cost_usd=result.estimated_cost_usd,
            latency_ms=result.latency_ms,
        )
