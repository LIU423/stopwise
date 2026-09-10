"""Small, provider-light StopWise analyzer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import ValidationError

from .prompts import load_system_prompt
from .schemas import StopWiseResult


Transport: TypeAlias = Literal["auto", "responses", "chat_completions"]
GeneratorOutput: TypeAlias = str | Mapping[str, Any] | StopWiseResult


class CompletionGenerator(Protocol):
    """Provider-neutral hook for SDKs without an OpenAI-compatible surface."""

    def __call__(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
        request_options: Mapping[str, Any],
    ) -> GeneratorOutput: ...


class StopWiseError(RuntimeError):
    """Raised when an LLM response cannot be obtained or validated."""


class StopWise:
    """Analyze a conversation using a compatible client or custom generator.

    The client may expose either ``responses.create`` or
    ``chat.completions.create``. Other SDKs can be integrated through
    ``generator``. No provider SDK is required by the core package.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        transport: Transport = "auto",
        generator: CompletionGenerator | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if (client is None) == (generator is None):
            raise ValueError("provide exactly one of client or generator")
        if transport not in {"auto", "responses", "chat_completions"}:
            raise ValueError(f"unsupported transport: {transport}")
        if generator is not None and transport != "auto":
            raise ValueError("transport only applies to compatible clients")

        self.client = client
        self.generator = generator
        self.model = model
        self.system_prompt = system_prompt or load_system_prompt()
        self.transport = transport
        self.request_options = dict(request_options or {})

    def analyze(self, messages: Sequence[Mapping[str, Any]]) -> StopWiseResult:
        normalized = _normalize_messages(messages)
        raw = self._request(normalized)
        try:
            if isinstance(raw, StopWiseResult):
                return raw
            if isinstance(raw, Mapping):
                return StopWiseResult.model_validate(raw)
            if isinstance(raw, str):
                return StopWiseResult.model_validate_json(raw)
            raise TypeError(f"unsupported generator output: {type(raw).__name__}")
        except ValidationError as exc:
            raise StopWiseError(f"LLM returned invalid StopWise JSON: {exc}") from exc

    def _request(self, messages: list[dict[str, str]]) -> GeneratorOutput:
        schema = StopWiseResult.model_json_schema()

        if self.generator is not None:
            return self.generator(
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                json_schema=schema,
                request_options=self.request_options,
            )

        responses = getattr(self.client, "responses", None)
        use_responses = self.transport in {"auto", "responses"}
        if use_responses and responses is not None and hasattr(responses, "create"):
            payload = {
                "model": self.model,
                "instructions": self.system_prompt,
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "stopwise_analysis",
                        "schema": schema,
                        "strict": True,
                    }
                },
            }
            payload.update(self.request_options)
            response = responses.create(**payload)
            output_text = getattr(response, "output_text", None)
            if not isinstance(output_text, str) or not output_text.strip():
                raise StopWiseError("Responses API returned no output_text")
            return output_text

        chat = getattr(self.client, "chat", None)
        completions = getattr(chat, "completions", None)
        use_chat = self.transport in {"auto", "chat_completions"}
        if use_chat and completions is not None and hasattr(completions, "create"):
            payload = {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *messages,
                ],
            }
            payload.update(self.request_options)
            response = completions.create(**payload)
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError) as exc:
                raise StopWiseError("Chat Completions API returned no message") from exc
            if not isinstance(content, str) or not content.strip():
                raise StopWiseError("Chat Completions API returned empty content")
            return content

        raise StopWiseError(
            f"client does not support the requested transport: {self.transport}"
        )


def analyze_conversation(
    messages: Sequence[Mapping[str, Any]],
    client: Any | None = None,
    *,
    model: str = "gpt-4o-mini",
    system_prompt: str | None = None,
    transport: Transport = "auto",
    generator: CompletionGenerator | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> StopWiseResult:
    """Convenience wrapper around :class:`StopWise`."""

    return StopWise(
        client=client,
        model=model,
        system_prompt=system_prompt,
        transport=transport,
        generator=generator,
        request_options=request_options,
    ).analyze(messages)


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of role/content mappings")
    if not messages:
        raise ValueError("messages must not be empty")

    normalized: list[dict[str, str]] = []
    allowed_roles = {"user", "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in allowed_roles:
            raise ValueError(f"message {index} has unsupported role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} must contain non-empty text")
        normalized.append({"role": role, "content": content})
    return normalized
