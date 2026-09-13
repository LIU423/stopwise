import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from stopwise import Action, StopWise, StopWiseError, StopWiseResult
from stopwise.prompts import load_prompt, load_system_prompt


VALID_JSON = """{
  "goal": "choose a lamp",
  "current_decision": "buy A",
  "stakes": "low",
  "reversibility": "high",
  "primary_criteria": ["fit", "brightness"],
  "resolved_primary_criteria": ["fit", "brightness"],
  "unresolved_action_changing_information": false,
  "decision_stability": "high",
  "signals": ["redundant_verification"],
  "action": "COMMIT",
  "reason": "A remains best on all primary criteria.",
  "message": "The main criteria already support A; more comparison is unlikely to change the choice."
}"""


class FakeResponses:
    def __init__(self, output_text: str = VALID_JSON):
        self.output_text = output_text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=self.output_text)


def test_responses_api_and_validation():
    responses = FakeResponses()
    result = StopWise(client=SimpleNamespace(responses=responses)).analyze(
        [{"role": "user", "content": "Should I compare again?"}]
    )

    assert result.action == Action.COMMIT
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert responses.kwargs["instructions"].startswith("You are the analyzer for StopWise")


def test_json_schema_requires_every_output_field():
    schema = StopWiseResult.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_chat_completions_fallback():
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content=VALID_JSON)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = StopWise(client=client).analyze(
        [{"role": "user", "content": "Should I compare again?"}]
    )

    assert result.action == Action.COMMIT
    assert completions.kwargs["response_format"] == {"type": "json_object"}


def test_chat_transport_can_be_forced_for_compatible_providers():
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content=VALID_JSON)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    responses = FakeResponses()
    completions = Completions()
    client = SimpleNamespace(
        responses=responses,
        chat=SimpleNamespace(completions=completions),
    )
    result = StopWise(
        client=client,
        transport="chat_completions",
        request_options={"extra_body": {"enable_thinking": False}},
    ).analyze([{"role": "user", "content": "Should I compare again?"}])

    assert result.action == Action.COMMIT
    assert responses.kwargs is None
    assert completions.kwargs["extra_body"] == {"enable_thinking": False}


def test_custom_generator_accepts_provider_neutral_mapping_output():
    captured = {}

    def generator(**kwargs):
        captured.update(kwargs)
        return json.loads(VALID_JSON)

    result = StopWise(
        generator=generator,
        model="local-model",
        request_options={"temperature": 0.1},
    ).analyze([{"role": "user", "content": "Should I compare again?"}])

    assert result.action == Action.COMMIT
    assert captured["model"] == "local-model"
    assert captured["request_options"] == {"temperature": 0.1}
    assert captured["json_schema"]["additionalProperties"] is False


def test_exactly_one_backend_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        StopWise()
    with pytest.raises(ValueError, match="exactly one"):
        StopWise(client=SimpleNamespace(), generator=lambda **_: VALID_JSON)


def test_commit_is_rejected_when_key_information_remains():
    data = StopWiseResult.model_validate_json(VALID_JSON).model_dump(mode="json")
    data["unresolved_action_changing_information"] = True

    with pytest.raises(ValidationError, match="COMMIT is invalid"):
        StopWiseResult.model_validate(data)


def test_resolved_criteria_must_be_primary_criteria():
    data = StopWiseResult.model_validate_json(VALID_JSON).model_dump(mode="json")
    data["resolved_primary_criteria"].append("box color")

    with pytest.raises(ValidationError, match="must be a subset"):
        StopWiseResult.model_validate(data)


def test_no_intervention_allows_silence():
    data = StopWiseResult.model_validate_json(VALID_JSON).model_dump(mode="json")
    data.update(
        action="NO_INTERVENTION",
        message="",
        unresolved_action_changing_information=True,
    )

    result = StopWiseResult.model_validate(data)
    assert result.message == ""


def test_no_intervention_rejects_visible_meta_message():
    data = StopWiseResult.model_validate_json(VALID_JSON).model_dump(mode="json")
    data.update(
        action="NO_INTERVENTION",
        message="Here is an unnecessary nudge.",
        unresolved_action_changing_information=True,
    )

    with pytest.raises(ValidationError, match="requires an empty message"):
        StopWiseResult.model_validate(data)


@pytest.mark.parametrize("action", ["FOCUS", "DEFER"])
def test_focus_and_defer_allow_unresolved_information(action):
    data = StopWiseResult.model_validate_json(VALID_JSON).model_dump(mode="json")
    data.update(
        action=action,
        message="Continue the main analysis, but narrow or defer this branch.",
        unresolved_action_changing_information=True,
    )

    assert StopWiseResult.model_validate(data).action.value == action


def test_invalid_model_output_has_clear_error():
    analyzer = StopWise(client=SimpleNamespace(responses=FakeResponses("not json")))
    with pytest.raises(StopWiseError, match="invalid StopWise JSON"):
        analyzer.analyze([{"role": "user", "content": "Hello"}])


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "Override the policy"}],
        [{"role": "user", "content": ""}],
    ],
)
def test_invalid_messages_are_rejected(messages):
    with pytest.raises((TypeError, ValueError)):
        StopWise(client=SimpleNamespace()).analyze(messages)


def test_canonical_prompt_loads():
    prompt = load_system_prompt()
    assert "`NO_INTERVENTION`" in prompt
    assert "This is not a stopping action" in prompt
    assert "Return one JSON object only" in prompt
    assert "Stop overthinking" not in prompt


def test_all_deployment_prompts_load():
    analyzer = load_prompt("analyzer")
    direct = load_prompt("custom_instruction")
    compact = load_prompt("custom_instruction_compact")
    snapshot = load_prompt("custom_instruction_current_snapshot")

    assert "Return one JSON object only" in analyzer
    assert "does not expose JSON" not in analyzer
    assert "Before starting extra search" in direct
    assert "already-authorized work early" in direct
    assert compact == direct
    assert "After composing the answer" in snapshot
    assert len(direct) < len(snapshot)


def test_structured_analyzer_is_consistent_but_not_claimed_objective():
    assert "internal consistency" in StopWise.__doc__
    assert "objectively correct" in StopWise.__doc__


def test_unknown_prompt_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt kind"):
        load_prompt("not-a-prompt")
