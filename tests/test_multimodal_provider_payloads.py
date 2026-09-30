import json

from tau_agent import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from tau_ai.anthropic import _build_messages_payload
from tau_ai.google import _build_google_payload
from tau_ai.mistral import _build_mistral_payload, _MistralStreamParser
from tau_ai.openai_codex import _build_codex_payload
from tau_ai.openai_compatible import _build_chat_payload, _build_responses_payload


def _image_result() -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[
            TextContent(text="Read image file [image/png]"),
            ImageContent(data="aW1hZ2U=", mime_type="image/png"),
        ],
    )


def test_anthropic_embeds_image_in_tool_result() -> None:
    payload = _build_messages_payload(
        model="claude",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["messages"][0]["content"][0]
    assert result["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U=",
        },
    }


def test_openai_responses_embeds_image_in_function_output() -> None:
    payload = _build_responses_payload(
        model="gpt",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["input"][0]
    assert result["output"][1] == {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,aW1hZ2U=",
    }


def test_codex_embeds_image_in_function_output() -> None:
    payload = _build_codex_payload(
        model="codex",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    result = payload["input"][0]
    assert result["output"][1]["type"] == "input_image"


def test_openai_chat_attaches_tool_images_in_followup_user_message() -> None:
    payload = _build_chat_payload(
        model="vision",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=True,
    )

    assert [message["role"] for message in payload["messages"]] == ["system", "tool", "user"]
    assert payload["messages"][2]["content"][1]["type"] == "image_url"


def test_google_embeds_image_in_gemini_3_function_response() -> None:
    payload = _build_google_payload(
        model="gemini-3-pro",
        system="system",
        messages=[_image_result()],
        tools=[],
        reasoning_effort=None,
        max_tokens=None,
        supports_images=True,
    )

    response = payload["contents"][0]["parts"][0]["functionResponse"]
    assert response["parts"][0] == {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}


def test_mistral_attaches_tool_images_in_followup_user_message() -> None:
    payload = _build_mistral_payload(
        model="pixtral",
        system="system",
        messages=[_image_result()],
        tools=[],
        reasoning_effort=None,
        max_tokens=None,
        supports_images=True,
    )

    assert [message["role"] for message in payload["messages"]] == ["system", "tool", "user"]
    assert payload["messages"][2]["content"][1]["type"] == "image_url"


def test_mistral_does_not_replay_reasoning_content() -> None:
    prior = AssistantMessage(
        content=[
            ThinkingContent(thinking="prior plan", thinking_signature="reasoning_content"),
            ToolCall(id="call-1", name="read", arguments={"path": "a.txt"}),
        ]
    )
    payload = _build_mistral_payload(
        model="zai-glm-5-3",
        system="system",
        messages=[prior],
        tools=[],
        reasoning_effort=None,
        max_tokens=None,
        supports_images=False,
    )

    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert "reasoning_content" not in assistant
    assert assistant["tool_calls"]


def test_mistral_keeps_tool_name_when_continuation_chunks_send_empty_name() -> None:
    parser = _MistralStreamParser()
    chunks = [
        {"id": "chatcmpl-tool-1", "function": {"name": "bash", "arguments": ""}},
        {"id": "", "function": {"name": "", "arguments": '{"command": '}},
        {"id": "", "function": {"name": "", "arguments": '"ls -la"}'}},
    ]
    for index, tool_call in enumerate(chunks):
        tool_call = {"index": 0, **tool_call}
        finish = "tool_calls" if index == len(chunks) - 1 else None
        parser.feed(
            json.dumps(
                {"choices": [{"delta": {"tool_calls": [tool_call]}, "finish_reason": finish}]}
            )
        )

    call = parser.finalize()[0].tool_call
    assert call.id == "chatcmpl-tool-1"
    assert call.name == "bash"
    assert call.arguments == {"command": "ls -la"}


def test_non_vision_models_receive_placeholder_instead_of_image() -> None:
    payload = _build_responses_payload(
        model="text-only",
        system="system",
        messages=[_image_result()],
        tools=[],
        supports_images=False,
    )

    result = payload["input"][0]
    assert "image contents are unavailable—do not infer or describe them" in result["output"]
    assert "switch to a vision-capable model" in result["output"]
    assert "aW1hZ2U=" not in str(payload)
