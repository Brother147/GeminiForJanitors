from gfjproxy.models import JaiMessage, JaiRequest
from gfjproxy.providers.openai_compat import (
    apply_openai_optional_params,
    deepseek_reasoning_params,
    gemini_parts,
    gemini_thinking_config,
    message_payloads,
    merge_extra_body,
)


def test_janitor_reasoning_fields_are_preserved():
    payload = {
        "model": "openrouter/deepseek/deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "xhigh",
        "reasoning": {"effort": "max", "exclude": False},
        "new_frontend_parameter": {"future": True},
    }

    request = JaiRequest.parse(payload)

    assert request.extra_params["reasoning_effort"] == "xhigh"
    assert request.extra_params["reasoning"] == {"effort": "max", "exclude": False}
    assert request.extra_params["new_frontend_parameter"] == {"future": True}


def test_message_extra_fields_and_multimodal_content_are_preserved():
    message = JaiMessage.parse(
        {
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "tool_call_id": "call_123",
            "name": "browser",
        }
    )

    assert message_payloads([message]) == [
        {
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "tool_call_id": "call_123",
            "name": "browser",
        }
    ]


def test_openrouter_reasoning_format_is_preserved_without_conflicting_alias():
    payload = {"model": "test", "messages": []}
    apply_openai_optional_params(
        payload,
        {
            "reasoning": {"effort": "high", "exclude": False},
            "reasoning_effort": "low",
        },
        provider="openrouter",
    )

    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert "reasoning_effort" not in payload


def test_openrouter_accepts_janitor_openai_thinking_toggle():
    payload = {"model": "test", "messages": []}
    apply_openai_optional_params(
        payload,
        {"thinking": {"type": "disabled"}},
        provider="openrouter",
    )

    assert payload["reasoning"] == {"effort": "none"}
    assert "thinking" not in payload


def test_deepseek_reasoning_level_mapping_matches_provider_contract():
    assert deepseek_reasoning_params({"reasoning_effort": "off"}) == {
        "reasoning_effort": "none",
        "thinking": {"type": "disabled"},
    }
    assert deepseek_reasoning_params({"reasoning_effort": "minimal"}) == {
        "reasoning_effort": "low",
        "thinking": {"type": "enabled"},
    }
    assert deepseek_reasoning_params({"reasoning": {"effort": "max"}}) == {
        "reasoning_effort": "max",
        "thinking": {"type": "enabled"},
    }


def test_gemini_thinking_effort_maps_by_model_generation():
    assert gemini_thinking_config("gemini-2.5-flash", {"reasoning_effort": "off"}) == {
        "thinkingBudget": 0
    }
    assert gemini_thinking_config("gemini-2.5-pro", {"reasoning_effort": "high"}) == {
        "thinkingBudget": 16384
    }
    assert gemini_thinking_config(
        "gemini-3.5-flash", {"reasoning_effort": "medium"}
    ) == {
        "thinkingLevel": "MEDIUM"
    }
    assert gemini_thinking_config(
        "gemini-3.1-pro-preview", {"reasoning_effort": "off"}
    ) == {
        "thinkingLevel": "LOW"
    }


def test_gemini_multimodal_text_and_inline_image_conversion():
    parts = gemini_parts(
        [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ]
    )

    assert parts[0] == {"text": "describe this"}
    assert parts[1] == {
        "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}
    }


def test_extra_body_is_flattened_for_raw_http_requests():
    payload = {"model": "test", "messages": []}
    apply_openai_optional_params(
        payload,
        {"extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "xhigh"}},
    )

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "xhigh"
    assert "extra_body" not in payload
    assert merge_extra_body({"extra_body": {"top_k": 30}})["top_k"] == 30


def test_openrouter_normalizes_janitor_level_aliases():
    payload = {"model": "test", "messages": []}
    apply_openai_optional_params(
        payload,
        {"reasoning_effort": "extra high"},
        provider="openrouter",
    )

    assert payload["reasoning_effort"] == "xhigh"


def test_gemini_four_off_and_max_clamp_to_model_supported_levels():
    assert gemini_thinking_config(
        "gemini-4-argon", {"reasoning_effort": "off"}
    ) == {"thinkingLevel": "LOW"}
    assert gemini_thinking_config(
        "gemini-4-neon", {"reasoning_effort": "max"}
    ) == {"thinkingLevel": "XHIGH"}


def test_gemini_25_pro_cannot_disable_thinking_and_uses_minimum_budget():
    assert gemini_thinking_config(
        "gemini-2.5-pro", {"reasoning_effort": "off"}
    ) == {"thinkingBudget": 128}


def test_deepseek_reads_thinking_toggle_from_extra_body():
    assert deepseek_reasoning_params(
        {"extra_body": {"thinking": {"type": "disabled"}}}
    ) == {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
    }


def test_gemini_clamps_levels_to_real_model_support():
    assert gemini_thinking_config(
        "gemini-3.8-flash", {"reasoning_effort": "minimal"}
    ) == {"thinkingLevel": "LOW"}
    assert gemini_thinking_config(
        "gemini-3.6-flash", {"reasoning_effort": "minimal"}
    ) == {"thinkingLevel": "MINIMAL"}
    assert gemini_thinking_config(
        "gemini-4-carbon", {"reasoning_effort": "xhigh"}
    ) == {"thinkingLevel": "HIGH"}


def test_assistant_reasoning_state_survives_message_round_trip():
    message = JaiMessage.parse(
        {
            "role": "assistant",
            "content": "Tool call follows",
            "reasoning_content": "provider reasoning state",
            "tool_calls": [{"id": "call_1", "type": "function"}],
        }
    )

    assert message_payloads([message])[0] == {
        "role": "assistant",
        "content": "Tool call follows",
        "reasoning_content": "provider reasoning state",
        "tool_calls": [{"id": "call_1", "type": "function"}],
    }


def test_reasoning_level_camel_case_alias_is_accepted():
    assert gemini_thinking_config(
        "gemini-4-argon", {"thinkingLevel": "medium"}
    ) == {"thinkingLevel": "MEDIUM"}
