"""Shared helpers for preserving OpenAI-compatible JanitorAI parameters."""

from typing import Any

from ..models import JaiMessage

# Common Chat Completions request options that can legitimately be supplied by
# JanitorAI or a compatible frontend. Provider adapters may still reject options
# not supported by their upstream model; this list prevents internal proxy state
# and arbitrary unknown keys from being leaked to external APIs.
OPENAI_OPTIONAL_PARAMS = frozenset(
    {
        "stop",
        "n",
        "seed",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
        "user",
        "service_tier",
        "metadata",
        "verbosity",
        "prediction",
        "modalities",
        "audio",
        "web_search_options",
        "top_a",
        "max_completion_tokens",
        "max_output_tokens",
        "typical_p",
        "min_p",
        "min_tokens",
        "reasoning_effort",
        "reasoningEffort",
        "reasoning_budget",
        "reasoning_summary",
        "reasoningLevel",
        "thinking_level",
        "thinkingLevel",
        "thinking_mode",
        "thinking_budget_tokens",
        "thinkingBudget",
        "enable_thinking",
        "reasoning",
        "thinking",
        "thinking_config",
        "thinkingConfig",
        "include_reasoning",
        "include_thoughts",
        "chat_template_kwargs",
    }
)

OPENROUTER_OPTIONAL_PARAMS = frozenset(
    {
        "provider",
        "route",
        "transforms",
        "models",
        "plugins",
        "session_id",
        "prompt_cache_key",
        "prompt_cache_options",
    }
)


def merge_extra_body(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten raw-HTTP ``extra_body`` parameters into a shallow settings copy."""
    if not settings:
        return {}
    source = dict(settings)
    extra_body = source.pop("extra_body", None)
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            source.setdefault(key, value)
    return source


def message_payloads(messages: list[JaiMessage]) -> list[dict[str, Any]]:
    """Serialize messages without dropping tool, name, image, or other fields."""
    return [message.to_payload() for message in messages]


def apply_openai_optional_params(
    payload: dict[str, Any],
    settings: dict[str, Any] | None,
    *,
    provider: str = "openai",
) -> dict[str, Any]:
    """Copy standard/provider-specific options into a raw HTTP chat request.

    Some OpenAI SDK clients express non-standard request fields through
    ``extra_body``. JanitorAI talks to this proxy over raw HTTP, so flatten that
    object instead of sending the SDK-only wrapper upstream. Unknown fields are
    retained at the intake layer but only provider-recognized options are sent
    to built-in providers, avoiding avoidable upstream 400 responses.
    """
    if not settings:
        return payload

    allowed = OPENAI_OPTIONAL_PARAMS
    if provider == "openrouter":
        allowed = allowed | OPENROUTER_OPTIONAL_PARAMS

    source = merge_extra_body(settings)

    for key in allowed:
        if key in source and key not in payload:
            payload[key] = source[key]

    # NVIDIA's GLM adapter sets a safe default for this object. Merge caller
    # values instead of silently discarding them or replacing that default.
    caller_template = source.get("chat_template_kwargs")
    if isinstance(caller_template, dict) and isinstance(
        payload.get("chat_template_kwargs"), dict
    ):
        payload["chat_template_kwargs"] = {
            **payload["chat_template_kwargs"],
            **caller_template,
        }

    # Janitor's OpenRouter mode uses ``reasoning: {effort: ...}``. OpenRouter
    # accepts ``reasoning_effort`` as a shorthand, but rejects conflicting
    # values. Keep the structured value authoritative when both are present.
    if provider == "openrouter":
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str):
            effort = reasoning.strip().lower().replace(" ", "_").replace("-", "_")
            payload["reasoning"] = {
                "effort": {"off": "none", "extra_high": "xhigh"}.get(
                    effort, effort
                )
            }
            reasoning = payload["reasoning"]
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            effort = (
                reasoning["effort"]
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            reasoning["effort"] = {"off": "none", "extra_high": "xhigh"}.get(
                effort, effort
            )
            payload.pop("reasoning_effort", None)
        elif isinstance(payload.get("reasoning_effort"), str):
            effort = (
                payload["reasoning_effort"]
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            payload["reasoning_effort"] = {"off": "none", "extra_high": "xhigh"}.get(
                effort, effort
            )

        # DeepSeek's OpenAI-compatible thinking toggle is not an OpenRouter
        # request field. Translate it to OpenRouter's native reasoning object.
        thinking = payload.pop("thinking", None)
        if isinstance(thinking, dict) and "reasoning" not in payload:
            thinking_type = thinking.get("type")
            if thinking_type == "disabled":
                payload["reasoning"] = {"effort": "none"}
            elif thinking_type == "enabled":
                effort = payload.pop("reasoning_effort", None) or "high"
                if effort == "none":
                    effort = "high"
                payload["reasoning"] = {"effort": effort}

    return payload


def get_reasoning_effort(settings: dict[str, Any] | None) -> str | None:
    """Read reasoning effort from Janitor, OpenAI, or OpenRouter request shapes."""
    if not settings:
        return None

    source = merge_extra_body(settings)

    effort = (
        source.get("reasoning_effort")
        or source.get("reasoningEffort")
        or source.get("reasoningLevel")
        or source.get("thinking_level")
        or source.get("thinkingLevel")
    )
    if isinstance(effort, str):
        normalized = effort.strip().lower().replace(" ", "_").replace("-", "_")
        return {"off": "none", "extra_high": "xhigh"}.get(normalized, normalized)

    reasoning = source.get("reasoning")
    if reasoning is None:
        reasoning = source.get("reasoningConfig")
    if isinstance(reasoning, str):
        normalized = reasoning.strip().lower().replace(" ", "_").replace("-", "_")
        return {"off": "none", "extra_high": "xhigh"}.get(normalized, normalized)
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort") or reasoning.get("level")
        if isinstance(effort, str):
            normalized = effort.strip().lower().replace(" ", "_").replace("-", "_")
            return {"off": "none", "extra_high": "xhigh"}.get(normalized, normalized)

    thinking = source.get("thinking")
    if thinking is None:
        thinking = source.get("thinkingConfig") or source.get("thinking_config")
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            return "none"
        if thinking.get("type") == "enabled":
            return "high"
        level = thinking.get("thinkingLevel") or thinking.get("level")
        if isinstance(level, str):
            normalized = level.strip().lower().replace(" ", "_").replace("-", "_")
            return {"off": "none", "extra_high": "xhigh"}.get(
                normalized, normalized
            )

    return None


def deepseek_reasoning_params(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Translate Janitor reasoning controls to DeepSeek's Chat Completions API."""
    if not settings:
        return {}

    source = merge_extra_body(settings)

    result: dict[str, Any] = {}
    thinking = source.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in {"enabled", "disabled"}:
        thinking_type = thinking["type"]
        result["thinking"] = {"type": thinking_type}

    effort = get_reasoning_effort(source)
    if effort is not None:
        effort_map = {
            "none": "none",
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "xhigh": "high",
            "max": "max",
        }
        mapped_effort = effort_map.get(effort, effort)
        thinking_type = result.get("thinking", {}).get("type")
        if thinking_type == "disabled":
            mapped_effort = "none"
        elif thinking_type == "enabled" and mapped_effort == "none":
            mapped_effort = "high"
        result["reasoning_effort"] = mapped_effort
        if "thinking" not in result:
            result["thinking"] = {
                "type": "disabled" if mapped_effort == "none" else "enabled"
            }

    return result


def gemini_thinking_config(
    model: str, settings: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Translate Janitor's levels to model-valid Gemini thinking settings.

    Gemini 2.5 uses a numeric thinking budget; Gemini 3+ uses discrete levels.
    Gemini 3+ cannot turn thinking fully off. Unsupported levels are clamped to
    the closest supported level for the specific model family.
    """
    effort = get_reasoning_effort(settings)
    if effort is None:
        return None

    model_id = model.lower()
    if "gemini-2.5" in model_id:
        if "pro" in model_id:
            # Gemini 2.5 Pro cannot disable thinking; its minimum budget is 128.
            budget_map = {
                "none": 128,
                "minimal": 128,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 32768,
                "max": 32768,
            }
        elif "flash-lite" in model_id:
            budget_map = {
                "none": 0,
                "minimal": 512,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 24576,
                "max": 24576,
            }
        else:
            budget_map = {
                "none": 0,
                "minimal": 1,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 24576,
                "max": 24576,
            }
        return {"thinkingBudget": budget_map.get(effort, 8192)}

    if any(marker in model_id for marker in ("gemini-3", "gemini-4")):
        # Model families have different supported levels. In particular, Gemini
        # 4 does not support MINIMAL; only Neon and Hydrogen accept XHIGH.
        if "gemini-4" in model_id:
            supported = ["LOW", "MEDIUM", "HIGH"]
            if "neon" in model_id or "hydrogen" in model_id:
                supported.append("XHIGH")
        elif "gemini-3.8" in model_id or "gemini-3.7" in model_id:
            supported = ["LOW", "MEDIUM", "HIGH"]
        elif "gemini-3.1-pro" in model_id:
            supported = ["LOW", "MEDIUM", "HIGH"]
        elif any(
            marker in model_id
            for marker in ("gemini-3.1-flash-lite-image", "gemini-3.1-flash-image")
        ):
            supported = ["MINIMAL", "HIGH"]
        elif "gemini-3-pro-image" in model_id:
            supported = ["HIGH"]
        else:
            supported = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]

        desired = {
            "none": "MINIMAL",
            "minimal": "MINIMAL",
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
            "xhigh": "XHIGH",
            "max": "XHIGH",
        }.get(effort, "HIGH")

        order = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"]
        if desired not in supported:
            # Clamp an unsupported level down to the nearest supported level.
            desired_index = order.index(desired)
            lower = [
                level for level in supported if order.index(level) <= desired_index
            ]
            desired = lower[-1] if lower else supported[0]
        return {"thinkingLevel": desired}

    return None


def gemini_parts(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-style text/image content into Gemini Part objects."""
    import base64
    import re

    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return []

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            parts.append({"text": item["text"]})
            continue
        if item_type == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str):
                continue
            if url.startswith("data:"):
                match = re.match(r"^data:([^;,]+);base64,(.+)$", url, re.DOTALL)
                if not match:
                    continue
                mime_type, encoded = match.groups()
                try:
                    base64.b64decode(encoded, validate=True)
                except (ValueError, base64.binascii.Error):
                    continue
                parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
            continue
        # Native Gemini parts already have a compatible shape.
        if any(
            key in item
            for key in (
                "text", "inlineData", "fileData", "functionCall", "functionResponse"
            )
        ):
            parts.append(item)

    return parts


def extract_text_content(value: Any) -> str:
    """Extract text-only output from OpenAI-compatible content values."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for part in value:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""
