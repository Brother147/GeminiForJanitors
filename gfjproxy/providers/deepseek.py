from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..streaming import openai_chat_completion
from ..utils import safe_response_json
from ..xuiduser import XUID


def deepseek_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around DeepSeek's Chat Completions API.

    User paramater is only used for logging."""

    stream = bool((settings or {}).get("stream", False))

    deepseek_request = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in messages
        ],
    }

    for key, value in (settings or {}).items():
        if key == "temperature":
            deepseek_request["temperature"] = value
        elif key == "max_tokens":
            deepseek_request["max_tokens"] = value
        elif key == "top_p":
            deepseek_request["top_p"] = value
        elif key == "frequency_penalty":
            deepseek_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            # Preserve JanitorAI's historical setting mapping.
            deepseek_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        deepseek_result = openai_chat_completion(
            "https://api.deepseek.com/chat/completions",
            request=deepseek_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        if stream:
            return JaiResult(200, "", stream=deepseek_result)

    except httpx2.TimeoutException:
        track_stats("deepseek.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from DeepSeek"
        extras = ""

        error = safe_response_json(e.response)
        if isinstance(error, dict) and error:
            if "error" in error:
                error = error["error"]
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        else:
            xlog(user, f"{message}: {error!r}")

        if e.response.is_client_error:
            track_stats("deepseek.failed.client")
        elif e.response.is_server_error:
            track_stats("deepseek.failed.server")
        else:
            track_stats("deepseek.failed.unknown")

        return JaiResult(e.response.status_code, message, extras=extras)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("deepseek.failed.exception")
        return JaiResult(502, "Unhandled exception from DeepSeek.")

    if not isinstance(deepseek_result, dict):
        xlog(user, f"Invalid response shape from DeepSeek: {type(deepseek_result).__name__}")
        track_stats("deepseek.rejected")
        return JaiResult(502, "Invalid response from DeepSeek.")

    try:
        text = str(deepseek_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := deepseek_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens"
            ),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {deepseek_result!r}")
        track_stats("deepseek.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("deepseek.succeeded")
    return JaiResult(200, text, metadata=metadata)
