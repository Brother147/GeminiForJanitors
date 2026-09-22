from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT, PROXY_NAME, PROXY_URL
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..streaming import openai_chat_completion
from ..utils import safe_response_json
from ..xuiduser import XUID


def openrouter_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around OpenRouter's Chat Completions API.

    User paramater is only used for logging."""

    stream = bool((settings or {}).get("stream", False))

    openrouter_request = {
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
            openrouter_request["temperature"] = value
        elif key == "max_tokens":
            openrouter_request["max_tokens"] = value
        elif key == "top_k":
            openrouter_request["top_k"] = value
        elif key == "top_p":
            openrouter_request["top_p"] = value
        elif key == "frequency_penalty":
            openrouter_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            # Preserve JanitorAI's historical setting mapping.
            openrouter_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": PROXY_URL,
        "X-Title": PROXY_NAME,
    }

    try:
        openrouter_result = openai_chat_completion(
            "https://openrouter.ai/api/v1/chat/completions",
            request=openrouter_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        if stream:
            return JaiResult(200, "", stream=openrouter_result)

    except httpx2.TimeoutException:
        track_stats("openrouter.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from OpenRouter"
        extras = ""

        response_json = safe_response_json(e.response)
        if isinstance(response_json, dict) and isinstance(
            error := response_json.get("error"), dict
        ):
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
            if error_metadata := error.get("metadata"):
                if isinstance(error_metadata, dict):
                    extras = str(error_metadata.get("raw", ""))
                else:
                    extras = str(error_metadata)
        else:
            xlog(user, f"{message}: {e.response.text!r}")

        if e.response.is_client_error:
            track_stats("openrouter.failed.client")
        elif e.response.is_server_error:
            track_stats("openrouter.failed.server")
        else:
            track_stats("openrouter.failed.unknown")

        return JaiResult(e.response.status_code, message, extras=extras)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("openrouter.failed.exception")
        return JaiResult(502, "Unhandled exception from OpenRouter.")

    if not isinstance(openrouter_result, dict):
        xlog(user, f"Invalid response shape from OpenRouter: {type(openrouter_result).__name__}")
        track_stats("openrouter.failed.anomalous")
        return JaiResult(502, "Invalid response from OpenRouter.")

    if isinstance(error := openrouter_result.get("error"), dict) and error:
        message = "Error from OpenRouter"
        if error_code := error.get("code"):
            message += f" ({error_code})"
        if error_message := error.get("message"):
            message += f": {error_message}"
        xlog(user, f"Error despite successful status: {openrouter_result!r}")
        track_stats("openrouter.failed.anomalous")
        return JaiResult(502, message)

    try:
        text = str(openrouter_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := openrouter_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {openrouter_result!r}")
        track_stats("openrouter.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("openrouter.succeeded")
    return JaiResult(200, text, metadata=metadata)
