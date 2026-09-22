from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..streaming import openai_chat_completion
from ..utils import safe_response_json
from ..xuiduser import XUID


def cerebras_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Cerebras' Chat Completions API.

    User paramater is only used for logging."""

    stream = bool((settings or {}).get("stream", False))

    cerebras_request = {
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

    # As of April 6, 2026, Cerebras does not support top_k
    # Pass the value(s) anyway and let the user get a relevant error

    for key, value in (settings or {}).items():
        if key == "temperature":
            cerebras_request["temperature"] = value
        elif key == "max_tokens":
            cerebras_request["max_completion_tokens"] = value
        elif key == "top_k":
            cerebras_request["top_k"] = value
        elif key == "top_p":
            cerebras_request["top_p"] = value
        elif key == "frequency_penalty":
            cerebras_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            cerebras_request["presence_penalty"] = value

    try:
        cerebras_result = openai_chat_completion(
            "https://api.cerebras.ai/v1/chat/completions",
            request=cerebras_request,
            headers={"Authorization": f"Bearer {api_key.removeprefix('cerebras/')}"},
            timeout=PROCESS_TIMEOUT,
        )
        if stream:
            return JaiResult(200, "", stream=cerebras_result)

    except httpx2.TimeoutException:
        track_stats("cerebras.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Cerebras"

        error = safe_response_json(e.response)
        if isinstance(error, dict) and error:
            if isinstance(error.get("error"), dict):
                error = error["error"]

            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        else:
            xlog(user, f"{message}: {e.response.text!r}")

        if e.response.is_client_error:
            track_stats("cerebras.failed.client")
        elif e.response.is_server_error:
            track_stats("cerebras.failed.server")
        else:
            track_stats("cerebras.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("cerebras.failed.exception")
        return JaiResult(502, "Unhandled exception from Cerebras.")

    try:
        text = str(cerebras_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := cerebras_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        # Rejection?
        xlog(user, f"No result text: {cerebras_result!r}")
        track_stats("cerebras.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("cerebras.succeeded")
    return JaiResult(200, text, metadata=metadata)
