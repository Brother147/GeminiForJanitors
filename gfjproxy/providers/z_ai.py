from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..streaming import openai_chat_completion
from ..utils import safe_response_json
from ..xuiduser import XUID


def z_ai_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Z.AI's Chat Completions API.
    API keys must be prefixed with "z_ai/" to help the handlers distinguish them,
    as official Z.AI's API keys don't have any prefix.

    User paramater is only used for logging."""

    stream = bool((settings or {}).get("stream", False))

    z_ai_request = {
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

    # Z.AI's current OpenAI-compatible API supports these sampling controls
    # for current GLM models. Parameters are forwarded unchanged below.

    for key, value in (settings or {}).items():
        if key == "temperature":
            z_ai_request["temperature"] = value
        elif key == "max_tokens":
            z_ai_request["max_tokens"] = value
        elif key == "top_k":
            z_ai_request["top_k"] = value
        elif key == "top_p":
            z_ai_request["top_p"] = value
        elif key == "frequency_penalty":
            z_ai_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            z_ai_request["repetition_penalty"] = value

    try:
        z_ai_result = openai_chat_completion(
            "https://api.z.ai/api/paas/v4/chat/completions",
            request=z_ai_request,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=PROCESS_TIMEOUT,
        )
        if stream:
            return JaiResult(200, "", stream=z_ai_result)

    except httpx2.TimeoutException:
        track_stats("z_ai.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Z.AI"

        response_json = safe_response_json(e.response)
        error = response_json.get("error") if isinstance(response_json, dict) else None
        if isinstance(error, dict):
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
        else:
            xlog(user, f"{message}: {e.response.text!r}")

        if e.response.is_client_error:
            track_stats("z_ai.failed.client")
        elif e.response.is_server_error:
            track_stats("z_ai.failed.server")
        else:
            track_stats("z_ai.failed.unknown")

        return JaiResult(e.response.status_code, message)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("z_ai.failed.exception")
        return JaiResult(502, "Unhandled exception from Z.AI.")

    text = ""
    extras = ""
    metadata = JaiResultMetadata()

    if not isinstance(z_ai_result, dict):
        xlog(user, f"Invalid response shape from Z.AI: {type(z_ai_result).__name__}")
        track_stats("z_ai.rejected")
        return JaiResult(502, "Invalid response from Z.AI.", metadata=metadata)

    message = {}

    if (
        isinstance(z_ai_result, dict)
        and isinstance(choices := z_ai_result.get("choices"), list)
        and len(choices) > 0
        and isinstance(choices[0], dict)
        and isinstance(message := choices[0].get("message"), dict)
    ):
        text = message.get("content", "")

    if usage := z_ai_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens"
            ),
            total_tokens=usage.get("total_tokens"),
        )

    if not text and isinstance(message, dict):
        # The `//think on` jailbreak seems to make Z.AI models spill
        # the response into "reasoning_content" instead of "content".
        # Hopefully handlers.py can handle this down the line.
        text = message.get("reasoning_content", "")
        if "<response>" not in text and "</response>" not in text:
            extras = "Z.AI returned anomalous response. The `//think` command might cause this."

    if not text:
        # Rejection?
        xlog(user, f"No result text: {z_ai_result!r}")
        track_stats("z_ai.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("z_ai.succeeded")
    return JaiResult(200, text, extras=extras, metadata=metadata)
