from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..streaming import openai_chat_completion
from ..xuiduser import XUID


def groq_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Groq's OpenAI-compatible Chat Completions API.

    The model is supplied by JanitorAI through the normal provider/model
    syntax; this provider does not select or hard-code a model.
    """

    stream = bool((settings or {}).get("stream", False))

    groq_request = {
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
            groq_request["temperature"] = value
        elif key == "max_tokens":
            # Groq recommends max_completion_tokens; max_tokens is deprecated.
            groq_request["max_completion_tokens"] = value
        elif key == "top_p":
            groq_request["top_p"] = value
        elif key == "frequency_penalty":
            # Groq currently documents this parameter but does not support it
            # on its models. Do not send it and cause a 400.
            continue
        elif key == "repetition_penalty":
            # JanitorAI exposes repetition_penalty, while Groq does not
            # currently support an equivalent parameter. Do not incorrectly
            # translate it to presence_penalty, which also causes unsupported
            # behavior on Groq models.
            continue

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('groq/').strip()}",
        "Content-Type": "application/json",
    }

    try:
        groq_result = openai_chat_completion(
            "https://api.groq.com/openai/v1/chat/completions",
            request=groq_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        if stream:
            return JaiResult(200, "", stream=groq_result)

    except httpx2.TimeoutException:
        track_stats("groq.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message = "Error from Groq"
        extras = ""

        try:
            error = e.response.json()
        except Exception:  # ruff: ignore[BLE001]
            error = None

        if isinstance(error, dict):
            if isinstance(error.get("error"), dict):
                error = error["error"]

            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {error_message}"
            if error_type := error.get("type"):
                extras = f"Groq error type: {error_type}"
        else:
            xlog(user, f"{message}: {e.response.text!r}")
            extras = e.response.text

        if e.response.is_client_error:
            track_stats("groq.failed.client")
        elif e.response.is_server_error:
            track_stats("groq.failed.server")
        else:
            track_stats("groq.failed.unknown")

        return JaiResult(e.response.status_code, message, extras=extras)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("groq.failed.exception")
        return JaiResult(502, "Unhandled exception from Groq.")

    try:
        text = str(groq_result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()
    if usage := groq_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens"
            ),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        xlog(user, f"No result text: {groq_result!r}")
        track_stats("groq.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("groq.succeeded")
    return JaiResult(200, text, metadata=metadata)
