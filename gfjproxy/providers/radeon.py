from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID

RADEON_CHAT_COMPLETIONS_URL = (
    "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
)


def _redact_api_key(value: str, api_key: str) -> str:
    return value.replace(api_key, "[REDACTED]") if api_key else value


def _log(user: XUID | str | None, message: str) -> None:
    """Log with a valid XUID while keeping provider tests/callers robust."""
    xlog(user if isinstance(user, XUID) else None, message)


def _extract_error(
    user: XUID, response: httpx2.Response, api_key: str
) -> tuple[str, str]:
    """Extract a useful AMD error message without exposing credentials."""
    message = "Error from AMD Radeon Cloud"
    extras = ""

    try:
        error = response.json()
    except Exception:  # ruff: ignore[BLE001]
        body = _redact_api_key(response.text, api_key)
        if body:
            message += f": {body}"
        return message, extras

    if isinstance(error, dict):
        detail = error.get("detail")
        if isinstance(error.get("error"), dict):
            error = error["error"]
        elif isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            error = detail["error"]
        elif isinstance(detail, str):
            message += f": {_redact_api_key(detail, api_key)}"
            error = None

        if isinstance(error, dict):
            if error_code := error.get("code"):
                message += f" ({error_code})"
            if error_message := error.get("message"):
                message += f": {_redact_api_key(str(error_message), api_key)}"
            if error_type := error.get("type"):
                extras = (
                    "AMD Radeon Cloud error type: "
                    f"{_redact_api_key(str(error_type), api_key)}"
                )

    if message == "Error from AMD Radeon Cloud":
        _log(user, f"{message}: {_redact_api_key(response.text, api_key)!r}")

    return message, extras


def radeon_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around AMD Radeon Cloud's OpenAI-compatible Chat Completions API.

    The model is supplied by JanitorAI using the normal provider/model syntax,
    e.g. ``radeon/DeepSeek-V4-Flash``. The request reaches AMD with the model
    name after the provider prefix, matching the Groq provider's architecture.
    """

    radeon_request = {
        "model": model,
        "stream": False,
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
            radeon_request["temperature"] = value
        elif key == "max_tokens":
            radeon_request["max_tokens"] = value
        elif key == "top_p":
            radeon_request["top_p"] = value
        elif key == "frequency_penalty":
            # Radeon Cloud's shared endpoint does not document this field.
            # Do not send unsupported settings.
            continue
        elif key == "repetition_penalty":
            # AMD Radeon Cloud does not document repetition_penalty.
            # Do not translate it to presence_penalty: they are not equivalent.
            continue

    headers = {
        "Authorization": f"Bearer {api_key.removeprefix('radeon/').strip()}",
        "Content-Type": "application/json",
    }

    try:
        radeon_response = http_client.post(
            RADEON_CHAT_COMPLETIONS_URL,
            json=radeon_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        radeon_response.raise_for_status()
        try:
            radeon_result = radeon_response.json()
        except (ValueError, TypeError):
            _log(
                user,
                f"Invalid AMD response JSON: "
                f"{_redact_api_key(radeon_response.text, api_key)!r}",
            )
            track_stats("radeon.failed.invalid_response")
            return JaiResult(502, "Invalid response from AMD Radeon Cloud.")
    except httpx2.TimeoutException:
        track_stats("radeon.time_out")
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        message, extras = _extract_error(user, e.response, api_key)

        if e.response.is_client_error:
            track_stats("radeon.failed.client")
        elif e.response.is_server_error:
            track_stats("radeon.failed.server")
        else:
            track_stats("radeon.failed.unknown")

        return JaiResult(e.response.status_code, message, extras=extras)
    except Exception as e:  # ruff: ignore[BLE001]
        _log(user, _redact_api_key(repr(e), api_key))
        track_stats("radeon.failed.exception")
        return JaiResult(502, "Unhandled exception from AMD Radeon Cloud.")

    if not isinstance(radeon_result, dict):
        _log(
            user,
            "Unexpected AMD response: "
            f"{_redact_api_key(repr(radeon_result), api_key)}",
        )
        track_stats("radeon.failed.invalid_response")
        return JaiResult(502, "Invalid response from AMD Radeon Cloud.")

    metadata = JaiResultMetadata()
    usage = radeon_result.get("usage")
    if isinstance(usage, dict):
        reasoning_tokens = usage.get("reasoning_tokens")
        if reasoning_tokens is None and isinstance(
            completion_details := usage.get("completion_tokens_details"), dict
        ):
            reasoning_tokens = completion_details.get("reasoning_tokens")

        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage.get("total_tokens"),
        )

    choices = radeon_result.get("choices")
    if not isinstance(choices, list):
        _log(
            user,
            "Unexpected AMD response choices: "
            f"{_redact_api_key(repr(choices), api_key)}",
        )
        track_stats("radeon.failed.invalid_response")
        return JaiResult(
            502, "Invalid response from AMD Radeon Cloud.", metadata=metadata
        )

    if not choices:
        text = ""
    else:
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            text = ""
        else:
            message = first_choice.get("message")
            if not isinstance(message, dict):
                text = ""
            else:
                text = str(message.get("content") or "")

    if not text:
        _log(
            user,
            f"No result text: {_redact_api_key(repr(radeon_result), api_key)}",
        )
        track_stats("radeon.rejected")
        return JaiResult(502, "Response blocked/empty.", metadata=metadata)

    track_stats("radeon.succeeded")
    return JaiResult(200, text, metadata=metadata)
