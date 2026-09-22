from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT, PROXY_NAME, PROXY_URL
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult
from ..streaming import openai_chat_completion
from ..xuiduser import XUID


def proxy_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Forwards the request to another URL.

    This provider is for testing only.
    User paramater is only used for logging."""

    try:
        api_key, url = api_key.split("@", maxsplit=1)
    except ValueError:
        return JaiResult(400, "Proxy API key must use the format <api_key>@<url>")

    stream = bool((settings or {}).get("stream", False))

    proxy_request = {
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
            proxy_request["temperature"] = value
        elif key == "max_tokens":
            proxy_request["max_tokens"] = value
        elif key == "top_k":
            proxy_request["top_k"] = value
        elif key == "top_p":
            proxy_request["top_p"] = value
        elif key == "frequency_penalty":
            proxy_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            # Preserve JanitorAI's historical setting mapping.
            proxy_request["presence_penalty"] = value

    headers = {
        "Accept": "text/event-stream" if stream else "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": PROXY_URL,
        "X-Title": PROXY_NAME,
    }

    try:
        if stream:
            proxy_result = openai_chat_completion(
                url,
                request=proxy_request,
                headers=headers,
                timeout=PROCESS_TIMEOUT,
            )
            return JaiResult(200, "", stream=proxy_result)

        proxy_response = http_client.post(
            url,
            json=proxy_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )
        proxy_response.raise_for_status()
    except httpx2.TimeoutException:
        return JaiResult(504, "Gateway Timeout")
    except httpx2.HTTPStatusError as e:
        return JaiResult(e.response.status_code, e.response.text)
    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        return JaiResult(502, "Unhandled exception from proxy.")

    return JaiResult(proxy_response.status_code, proxy_response.text)
