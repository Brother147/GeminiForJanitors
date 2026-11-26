from ipaddress import ip_address
from socket import SOCK_STREAM, getaddrinfo
from typing import Any
from urllib.parse import urlsplit

import httpx2

from .._globals import (
    PROCESS_TIMEOUT,
    PROXY_NAME,
    PROXY_URL,
    TEST_PROXY_ALLOWED_HOSTS,
)
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult
from ..streaming import openai_chat_completion
from ..xuiduser import XUID
from .openai_compat import apply_openai_optional_params, message_payloads


def _validate_target_url(url: str) -> str | None:
    """Return a safe error message when a test-proxy target is not trusted.

    This provider is deliberately opt-in and host-allowlisted to avoid turning
    the public proxy into an SSRF relay.
    """
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port if parsed.port is not None else 443
    except ValueError:
        return "Test proxy URL is malformed."

    if parsed.scheme.lower() != "https" or not host:
        return "Test proxy URLs must use HTTPS."
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return "Test proxy URL must not contain user information or a fragment."
    if host not in TEST_PROXY_ALLOWED_HOSTS:
        return "Test proxy target host is not in GFJPROXY_TEST_PROXY_ALLOWED_HOSTS."
    # Restrict test forwarding to the standard HTTPS port. This keeps the
    # allowlist narrowly scoped and avoids exposing arbitrary network services.
    if port != 443:
        return "Test proxy URLs must use the standard HTTPS port (443)."
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return "Test proxy cannot target local or internal hosts."

    try:
        addresses = {ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ip_address(result[4][0])
                for result in getaddrinfo(host, port, type=SOCK_STREAM)
            }
        except (OSError, ValueError):
            return "Test proxy target host could not be resolved safely."

    if not addresses or any(not address.is_global for address in addresses):
        return "Test proxy cannot target private, loopback, or reserved IP addresses."
    return None


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

    if (
        not api_key
        or len(api_key) > 4096
        or any(char in api_key for char in "\r\n\x00")
    ):
        return JaiResult(400, "Proxy API key is missing or invalid.")

    if error := _validate_target_url(url):
        return JaiResult(400, error)

    stream = bool((settings or {}).get("stream", False))

    proxy_request = {
        "model": model,
        "stream": stream,
        "messages": message_payloads(messages),
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

    apply_openai_optional_params(proxy_request, settings)
    if settings:
        for key, value in settings.items():
            if key == "extra_body" and isinstance(value, dict):
                for extra_key, extra_value in value.items():
                    proxy_request.setdefault(extra_key, extra_value)
            elif key not in {
                "stream", "temperature", "max_tokens", "top_k", "top_p",
                "frequency_penalty", "repetition_penalty", "extra_body",
            }:
                proxy_request.setdefault(key, value)

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
