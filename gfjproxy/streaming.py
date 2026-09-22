import json
from collections.abc import Iterator
from typing import Any

from .http_client import http_client


def _decode_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return line


def openai_chat_completion(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any] | Iterator[str]:
    """Send an OpenAI-compatible chat request.

    Non-streaming requests return the normal decoded JSON response.
    Streaming requests return a lazy iterator of text deltas which keeps the
    upstream HTTP connection open while the caller forwards the chunks.
    """
    if not request.get("stream"):
        response = http_client.post(url, json=request, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def generate() -> Iterator[str]:
        with http_client.stream(
            "POST", url, json=request, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                line = _decode_line(raw_line).strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content

    return generate()


def gemini_sse_completion(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """Stream text deltas from Google's Gemini SSE endpoint."""

    def generate() -> Iterator[str]:
        with http_client.stream(
            "POST", url, json=request, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                line = _decode_line(raw_line).strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for candidate in chunk.get("candidates", []):
                    if not isinstance(candidate, dict):
                        continue
                    content = candidate.get("content")
                    if not isinstance(content, dict):
                        continue
                    for part in content.get("parts", []):
                        if not isinstance(part, dict):
                            continue
                        if part.get("thought", False):
                            continue
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            yield text

    return generate()
