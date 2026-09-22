"""Helpers for streaming OpenAI-compatible and Gemini SSE responses."""

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, suppress
from typing import Any

import httpx2

from .http_client import http_client


class StreamingHTTPError(RuntimeError):
    """Raised when an upstream streaming connection fails."""

    def __init__(self, status_code: int | None = None):
        self.status_code = status_code
        if status_code is None:
            message = "Upstream provider stream failed"
        else:
            message = f"Upstream provider returned HTTP {status_code}"
        super().__init__(message)


class _ManagedStream(Iterator[str]):
    """Lazy HTTP stream with explicit ownership and cleanup.

    The upstream request is intentionally opened on the first ``next()`` rather
    than while building the Flask response.  This allows the downstream SSE
    response to start immediately even when a model needs a long time before
    sending its first token.
    """

    def __init__(
        self,
        open_factory: Callable[[], tuple[AbstractContextManager, Any]],
        iterator_factory: Callable[[Any], Iterator[str]],
    ):
        self._open_factory = open_factory
        self._iterator_factory = iterator_factory
        self._context: AbstractContextManager | None = None
        self._iterator: Iterator[str] | None = None
        self._closed = False

    def __iter__(self) -> "_ManagedStream":
        return self

    def _ensure_started(self) -> None:
        if self._iterator is not None or self._closed:
            return

        context, response = self._open_factory()
        self._context = context
        try:
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            with suppress(Exception):
                response.read()
            self.close()
            raise StreamingHTTPError(exc.response.status_code) from exc

        self._iterator = self._iterator_factory(response)

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration

        try:
            self._ensure_started()
            if self._iterator is None:
                raise StopIteration
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except httpx2.HTTPError as exc:
            self.close()
            raise StreamingHTTPError() from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

        if self._context is not None:
            _close_stream(self._context)
            self._context = None


def _decode_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return line


def _open_context(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[AbstractContextManager, Any]:
    stream_headers = dict(headers)
    stream_headers.setdefault("Accept", "text/event-stream")
    context = http_client.stream(
        "POST", url, json=request, headers=stream_headers, timeout=timeout
    )
    try:
        return context, context.__enter__()
    except BaseException:
        with suppress(Exception):
            context.__exit__(None, None, None)
        raise


def _close_stream(context: AbstractContextManager) -> None:
    with suppress(Exception):
        context.__exit__(None, None, None)


def _extract_openai_content(choice: dict[str, Any]) -> str | None:
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    text = choice.get("text")
    return text if isinstance(text, str) else None


def _iter_openai_text(response: Any) -> Iterator[str]:
    for raw_line in response.iter_lines():
        line = _decode_line(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            return

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if not isinstance(chunk, dict):
            continue

        content_found = False
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            content = _extract_openai_content(choice)
            if content:
                content_found = True
                yield content

        if not content_found:
            # Reasoning-only/non-text events are still useful as a heartbeat.
            yield ""


def _iter_gemini_text(response: Any) -> Iterator[str]:
    for raw_line in response.iter_lines():
        line = _decode_line(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            return

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if not isinstance(chunk, dict):
            continue

        content_found = False
        for candidate in chunk.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict) or part.get("thought", False):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    content_found = True
                    yield text

        if not content_found:
            yield ""


def _managed_stream(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    parser: Callable[[Any], Iterator[str]],
) -> _ManagedStream:
    return _ManagedStream(
        lambda: _open_context(
            url, request=request, headers=headers, timeout=timeout
        ),
        parser,
    )


def openai_chat_completion(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any] | Iterator[str]:
    """Send an OpenAI-compatible chat request."""

    if not request.get("stream"):
        response = http_client.post(url, json=request, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    return _managed_stream(
        url,
        request=request,
        headers=headers,
        timeout=timeout,
        parser=_iter_openai_text,
    )


def gemini_sse_completion(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """Stream text deltas from a Gemini-compatible SSE endpoint."""

    return _managed_stream(
        url,
        request=request,
        headers=headers,
        timeout=timeout,
        parser=_iter_gemini_text,
    )
