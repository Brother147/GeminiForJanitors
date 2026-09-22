"""Helpers for streaming OpenAI-compatible and Gemini SSE responses."""

import json
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any, Callable

import httpx2

from .http_client import http_client


class StreamingHTTPError(RuntimeError):
    """Raised when an upstream streaming connection fails after opening."""


class _ManagedStream(Iterator[str]):
    """Iterator that owns an already-open HTTP streaming context.

    A plain generator is not enough here: calling ``close()`` on a generator
    that has never been started does not execute its ``finally`` block. The
    downstream Flask generator can be closed immediately after its first SSE
    heartbeat, so the HTTP context needs an explicit close method that works
    even when ``__next__`` has never run.
    """

    def __init__(
        self,
        context: AbstractContextManager,
        iterator_factory: Callable[[], Iterator[str]],
    ):
        self._context = context
        self._iterator = iterator_factory()
        self._closed = False

    def __iter__(self) -> "_ManagedStream":
        return self

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration

        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except httpx2.HTTPError as exc:
            self.close()
            raise StreamingHTTPError("Upstream provider stream failed") from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        close = getattr(self._iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - cleanup only
                pass

        _close_stream(self._context)


def _decode_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return line


def _open_stream(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[AbstractContextManager, Any]:
    """Open an upstream HTTP stream and validate its status before returning."""

    stream_headers = dict(headers)
    stream_headers.setdefault("Accept", "text/event-stream")

    context = http_client.stream(
        "POST", url, json=request, headers=stream_headers, timeout=timeout
    )

    try:
        response = context.__enter__()
        response.raise_for_status()
    except BaseException:
        try:
            context.__exit__(*sys.exc_info())
        except Exception:  # pragma: no cover - defensive cleanup only
            pass
        raise

    return context, response


def _close_stream(context: AbstractContextManager) -> None:
    try:
        context.__exit__(None, None, None)
    except Exception:
        # Cleanup must never mask the original streaming/disconnect result.
        pass


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
        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_found = True
                yield content

        if not content_found:
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

    context, response = _open_stream(
        url,
        request=request,
        headers=headers,
        timeout=timeout,
    )

    return _ManagedStream(context, lambda: _iter_openai_text(response))


def gemini_sse_completion(
    url: str,
    *,
    request: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """Stream text deltas from a Gemini-compatible SSE endpoint."""

    context, response = _open_stream(
        url,
        request=request,
        headers=headers,
        timeout=timeout,
    )

    return _ManagedStream(context, lambda: _iter_gemini_text(response))
