"""Helpers for streaming OpenAI-compatible and Gemini SSE responses."""

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, suppress
from queue import Empty, Full, Queue
from typing import Any

import httpx2

from .http_client import http_client

STREAM_HEARTBEAT_INTERVAL = 10.0
_QUEUE_END = object()
_QUEUE_ERROR = object()


class StreamingHTTPError(RuntimeError):
    """Raised when an upstream streaming connection fails."""

    def __init__(self, status_code: int | None = None):
        self.status_code = status_code
        if status_code is None:
            message = "Upstream provider stream failed"
        else:
            message = f"Upstream provider returned HTTP {status_code}"
        super().__init__(message)


class StreamingProviderError(RuntimeError):
    """Raised when an upstream SSE event reports an application-level error."""


class StreamingTimeoutError(RuntimeError):
    """Raised when an upstream streaming request times out."""


class _ManagedStream(Iterator[str]):
    """Lazy HTTP stream with explicit ownership, heartbeats, and cleanup.

    The upstream request is opened on the first ``next()`` but in a background
    worker. The consumer can therefore emit a heartbeat while connection setup,
    response headers, or the provider's first token are still pending.
    """

    def __init__(
        self,
        open_factory: Callable[[], tuple[AbstractContextManager, Any]],
        iterator_factory: Callable[[Any], Iterator[str]],
        *,
        heartbeat_interval: float = STREAM_HEARTBEAT_INTERVAL,
    ):
        self._open_factory = open_factory
        self._iterator_factory = iterator_factory
        self._heartbeat_interval = heartbeat_interval
        self._context: AbstractContextManager | None = None
        self._iterator: Iterator[str] | None = None
        self._worker: threading.Thread | None = None
        self._queue: Queue[tuple[object, Any]] = Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._closed = False
        self._started = False
        self._state_lock = threading.Lock()

    def __iter__(self) -> "_ManagedStream":
        return self

    def _ensure_started(self) -> None:
        with self._state_lock:
            if self._closed or self._started:
                return
            self._started = True
            self._worker = threading.Thread(
                target=self._pump,
                name="gfjproxy-stream",
                daemon=True,
            )
            self._worker.start()

    def _put(self, kind: object, value: Any) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put((kind, value), timeout=0.25)
                return
            except Full:
                continue

    def _install_context(self, context: AbstractContextManager) -> bool:
        with self._state_lock:
            if self._closed or self._stop_event.is_set():
                return False
            self._context = context
            return True

    def _install_upstream(
        self, context: AbstractContextManager, iterator: Iterator[str]
    ) -> bool:
        with self._state_lock:
            if self._closed or self._stop_event.is_set():
                return False
            self._context = context
            self._iterator = iterator
            return True

    def _pump(self) -> None:
        context: AbstractContextManager | None = None
        iterator: Iterator[str] | None = None

        try:
            context, response = self._open_factory()
            if self._stop_event.is_set():
                _close_stream(context)
                return

            if not self._install_context(context):
                _close_stream(context)
                return

            try:
                response.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                with suppress(Exception):
                    response.read()
                status_code = getattr(exc.response, "status_code", None)
                self._put(
                    _QUEUE_ERROR,
                    StreamingHTTPError(
                        status_code if isinstance(status_code, int) else None
                    ),
                )
                return

            if self._stop_event.is_set():
                return

            iterator = self._iterator_factory(response)
            if not self._install_upstream(context, iterator):
                close = getattr(iterator, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
                _close_stream(context)
                return

            for chunk in iterator:
                if self._stop_event.is_set():
                    break
                self._put(None, chunk)
        except httpx2.TimeoutException:
            if not self._stop_event.is_set():
                self._put(_QUEUE_ERROR, StreamingTimeoutError())
        except httpx2.HTTPError:
            if not self._stop_event.is_set():
                self._put(_QUEUE_ERROR, StreamingHTTPError())
        except Exception as exc:  # noqa: BLE001 - propagate provider/parser errors
            if not self._stop_event.is_set():
                self._put(_QUEUE_ERROR, exc)
        finally:
            if not self._stop_event.is_set():
                self._put(_QUEUE_END, None)

    def __next__(self) -> str:
        with self._state_lock:
            if self._closed:
                raise StopIteration

        self._ensure_started()

        while True:
            try:
                kind, value = self._queue.get(timeout=self._heartbeat_interval)
            except Empty:
                # Empty text is intentionally used by the response layer as an
                # explicit downstream heartbeat.
                return ""

            if kind is _QUEUE_END:
                self.close()
                raise StopIteration
            if kind is _QUEUE_ERROR:
                self.close()
                if isinstance(value, BaseException):
                    raise value
                raise RuntimeError("Unknown streaming provider error")
            return value

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            iterator = self._iterator
            context = self._context
            worker = self._worker
            self._iterator = None
            self._context = None
            self._worker = None

        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

        if context is not None:
            _close_stream(context)

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)


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


def _raise_sse_error(chunk: dict[str, Any], provider: str) -> None:
    error = chunk.get("error")
    if not isinstance(error, dict):
        return
    code = error.get("code")
    message = error.get("message")
    details = ""
    if code:
        details += f" ({code})"
    if message:
        details += f": {message}"
    raise StreamingProviderError(f"{provider} returned an SSE error{details}")


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
            # OpenAI-compatible SSE streams should contain one JSON object per
            # data event. Ignore malformed vendor heartbeats rather than killing
            # an otherwise healthy response.
            continue

        if not isinstance(chunk, dict):
            continue

        _raise_sse_error(chunk, "OpenAI-compatible provider")

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            # Some providers send usage-only terminal events. There is no text
            # for JanitorAI to display, so treat them as a heartbeat.
            yield ""
            continue

        content_found = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            content = _extract_openai_content(choice)
            if content is not None:
                if content:
                    content_found = True
                    yield content
                else:
                    yield ""

        if not content_found and not choices:
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

        _raise_sse_error(chunk, "Gemini-compatible provider")

        content_found = False
        candidates = chunk.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
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
