"""Utilities."""

import atexit
import base64
import datetime
import json
import re
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from itertools import groupby

from flask import Response
from httpx2 import HTTPError

from .http_client import http_client

################################################################################


class MessageKind(Enum):
    CHAT = 0
    ERROR = 1
    PROXY = 2


@dataclass(frozen=True, kw_only=True)
class ResponseMessage:
    """Response message model."""

    kind: MessageKind

    text: str

    status_code: int | None = None


class ResponseHelper:
    """Response helper to provide JanitorAI with valid responses."""

    # U+200B ZERO WIDTH SPACE
    PROXY_TAG_OPEN = "\u200b<proxy>\n"
    PROXY_TAG_CLOSE = "\n\u200b</proxy>"

    def __init__(self, *, use_stream: bool = False, wrap_errors: bool = False):
        self._messages = []
        self._status = 200
        self._use_stream = use_stream
        self._wrap_errors = wrap_errors
        self._stream = None

    def add_error(self, message, status_code: int):
        self._messages.append(
            ResponseMessage(
                kind=MessageKind.ERROR,
                text=str(message),
                status_code=status_code,
            )
        )
        self._status = status_code
        return self

    def add_message(self, *messages):
        for message in messages:
            self._messages.append(
                ResponseMessage(kind=MessageKind.CHAT, text=str(message))
            )
        return self

    def add_proxy_message(self, *messages):
        for message in messages:
            self._messages.append(
                ResponseMessage(kind=MessageKind.PROXY, text=str(message))
            )
        return self

    def add_stream(self, chunks):
        self._stream = chunks
        return self

    @staticmethod
    def _format_sse_delta(text: str) -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    @staticmethod
    def _format_sse_finish() -> str:
        return (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    @staticmethod
    def _stream_error_message(exc: Exception) -> str:
        from .streaming import StreamingTimeoutError

        if isinstance(exc, StreamingTimeoutError):
            return "Streaming provider timed out. Please retry."

        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code == 429:
                return "Streaming provider rate limit reached. Please retry later."
            if 400 <= status_code < 500:
                return f"Streaming provider rejected the request (HTTP {status_code})."
            if status_code >= 500:
                return f"Streaming provider is unavailable (HTTP {status_code})."
        return "Streaming provider error. Please retry."

    def build(self) -> Response:
        if self._status != 200 and len(self._messages) == 1:
            if self._wrap_errors:
                return Response(
                    response=[json.dumps({"error": self.message.strip()})],
                    status=self._status,
                    content_type="application/json; charset=utf-8",
                )
            return Response(
                response=[self.message],
                status=self._status,
                content_type="text/plain; charset=utf-8",
            )
        elif self._use_stream:
            if self._stream is not None:

                def generate():
                    stream = iter(self._stream)
                    completed = False
                    provider_chunks = 0
                    provider_chars = 0

                    try:
                        # Use an actual SSE data event for heartbeats. Some
                        # clients, including strict OpenAI-compatible parsers,
                        # ignore comment-only SSE frames.
                        yield self._format_sse_delta("")

                        for chunk in stream:
                            if chunk:
                                provider_chunks += 1
                                provider_chars += len(chunk)
                                yield self._format_sse_delta(chunk)
                            else:
                                yield self._format_sse_delta("")

                        tail = self.message
                        if tail:
                            yield self._format_sse_delta(tail)
                        if provider_chunks == 0 and not tail:
                            from .logging import xlog

                            xlog(
                                None,
                                "Streaming response ended without visible content",
                            )
                            yield self._format_sse_delta(
                                f"{self.PROXY_TAG_OPEN}Streaming provider returned no visible content. Please retry.{self.PROXY_TAG_CLOSE}"
                            )
                        completed = True

                        from .logging import xlog

                        xlog(
                            None,
                            f"Streaming response completed: {provider_chunks} provider chunk(s), {provider_chars} character(s)",
                        )
                    except GeneratorExit:
                        raise
                    except Exception as exc:  # noqa: BLE001 - stream boundary
                        from .logging import xlog

                        status_code = getattr(exc, "status_code", None)
                        status_text = (
                            f", status={status_code}"
                            if isinstance(status_code, int)
                            else ""
                        )
                        xlog(
                            None,
                            f"Streaming response failed: {type(exc).__name__}{status_text}",
                        )
                        yield self._format_sse_delta(
                            f"{self.PROXY_TAG_OPEN}{self._stream_error_message(exc)}{self.PROXY_TAG_CLOSE}"
                        )
                        completed = True
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            with suppress(Exception):  # pragma: no cover - cleanup only
                                close()

                    if completed:
                        yield self._format_sse_finish()
                        yield "data: [DONE]\n\n"

                return Response(
                    response=generate(),
                    status=200,
                    content_type="text/event-stream; charset=utf-8",
                    headers={
                        "Cache-Control": "no-store, no-cache, no-transform, max-age=0",
                        "Content-Encoding": "identity",
                        "X-Accel-Buffering": "no",
                    },
                    direct_passthrough=True,
                )

            return Response(
                response=self._format_sse_delta(self.message)
                + self._format_sse_finish()
                + "data: [DONE]\n\n",
                status=200,
                content_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-store, no-cache, no-transform, max-age=0",
                    "Content-Encoding": "identity",
                    "X-Accel-Buffering": "no",
                },
                direct_passthrough=True,
            )
        else:
            return Response(
                response=[
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": self.message,
                                    },
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    )
                ],
                status=200,
                content_type="application/json; charset=utf-8",
            )

    def build_error(self, message, status_code: int):
        return self.add_error(message, status_code).build()

    def build_message(self, *messages):
        return self.add_message(*messages).build()

    @property
    def message(self) -> str:
        if not self._messages:
            return ""

        proxy_open = ResponseHelper.PROXY_TAG_OPEN
        proxy_close = ResponseHelper.PROXY_TAG_CLOSE

        if len(self._messages) == 1:
            if self._messages[0].kind == MessageKind.CHAT:
                return self._messages[0].text
            if self._messages[0].kind == MessageKind.ERROR:
                if self._wrap_errors:
                    return f"PROXY ERROR {self._messages[0].status_code}: {self._messages[0].text}"
                return self._messages[0].text
            return f"{proxy_open}{self._messages[0].text}{proxy_close}"

        def do_wrap_proxy(msg):
            return msg.kind in (MessageKind.PROXY, MessageKind.ERROR)

        result = []

        for wrap_proxy, msg_group in groupby(self._messages, do_wrap_proxy):
            if wrap_proxy:
                content = []
                for msg in msg_group:
                    if msg.kind == MessageKind.ERROR:
                        if msg.text.startswith("Error from"):
                            text = msg.text
                        else:
                            text = f"Error {msg.status_code}: {msg.text}"
                    else:  # PROXY message
                        text = msg.text
                    content.append(text)
                result.append(f"{proxy_open}{'\n'.join(content)}{proxy_close}")
            else:  # CHAT messages
                result.append("\n".join(msg.text for msg in msg_group))

        return "\n".join(result)

    @property
    def status(self) -> int:
        return self._status

    @property
    def status_code(self) -> int:
        return self.build().status_code

    @property
    def response(self):
        return self.build().response

################################################################################


def is_proxy_test(request_json: dict) -> bool:
    # A normal chat request has 2 or more messages and the first one always has
    # "role" set to "system" (this being the bot description). Meanwhile, a
    # proxy test request looks like this:
    #   {
    #     "max_tokens": 10,
    #     "messages": [{"content": "Just say TEST", "role": "user"}],
    #     "model": "gemini-2.5-pro",
    #     "temperature": 0
    #   }
    # We need to inspect the "messages" key. Everything else can vary.
    # A false negative will lead the request down the regular chat path, which
    # isn't really a big deal, considering the error feedback UI will only fail
    # to display any proxy errors and will show something else instead.

    messages = request_json.get("messages")
    if isinstance(messages, list) and len(messages) == 1:
        message = messages[0]
        if isinstance(message, dict):
            text = message.get("content")
            role = message.get("role")
            if text == "Just say TEST" and role == "user":
                # Yep, looks like a proxy test
                return True

    # Most likely not a proxy test request
    return False


################################################################################


def comma_split(s: str) -> list[str]:
    return [t for t in map(str.strip, s.split(",")) if t]


def safe_response_json(response) -> object:
    """Return decoded JSON from an HTTP response, or an empty object on failure."""
    try:
        return response.json()
    except (TypeError, ValueError):
        return {}


################################################################################


def _runner(cloudflared: str):
    from .logging import xlog

    xlog(None, "Running cloudflared ...")

    process = subprocess.Popen(
        [
            cloudflared,
            "tunnel",
            "--metrics",
            "127.0.0.1:5001",
            "--url",
            "http://127.0.0.1:5000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    atexit.register(process.terminate)

    pattern = re.compile(r"(?P<url>https?:\/\/[^\s]+.trycloudflare.com)")

    for _ in range(10):
        try:
            metrics = http_client.get("http://127.0.0.1:5001/metrics").text
            if match := pattern.search(metrics):
                url = match.group("url")
                xlog(None, f"Cloudflared tunnel on {url}")
                return
            else:
                xlog(None, "Pattern search returned no match")
        except HTTPError:
            time.sleep(1)
    xlog(None, "Couldn't get cloudflared tunnel")


def run_cloudflared(cloudflared: str):
    runner_thread = threading.Thread(target=_runner, args=(cloudflared,), daemon=True)
    runner_thread.start()


################################################################################


# https://github.com/googleapis/google-cloud-python/blob/b9466f9c85c94331ffc39e1da3cf98fb5ff7d612/packages/google-auth/google/auth/_helpers.py#L111
def utcnow() -> datetime.datetime:
    """Returns the current UTC datetime.

    Returns:
        datetime: The current time in UTC.
    """
    return datetime.datetime.now(datetime.UTC)


# https://github.com/googleapis/google-cloud-python/blob/b9466f9c85c94331ffc39e1da3cf98fb5ff7d612/packages/google-auth/google/auth/_helpers.py#L127
def utcfromtimestamp(timestamp: float) -> datetime.datetime:
    """Returns the UTC datetime from a timestamp.

    Args:
        timestamp (float): The timestamp, in fractional seconds, to convert.

    Returns:
        datetime: The time in UTC.
    """
    return datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)


def utctimestamp() -> float:
    """Returns the current UTC timestamp.

    Returns:
        float: The current time in UTC in fractional seconds.
    """
    return time.time()


################################################################################


def base64url_encode(input: str | bytes) -> str:
    if isinstance(input, str):
        input = input.encode("utf-8")

    return base64.urlsafe_b64encode(input).decode("ascii").rstrip("=")


def base64url_decode(input: str | bytes) -> bytes:
    if isinstance(input, str):
        input = input.encode("utf-8")

    padding = len(input) % 4
    if padding > 0 and not input.endswith(b"="):
        input += b"=" * (4 - padding)

    return base64.urlsafe_b64decode(input)


################################################################################
