import httpx2
import pytest
from flask import Flask

from gfjproxy.streaming import (
    StreamingProviderError,
    StreamingTimeoutError,
    _ManagedStream,
    gemini_sse_completion,
    openai_chat_completion,
)
from gfjproxy.utils import ResponseHelper


def _stream_context(mocker, lines, *, status_error=None):
    response = mocker.MagicMock()
    response.iter_lines.return_value = lines
    response.raise_for_status.side_effect = status_error
    context = mocker.MagicMock()
    context.__enter__.return_value = response
    mocker.patch("gfjproxy.streaming.http_client.stream", return_value=context)
    return response, context


def test_openai_stream_is_lazy_until_first_next(mocker):
    response, context = _stream_context(
        mocker,
        [
            b'data: {"choices":[{"delta":{"content":"hello"}}]}',
            b'data: [DONE]',
        ],
    )

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={"Authorization": "Bearer test"},
        timeout=123,
    )

    context.__enter__.assert_not_called()
    assert next(stream) == "hello"
    context.__enter__.assert_called_once()
    response.raise_for_status.assert_called_once()
    with pytest.raises(StopIteration):
        next(stream)
    context.__exit__.assert_called_once()


def test_stream_closes_even_before_first_next(mocker):
    _, context = _stream_context(mocker, [])

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={},
        timeout=123,
    )
    stream.close()

    context.__enter__.assert_not_called()
    context.__exit__.assert_not_called()


def test_stream_reads_error_body_before_reraising_status_error(mocker):
    request = httpx2.Request("POST", "https://example.test")
    response = mocker.MagicMock()
    response.status_code = 429
    response.raise_for_status.side_effect = httpx2.HTTPStatusError(
        "429", request=request, response=response
    )
    context = mocker.MagicMock()
    context.__enter__.return_value = response
    mocker.patch("gfjproxy.streaming.http_client.stream", return_value=context)

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={},
        timeout=123,
    )

    with pytest.raises(Exception) as caught:
        next(stream)
    assert type(caught.value).__name__ == "StreamingHTTPError"
    assert caught.value.status_code == 429
    response.read.assert_called_once()
    context.__exit__.assert_called_once()


def test_gemini_stream_skips_thoughts_and_keeps_heartbeat(mocker):
    _, context = _stream_context(
        mocker,
        [
            b'data: {"candidates":[{"content":{"parts":[{"text":"hidden","thought":true}]}}]}',
            b'data: {"candidates":[{"content":{"parts":[{"text":"visible"}]}}]}',
        ],
    )

    stream = gemini_sse_completion(
        "https://example.test:streamGenerateContent",
        request={"contents": []},
        headers={},
        timeout=123,
    )

    assert next(stream) == ""
    assert next(stream) == "visible"
    stream.close()
    context.__exit__.assert_called_once()


def test_openai_stream_accepts_final_message_content(mocker):
    _response, context = _stream_context(
        mocker,
        [b'data: {"choices":[{"message":{"content":"final"}}]}'],
    )

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={},
        timeout=123,
    )

    assert next(stream) == "final"
    stream.close()
    context.__exit__.assert_called_once()


def test_response_helper_does_not_yield_after_generatorexit():
    app = Flask(__name__)

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(iter(["hello"])).build()
        iterator = response.response
        first = next(iterator)
        assert first.startswith("data: ")
        assert '"delta": {"content": ""}' in first
        iterator.close()


def test_response_helper_converts_stream_error_to_safe_sse():
    app = Flask(__name__)

    def broken_stream():
        yield "hello"
        raise RuntimeError("secret upstream details")

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(broken_stream()).build()
        chunks = list(response.response)

    body = "".join(chunks)
    assert "Streaming provider error. Please retry." in body
    assert "secret upstream details" not in body
    assert body.endswith("data: [DONE]\n\n")


def test_response_helper_sends_heartbeat_before_slow_stream():
    app = Flask(__name__)

    def stream():
        yield "answer"

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(stream()).build()
        iterator = iter(response.response)
        first = next(iterator)
        second = next(iterator)

    assert first.startswith("data: ")
    assert '"delta": {"content": ""}' in first
    assert '"content": "answer"' in second


def test_stream_emits_periodic_heartbeat_while_upstream_is_silent():
    import threading

    release = threading.Event()
    started = threading.Event()
    closed = threading.Event()

    def open_factory():
        started.set()
        release.wait(timeout=2)

        class Context:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                closed.set()
                return False

        class Response:
            def raise_for_status(self):
                return None

        return Context(), Response()

    def iterator_factory(_response):
        yield "answer"

    stream = _ManagedStream(
        open_factory,
        iterator_factory,
        heartbeat_interval=0.01,
    )

    first = next(stream)
    assert first == ""
    assert started.is_set()

    release.set()
    assert next(stream) == "answer"
    stream.close()
    assert closed.wait(timeout=1)


def test_stream_raises_on_provider_sse_error(mocker):
    _response, context = _stream_context(
        mocker,
        [b'data: {"error":{"code":"rate_limited","message":"too many requests"}}'],
    )

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={},
        timeout=123,
    )

    with pytest.raises(StreamingProviderError, match="rate_limited"):
        next(stream)
    context.__exit__.assert_called_once()


def test_stream_reports_upstream_timeout(mocker):
    _response, context = _stream_context(
        mocker,
        [],
        status_error=httpx2.ReadTimeout("timed out"),
    )

    stream = openai_chat_completion(
        "https://example.test/v1/chat/completions",
        request={"model": "test", "stream": True},
        headers={},
        timeout=123,
    )

    with pytest.raises(StreamingTimeoutError):
        next(stream)
    context.__exit__.assert_called_once()


def test_response_helper_reports_empty_provider_stream():
    app = Flask(__name__)

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(iter([])).build()
        body = "".join(response.response)

    assert "Streaming provider returned no visible content. Please retry." in body
    assert body.endswith("data: [DONE]\n\n")


def test_response_helper_starts_downstream_before_upstream_connection():
    opened = False

    def open_factory():
        nonlocal opened
        opened = True
        return _Context(), _Response()

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Response:
        def raise_for_status(self):
            return None

    def iterator_factory(_response):
        yield "answer"

    stream = _ManagedStream(
        open_factory, iterator_factory, heartbeat_interval=0.1
    )
    app = Flask(__name__)

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(stream).build()
        iterator = iter(response.response)
        first = next(iterator)
        assert not opened
        assert '"content": ""' in first
        second = next(iterator)
        assert opened
        assert '"content": "answer"' in second
        iterator.close()


def test_response_helper_stream_headers_disable_proxy_buffering():
    app = Flask(__name__)
    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(iter(["answer"])).build()

    assert response.content_type.startswith("text/event-stream")
    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, no-transform, max-age=0"
    )
    assert response.headers["Content-Encoding"] == "identity"
    assert response.headers["X-Accel-Buffering"] == "no"
