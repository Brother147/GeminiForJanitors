import json

import httpx2
import pytest
from flask import Flask

from gfjproxy.streaming import gemini_sse_completion, openai_chat_completion
from gfjproxy.utils import ResponseHelper


def _stream_context(mocker, lines, *, status_error=None):
    response = mocker.MagicMock()
    response.iter_lines.return_value = lines
    response.raise_for_status.side_effect = status_error
    context = mocker.MagicMock()
    context.__enter__.return_value = response
    mocker.patch("gfjproxy.streaming.http_client.stream", return_value=context)
    return response, context


def test_openai_stream_opens_and_validates_before_returning(mocker):
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

    context.__enter__.assert_called_once()
    response.raise_for_status.assert_called_once()
    assert next(stream) == "hello"
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

    context.__exit__.assert_called_once()


def test_stream_http_status_error_happens_before_return(mocker):
    request = httpx2.Request("POST", "https://example.test")
    response = httpx2.Response(401, request=request)
    error = httpx2.HTTPStatusError("401", request=request, response=response)
    _, context = _stream_context(mocker, [], status_error=error)

    with pytest.raises(httpx2.HTTPStatusError):
        openai_chat_completion(
            "https://example.test/v1/chat/completions",
            request={"model": "test", "stream": True},
            headers={},
            timeout=123,
        )

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


def test_response_helper_does_not_yield_after_generatorexit():
    app = Flask(__name__)

    with app.test_request_context("/"):
        response = ResponseHelper(use_stream=True).add_stream(iter(["hello"])).build()
        iterator = response.response
        assert next(iterator).startswith(
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}'
        )
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
    assert json.dumps("Streaming provider error. Please retry.") in body
    assert "secret upstream details" not in body
    assert body.endswith("data: [DONE]\n\n")
