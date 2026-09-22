import httpx2
import pytest
from httpx2 import ReadTimeout

from gfjproxy.models import JaiMessage
from gfjproxy.providers.radeon import (
    RADEON_CHAT_COMPLETIONS_URL,
    radeon_generate_content,
)


def test_radeon_provider_sends_openai_compatible_request(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
            "reasoning_tokens": 7,
        },
    }
    post = mocker.patch(
        "gfjproxy.streaming.http_client.post", return_value=response
    )

    result = radeon_generate_content(
        "test-user",
        "radeon/rc-secret-key",
        "DeepSeek-V4-Flash",
        [
            JaiMessage(role="system", content="system"),
            JaiMessage(role="user", content="hello"),
        ],
        {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
        },
    )

    assert result.status == 200
    assert result.text == "ok"
    assert result.metadata.token_usage.prompt_tokens == 11
    assert result.metadata.token_usage.completion_tokens == 22
    assert result.metadata.token_usage.reasoning_tokens == 7
    assert result.metadata.token_usage.total_tokens == 33

    assert post.call_args.args[0] == RADEON_CHAT_COMPLETIONS_URL
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer rc-secret-key"
    payload = post.call_args.kwargs["json"]
    assert payload == {
        "model": "DeepSeek-V4-Flash",
        "stream": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
    }


def test_radeon_provider_uses_process_timeout(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    post = mocker.patch(
        "gfjproxy.streaming.http_client.post", return_value=response
    )
    mocker.patch("gfjproxy.providers.radeon.PROCESS_TIMEOUT", 123)

    radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert post.call_args.kwargs["timeout"] == 123


def test_radeon_provider_handles_timeout(mocker):
    mocker.patch(
        "gfjproxy.streaming.http_client.post", side_effect=ReadTimeout("")
    )

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 504
    assert result.error == "Gateway Timeout"


def test_radeon_provider_preserves_amd_error_and_redacts_key(mocker):
    secret = "rc-secret-key"
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        429,
        json={
            "error": {
                "code": "rate_limited",
                "message": f"quota exceeded for {secret}",
                "type": "rate_limit_error",
            }
        },
        request=request,
    )
    error = httpx2.HTTPStatusError("HTTP 429", request=request, response=response)
    mocker.patch("gfjproxy.streaming.http_client.post", side_effect=error)

    result = radeon_generate_content(
        "test-user", secret, "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 429
    assert "rate_limited" in result.error
    assert "quota exceeded" in result.error
    assert "rate_limit_error" in result.extras
    assert secret not in result.error
    assert secret not in result.extras


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503])
def test_radeon_provider_handles_streaming_http_error_response(mocker, status):
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        status,
        json={
            "error": {
                "code": "rate_limited",
                "message": "too many requests",
            }
        },
        request=request,
    )
    error = httpx2.HTTPStatusError(f"HTTP {status}", request=request, response=response)

    context = mocker.MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    mocker.patch("gfjproxy.streaming.http_client.stream", return_value=context)
    response.raise_for_status = mocker.Mock(side_effect=error)
    response.read = mocker.Mock(wraps=response.read)

    result = radeon_generate_content(
        "test-user",
        "rc-secret-key",
        "some-model",
        [JaiMessage(content="hello")],
        {"stream": True},
    )

    assert result.status == status
    assert "rate_limited" in result.error
    assert "too many requests" in result.error
    response.read.assert_called_once()
    context.__exit__.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503])
def test_radeon_provider_preserves_http_error_status(mocker, status):
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        status,
        json={"error": {"code": "amd_error", "message": "request failed"}},
        request=request,
    )
    error = httpx2.HTTPStatusError(f"HTTP {status}", request=request, response=response)
    mocker.patch("gfjproxy.streaming.http_client.post", side_effect=error)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == status
    assert "request failed" in result.error


def test_radeon_provider_never_logs_api_key(mocker):
    secret = "rc-secret-key"
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        500,
        text=f"AMD upstream error included {secret} unexpectedly",
        request=request,
    )
    error = httpx2.HTTPStatusError("HTTP 500", request=request, response=response)
    mock_post = mocker.patch(
        "gfjproxy.streaming.http_client.post", side_effect=error
    )
    mock_log = mocker.patch("gfjproxy.providers.radeon.xlog")

    result = radeon_generate_content(
        "test-user", secret, "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 500
    mock_post.assert_called_once()
    for call in mock_log.call_args_list:
        assert secret not in str(call)


def test_radeon_provider_handles_detail_message(mocker):
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        400,
        json={"detail": "Invalid request parameter"},
        request=request,
    )
    error = httpx2.HTTPStatusError("HTTP 400", request=request, response=response)
    mocker.patch("gfjproxy.streaming.http_client.post", side_effect=error)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 400
    assert "Invalid request parameter" in result.error


def test_radeon_provider_handles_detail_error(mocker):
    request = httpx2.Request("POST", RADEON_CHAT_COMPLETIONS_URL)
    response = httpx2.Response(
        403,
        json={
            "detail": {
                "error": {
                    "code": "account_not_verified",
                    "message": "Account needs verification",
                    "type": "permission_error",
                }
            }
        },
        request=request,
    )
    error = httpx2.HTTPStatusError("HTTP 403", request=request, response=response)
    mocker.patch("gfjproxy.streaming.http_client.post", side_effect=error)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 403
    assert "account_not_verified" in result.error
    assert "Account needs verification" in result.error


def test_radeon_provider_handles_invalid_json(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("not json")
    mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 502
    assert result.error == "Invalid response from AMD Radeon Cloud."


def test_radeon_provider_reads_nested_reasoning_tokens(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }
    mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 200
    assert result.metadata.token_usage.reasoning_tokens == 4


def test_radeon_provider_handles_empty_choices(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": []}
    mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 502
    assert result.error == "Response blocked/empty."


def test_radeon_provider_handles_invalid_choices_type(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": {}}
    mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 502
    assert result.error == "Invalid response from AMD Radeon Cloud."


def test_radeon_provider_handles_missing_message_content(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {}}]}
    mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = radeon_generate_content(
        "test-user", "rc-secret-key", "some-model", [JaiMessage(content="hello")]
    )

    assert result.status == 502
    assert result.error == "Response blocked/empty."
