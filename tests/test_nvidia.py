import httpx2

from gfjproxy.models import JaiMessage
from gfjproxy.providers.nvidia import nvidia_generate_content


def test_glm_53_adds_clear_thinking_request_option(mocker):
    response = mocker.MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
    }
    post = mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = nvidia_generate_content(
        "test-user",
        "nvapi-secret",
        "z-ai/glm-5.3",
        [JaiMessage(content="hello", role="user")],
    )

    assert result.status == 200
    payload = post.call_args.kwargs["json"]
    assert payload["chat_template_kwargs"] == {"clear_thinking": True}


def test_streaming_nvidia_errors_are_not_reported_as_internal_exception(mocker):
    request = httpx2.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx2.Response(429, request=request, json={"error": {"message": "rate limited"}})
    context = mocker.MagicMock()
    context.__enter__.return_value = response
    mocker.patch("gfjproxy.streaming.http_client.stream", return_value=context)

    result = nvidia_generate_content(
        "test-user",
        "nvapi-secret",
        "z-ai/glm-5.3",
        [JaiMessage(content="hello", role="user")],
        {"stream": True},
    )

    assert result.status == 200
    stream = result.stream
    assert stream is not None
    try:
        next(stream)
    except Exception as exc:
        assert "429" in str(exc)
