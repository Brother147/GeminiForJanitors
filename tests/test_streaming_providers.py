import pytest

from gfjproxy.models import JaiMessage
from gfjproxy.providers.cerebras import cerebras_generate_content
from gfjproxy.providers.deepseek import deepseek_generate_content
from gfjproxy.providers.gemini import gemini_generate_content
from gfjproxy.providers.gemini_cli import gemini_cli_generate_content_ex
from gfjproxy.providers.groq import groq_generate_content
from gfjproxy.providers.nvidia import nvidia_generate_content
from gfjproxy.providers.openrouter import openrouter_generate_content
from gfjproxy.providers.proxy import proxy_generate_content
from gfjproxy.providers.radeon import radeon_generate_content
from gfjproxy.providers.z_ai import z_ai_generate_content


@pytest.mark.parametrize(
    "provider,func,api_key,model",
    [
        ("cerebras", cerebras_generate_content, "cerebras/key", "test-model"),
        ("deepseek", deepseek_generate_content, "deepseek/key", "test-model"),
        ("groq", groq_generate_content, "groq/key", "test-model"),
        ("nvidia", nvidia_generate_content, "nvapi-key", "test-model"),
        ("openrouter", openrouter_generate_content, "openrouter/key", "test-model"),
        ("radeon", radeon_generate_content, "radeon/key", "DeepSeek-V4-Flash"),
        ("z_ai", z_ai_generate_content, "z_ai/key", "test-model"),
    ],
)
def test_openai_compatible_providers_return_a_lazy_stream(
    mocker, provider, func, api_key, model
):
    marker = iter(["answer"])
    stream_call = mocker.patch(
        f"gfjproxy.providers.{provider}.openai_chat_completion",
        return_value=marker,
    )

    result = func(
        "test-user",
        api_key,
        model,
        [JaiMessage(content="hello")],
        {"stream": True},
    )

    assert result.status == 200
    assert result.stream is marker
    request = stream_call.call_args.kwargs["request"]
    assert request["stream"] is True


def test_proxy_provider_returns_a_lazy_stream(mocker):
    marker = iter(["answer"])
    stream_call = mocker.patch(
        "gfjproxy.providers.proxy.openai_chat_completion",
        return_value=marker,
    )

    result = proxy_generate_content(
        "test-user",
        "secret@https://example.test/v1/chat/completions",
        "test-model",
        [JaiMessage(content="hello")],
        {"stream": True},
    )

    assert result.status == 200
    assert result.stream is marker
    assert stream_call.call_args.kwargs["request"]["stream"] is True


def test_google_provider_returns_a_lazy_stream(mocker):
    marker = iter(["answer"])
    stream_call = mocker.patch(
        "gfjproxy.providers.gemini.gemini_sse_completion",
        return_value=marker,
    )

    result = gemini_generate_content(
        "test-user",
        "google-key",
        "gemini-2.5-flash",
        [JaiMessage(content="hello")],
        {"stream": True},
    )

    assert result.status == 200
    assert result.stream is marker
    assert stream_call.call_args.kwargs["request"]["contents"]


def test_gemini_cli_provider_returns_a_lazy_stream(mocker):
    marker = iter(["answer"])
    stream_call = mocker.patch(
        "gfjproxy.providers.gemini_cli.gemini_sse_completion",
        return_value=marker,
    )

    result = gemini_cli_generate_content_ex(
        None,
        "access-token",
        "project-id",
        "gemini-2.5-flash",
        [JaiMessage(content="hello")],
        {"stream": True},
    )

    assert result.success
    assert result.value["_stream"] is marker
    assert stream_call.call_args.kwargs["request"]["request"]["contents"]
