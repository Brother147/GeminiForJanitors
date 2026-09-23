from typing import Any

import httpx2
import pytest
from httpx2 import ReadTimeout
from pytest_mock import MockerFixture

from gfjproxy._globals import BANNER, BANNER_VERSION
from gfjproxy.handlers import (
    _handle_request,
    _resolve_provider,
    handle_chat_message,
    handle_proxy_test,
)
from gfjproxy.models import JaiMessage, JaiRequest, JaiResult
from gfjproxy.utils import ResponseHelper
from gfjproxy.xuiduser import XUID, LocalUserStorage, UserSettings

################################################################################


def make_mock_response(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def make_http_error(code: int, response_json: dict[str, Any]) -> httpx2.HTTPStatusError:
    req = httpx2.Request("POST", "https://generativelanguage.googleapis.com/")
    resp = httpx2.Response(code, json=response_json, request=req)
    return httpx2.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


def make_expected_error(code: int, status: str, message: str) -> tuple[str, int]:
    return (f"Error from Google AI ({code}): {status}\n{message}", code)


################################################################################

# Any of these errors could occur during proxy test and chat message.
# Thus, that are to be tested on both handlers.

COMMON_ERRORS = [
    {
        "generate_content_mock": ReadTimeout(""),
        "expected_result": ("Gateway Timeout", 504),
    },
    {
        "generate_content_mock": Exception("I'm a teapot"),
        "expected_result": ("Unhandled exception from Google AI.", 502),
    },
    {
        "generate_content_mock": make_http_error(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        ),
        "expected_result": make_expected_error(
            400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."
        ),
    },
    {
        "generate_content_mock": make_http_error(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {
                                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                    "quotaDimensions": {
                                        "model": "gemini-2.5-pro",
                                        "location": "global",
                                    },
                                    "quotaValue": "2",
                                },
                            ],
                        },
                    ],
                }
            },
        ),
        "expected_result": make_expected_error(
            429, "RESOURCE_EXHAUSTED", "Requests per Minute quota exceeded."
        ),
    },
    {
        "generate_content_mock": make_http_error(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {
                                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                    "quotaDimensions": {
                                        "location": "global",
                                        "model": "gemini-2.5-pro",
                                    },
                                    "quotaValue": "50",
                                }
                            ],
                        },
                    ],
                }
            },
        ),
        "expected_result": make_expected_error(
            429, "RESOURCE_EXHAUSTED", "Requests per Day quota exceeded."
        ),
    },
    {
        "generate_content_mock": make_http_error(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Generative Language API has not been used in project * before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=182995638091 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.",
                    "status": "PERMISSION_DENIED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "SERVICE_DISABLED",
                            "domain": "googleapis.com",
                            "metadata": {
                                "activationUrl": "https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=182995638091",
                                "service": "generativelanguage.googleapis.com",
                                "serviceTitle": "Generative Language API",
                                "containerInfo": "*",
                                "consumer": "projects/*",
                            },
                        },
                    ],
                }
            },
        ),
        "expected_result": make_expected_error(
            403, "PERMISSION_DENIED", "Generative Language API needs to be enabled"
        ),
    },
    {
        "generate_content_mock": make_http_error(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Permission denied: Consumer 'api_key:*' has been suspended.",
                    "status": "PERMISSION_DENIED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "CONSUMER_SUSPENDED",
                            "domain": "googleapis.com",
                            "metadata": {
                                "service": "generativelanguage.googleapis.com",
                                "containerInfo": "api_key:*",
                                "consumer": "projects/*",
                            },
                        },
                    ],
                }
            },
        ),
        "expected_result": make_expected_error(
            403, "PERMISSION_DENIED", "Customer suspended. You might be banned."
        ),
    },
    {
        "generate_content_mock": make_http_error(
            500,
            {
                "error": {
                    "code": 500,
                    "message": "Some internal error.",
                    "status": "INTERNAL",
                }
            },
        ),
        "expected_result": make_expected_error(500, "INTERNAL", "Some internal error."),
    },
    {
        "generate_content_mock": make_http_error(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "The model is overloaded. Please try again later.",
                    "status": "UNAVAILABLE",
                }
            },
        ),
        "expected_result": make_expected_error(
            503, "UNAVAILABLE", "The model is overloaded. Please try again later."
        ),
    },
]

################################################################################

PROXY_TESTS = [
    {
        "generate_content_mock": make_mock_response("TEST"),
        "expected_result": ("TEST", 200),
    },
]


@pytest.mark.parametrize(
    "params",
    COMMON_ERRORS + PROXY_TESTS,
)
def test_proxy_test(mocker: MockerFixture, params: dict[str, Any]):
    generate_content_mock = params["generate_content_mock"]
    expected_message, expected_status = params["expected_result"]

    mock_post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    if isinstance(generate_content_mock, Exception):
        mock_post.side_effect = generate_content_mock
    else:
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = generate_content_mock
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

    storage = LocalUserStorage()
    xuid = XUID("john", "smith")
    user = UserSettings(storage, xuid)

    jai_req = JaiRequest(
        api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        models={"google": "gemini-2.5-flash"},
    )

    response = handle_proxy_test(user, jai_req, ResponseHelper(wrap_errors=True))

    assert (response.message, response.status) == (
        expected_message
        if expected_status == 200
        else f"PROXY ERROR {expected_status}: {expected_message}",
        expected_status,
    )


################################################################################


CHAT_MESSAGE_TESTS = [
    {  # Blank-slate users should get the bot response plus the latest banner
        "generate_content_mock": make_mock_response("Bot response."),
        "expected_result": ("Bot response.\n" + BANNER, 200),
        "extra_settings": [],
    },
    {  # Blank-slate users on /quiet/ should not see any banner
        "generate_content_mock": make_mock_response("Bot response."),
        "expected_result": ("Bot response.", 200),
        "extra_settings": [
            ("jai_req_quiet", True),
        ],
    },
    {  # Users that already saw the latest banner should not see it again
        "generate_content_mock": make_mock_response("Bot response."),
        "expected_result": ("Bot response.", 200),
        "extra_settings": [
            ("call_do_show_banner", BANNER_VERSION),
        ],
    },
    {  # Users that saw a different banner should see the newest one
        "generate_content_mock": make_mock_response("Bot response."),
        "expected_result": ("Bot response.\n" + BANNER, 200),
        "extra_settings": [
            ("call_do_show_banner", BANNER_VERSION - 1),
        ],
    },
    {  # Ensure user set temperature is honored
        "generate_content_mock": make_mock_response("Bot response."),
        "expected_result": ("Bot response.", 200),
        "extra_settings": [
            (
                "jai_add_message",
                JaiMessage.parse({"role": "user", "content": "Message"}),
            ),
            ("jai_req_quiet", True),
            ("jai_req_quiet_commands", True),
        ],
    },
]


@pytest.mark.parametrize(
    "params",
    COMMON_ERRORS + CHAT_MESSAGE_TESTS,
)
def test_chat_message(mocker: MockerFixture, params: dict[str, Any]):
    generate_content_mock = params["generate_content_mock"]
    expected_message, expected_status = params["expected_result"]
    user_messages = params.get("user_messages", [JaiMessage()])
    extra_settings = params.get("extra_settings", [])

    mock_post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    if isinstance(generate_content_mock, Exception):
        mock_post.side_effect = generate_content_mock
    else:
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = generate_content_mock
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

    user = UserSettings(LocalUserStorage(), XUID("john", "smith"))

    jai_req = JaiRequest(
        api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        models={"google": "gemini-2.5-flash"},
        messages=user_messages,
    )

    for key, value in extra_settings:
        if key == "call_do_show_banner":
            user.do_show_banner(value)
        elif key == "jai_add_message":
            jai_req.messages.append(value)
        elif key == "jai_req_quiet":
            jai_req.quiet = value
        elif key == "jai_req_quiet_commands":
            jai_req.quiet_commands = value
        else:
            assert 0, f"Invalid extra_settings key: {key}"

    response = handle_chat_message(user, jai_req, ResponseHelper(wrap_errors=False))

    assert (response.message, response.status) == (
        expected_message,
        expected_status,
    )

    _, kwargs = mock_post.call_args




def test_generation_settings_are_forwarded_without_commands(mocker: MockerFixture):
    mock_post = mocker.patch("gfjproxy.providers.gemini.http_client.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Bot response"}]}}],
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    user = UserSettings(LocalUserStorage(), XUID("settings", "user"))
    jai_req = JaiRequest(
        api_key="AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        models={"google": "gemini-2.5-flash"},
        messages=[
            JaiMessage.parse(
                {"role": "user", "content": "//max_tokens 1234 Message"}
            )
        ],
        temperature=0.8,
        top_p=0.9,
        quiet=True,
        quiet_commands=True,
    )

    response = handle_chat_message(user, jai_req, ResponseHelper(wrap_errors=False))

    assert response.status == 200
    _, kwargs = mock_post.call_args
    generation_config = kwargs["json"]["generationConfig"]
    assert generation_config["temperature"] == pytest.approx(0.8)
    assert generation_config["topP"] == pytest.approx(0.9)
    assert generation_config["maxOutputTokens"] == 1234



################################################################################


def test_resolve_radeon_provider_prefix():
    assert _resolve_provider("radeon/rc-secret-key") == ("radeon", "rc-secret-key")
    assert _resolve_provider("RaDeOn/rc-secret-key") == ("radeon", "rc-secret-key")
    assert _resolve_provider("rc-secret-key") == ("radeon", "rc-secret-key")


def test_handle_request_auto_detects_radeon_key_and_model(mocker):
    result = JaiResult(200, "ok")
    radeon = mocker.Mock(return_value=result)
    mocker.patch.dict(
        "gfjproxy.handlers.PROVIDER_FUNCS", {"radeon": radeon}, clear=False
    )

    response = _handle_request(
        XUID("test", "user"),
        "rc-secret-key",
        {"radeon": "DeepSeek-V4-Flash"},
        [JaiMessage(role="user", content="hello")],
    )

    assert response is result
    radeon.assert_called_once_with(
        XUID("test", "user"),
        "rc-secret-key",
        "DeepSeek-V4-Flash",
        [JaiMessage(role="user", content="hello")],
        None,
    )


def test_handle_request_dispatches_radeon(mocker):
    result = JaiResult(200, "ok")
    radeon = mocker.Mock(return_value=result)
    mocker.patch.dict(
        "gfjproxy.handlers.PROVIDER_FUNCS", {"radeon": radeon}, clear=False
    )

    response = _handle_request(
        XUID("test", "user"),
        "radeon/rc-secret-key",
        {"radeon": "DeepSeek-V4-Flash"},
        [JaiMessage(role="user", content="hello")],
    )

    assert response is result
    radeon.assert_called_once_with(
        XUID("test", "user"),
        "rc-secret-key",
        "DeepSeek-V4-Flash",
        [JaiMessage(role="user", content="hello")],
        None,
    )
