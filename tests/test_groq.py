from gfjproxy.models import JaiMessage
from gfjproxy.providers.groq import groq_generate_content


def test_groq_provider_accepts_model_from_janitor(mocker):
    response = mocker.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }

    post = mocker.patch("gfjproxy.streaming.http_client.post", return_value=response)

    result = groq_generate_content(
        "test-user",
        "gsk_test",
        "some-groq-model",
        [JaiMessage(role="user", content="hello")],
        {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1},
    )

    assert result.status == 200
    assert result.text == "ok"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "some-groq-model"
    assert payload["messages"][0]["content"] == "hello"
    assert "presence_penalty" not in payload
    assert "repetition_penalty" not in payload
