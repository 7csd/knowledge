import json

import httpx
import pytest

from knowledge_agent.config import Settings
from knowledge_agent.model import CompatibleChatModel, ModelError
from knowledge_agent.trace import Trace


def settings():
    return Settings(base_url="https://provider.example/v1", api_key="test-private-secret", model="test-model")


def test_http_contract_and_reasoning_not_persisted():
    def handle(request):
        assert str(request.url) == "https://provider.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-private-secret"
        body = json.loads(request.content)
        assert body["model"] == "test-model" and body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "private reasoning",
                            "tool_calls": [
                                {
                                    "id": "x",
                                    "type": "function",
                                    "function": {"name": "search_documents", "arguments": '{"query":"年假"}'},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    model = CompatibleChatModel(settings(), httpx.MockTransport(handle))
    try:
        result = model.respond([{"role": "user", "content": "问题"}], [])
        assert result.tool_calls[0].function.name == "search_documents"
        assert "reasoning" not in result.model_dump_json()
    finally:
        model.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, text="test-private-secret"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"role": "system"}}]}),
    ],
)
def test_provider_errors_are_sanitized(response):
    model = CompatibleChatModel(settings(), httpx.MockTransport(lambda _: response))
    try:
        with pytest.raises(ModelError) as caught:
            model.respond([], [])
        assert "test-private-secret" not in str(caught.value)
    finally:
        model.close()


def test_timeout_is_explicit():
    def handle(request):
        raise httpx.ReadTimeout("test-private-secret", request=request)

    model = CompatibleChatModel(settings(), httpx.MockTransport(handle))
    try:
        with pytest.raises(ModelError, match="model_timeout"):
            model.respond([], [])
    finally:
        model.close()


def test_trace_redacts_configured_secrets(tmp_path):
    trace = Trace("test", tmp_path / "trace.jsonl", ("sensitive-key",))
    trace.record("test", nested={"text": "sensitive-key"})
    assert "sensitive-key" not in (tmp_path / "trace.jsonl").read_text()


@pytest.mark.parametrize(
    "url",
    ["https://u:p@example.com/v1", "https://example.com/v1?key=x", "https://example.com/v1/chat/completions"],
)
def test_bad_api_root_rejected(url):
    with pytest.raises(ValueError):
        Settings(base_url=url, api_key="x", model="m").require_live_model()
