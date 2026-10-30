import json
from pathlib import Path

import httpx
import pytest

from knowledge_agent.cli import main
from knowledge_agent.config import Settings
from knowledge_agent.model import CompatibleChatModel


def test_json_credentials_reach_http_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "base_url": "https://configured.example/v1",
        "api_key": "local-test-secret",
        "model": "configured-model",
    }), encoding="utf-8-sig")

    def handle(request):
        assert str(request.url) == "https://configured.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer local-test-secret"
        assert json.loads(request.content)["model"] == "configured-model"
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    model = CompatibleChatModel(Settings.load(), httpx.MockTransport(handle))
    try:
        assert model.respond([], []).content == "ok"
    finally:
        model.close()


def test_configuration_precedence_and_no_environment_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    (tmp_path / ".env").write_text("LLM_MODEL=dotenv-model\nLLM_API_KEY=dotenv-secret\n", encoding="utf-8")
    assert Settings.load().model == "environment-model"
    assert Settings.load().api_key.get_secret_value() == "dotenv-secret"
    (tmp_path / "config.json").write_text('{"model":"file-model","api_key":""}', encoding="utf-8")
    configured = Settings.load()
    assert configured.model == "file-model"
    assert configured.api_key.get_secret_value() == ""
    (tmp_path / ".env").unlink()
    (tmp_path / "config.json").unlink()
    assert Settings.load().api_key.get_secret_value() == ""


@pytest.mark.parametrize("content", ['{"api_key":"sensitive-secret",}', '[]', '{"unknown":"sensitive-secret"}'])
def test_invalid_config_fails_without_exposing_contents(tmp_path, monkeypatch, capsys, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(content, encoding="utf-8")
    assert main(["ask", "问题", "--trace", "unused.jsonl"]) == 2
    assert "sensitive-secret" not in capsys.readouterr().err


def test_explicit_missing_config_fails(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.json"), "index"]) == 2
    assert "无法读取配置文件" in capsys.readouterr().err


def test_custom_config_and_cli_override(tmp_path, capsys):
    config = tmp_path / "custom.local.json"
    data = Path(__file__).resolve().parents[1] / "data/documents.json"
    config.write_text(json.dumps({"retrieval_mode": "hybrid", "data_path": str(data)}), encoding="utf-8")
    assert main(["--config", str(config), "--retrieval", "bm25", "index"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "bm25"
