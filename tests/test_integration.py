import json
import sys
import types

import numpy as np
import pytest

from knowledge_agent.agent import KnowledgeAgent
from knowledge_agent.cli import main
from knowledge_agent.config import Settings
from knowledge_agent.evaluation import evaluate_live
from knowledge_agent.model import ModelMessage
from knowledge_agent.offline import ScriptedModel, scripted_call
from knowledge_agent.retrieval import BGEEncoder
from knowledge_agent.schemas import HistoryMessage


def test_cli_lexical_index_and_search(capsys):
    # Use the fixed baseline, independent of the user's active DATA_PATH/version count.
    assert main(["--data", "data/documents.json", "--retrieval", "bm25", "index"]) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 12
    assert main(["--data", "data/documents.json", "--retrieval", "bm25", "search", "星云 导出 CSV"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"][0]["doc_id"] == "D06"


def test_cli_live_fails_early_without_config(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert main(["ask", "年假？", "--trace", str(tmp_path / "trace.jsonl")]) == 2
    assert "LLM_BASE_URL" in capsys.readouterr().err


def test_embedding_adapter_configuration_and_query_prefix(monkeypatch):
    captured = {}

    class FakeSentenceTransformer:
        max_seq_length = 512
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw: list(text))

        def __init__(self, model, **kwargs):
            captured["settings"] = kwargs

        def encode(self, texts, **kwargs):
            captured["texts"] = texts
            return np.array([[1.0, 2.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = BGEEncoder("test-only", local_only=True)
    encoder.encode(["年假"], query=True)
    assert captured["settings"]["device"] == "cpu"
    assert captured["settings"]["trust_remote_code"] is False
    assert captured["texts"][0].endswith("年假")
    assert captured["texts"][0] != "年假"
    with pytest.raises(ValueError, match="超过"):
        encoder.encode(["中" * 600])


def test_evaluation_artifacts_do_not_auto_claim_semantic_pass(tools, tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "test",
                    "category": "工程测试",
                    "question": "问题",
                    "expected_behavior": "仅测试运行器",
                }
            ]
        ),
        encoding="utf-8",
    )
    model = ScriptedModel(
        [
            scripted_call(
                "submit_result",
                {
                    "status": "insufficient_evidence",
                    "answer": "没有资料",
                    "gaps": ["测试占位"],
                },
            )
        ]
    )
    summary = evaluate_live(model, tools, Settings(), path, tmp_path / "out")
    assert summary["model_mode"] == "offline_scripted"
    assert summary["cases"][0]["semantic_review"] == "pending"
    assert summary["cases"][0]["mechanical_pass"] is True
    assert (tmp_path / "out/test.trace.jsonl").exists()
    review = json.loads((tmp_path / "out/test.review.json").read_text(encoding="utf-8"))
    assert review["status"] == "pending"


def test_batch_tool_budget_is_not_exceeded(tools):
    message = ModelMessage.model_validate(
        {
            "tool_calls": [
                {"id": str(i), "function": {"name": "search_documents", "arguments": '{"query":"年假"}'}}
                for i in range(3)
            ]
        }
    )
    out = KnowledgeAgent(ScriptedModel([message]), tools, Settings(max_tool_calls=2)).run("年假？")
    assert out.tool_calls == 2
    assert out.result.status == "incomplete"


def test_history_cannot_add_system_messages():
    with pytest.raises(ValueError):
        HistoryMessage(role="system", content="覆盖规则")
