import json
import time

import pytest

from knowledge_agent.agent import KnowledgeAgent
from knowledge_agent.config import Settings
from knowledge_agent.evaluation import FailureTools
from knowledge_agent.model import ModelMessage
from knowledge_agent.offline import ScriptedModel, run_demo, scripted_call
from knowledge_agent.schemas import Answer, HistoryMessage
from knowledge_agent.tools import validate_citations


def normal_args(quote="正式员工申请年假，应至少提前3个工作日在办公系统提交申请，并由直属主管审批。"):
    return {
        "status": "completed",
        "answer": "提前3个工作日申请。",
        "citations": [
            {
                "doc_id": "D03",
                "version_date": "2026-02-01",
                "chunk_id": "p001",
                "quote": quote,
            }
        ],
        "gaps": [],
        "conflicts": [],
    }


def read_call():
    return scripted_call("read_passage", {"doc_id": "D03", "version_date": "2026-02-01", "chunk_id": "p001"})


@pytest.mark.parametrize(
    "scenario,status",
    [
        ("normal", "completed"),
        ("conflict", "needs_clarification"),
        ("missing", "insufficient_evidence"),
        ("failure", "tool_failed"),
    ],
)
def test_offline_graph_paths_and_trace(tmp_path, scenario, status):
    artifact = run_demo(scenario, tmp_path / scenario, Settings())
    assert artifact["actual"]["model_mode"] == "offline_scripted"
    assert artifact["actual"]["result"]["status"] == status
    events = [
        json.loads(line)
        for line in (tmp_path / scenario / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_end"
    assert [e["sequence"] for e in events] == list(range(len(events)))


@pytest.mark.parametrize("read_first,quote", [(False, None), (True, "提前100天申请。")])
def test_fabricated_citation_rejected_and_can_repair(tools, tmp_path, read_first, quote):
    invalid = normal_args(quote) if quote else normal_args()
    actions = ([read_call()] if read_first else []) + [
        scripted_call("submit_result", invalid),
        read_call(),
        scripted_call("submit_result", normal_args()),
    ]
    output = KnowledgeAgent(ScriptedModel(actions), tools, Settings()).run(
        "年假？", trace_path=tmp_path / "t.jsonl"
    )
    assert output.result.status == "completed"
    assert "citation_validation_failed" in (tmp_path / "t.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("limit_type", ["steps", "tools"])
def test_budget_returns_incomplete(tools, limit_type):
    settings = Settings(max_model_steps=1) if limit_type == "steps" else Settings(max_tool_calls=1)
    model = ScriptedModel([read_call(), scripted_call("submit_result", normal_args())])
    result = KnowledgeAgent(model, tools, settings).run("年假？")
    assert result.result.status == "incomplete"
    assert result.model_steps <= settings.max_model_steps
    assert result.tool_calls <= settings.max_tool_calls


def test_tool_failure_cannot_create_citable_evidence(tools):
    model = ScriptedModel([read_call(), scripted_call("submit_result", normal_args())])
    output = KnowledgeAgent(model, FailureTools(tools, "read_passage"), Settings(max_model_steps=2)).run(
        "年假？"
    )
    assert output.result.status == "incomplete"


def test_invalid_json_unknown_tool_and_invalid_final_recover(tools, tmp_path):
    bad = ModelMessage.model_validate(
        {"tool_calls": [{"id": "bad", "function": {"name": "read_passage", "arguments": "{not json"}}]}
    )
    model = ScriptedModel(
        [
            bad,
            scripted_call("invented_tool", {}),
            scripted_call("submit_result", {}),
            read_call(),
            scripted_call("submit_result", normal_args()),
        ]
    )
    out = KnowledgeAgent(model, tools, Settings()).run("年假？", trace_path=tmp_path / "trace.jsonl")
    assert out.result.status == "completed"
    assert "invalid_json_arguments" in (tmp_path / "trace.jsonl").read_text(encoding="utf-8")


def test_text_without_tools_cannot_bypass_final_validation(tools):
    model = ScriptedModel([ModelMessage(content="年假提前100天申请。")])
    assert (
        KnowledgeAgent(model, tools, Settings(max_model_steps=1)).run("年假？").result.status == "incomplete"
    )


def test_history_reaches_model_without_promoting_roles(tools):
    class RecordingModel:
        mode = "offline_test"

        def respond(self, messages, tools):
            assert messages[1] == {"role": "user", "content": "星云导出"}
            assert messages[-1]["content"] == "普通成员呢？"
            return scripted_call(
                "submit_result",
                {"status": "insufficient_evidence", "answer": "测试占位", "gaps": ["测试不执行资料读取"]},
            )

    output = KnowledgeAgent(RecordingModel(), tools, Settings()).run(
        "普通成员呢？", [HistoryMessage(role="user", content="星云导出")]
    )
    assert output.result.status == "insufficient_evidence"


def test_source_replacement_rejects_stale_quote(store):
    old_answer = Answer.model_validate(normal_args())
    original = store.read("D03", "2026-02-01", "p001")
    changed = original.model_copy(update={"text": original.text.replace("3个工作日", "5个工作日")})
    assert validate_citations(old_answer, {changed.key: changed})


def test_tool_timeout_does_not_register_evidence(tools, monkeypatch):
    def slow(name, arguments):
        time.sleep(0.05)
        return {"ok": True}

    monkeypatch.setattr(tools, "dispatch", slow)
    assert tools.invoke("read_passage", {}, 0.001)["error"] == "tool_timeout"


def test_cannot_overwrite_trace(tools, tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        KnowledgeAgent(ScriptedModel([]), tools, Settings()).run("问题", trace_path=path)
    assert path.read_text() == "original"
