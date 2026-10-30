"""Explicitly scripted fixtures. These are NOT model-driven decision evidence."""

import json
from pathlib import Path

from .agent import KnowledgeAgent
from .evaluation import FailureTools
from .model import ModelMessage
from .retrieval import HybridRetriever
from .schemas import Answer, Citation
from .store import DocumentStore
from .tools import KnowledgeTools


def scripted_call(name: str, arguments: dict, call_id: str = "fixture_call") -> ModelMessage:
    return ModelMessage.model_validate(
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
        }
    )


class ScriptedModel:
    mode = "offline_scripted"

    def __init__(self, messages: list[ModelMessage]):
        self.messages = iter(messages)

    def respond(self, messages, tools):
        from .model import ModelError

        try:
            return next(self.messages)
        except StopIteration:
            raise ModelError("offline_script_exhausted") from None


def run_demo(scenario: str, output_dir: Path, settings) -> dict:
    if scenario not in {"normal", "conflict", "missing", "failure"}:
        raise ValueError("未知离线场景")
    output_dir.mkdir(parents=True, exist_ok=False)
    store = DocumentStore.load(settings.data_path)
    tools = KnowledgeTools(store, HybridRetriever(store))
    readable = list(getattr(store, "parents", store.passages).values())
    normal = next(p for p in readable if p.doc_id == "D03" and "提前3个工作日" in p.text)
    left = next(p for p in readable if p.doc_id == "D01" and "报销期限为费用发生后30个自然日" in p.text)
    right = next(p for p in readable if p.doc_id == "D02" and "报销期限为费用发生后15个自然日" in p.text)

    def cite(p):
        needle = "提前3个工作日" if p.doc_id == "D03" else "报销期限为费用发生后"
        quote = next((loc.text for loc in p.source_locations if needle in loc.text), p.text)
        return Citation(doc_id=p.doc_id, version_date=p.version_date, chunk_id=p.chunk_id, quote=quote)

    def read(p):
        return scripted_call(
            "read_passage", {"doc_id": p.doc_id, "version_date": p.version_date, "chunk_id": p.chunk_id}
        )

    def related(p):
        return scripted_call("list_related_documents", {"doc_id": p.doc_id, "version_date": p.version_date})

    if scenario == "normal":
        question = "申请年假需要提前多久？"
        actions = [scripted_call("search_documents", {"query": question}), read(normal), related(normal)]
        answer = Answer(
            status="completed",
            answer="应至少提前3个工作日在办公系统提交，并由直属主管审批。",
            citations=[cite(normal)],
        )
    elif scenario == "conflict":
        question = "报销期限多久？"
        actions = [
            scripted_call("search_documents", {"query": question}),
            read(left),
            related(left),
            read(right),
            scripted_call(
                "search_documents", {"query": "报销期限 适用部门 替代 优先关系", "topic": "报销期限"}
            ),
        ]
        answer = Answer(
            status="needs_clarification",
            answer="资料分别规定30与15个自然日，请确认适用哪份制度。",
            citations=[cite(left), cite(right)],
            gaps=["适用范围和制度优先级未说明"],
            conflicts=["D01规定30个自然日，D02规定15个自然日"],
        )
    elif scenario == "missing":
        question = "公司购房补贴多少钱？"
        actions = [
            scripted_call("search_documents", {"query": question}),
            scripted_call("search_documents", {"query": "员工 买房 住房补助"}),
        ]
        answer = Answer(
            status="insufficient_evidence",
            answer="当前知识库没有可确认的购房补贴规定。",
            gaps=["缺少购房补贴制度"],
        )
    else:
        question = "申请年假需要提前多久？"
        actions = [scripted_call("search_documents", {"query": question}), read(normal)]
        answer = Answer(
            status="tool_failed",
            answer="原文读取失败，暂时无法核实年假申请规定。",
            gaps=["原文读取工具不可用"],
        )
    actions.append(scripted_call("submit_result", answer.model_dump()))
    try:
        active_tools = FailureTools(tools, "read_passage") if scenario == "failure" else tools
        result = (
            KnowledgeAgent(ScriptedModel(actions), active_tools, settings)
            .run(question, trace_path=output_dir / "trace.jsonl")
            .model_dump()
        )
        artifact = {
            "warning": "离线预设动作回放，仅验证工程流程；不代表真实模型决策或语义检索效果。",
            "scenario": scenario,
            "input": question,
            "actual": result,
        }
        (output_dir / "result.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return artifact
    finally:
        tools.close()
