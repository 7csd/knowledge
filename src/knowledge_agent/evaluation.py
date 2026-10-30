import json
from pathlib import Path

from pydantic import Field, TypeAdapter

from .agent import KnowledgeAgent
from .answer_evaluation import audit_run, load_trace, review_template
from .schemas import HistoryMessage, StrictModel


class EvaluationCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    category: str
    question: str
    history: list[HistoryMessage] = Field(default_factory=list)
    expected_behavior: str
    fail_tool: str | None = None
    acceptable_statuses: list[str] = Field(default_factory=list)
    required_read_docs: list[str] = Field(default_factory=list)
    required_citation_docs: list[str] = Field(default_factory=list)
    required_source_pages: dict[str, list[int]] = Field(default_factory=dict)
    require_followup_after_conflict: bool = False
    max_model_steps: int | None = Field(default=None, ge=1, le=100)
    max_tool_calls: int | None = Field(default=None, ge=1, le=200)


class FailureTools:
    """Explicit failure injection for evaluation only; never rewrites other observations."""

    def __init__(self, tools, fail_tool):
        self.store, self.retriever = tools.store, tools.retriever
        self.tools, self.fail_tool = tools, fail_tool

    def invoke(self, name, arguments, timeout):
        if name == self.fail_tool:
            return {"ok": False, "error": "injected_tool_failure", "message": "验收故障注入"}
        return self.tools.invoke(name, arguments, timeout)


def evaluate_live(model, tools, settings, cases_path: Path, output_dir: Path) -> dict:
    cases = TypeAdapter(list[EvaluationCase]).validate_json(cases_path.read_text(encoding="utf-8-sig"))
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("验收用例 ID 不允许重复")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        active_tools = FailureTools(tools, case.fail_tool) if case.fail_tool else tools
        limits = {k: getattr(case, k) for k in ["max_model_steps", "max_tool_calls"] if getattr(case, k) is not None}
        case_settings = settings.model_copy(update=limits)
        output = KnowledgeAgent(model, active_tools, case_settings).run(
            case.question, case.history, output_dir / f"{case.case_id}.trace.jsonl"
        )
        artifact = {
            "input": case.model_dump(),
            "actual": output.model_dump(),
            "semantic_review": "pending",
            "note": "状态和引文形式校验不等于答案语义正确，需审阅原文与动作",
        }
        audit = audit_run(output.model_dump(), load_trace(output_dir / f"{case.case_id}.trace.jsonl"),
                          case.model_dump())
        artifact["mechanical_audit"] = audit
        (output_dir / f"{case.case_id}.review.json").write_text(
            json.dumps(review_template(output.model_dump()), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / f"{case.case_id}.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append({"case_id": case.case_id, "status": output.result.status, "semantic_review": "pending",
                     "mechanical_pass": audit["mechanical_pass"]})
    summary = {"model_mode": model.mode, "retrieval_mode": tools.retriever.mode, "cases": rows}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
