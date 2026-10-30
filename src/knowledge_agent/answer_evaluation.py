"""Separate mechanical trace checks from reviewed semantic judgments; never invent scores."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .schemas import Passage, RunResult, StrictModel
from .tools import validate_citations


def audit_run(actual: dict, events: list[dict], expectations: dict | None = None) -> dict:
    run = RunResult.model_validate(actual)
    expectations = expectations or {}
    evidence, reads, cited_pages = {}, set(), {}
    actions, failed_tools = [], []
    for event in events:
        if event["event"] == "tool_request":
            actions.append(event["name"])
        if event["event"] != "tool_result":
            continue
        observation = event["observation"]
        if not observation.get("ok"):
            failed_tools.append(event["name"])
        if event["name"] == "read_passage" and observation.get("ok"):
            for item in [observation["passage"], *observation.get("related_passages", [])]:
                p = Passage.model_validate(item)
                evidence[p.key] = p
                reads.add(p.doc_id)
    for citation in run.result.citations:
        for loc in citation.source_locations:
            if loc.page is not None:
                cited_pages.setdefault(citation.doc_id, set()).add(loc.page)
    starts = [e for e in events if e["event"] == "run_start"]
    ends = [e for e in events if e["event"] == "run_end"]
    citation_errors = validate_citations(run.result, evidence)
    checks = {
        "output_schema": True,
        "trace_has_start_and_end": len(starts) == len(ends) == 1,
        "run_identity_consistent": bool(events) and all(e["run_id"] == run.run_id for e in events),
        "trace_sequence_continuous": [e["sequence"] for e in events] == list(range(len(events))),
        "trace_output_matches": bool(ends) and ends[-1]["output"] == actual,
        "tool_count_matches_trace": len(actions) == run.tool_calls,
        "model_count_matches_trace": sum(e["event"] == "model_request" for e in events) == run.model_steps,
        "citations_read_and_verbatim": not citation_errors,
        "noncompletion_has_gaps": run.result.status == "completed" or bool(run.result.gaps),
        "no_unresolved_conflicts_marked_completed": not run.result.conflicts or run.result.status != "completed",
    }
    if starts:
        start = starts[0]
        checks["execution_budget_respected"] = (
            run.model_steps <= start["max_model_steps"] and run.tool_calls <= start["max_tool_calls"])
        checks["mode_matches_trace"] = start["model_mode"] == run.model_mode
    expected_status = expectations.get("acceptable_statuses", [])
    if expected_status:
        checks["expected_status"] = run.result.status in expected_status
    if expectations.get("required_read_docs"):
        checks["required_documents_read"] = set(expectations["required_read_docs"]) <= reads
    if expectations.get("required_citation_docs"):
        checks["required_documents_cited"] = set(expectations["required_citation_docs"]) <= {
            c.doc_id for c in run.result.citations}
    if expectations.get("required_source_pages"):
        checks["required_pages_cited"] = all(set(pages) <= cited_pages.get(doc, set())
                                               for doc, pages in expectations["required_source_pages"].items())
    if expectations.get("require_followup_after_conflict"):
        required = set(expectations.get("required_read_docs", []))
        observed, complete_at = set(), None
        for event in events:
            if (event["event"] == "tool_result" and event["name"] == "read_passage"
                    and event["observation"].get("ok")):
                observed.add(event["observation"]["passage"]["doc_id"])
                if required and required <= observed and complete_at is None:
                    complete_at = event["sequence"]
        checks["followup_observation_after_both_sources"] = complete_at is not None and any(
            e["sequence"] > complete_at and e["event"] == "tool_result"
            and e["name"] in {"search_documents", "list_related_documents", "read_passage"} for e in events)
    if expectations.get("fail_tool"):
        checks["injected_failure_observed"] = expectations["fail_tool"] in failed_tools
    return {"checks": checks, "mechanical_pass": all(checks.values()), "citation_errors": citation_errors,
            "read_documents": sorted(reads), "actions": actions, "observed_failed_tools": failed_tools,
            "live_model_run": run.model_mode == "live", "semantic_pass": None,
            "note": "引用存在/动作发生不是忠实度、相关性或补查有效性的语义证明"}


def review_template(actual: dict) -> dict:
    run = RunResult.model_validate(actual)
    digest = hashlib.sha256(json.dumps(actual, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"run_id": run.run_id, "artifact_sha256": digest, "status": "pending", "reviewer": "",
            "all_factual_claims_reviewed": False, "claims": [], "citation_judgments": [],
            "answer_correctness": None, "answer_relevance": None, "status_appropriate": None,
            "conflict_handling": None, "clarification_quality": None, "refusal_appropriate": None,
            "notes": "对照原问题、参考事实、完整原文与Trace审阅；无事实断言时忠实度为null，不能计满分。"}


class ClaimReview(StrictModel):
    answer_span: str = Field(min_length=1)
    supported_by_read_evidence: bool
    supporting_read_keys: list[str] = Field(default_factory=list)
    supporting_citation_indices: list[int] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class CitationJudgment(StrictModel):
    citation_index: int = Field(ge=0)
    supports_associated_claim: bool
    rationale: str = Field(min_length=1)


class SemanticReview(StrictModel):
    run_id: str
    artifact_sha256: str
    status: Literal["pending", "reviewed"]
    reviewer: str
    all_factual_claims_reviewed: bool
    claims: list[ClaimReview]
    citation_judgments: list[CitationJudgment]
    answer_correctness: float | None = Field(ge=0, le=1)
    answer_relevance: float | None = Field(ge=0, le=1)
    status_appropriate: bool | None
    conflict_handling: bool | None
    clarification_quality: bool | None
    refusal_appropriate: bool | None
    notes: str


def score_review(actual: dict, events: list[dict], raw_review: dict) -> dict:
    review = SemanticReview.model_validate(raw_review)
    expected = review_template(actual)
    if review.run_id != expected["run_id"] or review.artifact_sha256 != expected["artifact_sha256"]:
        raise ValueError("审阅记录对应的运行或回答已变化，不能复用旧评分")
    if review.status == "pending":
        return {"review_status": "pending", "faithfulness": None, "citation_precision": None,
                "citation_coverage": None, "answer_correctness": None, "answer_relevance": None}
    if not review.reviewer.strip() or not review.all_factual_claims_reviewed:
        raise ValueError("语义评分需审阅人标识及全部事实断言审阅声明")
    if review.answer_correctness is None or review.answer_relevance is None or review.status_appropriate is None:
        raise ValueError("已审阅记录必须包含正确性、相关性和状态适当性")
    run = RunResult.model_validate(actual)
    citations = run.result.citations
    evidence = {}
    for event in events:
        if event["event"] == "tool_result" and event["name"] == "read_passage" and event["observation"].get("ok"):
            obs = event["observation"]
            for item in [obs["passage"], *obs.get("related_passages", [])]:
                p = Passage.model_validate(item)
                evidence[p.key] = p
    judged = [j.citation_index for j in review.citation_judgments]
    if sorted(judged) != list(range(len(citations))):
        raise ValueError("必须逐条审阅所有引文，不得重复或遗漏")
    valid_citation_indices = {j.citation_index for j in review.citation_judgments if j.supports_associated_claim}
    for claim in review.claims:
        if claim.answer_span not in run.result.answer:
            raise ValueError("审阅断言必须定位到实际回答文字")
        if any(key not in evidence for key in claim.supporting_read_keys):
            raise ValueError("审阅依据不在本次已读取证据内")
        if claim.supported_by_read_evidence and not claim.supporting_read_keys:
            raise ValueError("支持断言的判断必须列出已读取证据ID")
        if any(i not in valid_citation_indices for i in claim.supporting_citation_indices):
            raise ValueError("引用覆盖不能关联不存在或被判无支持性的引文")
        if claim.supporting_citation_indices and not claim.supported_by_read_evidence:
            raise ValueError("引文支持与已读取证据支持的判断相互矛盾")
        if any(citations[i].key not in claim.supporting_read_keys for i in claim.supporting_citation_indices):
            raise ValueError("引文与断言的已读依据必须对应")
    n = len(review.claims)
    result = {"review_status": "reviewed", "reviewer": review.reviewer,
              "faithfulness": sum(c.supported_by_read_evidence for c in review.claims) / n if n else None,
              "citation_precision": len(valid_citation_indices) / len(citations) if citations else None,
              "citation_coverage": sum(bool(c.supporting_citation_indices) for c in review.claims) / n if n else None,
              "answer_correctness": review.answer_correctness, "answer_relevance": review.answer_relevance,
              "status_appropriate": review.status_appropriate, "conflict_handling": review.conflict_handling,
              "clarification_quality": review.clarification_quality, "refusal_appropriate": review.refusal_appropriate,
              "claim_count": n, "score_origin": "review_annotations_not_automatic_entailment"}
    result["critical_failure"] = (
        not audit_run(actual, events)["mechanical_pass"]
        or
        (result["faithfulness"] is not None and result["faithfulness"] < 1)
        or (result["citation_precision"] is not None and result["citation_precision"] < 1)
        or not review.status_appropriate or review.conflict_handling is False
    )
    return result


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
