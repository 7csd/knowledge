import copy

import pytest

from knowledge_agent.answer_evaluation import audit_run, load_trace, review_template, score_review
from knowledge_agent.config import Settings
from knowledge_agent.offline import run_demo
from knowledge_agent.retrieval_evaluation import metrics
from knowledge_agent.schemas import Answer


@pytest.fixture
def actual_run(tmp_path):
    artifact = run_demo("normal", tmp_path / "normal", Settings())
    return artifact["actual"], load_trace(tmp_path / "normal/trace.jsonl")


def test_pending_review_never_implies_perfect_faithfulness(actual_run):
    actual, events = actual_run
    result = score_review(actual, events, review_template(actual))
    assert result["faithfulness"] is None and result["citation_precision"] is None
    assert audit_run(actual, events)["semantic_pass"] is None


def test_valid_quote_does_not_prove_answer_faithfulness(actual_run):
    actual, events = copy.deepcopy(actual_run)
    actual["result"]["answer"] = "年假要提前100天申请。"
    events[-1]["output"] = actual
    assert audit_run(actual, events)["mechanical_pass"]
    review = review_template(actual)
    review.update(status="reviewed", reviewer="unit-test-fixture-not-model-evaluation",
                  all_factual_claims_reviewed=True, answer_correctness=0, answer_relevance=1,
                  status_appropriate=False,
                  claims=[{"answer_span": "提前100天", "supported_by_read_evidence": False,
                           "supporting_read_keys": [], "supporting_citation_indices": [],
                           "rationale": "引文规定3个工作日，并不支持100天"}],
                  citation_judgments=[{"citation_index": 0, "supports_associated_claim": False,
                                       "rationale": "原文存在，但不支持回答中的期限"}])
    result = score_review(actual, events, review)
    assert result["faithfulness"] == result["citation_precision"] == result["citation_coverage"] == 0
    assert result["critical_failure"]


def test_faithful_annotation_requires_actual_evidence_keys(actual_run):
    actual, events = actual_run
    review = review_template(actual)
    review.update(status="reviewed", reviewer="unit-fixture", all_factual_claims_reviewed=True,
                  answer_correctness=1, answer_relevance=1, status_appropriate=True,
                  claims=[{"answer_span": "提前3个工作日", "supported_by_read_evidence": True,
                           "supporting_read_keys": ["D03@2026-02-01#p001"],
                           "supporting_citation_indices": [0], "rationale": "与已读原文一致"}],
                  citation_judgments=[{"citation_index": 0, "supports_associated_claim": True,
                                       "rationale": "原文直接规定提前3个工作日"}])
    result = score_review(actual, events, review)
    assert result["faithfulness"] == result["citation_precision"] == result["citation_coverage"] == 1
    review["claims"][0]["supporting_read_keys"] = ["invented"]
    with pytest.raises(ValueError, match="已读取"):
        score_review(actual, events, review)


def test_review_cannot_be_reused_after_answer_changes(actual_run):
    actual, events = actual_run
    review = review_template(actual)
    actual["result"]["answer"] += "另一条事实。"
    with pytest.raises(ValueError, match="已变化"):
        score_review(actual, events, review)


def test_empty_claim_set_is_not_faithfulness_one(tmp_path):
    artifact = run_demo("missing", tmp_path / "missing", Settings())
    actual = artifact["actual"]
    events = load_trace(tmp_path / "missing/trace.jsonl")
    review = review_template(actual)
    review.update(status="reviewed", reviewer="unit-fixture", all_factual_claims_reviewed=True,
                  answer_correctness=1, answer_relevance=1, status_appropriate=True,
                  refusal_appropriate=True)
    assert score_review(actual, events, review)["faithfulness"] is None


def test_expectations_require_both_conflict_documents(actual_run):
    actual, events = actual_run
    result = audit_run(actual, events, {"required_citation_docs": ["D01", "D02"]})
    assert result["checks"]["required_documents_cited"] is False
    assert not result["mechanical_pass"]


def test_unresolved_conflicts_cannot_be_completed():
    with pytest.raises(ValueError, match="未解决冲突"):
        Answer(status="completed", answer="采用15天。", conflicts=["30与15天冲突"], citations=[
            {"doc_id": "D02", "version_date": "2026-03-01", "chunk_id": "p001", "quote": "15天"}])


def test_precision_ndcg_require_complete_relevance_labels():
    incomplete = metrics(["a"], ["a", "b"])
    assert incomplete["precision@3"] is None and incomplete["ndcg@3"] is None
    complete = metrics(["a"], ["a", "b", "x"], ["a", "b"])
    assert complete["precision@3"] == pytest.approx(2 / 3)
    assert complete["ndcg@3"] == 1
    reordered = metrics(["a"], ["x", "b", "a"], ["a", "b"])
    assert reordered["ndcg@3"] < complete["ndcg@3"]


def test_evaluation_case_budget_override(tools, tmp_path):
    import json

    from knowledge_agent.evaluation import evaluate_live
    from knowledge_agent.offline import ScriptedModel, scripted_call

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"case_id": "limit", "category": "预算", "question": "年假？",
                                  "expected_behavior": "未完成", "max_model_steps": 1,
                                  "acceptable_statuses": ["incomplete"]}]), encoding="utf-8")
    model = ScriptedModel([scripted_call("search_documents", {"query": "年假"})])
    summary = evaluate_live(model, tools, Settings(), cases, tmp_path / "evaluation")
    assert summary["cases"][0]["status"] == "incomplete"
    assert summary["cases"][0]["mechanical_pass"] is True
