"""Additional descriptive statistics, without relabeling or rerunning retrieval."""
import json
from pathlib import Path

import numpy as np

from knowledge_agent.answer_evaluation import audit_run, load_trace, review_template, score_review

ROOT = Path("examples/assistant-evaluation")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


raw = json.loads((ROOT / "retrieval-2026-10-30/results.json").read_text(encoding="utf-8"))
stats = {}
for method, results in raw["results"].items():
    values = [r["elapsed_ms"] for r in results["cases"]]
    stats[method] = {"single_run_query_ms": {"p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "mean": float(np.mean(values))},
        "not_all_required_at_6": [r["case_id"] for r in results["cases"]
                                  if r["required"] and not r["metrics"]["all_required@6"]]}
positive = [r for r in raw["results"]["hybrid"]["cases"] if r["required"]]
for row in positive:
    events = load_trace(ROOT / "2026-10-30" / row["case_id"] / "trace.jsonl")
    read = set()
    for event in events:
        if event["event"] == "tool_result" and event["name"] == "read_passage" and event["observation"].get("ok"):
            obs = event["observation"]
            read.update(f"{p['doc_id']}@{p['version_date']}#{p['chunk_id']}" for p in [obs["passage"], *obs.get("related_passages", [])])
    row["all_required_read_after_followup"] = all(k.replace("#c", "#r") in read for k in row["required"])
stats["assistant_followup"] = {"eligible": len(positive),
    "all_required_read_count": sum(r["all_required_read_after_followup"] for r in positive),
    "note": "补查及父块/共同注释展开后的读取覆盖，不是首轮Recall"}
save(ROOT / "retrieval-2026-10-30/statistics.json", stats)

baseline = ROOT / "version-baseline-2026-10-30/v01"
actual = json.loads((baseline / "actual.json").read_text(encoding="utf-8"))
events = load_trace(baseline / "trace.jsonl")
review = review_template(actual)
review.update(status="reviewed", reviewer="Codex current assistant (non-independent)",
    all_factual_claims_reviewed=True, answer_correctness=1, answer_relevance=1, status_appropriate=True,
    notes="对照原文及仅单版本的相关文档列表审阅；不纳入主集30题平均分。")
for span in ["工作日18点前", "一楼收发室领取"]:
    review["claims"].append({"answer_span": span, "supported_by_read_evidence": True,
        "supporting_read_keys": ["D11@2026-01-05#r001"], "supporting_citation_indices": [0],
        "rationale": "基线仅有旧版，已读原文明确支持时间和地点"})
review["citation_judgments"] = [{"citation_index": 0, "supports_associated_claim": True,
                                "rationale": "原文支持工作日18点及收发室地点"}]
save(baseline / "review.json", review)
save(baseline / "score.json", score_review(actual, events, review))
save(baseline / "audit.json", audit_run(actual, events, {"acceptable_statuses": ["completed"]}))
save(ROOT / "version-comparison.json", {
    "mode": "assistant_in_the_loop", "question": "现在个人快递最晚几点领？",
    "baseline": {"data": "data/processed/corpus.json", "versions": 12, "passages": 107,
                 "actual": actual["result"]},
    "incremental": {"data": "data/processed-incremental-verified/corpus.json", "versions": 13,
        "passages": 108, "actual": json.loads((ROOT / "2026-10-30/c03/actual.json").read_text(encoding="utf-8"))["result"]},
    "conclusion": "旧版未被覆盖；新增17点文本后由18点完成答案变为双版本冲突追问；当前助手代行，不是HTTP模型验收"})
save(ROOT / "2026-10-30/annotation-errata.json", {
    "case_id": "n04", "original_expected": "主管审批后IT发放", "corrected_expected": "直属主管审批后行政部门发放",
    "reason": "出题参考了旧JSON印象，当前快照D07原文为行政部门。原input.json保留，正式数据集及汇总使用当前证据修正。",
    "audit_fix": "初次内存审计使用Python tuple坐标，与Trace JSON list比较误判6例；改用JSON模式并从实际文件重算，未修改任何工具或模型Trace。"})
print(json.dumps(stats, ensure_ascii=False, indent=2))
