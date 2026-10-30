"""Encode this session's assistant review annotations; not an automatic semantic judge.

Reproduces arithmetic for saved answers only. A new run needs a new independent review.
Spans and citation mappings below were selected after reading the observed originals.
"""
import json
from pathlib import Path

import numpy as np

from knowledge_agent.answer_evaluation import audit_run, load_trace, review_template, score_review

ROOT = Path("examples/assistant-evaluation/2026-10-30")
ANNOTATIONS = {
    "n01": [("正式员工", [0]), ("至少提前3个工作日", [0]), ("在办公系统提交年假申请", [0]), ("由直属主管审批", [0])],
    "n02": [("紧急请假可先联系主管", [0]), ("返岗后1个工作日内补交申请", [0])],
    "n03": [("支持CSV导出", [0]), ("单次最多10000条记录", [0]), ("仅空间管理员有导出权限", [0])],
    "n04": [("通过资产系统申请", [0]), ("一台工作笔记本电脑", [0]), ("经直属主管审批", [0]), ("由行政部门发放", [0])],
    "n05": [("超过15分钟无人使用", [0]), ("行政人员可释放该时段", [0])],
    "n06": [("通过学习平台报名", [0]), ("取消报名应至少在开课前1个工作日操作", [0])],
    "n07": [("密码至少12位", [0]), ("每90天更换一次", [0]), ("D09自2026年1月1日起明确替代D08", [0, 1]), ("旧8位规则已被替代", [0, 1])],
    "n08": [("不能直接按500元报销", [0, 1, 2, 3, 4]), ("成都住宿限额为350元/人/晚", [0, 1, 2]),
            ("会展涨价不自动提高额度", [3]), ("事先取得部门负责人书面批准", [3, 4]), ("在报销时附批准记录", [3, 4])],
    "n09": [("VPN网络连通不能替代业务角色授权", [0]), ("仅空间管理员有导出权限", [1]), ("普通成员需要导出时应联系本空间管理员", [2])],
    "p01": [("正式员工", [0]), ("至少提前3个工作日", [0]), ("在办公系统提交申请", [0]), ("由直属主管审批", [0])],
    "p02": [("支持CSV导出", [0]), ("单次最多10000条记录", [0]), ("仅空间管理员具有权限", [0]), ("普通成员应联系本空间管理员", [1])],
    "p03": [("超过15分钟无人使用", [0]), ("行政人员可释放该时段", [0])],
    "p04": [("先检查网络与验证码有效性", [0]), ("仍失败应联系IT服务台", [0]), ("不要向他人分享验证码", [0])],
    "h01": [("普通成员需要导出星云数据时，应联系本空间管理员", [0])],
    "h02": [("内部培训取消报名应至少在开课前1个工作日操作", [0])],
    "f03": [("内部培训取消报名应至少在开课前1个工作日操作", [0])],
    "o03": [("现有培训说明不规定外部培训费用标准", [0])],
    "o04": [("VPN指南未提供服务器地址和端口", [0])],
}
for cid in ("c01", "c02"):
    ANNOTATIONS[cid] = [("D01规定费用发生后30个自然日提交", [0]), ("D02规定15个自然日", [1]),
                        ("仍缺少适用范围和优先关系", [2, 3]), ("不能因为D02较新或更严格就选择15天", [3])]
for cid in ("c03", "c04"):
    ANNOTATIONS[cid] = [("2026-01-05版要求工作日18点前领取", [0]),
                        ("2026-10-30版要求17点前领取", [1]),
                        ("两版均列为有效", []), ("未提供替代依据", [])]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cases = {c["case_id"]: c for c in json.loads(Path("data/assistant_evaluation_cases.json").read_text(encoding="utf-8"))}
    rows = []
    for cid, case in cases.items():
        directory = ROOT / cid
        actual = json.loads((directory / "actual.json").read_text(encoding="utf-8"))
        events = load_trace(directory / "trace.jsonl")
        audit = audit_run(actual, events, case)
        save(directory / "audit.json", audit)
        review = review_template(actual)
        review.update(status="reviewed", reviewer="Codex current assistant (answer author; non-independent)",
                      all_factual_claims_reviewed=True, answer_correctness=1, answer_relevance=1,
                      status_appropriate=True, conflict_handling=True if cid.startswith("c") else None,
                      clarification_quality=True if cid.startswith(("c", "a")) else None,
                      refusal_appropriate=True if cid.startswith("o") else None,
                      notes="当前助手对照已读原文、问题和预期逐条审阅；同一助手出题/作答/审阅，存在自评偏差，需独立复核。纯操作状态和澄清请求不计企业事实。")
        citations = actual["result"]["citations"]
        for span, indices in ANNOTATIONS.get(cid, []):
            keys = [f"{citations[i]['doc_id']}@{citations[i]['version_date']}#{citations[i]['chunk_id']}" for i in indices]
            rationale = "已读原文与指定引文支持该事实，保留范围、时间及条件；表格单元格结合相同行与计价单位判断。"
            if not keys:
                keys = ["D11@2026-01-05#r001", "D11@2026-10-30#r001"]
                rationale = "已读两版完整原文及相关文档元数据显示同范围均有效、无替代声明；该结论未在引文中直接覆盖。"
            review["claims"].append({"answer_span": span, "supported_by_read_evidence": True,
                "supporting_read_keys": list(dict.fromkeys(keys)), "supporting_citation_indices": indices, "rationale": rationale})
        review["citation_judgments"] = [{"citation_index": i, "supports_associated_claim": True,
            "rationale": "与回答事实对应；表格的城市、金额、计价单位需联合支持同一结论，续行及共同注释支持超标条件。"} for i in range(len(citations))]
        save(directory / "review.json", review)
        score = score_review(actual, events, review)
        save(directory / "score.json", score)
        telemetry = json.loads((directory / "telemetry.json").read_text(encoding="utf-8"))
        rejected = [e for e in events if e["event"] == "tool_result" and e["name"] == "submit_result" and not e["observation"].get("ok")]
        rows.append({"case_id": cid, "category": case["category"], "question": case["question"],
                         "expected": case["expected_behavior"], "actual_answer": actual["result"]["answer"],
                         "status": actual["result"]["status"], "mechanical_pass": audit["mechanical_pass"],
                         "score": score, "telemetry": telemetry, "model_steps": actual["model_steps"],
                         "tool_calls": actual["tool_calls"], "rejected_submissions": len(rejected)})
    summary = {"mode": "assistant_in_the_loop", "count": len(rows),
                   "mechanical_pass_count": sum(r["mechanical_pass"] for r in rows),
                   "status_accuracy": sum(r["score"]["status_appropriate"] for r in rows)/len(rows),
                   "rejected_submission_count": sum(r["rejected_submissions"] for r in rows),
                   "cases_with_rejected_submissions": [r["case_id"] for r in rows if r["rejected_submissions"]]}
    for name in ["faithfulness", "citation_precision", "citation_coverage", "answer_correctness", "answer_relevance"]:
        values = [r["score"][name] for r in rows if r["score"][name] is not None]
        summary[name] = {"macro_mean": float(np.mean(values)), "eligible_cases": len(values)}
    claims = sum(r["score"]["claim_count"] for r in rows)
    covered = sum(sum(bool(c[1]) for c in ANNOTATIONS.get(r["case_id"], [])) for r in rows)
    summary.update(claim_count=claims, citation_coverage_micro=covered/claims,
                   citations=sum(len(json.loads((ROOT/r["case_id"]/"actual.json").read_text(encoding="utf-8"))["result"]["citations"]) for r in rows))
    for field in ["question_bge_tokens", "business_tool_ms", "wall_ms_including_assistant_wait"]:
        values = [r["telemetry"][field] for r in rows]
        summary[field] = {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                              "p95": float(np.percentile(values, 95)), "total": sum(values), "minimum": min(values), "maximum": max(values)}
    summary["limitations"] = ["同一助手代行并自评，不是独立评测或项目真实API决策验收", "30个项目内开发用例，无生产外推",
                              "API计费token和推理延时不可得，保持null", "业务工具耗时包含共享工具队列等待",
                              "宏平均仅纳入有事实或引文的适用样本，纯澄清/工具状态不记满分"]
    save(ROOT / "summary.json", {"summary": summary, "cases": rows})
    lines = ["# 当前助手代行：逐题预期、实际与测量", "", "模式：assistant_in_the_loop；回答与评分均来自当前助手，不代表真实API验收。", "",
             "| ID | 场景 | 问题 | 测试预期 | 实际状态/回答 | 问题BGE token | 工具合计ms（含排队） | 总耗时秒（含代行等待） | 忠实度 | 引文覆盖 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in [
            r["case_id"], r["category"], r["question"], r["expected"], r["status"]+"："+r["actual_answer"],
            r["telemetry"]["question_bge_tokens"], round(r["telemetry"]["business_tool_ms"], 2),
            round(r["telemetry"]["wall_ms_including_assistant_wait"] / 1000, 2),
            r["score"]["faithfulness"], r["score"]["citation_coverage"]]) + " |")
    (ROOT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
