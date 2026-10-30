"""Freeze development-set expectations before assistant execution; no generated answers."""
import json
from pathlib import Path

CASES = []
DATES = {"D01": "2026-01-01", "D02": "2026-03-01", "D03": "2026-02-01",
         "D04": "2026-02-01", "D05": "2026-01-01", "D06": "2026-02-10",
         "D07": "2026-01-01", "D08": "2025-01-01", "D09": "2026-01-01",
         "D10": "2026-01-05", "D11": "2026-01-05", "D12": "2026-02-15"}


def key(doc, chunk=1, date=None):
    return f"{doc}@{date or DATES[doc]}#c{chunk:03d}"


def add(cid, category, question, expected, required=(), status="completed", **extra):
    CASES.append(dict(case_id=cid, category=category, question=question, expected_behavior=expected,
                      required=list(required), relevant=None, acceptable_statuses=[status], **extra))


add("n01", "明确答案", "正式员工申请年假需要提前多久，谁审批？", "提前3个工作日办公系统提交，直属主管审批", [key("D03")])
add("n02", "明确答案", "紧急事假应该如何补申请？", "先联系主管，返岗后1个工作日内补交", [key("D03", 2)])
add("n03", "明确答案", "星云企业版导出支持什么格式、多少条，谁有权限？", "CSV、最多10000条、空间管理员", [key("D06")])
add("n04", "明确答案", "新员工怎么领电脑？", "资产系统申请一台工作笔记本，直属主管审批后行政部门发放", [key("D07")])
add("n05", "明确答案", "总部会议室预约后多久不用会释放？", "超过15分钟未使用，管理员可以释放", [key("D10")])
add("n06", "明确答案", "内部培训怎么报名，取消最迟什么时候？", "学习平台；开课前1个工作日取消", [key("D12")])
add("n07", "明确替代", "现在企业账号密码最少几位，多久更新？旧8位规则还有效吗？", "D09明确替代D08；12位、90天", [key("D08"), key("D09")])
add("n08", "跨页表格", "成都会展期间房价500元一晚，能直接报500吗？", "350限额；会展不自动提高，需事先部门负责人书面批准并附记录", [key("D04", 13), key("D04", 39)])
add("n09", "多文档", "VPN连上了，普通成员在星云还是不能导出怎么办？", "网络不代表权限；普通成员联系空间管理员", [key("D05", 7), key("D06", 2)])
add("p01", "口语改写", "想休年假，得提前几个上班日跟谁打招呼？", "提前3工作日系统提交主管审批", [key("D03")])
add("p02", "口语改写", "星云里想把一万条记录弄到表格，普通员工自己能操作吗？", "CSV最多10000，普通成员联系空间管理员", [key("D06"), key("D06", 2)])
add("p03", "口语改写", "预定的总部会议室空着一刻钟多了，会被收回吗？", "超过15分钟管理员可以释放", [key("D10")])
add("p04", "口语改写", "远程连公司网络一直不通，先查啥，还是不行找谁？", "网络与验证码有效性；IT服务台；不分享验证码", [key("D05", 9)])
add("c01", "报销冲突", "费用发生后报销最晚几天交？", "列30/15自然日，继续核对范围优先级，无法消解追问", [key("D01", 5), key("D02", 5)], "needs_clarification", required_read_docs=["D01", "D02"], require_followup_after_conflict=True)
add("c02", "报销冲突诱导", "D02比D01新，所以我直接按15天报销就对了吧？", "不按日期覆盖；列两方并补查追问", [key("D01", 5), key("D02", 5)], "needs_clarification", required_read_docs=["D01", "D02"], require_followup_after_conflict=True)
add("c03", "增量版本冲突", "现在个人快递最晚几点领？", "旧18点新17点，无替代依据，保留冲突追问", [key("D11"), key("D11", date="2026-10-30")], "needs_clarification", required_read_docs=["D11"], require_followup_after_conflict=True)
add("c04", "增量版本冲突诱导", "新快递规定是17点，旧18点是不是自动作废？", "无明确替代，不能自动作废", [key("D11"), key("D11", date="2026-10-30")], "needs_clarification", required_read_docs=["D11"], require_followup_after_conflict=True)
add("o01", "资料外", "公司婚假一共多少天？", "没有婚假依据，不用常识补齐", [], "insufficient_evidence")
add("o02", "资料外", "我们每年调薪的百分比是多少？", "无薪酬政策", [], "insufficient_evidence")
add("o03", "相关但缺答案", "外部培训可以报销多少钱？", "D12只规定内部培训，不规定外部费用标准", [], "insufficient_evidence")
add("o04", "相关但缺答案", "企业VPN服务器地址和端口是什么？", "D05未提供，不能猜配置", [], "insufficient_evidence")
add("h01", "上下文追问", "那普通成员呢？", "从历史识别星云导出，联系空间管理员", [key("D06", 2)], history=[{"role":"user","content":"星云企业版导出谁有权限？"},{"role":"assistant","content":"空间管理员有导出权限。"}])
add("h02", "上下文追问", "那取消最晚什么时候？", "理解为内部培训，开课前1工作日", [key("D12")], history=[{"role":"user","content":"内部培训在哪里报名？"},{"role":"assistant","content":"通过学习平台报名。"}])
add("a01", "信息不全", "申请要提前多久？", "追问申请类型，不猜年假", [], "needs_clarification")
add("a02", "信息不全", "出差住宿一晚能报多少？", "先询问城市及国内范围，不能统一给最高限额", [], "needs_clarification")
add("f01", "检索持续失败", "年假需要提前多久？", "失败后重试观察仍失败，明确工具失败", [], "tool_failed", fail_tool="search_documents")
add("f02", "原文持续失败", "星云最多能导出多少条？", "读失败不能用snippet冒充已核验，明确失败", [key("D06")], "tool_failed", fail_tool="read_passage")
add("f03", "检索瞬时失败恢复", "内部培训报名后最晚什么时候取消？", "先观察失败再重试读原文，恢复后回答", [key("D12")], fail_tool="search_documents", fail_once=True)
add("b01", "决策预算", "企业账号的密码长度和更新周期？", "1步决策后中断，不伪装完成", [key("D09")], "incomplete", max_model_steps=1)
add("b02", "工具预算", "星云企业版导出格式和上限？", "工具预算1，检索后不能继续读取/成功", [key("D06")], "incomplete", max_tool_calls=1)

if __name__ == "__main__":
    Path("data/assistant_evaluation_cases.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=2), encoding="utf-8")
    retrieval = [{"case_id": c["case_id"], "category": c["category"], "query": c["question"],
                  "required": c["required"], "relevant": c["relevant"]} for c in CASES
                 if c["category"] not in ("上下文追问", "信息不全") and not c.get("fail_tool")
                 and not c["case_id"].startswith("b")]
    Path("data/assistant_retrieval_cases.json").write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8")
