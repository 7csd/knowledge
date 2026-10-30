# 测试记录

我于2026-10-30完成本轮测试记录。环境：Windows、Python 3.10.8；默认快照12个文档ID、13个版本、108块。

## 工程与数据测试

| 测试 | 预期 | 实际结果与记录 |
| --- | --- | --- |
| pytest回归 | 检索、引用、历史、失败、预算、增量等断言通过 | 本次72项通过，见 [tests.xml](../examples/final-review/tests.xml) |
| Ruff / pip check | 无静态错误、无依赖冲突 | 本次均通过，见 [checks.json](../examples/final-review/checks.json) |
| 离线回放 | 正常、冲突、资料不足、工具失败状态正确 | 四组输出与Trace见 [final-check](../examples/ingestion/final-check/)，模式offline_scripted |
| 解析入库 | OCR、跨页续行和注释可核对，来源变化可检测 | 已验证，见 [报告](../examples/ingestion/final-check/report.json) |
| 增量去重 | 保留旧版，仅处理新增；重复导入不新增 | 复用107块、新增1块；再次编码及追加均为0，见 [summary.json](../examples/incremental-update-verified/summary.json) |

## 交互式代行决策测试

我通过文件桥接，让Codex逐轮读取实际LangGraph/BGE/Qdrant工具观察并选择动作，模式assistant_in_the_loop。主集30题，另有1题版本对照；同一AI参与出题、作答及评分，属于开发集测试，评分具有自评偏差。

| 场景 / 用例 | 预期行为 | 实际结果 |
| --- | --- | --- |
| 明确答案 n01–n06、n09 | 读取原文，按条件回答 | 7题completed，有支持引用 |
| 明确替代 n07 | 凭原文替代声明选择新版 | completed，核对D08/D09 |
| 跨页表格 n08 | 联读限额、续行和共同注释 | completed；首次拼接引文被拒，拆分后成功 |
| 相似改写 p01–p04 | 口语问题找到对应条款 | 4题completed |
| 冲突 c01–c04 | 补查后无法消解则列双方并追问 | 4题needs_clarification，未按日期选择制度 |
| 资料外及缺答案 o01–o04 | 不编造数值，说明缺口 | 4题insufficient_evidence |
| 上下文 h01–h02 | 结合历史理解导出和取消事项 | 2题completed，实际改写查询 |
| 缺信息 a01–a02 | 追问事项、城市等必要信息 | 2题needs_clarification |
| 工具失败 f01–f03 | 观察错误后重试或明确失败 | 持续失败2题tool_failed，瞬时失败1题恢复完成 |
| 上限 b01–b02 | 限制轮次或工具次数后明确未完成 | 2题incomplete |
| 文档变化 v01 / c03 | 新增冲突版本后答案和动作变化 | 旧版答18点；加入17点版后补查追问，见 [版本对照](../examples/assistant-evaluation/version-comparison.json) |

每题输入、预期、实际回答、引用和Trace见 [逐题记录](../examples/assistant-evaluation/2026-10-30/report.md)。动态路径包括n01“检索→读取→回答”、c01“读取双方→补查→追问”、f03“失败→重试→读取→回答”。

机械审计30/30通过。本轮AI自评：63项事实忠实度100%，38条最终引文精确率100%，引文覆盖宏平均95.45%（微平均93.65%）。c03/c04未直接引用有效性等全部依据；n04参考答案标注更正、n08首次失败均留痕。见 [汇总](../examples/assistant-evaluation/2026-10-30/summary.json) 和 [更正](../examples/assistant-evaluation/2026-10-30/annotation-errata.json)。

## 执行路径与上限记录

我选取以下实际记录对应考核中的执行路径要求。表中概括工具观察与后续动作，完整参数、返回值及简短行动依据保存在各Trace中，未记录模型内部思维过程。

| 路径 | 工具参数与返回摘要 | 后续动作及依据 | 原始记录 |
| --- | --- | --- | --- |
| 正常 n01 | search_documents(query="正式员工申请年假需要提前多久，谁审批？")返回D03；read_passage(D03, 2026-02-01, c001)读到提前3工作日、主管审批 | 原文充分，submit_result提交completed | [Trace](../examples/assistant-evaluation/2026-10-30/n01/trace.jsonl)、[输出](../examples/assistant-evaluation/2026-10-30/n01/actual.json) |
| 冲突 c01 | search_documents(query="费用发生后报销最晚几天交？")；读取D01@2026-01-01/c005和D02@2026-03-01/c005，分别为30、15自然日 | 调用list_related_documents补查双方，再读D01/c001、D02/c008；范围和优先级仍缺，提交needs_clarification | [Trace](../examples/assistant-evaluation/2026-10-30/c01/trace.jsonl)、[输出](../examples/assistant-evaluation/2026-10-30/c01/actual.json) |
| 失败恢复 f03 | 首次search_documents返回injected_tool_failure；改查query="内部培训 取消报名"成功，读取D12@2026-02-15/c001 | 根据失败观察重试，恢复后核验原文，回答开课前1工作日取消，提交completed | [Trace](../examples/assistant-evaluation/2026-10-30/f03/trace.jsonl)、[输出](../examples/assistant-evaluation/2026-10-30/f03/actual.json) |

默认上限为8次模型决策、12次工具调用，配置在.env.example及Settings，执行控制在agent.py。b01将模型步数限为1，返回incomplete和“达到模型决策步数上限”；b02将工具次数限为1，返回incomplete和“达到工具调用次数上限”。见 [b01输出](../examples/assistant-evaluation/2026-10-30/b01/actual.json)、[b01 Trace](../examples/assistant-evaluation/2026-10-30/b01/trace.jsonl)、[b02输出](../examples/assistant-evaluation/2026-10-30/b02/actual.json)、[b02 Trace](../examples/assistant-evaluation/2026-10-30/b02/trace.jsonl)。

## 检索实测

我从主集选取21条首轮检索问题（17条有指定证据、4条缺答案），top-k=6：

| 方法 | 指定证据Recall@6 | MRR@6 |
| --- | ---: | ---: |
| BM25 | 85.29% | 0.7745 |
| BGE | 94.12% | 0.8725 |
| 混合 | 94.12% | 0.8843 |

混合首轮漏齐n08注释、c02指定期限片段，后续补查及关联读取后17/17读齐，不能将首轮召回改记100%。见 [检索记录](../examples/assistant-evaluation/retrieval-2026-10-30/results.json)。缺少完整相关性标签，Precision/nDCG未评分；API计费token和推理延时不可得，保留null。

## 测试范围

本次交互式测试使用assistant_in_the_loop模式，实际经过项目工具循环、BGE检索和Qdrant原文读取；未调用外部chat/completions接口。结果来自小规模合成开发集及AI自评，不外推其他模型或生产资料。模型URL、Key和名称由运行者在本地配置，不随仓库提交。
