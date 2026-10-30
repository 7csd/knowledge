# 合成知识材料

全部资料为虚构企业“澄明科技”的制度和FAQ，不含真实员工数据。PDF及模拟扫描件为程序生成的测试资料，不能代表真实复杂扫描场景。

## 当前数据

- 14个物理文件：9份Markdown、3份原生PDF、2份模拟扫描PDF，共12个文档ID、13个版本。
- manifest.json登记版本、范围、状态、MD5和SHA256。默认纳入13份canonical文件；D05扫描件为格式对照，不重复索引。
- 当前快照：`data/processed-incremental-verified/corpus.json`，108个父块、108个检索块。
- `data/processed/corpus.json`为107块历史基线；原JSON基线为data/documents.json。

## 内容与验证用途

| 文档 | 内容 |
| --- | --- |
| D01 / D02 | 报销期限30 / 15自然日，无明确优先级的冲突 |
| D03 | 年假及紧急请假申请 |
| D04 | 住宿限额、32行跨页表格、续行和共同注释 |
| D05 | VPN指南，原生及模拟扫描对照 |
| D06 / D07 | 星云导出FAQ、设备申领 |
| D08 / D09 | 密码旧新版，原文明示替代 |
| D10 / D12 | 会议室规范、培训报名 |
| D11两版 | 工作日18点 / 17点领取快递，保留双方供查证 |

5份PDF共24页。D04的T08在第2、3页续行，共同注释在第5页；D02、D05扫描件无隐藏文字层，已执行实际OCR。

源文件检查见 [检查结果](../../examples/source-materials/file-checks.json)，解析、OCR与跨页验证见 [测试报告](../../examples/ingestion/final-check/report.json)。模型测试见 [测试记录](../../docs/03-evaluation.md)。
