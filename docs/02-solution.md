# 实现方案

## 技术选择

| 组件 | 实现及职责 |
| --- | --- |
| Python 3.10.8 | 已验证的本地开发环境，提供CLI |
| LangGraph 1.2.18 | StateGraph编排模型、工具、观察和结束状态 |
| 模型适配 | 配置URL、Key、模型名，兼容chat/completions的tools/tool_calls |
| 混合检索 | jieba + BM25、BGE-small-zh-v1.5、RRF融合 |
| 存储与解析 | 本地Qdrant、pdfplumber、RapidOCR，保留来源定位 |
| 校验与记录 | Pydantic校验输出，JSONL记录工具参数、观察和简短行动依据 |

我采用单Agent的Agentic RAG，集中实现证据核验和可回放过程，依赖版本见根目录锁文件。

## 数据与检索

默认快照为 `data/processed-incremental-verified/corpus.json`：12个文档ID、13个版本、108个父块及108个检索块，源文件见data/source_materials/。

我按条款、FAQ或表格逻辑行切分，跨页续行及共同注释关联读取。正文目标300 token、完整Embedding输入上限450、长条款重叠40、父块上限1000。多数条款较短，未强制多子块。

BM25与向量两路各取12个候选，RRF常数60，按父块去重后默认返回6个。BGE固定revision为7999e1d3359715c523056ef9478215996d62a620，未使用Rerank。

我采用MD5进行文件去重、SHA256校验来源完整性。ingest --base保留旧版，仅处理新文件，复用Embedding并追加缺失向量。D11的18点、17点两版均保留，不按日期覆盖。

## 决策流程

问题与历史 → 模型选择工具 → 执行并返回观察 → 模型决定补查、读取或提交结果，循环至结束或达到预算。

- search_documents：返回候选、版本、适用范围和片段。
- read_passage：读取可引用的父条款及关联注释。
- list_related_documents：列出同主题和同文档其他版本，供模型查证。
- submit_result：提交结果；引文校验失败作为观察返回模型修正。

工具失败也返回模型观察，由模型选择重试、补查或停止。默认最多8次模型决策、12次工具调用，触达上限返回incomplete。历史用于理解追问，回答仍需核验当前原文。

## 输出约束与边界

结果为answer/citations/gaps/conflicts/status。状态包括completed、needs_clarification、insufficient_evidence、tool_failed、incomplete。

引文须来自本轮成功读取的文档、版本和片段，摘录须存在于原文；程序补齐文件哈希、PDF页码坐标或Markdown行号。未解决的conflicts禁止completed。形式合法不等于语义支持，仍需审阅。

PDF规则针对本批合成模板，不保证任意版式或扫描表格质量；检索不保证穷尽冲突。HTTP适配器支持兼容接口，我将模型配置放在本地config.json，直接填写URL、API Key和模型名，保留环境变量兼容；配置及运行命令见README。
