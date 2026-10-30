# 企业知识问答 Agent

面向企业制度和产品FAQ的单Agent问答项目。使用Python + LangGraph、BM25/BGE混合召回和原文核验，提供CLI、结构化答案和JSONL Trace。

已完成核心代码、72项工程测试、30题交互式决策测试及1题版本对照。模型通过本地环境变量配置，仓库不包含API密钥。实际结果见 [测试记录](docs/03-evaluation.md)。

## 安装与配置

在项目根目录执行，已验证环境为Windows、Python 3.10.8：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ingestion-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

完整锁文件包含检索、PDF/OCR及测试依赖；仅运行基础BM25工程可改用requirements-lock.txt。Linux/macOS将Python路径替换为 `.venv/bin/python`。

默认数据为data/processed-incremental-verified/corpus.json，含12个文档ID、13个版本、108块。首次混合检索需下载BGE权重并构建本地Qdrant索引；.cache/不随仓库提交。模型不可用时明确报错，可显式选择BM25。

## 无需对话模型Key的入口

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m knowledge_agent --retrieval bm25 search "报销期限多久？"
.\.venv\Scripts\python.exe -m knowledge_agent --retrieval hybrid index
.\.venv\Scripts\python.exe -m knowledge_agent --retrieval hybrid search "年假要提前几个上班日申请？"
.\.venv\Scripts\python.exe -m knowledge_agent demo --scenario conflict --output examples/demo-new
```

demo支持normal/conflict/missing/failure，输出目录须未存在。offline_scripted是预设流程回放；assistant_in_the_loop是交互式模型决策测试，逐轮观察真实工具结果并选择动作，记录保存在examples/assistant-evaluation/。

## 配置模型并运行问答

在本地.env填写LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。接口须支持chat/completions的tools/tool_calls；BASE_URL填API根地址，按服务商要求包含/v1，不包含/chat/completions。密钥不提交仓库。

```powershell
.\.venv\Scripts\python.exe -m knowledge_agent ask "报销期限多久？" --trace examples/live/question-001.jsonl
.\.venv\Scripts\python.exe -m knowledge_agent ask "普通成员需要导出怎么办？" --history examples/history.json --trace examples/live/followup-001.jsonl
.\.venv\Scripts\python.exe -m knowledge_agent evaluate --output examples/live/evaluation-001
```

Trace路径和验收目录须未存在。evaluate运行data/evaluation_cases.json的13条输入，保存实际输出、Trace和待填写的语义审阅表，需对照原文审阅。examples/live/默认忽略，提交前将可公开的合成资料运行记录另行归档。

结果包含answer/citations/gaps/conflicts/status，区分完成、需澄清、资料不足、工具失败和未完成。默认上限8次模型决策、12次工具调用；引用须来自本轮成功读取的原文，附ID、版本和定位。程序校验来源与摘录，语义支持性仍需审阅。

## 更新知识文件

在data/source_materials/manifest.json登记独立的新版本文件及哈希，保留旧源文件：

```powershell
.\.venv\Scripts\python.exe -m knowledge_agent ingest --base data/processed-incremental-verified/corpus.json --output data/next-version
.\.venv\Scripts\python.exe -m knowledge_agent --data data/next-version/corpus.json --retrieval hybrid index
```

确认后修改.env的DATA_PATH，首次全量解析省略--base，输出须为新目录。MD5相同文件跳过，变更版本增量保留，不按日期覆盖。目前支持本批Markdown、原生PDF、模拟扫描PDF和跨页表格，不保证任意版式解析。

## 评审入口

| 目录 / 文件 | 内容 |
| --- | --- |
| src/knowledge_agent/、tests/ | 实现与可执行测试 |
| data/source_materials/ | 合成源文件及清单 |
| examples/assistant-evaluation/ | 30题、版本对照、逐题输入输出和Trace |
| examples/incremental-update-verified/ | 增量去重最终实测 |
| [需求](docs/01-requirements.md)、[方案](docs/02-solution.md)、[测试](docs/03-evaluation.md) | 要求、实现和实际结果 |
| [AI使用](docs/04-ai-usage.md)、[复盘](docs/05-retrospective.md) | 协作范围与改进记录 |

其他examples/子目录保留历史测试结果，以测试记录链接的最终材料为入口。

仓库：[7csd/knowledge](https://github.com/7csd/knowledge)。仓库公开，评审可直接查看代码和运行记录，最终commit SHA以Git记录为准。
