import argparse
import json
import sys
from pathlib import Path

from knowledge_agent.config import Settings
from knowledge_agent.retrieval_evaluation import evaluate_retrieval

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(description="对比BM25、真实BGE向量与混合召回；不调用对话模型")
parser.add_argument("--cases", type=Path, default=Path("data/retrieval_cases.json"))
parser.add_argument("--data", type=Path, default=Path("data/documents.json"),
                    help="标注集对应语料；默认保留原JSON基准，避免与新快照片段ID混用")
parser.add_argument("--output", type=Path, required=True, help="新目录，避免覆盖已有记录")
parser.add_argument("--lexical-only", action="store_true", help="仅测BM25，不下载向量模型")
parser.add_argument("--embedding-revision", help="固定Embedding模型的commit，便于复现")
args = parser.parse_args()
settings = Settings.from_env()
settings.data_path = args.data
if args.embedding_revision:
    settings.embedding_revision = args.embedding_revision
result = evaluate_retrieval(settings, args.cases, args.output, args.lexical_only)
print(
    json.dumps(
        {
            "semantic_status": result["semantic_status"],
            "semantic_error": result["semantic_error"],
            "results": {method: data["summary"] for method, data in result["results"].items()},
        },
        ensure_ascii=False,
        indent=2,
    )
)
