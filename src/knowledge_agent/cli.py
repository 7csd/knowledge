import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .agent import KnowledgeAgent
from .config import Settings
from .model import CompatibleChatModel
from .retrieval import BGEEncoder, HybridRetriever
from .schemas import HistoryMessage
from .store import DocumentStore
from .tools import KnowledgeTools


def build_tools(settings: Settings) -> KnowledgeTools:
    store = DocumentStore.load(settings.data_path)
    store.verify_sources()
    encoder = (
        BGEEncoder(settings.embedding_model, settings.embedding_revision, settings.embedding_local_only)
        if settings.retrieval_mode == "hybrid"
        else None
    )
    return KnowledgeTools(
        store, HybridRetriever(store, encoder, settings.cache_dir, settings.vector_min_score,
                               settings.vector_db_path if settings.vector_backend == "qdrant" else None)
    )


def main(argv: list[str] | None = None) -> int:
    # Keep redirected JSON portable on Windows, where the default may be CP936.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="企业知识问答 Agent（真实模型或显式离线回放）")
    parser.add_argument("--data", type=Path, help="原JSON或多格式解析快照路径")
    parser.add_argument("--retrieval", choices=["hybrid", "bm25"], help="覆盖检索模式，不自动降级")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="验证语料并构建检索索引")
    ingest_parser = sub.add_parser("ingest", help="解析Markdown/PDF及实际OCR，建立可追溯父子块快照")
    ingest_parser.add_argument("--manifest", type=Path, default=Path("data/source_materials/manifest.json"))
    ingest_parser.add_argument("--output", type=Path, required=True, help="新建快照目录，拒绝覆盖")
    ingest_parser.add_argument("--base", type=Path, help="已有快照：保留所有旧版本，只处理新MD5文件")
    ingest_parser.add_argument("--scan-doc", action="append", default=[], help="用扫描对照件替换该文档，可重复")
    ingest_parser.add_argument("--target-tokens", type=int, default=300)
    ingest_parser.add_argument("--overlap", type=int, default=40)
    read = sub.add_parser("read", help="读取父条款和关联注释，不调用模型")
    read.add_argument("doc_id")
    read.add_argument("version_date")
    read.add_argument("chunk_id")
    search = sub.add_parser("search", help="仅检索，不调用对话模型")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, choices=range(1, 13), default=6)
    ask = sub.add_parser("ask", help="真实模型问答，需要 .env 配置")
    ask.add_argument("question")
    ask.add_argument("--history", type=Path, help="[{role: user/assistant, content: ...}] JSON")
    ask.add_argument("--trace", type=Path, required=True, help="新的 JSONL 文件，拒绝覆盖")
    evaluate = sub.add_parser("evaluate", help="执行六类真实模型验收输入；不自动判定答案语义正确")
    evaluate.add_argument("--cases", type=Path, default=Path("data/evaluation_cases.json"))
    evaluate.add_argument("--output", type=Path, required=True, help="新建目录，保存各次真实运行")
    demo = sub.add_parser("demo", help="离线脚本回放，仅验证工程流程，不验证模型决策")
    demo.add_argument("--scenario", choices=["normal", "conflict", "missing", "failure"], default="normal")
    demo.add_argument("--output", type=Path, required=True, help="新建目录")
    args = parser.parse_args(argv)
    tools = model = None
    try:
        settings = Settings.from_env()
        if args.data:
            settings.data_path = args.data
        if args.retrieval:
            settings.retrieval_mode = args.retrieval
        if args.command == "ingest":
            from .ingestion import ingest

            result = ingest(args.manifest, args.output, settings.cache_dir / "parsed",
                            settings.embedding_model, settings.embedding_revision,
                            settings.embedding_local_only, args.scan_doc, args.target_tokens, args.overlap,
                            args.base)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"ask", "evaluate"}:
            settings.require_live_model()  # fail before attempting an embedding download
        if args.command == "demo":
            from .offline import run_demo

            print(json.dumps(run_demo(args.scenario, args.output, settings), ensure_ascii=False, indent=2))
            return 0
        tools = build_tools(settings)
        if args.command == "index":
            result = {
                "mode": tools.retriever.mode,
                "documents": len(tools.store.documents),
                "passages": len(tools.store.passages),
                "corpus_hash": tools.store.fingerprint,
                "parents": len(getattr(tools.store, "parents", {})),
                "vector_backend": "qdrant_local" if tools.retriever.database else "numpy" if tools.retriever.encoder else "none",
                "vector_database": str(tools.retriever.database.path) if tools.retriever.database else None,
                "vector_count": len(tools.retriever.vectors) if tools.retriever.vectors is not None else 0,
                "embedding_stats": tools.retriever.embedding_stats,
                "vector_points_added": tools.retriever.database.added_count if tools.retriever.database else 0,
            }
        elif args.command == "read":
            result = tools.dispatch("read_passage", {"doc_id": args.doc_id,
                                                      "version_date": args.version_date,
                                                      "chunk_id": args.chunk_id})
        elif args.command == "search":
            result = {"mode": tools.retriever.mode, "hits": tools.retriever.search(args.query, args.top_k)}
        else:
            model = CompatibleChatModel(settings)
            if args.command == "evaluate":
                from .evaluation import evaluate_live

                result = evaluate_live(model, tools, settings, args.cases, args.output)
            else:
                history = (
                    TypeAdapter(list[HistoryMessage]).validate_json(
                        args.history.read_text(encoding="utf-8-sig")
                    )
                    if args.history
                    else []
                )
                result = (
                    KnowledgeAgent(model, tools, settings)
                    .run(args.question, history, args.trace)
                    .model_dump()
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except ValidationError:
        print("配置或输入结构校验失败，请核对 .env、语料、历史或用例格式。", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        message = str(exc)
        if "settings" in locals() and settings.api_key.get_secret_value():
            message = message.replace(settings.api_key.get_secret_value(), "[REDACTED]")
        print(f"运行失败：{message}", file=sys.stderr)
        return 2
    finally:
        if tools:
            tools.close()
        if model:
            model.close()


if __name__ == "__main__":
    raise SystemExit(main())
