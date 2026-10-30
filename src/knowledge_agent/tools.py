from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from pydantic import Field

from .retrieval import HybridRetriever
from .schemas import Answer, Passage, StrictModel
from .store import DocumentStore


class SearchArgs(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=12)
    topic: str | None = None


class ReadArgs(StrictModel):
    doc_id: str
    version_date: str
    chunk_id: str


class RelatedArgs(StrictModel):
    doc_id: str
    version_date: str


TOOL_MODELS = {
    "search_documents": (SearchArgs, "检索候选资料。片段仅为导航，回答前须用 read_passage 读取原文。"),
    "read_passage": (ReadArgs, "读取片段的完整父条款及关联注释。引用返回块的chunk_id和source_locations中的原文。OCR是识别文本，须注意质量警告。"),
    "list_related_documents": (RelatedArgs, "列出同主题文档与所有版本，查找可能冲突的范围、版本及条款。"),
    "submit_result": (Answer, "提交最终回答。必须引用本次 read_passage 成功读取的原文；冲突不明时追问。"),
}


def tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
            },
        }
        for name, (model, description) in TOOL_MODELS.items()
    ]


class KnowledgeTools:
    def __init__(self, store: DocumentStore, retriever: HybridRetriever):
        self.store = store
        self.retriever = retriever
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-tools")

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.retriever.close()

    def dispatch(self, name: str, arguments: dict) -> dict:
        if name not in TOOL_MODELS or name == "submit_result":
            raise ValueError("未知业务工具")
        args = TOOL_MODELS[name][0].model_validate(arguments)
        if isinstance(args, SearchArgs):
            return {
                "ok": True,
                "mode": self.retriever.mode,
                "hits": self.retriever.search(args.query, args.top_k, args.topic),
            }
        if isinstance(args, ReadArgs):
            passages = self.store.read_bundle(**args.model_dump())
            return {"ok": True, "passage": passages[0].model_dump(),
                    "related_passages": [p.model_dump() for p in passages[1:]]}
        return {"ok": True, "documents": self.store.related(**args.model_dump())}

    def invoke(self, name: str, arguments: dict, timeout: float) -> dict[str, Any]:
        future = self.pool.submit(self.dispatch, name, arguments)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            return {"ok": False, "error": "tool_timeout", "message": "工具调用超时，可改查或说明未完成"}
        except (ValueError, KeyError):
            return {"ok": False, "error": "invalid_arguments_or_source", "message": "参数不合法或来源不存在"}
        except Exception:  # noqa: BLE001 -- tool boundary returns safe observations to the model
            return {"ok": False, "error": "tool_failed", "message": "工具执行失败，不能视为没有相关资料"}


def validate_citations(answer: Answer, evidence: dict[str, Passage]) -> list[str]:
    errors = []
    for citation in answer.citations:
        source = evidence.get(citation.key)
        if source is None:
            errors.append(f"{citation.key}: 未在本次运行中成功读取")
        elif citation.quote not in source.text:
            errors.append(f"{citation.key}: quote 不是该片段的原文")
        elif source.source_locations and not any(
            citation.quote in loc.text for loc in source.source_locations
        ):
            errors.append(f"{citation.key}: 引用跨越了不连续来源，请分别引用原始单元格或页面段落")
        elif source.quality_warnings and any(w.startswith("blocked:") for w in source.quality_warnings):
            errors.append(f"{citation.key}: OCR质量不足，不能作为确定结论的证据")
        if source is not None and citation.source_locations:
            expected = [loc for loc in source.source_locations if citation.quote in loc.text]
            if citation.source_locations != expected:
                errors.append(f"{citation.key}: 引用定位与已读取来源不一致")
    if answer.conflicts and len({c.key for c in answer.citations}) < 2:
        errors.append("报告资料冲突时至少需要两处不同来源片段")
    return errors
