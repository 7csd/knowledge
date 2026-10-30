"""Passage-level retrieval benchmark, independent of answer generation and LLM decisions."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field, TypeAdapter

from .config import Settings
from .retrieval import BGEEncoder, HybridRetriever, normalized
from .schemas import StrictModel
from .store import DocumentStore


class RetrievalCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    category: str
    query: str = Field(min_length=1)
    required: list[str]
    relevant: list[str] | None = None


def passage_key(hit: dict) -> str:
    return f"{hit['doc_id']}@{hit['version_date']}#{hit['chunk_id']}"


def metrics(required: list[str], retrieved: list[str], relevant: list[str] | None = None) -> dict:
    gold = set(required)
    if not gold:
        return {"outside_returned_candidates": bool(retrieved)}
    first = next((i for i, key in enumerate(retrieved, start=1) if key in gold), None)
    result = {"reciprocal_rank": 1 / first if first else 0.0}
    for k in (1, 3, 6):
        found = gold.intersection(retrieved[:k])
        result[f"hit@{k}"] = bool(found)
        result[f"recall@{k}"] = len(found) / len(gold)
        result[f"all_required@{k}"] = found == gold
        result[f"precision@{k}"] = None
        result[f"ndcg@{k}"] = None
        if relevant is not None:
            labels = set(relevant)
            result[f"precision@{k}"] = len(labels.intersection(retrieved[:k])) / k
            seen = set()
            dcg = 0.0
            for rank, key in enumerate(retrieved[:k], 1):
                if key in labels and key not in seen:
                    dcg += 1 / math.log2(rank + 1)
                seen.add(key)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(labels)) + 1))
            result[f"ndcg@{k}"] = dcg / ideal if ideal else None
    return result


def summarize(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["required"]]
    negatives = [row for row in rows if not row["required"]]
    multi = [row for row in positives if len(row["required"]) > 1]
    summary = {"positive_count": len(positives), "outside_count": len(negatives)}
    for key in ("hit@1", "hit@3", "hit@6", "recall@1", "recall@3", "recall@6", "reciprocal_rank",
                "precision@1", "precision@3", "precision@6", "ndcg@1", "ndcg@3", "ndcg@6"):
        values = [row["metrics"][key] for row in positives if row["metrics"][key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    summary["multi_source_all@6"] = (
        sum(row["metrics"]["all_required@6"] for row in multi) / len(multi) if multi else None
    )
    summary["outside_candidate_rate"] = (
        sum(row["metrics"]["outside_returned_candidates"] for row in negatives) / len(negatives)
        if negatives
        else None
    )
    summary["mean_query_ms"] = sum(row["elapsed_ms"] for row in rows) / len(rows) if rows else None
    return summary


def vector_hits(index: HybridRetriever, query: str, top_k: int = 6) -> list[dict]:
    query_vector = normalized(index.encoder.encode([query], query=True))
    scores = index.vectors @ query_vector[0]
    candidates = [i for i in range(len(index.passages)) if scores[i] >= index.min_score]
    ranking = sorted(candidates, key=lambda i: (-scores[i], i))[:top_k]
    return [
        {
            **index.passages[i].model_dump(exclude={"text"}),
            "snippet": index.passages[i].text[:180],
            "vector_score": float(scores[i]),
        }
        for i in ranking
    ]


def evaluate_retrieval(
    settings: Settings, cases_path: Path, output_dir: Path, lexical_only: bool = False
) -> dict:
    store = DocumentStore.load(settings.data_path)
    cases = TypeAdapter(list[RetrievalCase]).validate_json(cases_path.read_text(encoding="utf-8-sig"))
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("召回用例不能为空，且 ID 必须唯一")
    for case in cases:
        if len(case.required) != len(set(case.required)) or any(
            key not in store.passages for key in case.required
        ):
            raise ValueError(f"用例 {case.case_id} 的标注来源不存在或重复")
        if case.relevant is not None and (
            len(case.relevant) != len(set(case.relevant))
            or any(key not in store.passages for key in case.relevant)
            or not set(case.required) <= set(case.relevant)
        ):
            raise ValueError(f"用例 {case.case_id} 的相关性标注必须包含所需证据且来源有效")
    output_dir.mkdir(parents=True, exist_ok=False)
    bm25 = HybridRetriever(store)
    methods = {"bm25": bm25.search}
    semantic_error = None
    semantic_status = "not_requested" if lexical_only else "pending"
    embedding_init_seconds = None
    if not lexical_only:
        started = time.perf_counter()
        try:
            encoder = BGEEncoder(
                settings.embedding_model, settings.embedding_revision, settings.embedding_local_only
            )
            hybrid = HybridRetriever(store, encoder, settings.cache_dir, settings.vector_min_score)
            # Exclude first query warmup and model/index construction from measured query latency.
            encoder.encode(["预热查询"], query=True)
            methods.update({"vector": lambda query: vector_hits(hybrid, query), "hybrid": hybrid.search})
            semantic_status = "real_embedding"
        except (ValueError, OSError):
            semantic_error = "真实 Embedding 初始化失败；向量及混合效果没有测得，不能视为0分或已通过。"
            semantic_status = "unavailable"
        embedding_init_seconds = time.perf_counter() - started
    evaluations = {}
    for method, search in methods.items():
        rows = []
        for case in cases:
            started = time.perf_counter()
            hits = search(case.query)
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    **case.model_dump(),
                    "hits": hits,
                    "elapsed_ms": round(elapsed, 3),
                    "relevant": case.relevant,
                    "metrics": metrics(case.required, [passage_key(hit) for hit in hits], case.relevant),
                }
            )
        evaluations[method] = {"summary": summarize(rows), "cases": rows}
    result = {
        "scope": "retrieval_only_no_llm_no_rerank",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_hash": store.fingerprint,
        "documents": len(store.documents),
        "passages": len(store.passages),
        "cases": len(cases),
        "semantic_status": semantic_status,
        "semantic_error": semantic_error,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
        "vector_min_score": settings.vector_min_score,
        "embedding_init_seconds": embedding_init_seconds,
        "results": evaluations,
        "limitations": [
            "人工编写的小规模合成资料与用例，没有独立生产测试集，不能外推生产质量。",
            "按支持答案的具体片段计算，不仅按文档ID；多来源要求同时找全。",
            "资料外仍召回候选不等于最终幻觉；拒答需模型读取后判断。",
            "这里不测试对话历史改写、工具失败恢复或真实模型决策。",
            "耗时为单次本机测量，排除索引初始化，不能作为稳定性能基准。",
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 召回评测实际结果",
        "",
        f"资料：{len(store.documents)}份文档、{len(store.passages)}个片段；用例：{len(cases)}条。",
        "",
        "按具体原文片段标注，未接入对话模型与Rerank。",
        "",
        "| 方法 | Hit@1 | Recall@3 | Recall@6 | 多来源全部召回@6 | 资料外返回候选比例 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, evaluation in evaluations.items():
        summary = evaluation["summary"]
        columns = [
            f"{summary[key]:.1%}" if summary[key] is not None else "—"
            for key in ("hit@1", "recall@3", "recall@6", "multi_source_all@6", "outside_candidate_rate")
        ]
        lines.append("| " + " | ".join([method, *columns]) + " |")
    if semantic_error:
        lines += ["", semantic_error]
    lines += [
        "",
        "库内命中与召回指标越高越好。资料外返回候选比例仅观察噪声，不能单凭该比例评判检索质量或拒答正确性；说明资料边界的片段也可能有帮助。",
        "",
        "## 未找全的正例与资料外候选",
        "",
    ]
    for method, evaluation in evaluations.items():
        for row in evaluation["cases"]:
            if (row["required"] and not row["metrics"]["all_required@6"]) or not row["required"]:
                ids = ", ".join(passage_key(hit) for hit in row["hits"]) or "无候选"
                lines.append(f"- {method} / {row['case_id']} / {row['query']}：{ids}")
    lines += ["", "## 边界", "", *["- " + note for note in result["limitations"]]]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
