import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Protocol

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .store import DocumentStore

jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    return [word.lower() for word in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", word)]


class Encoder(Protocol):
    identity: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class BGEEncoder:
    def __init__(self, model: str, revision: str = "main", local_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError('hybrid 模式需要先安装 pip install -e ".[semantic]"') from exc
        try:
            self.model = SentenceTransformer(
                model,
                revision=revision,
                device="cpu",
                local_files_only=local_only,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise ValueError("Embedding 加载失败：请检查本地模型路径或模型下载网络") from exc
        self.identity = f"{model}@{revision}:bge-query-instruction-v1"

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if query:
            texts = ["为这个句子生成表示以用于检索相关文章：" + text for text in texts]
        limit = self.model.max_seq_length
        if any(len(self.model.tokenizer.encode(text, add_special_tokens=True)) > limit for text in texts):
            raise ValueError("Embedding输入超过模型长度上限，拒绝静默截断，请重新切分或缩短查询")
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError("向量必须为有限数值二维矩阵")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("不允许零向量")
    return vectors / norms


def cached_embeddings(texts, encoder, cache_dir, seed=None):
    """Encode only unseen exact inputs; cache includes the model revision/instruction."""
    keys = [hashlib.sha256((encoder.identity + "\0" + t).encode()).hexdigest() for t in texts]
    directory = cache_dir / "embedding-inputs" if cache_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    values = {}
    if seed is not None:
        for key, vector in zip(keys, seed):
            values[key] = vector
    for key in dict.fromkeys(keys):
        path = directory / f"{key}.npy" if directory else None
        if key not in values and path and path.exists():
            try:
                values[key] = normalized(np.load(path, allow_pickle=False).reshape(1, -1))[0]
            except (OSError, ValueError):
                pass
    missing = list(dict.fromkeys(key for key in keys if key not in values))
    by_key = dict(zip(keys, texts))
    if missing:
        encoded = normalized(encoder.encode([by_key[key] for key in missing]))
        if len(encoded) != len(missing):
            raise ValueError("Embedding 返回条目数与资料数不一致")
        values.update(zip(missing, encoded))
    if directory:
        for key, vector in values.items():
            path = directory / f"{key}.npy"
            if not path.exists() or key in missing:
                np.save(path, vector, allow_pickle=False)
    return normalized(np.stack([values[key] for key in keys])), len(missing)


class HybridRetriever:
    def __init__(
        self,
        store: DocumentStore,
        encoder: Encoder | None = None,
        cache_dir: Path | None = None,
        vector_min_score: float = 0.45,
        database_path: Path | None = None,
    ):
        self.store = store
        self.encoder = encoder
        self.mode = "hybrid" if encoder else "bm25"
        self.min_score = vector_min_score
        self.database = None
        self.passages = list(store.passages.values())
        self.lexical = BM25Okapi([tokenize(p.index_text or (p.title + " " + p.text)) for p in self.passages])
        self.vectors = None
        self.cache_key = ""
        self.embedding_stats = {"encoded_inputs": 0, "reused_passages": 0}
        if encoder:
            self.cache_key = hashlib.sha256(
                (store.fingerprint + encoder.identity).encode("utf-8")
            ).hexdigest()
            path = cache_dir / f"vectors-{self.cache_key}.npz" if cache_dir else None
            if path and path.exists():
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        self.vectors = normalized(saved["vectors"])
                    if len(self.vectors) != len(self.passages):
                        self.vectors = None
                except (ValueError, OSError, KeyError):
                    self.vectors = None
            seed = self.vectors
            base_hash = getattr(store, "ingestion_config", {}).get("base_fingerprint")
            if seed is None and cache_dir and base_hash:
                base_key = hashlib.sha256((base_hash + encoder.identity).encode()).hexdigest()
                base_path = cache_dir / f"vectors-{base_key}.npz"
                if base_path.exists():
                    with np.load(base_path, allow_pickle=False) as saved:
                        seed = normalized(saved["vectors"])
                    if len(seed) > len(self.passages):
                        raise ValueError("增量向量基线条目数不合法")
            if self.vectors is None:
                self.vectors, encoded_count = cached_embeddings([
                    p.index_text or (p.title + "\n" + p.text) for p in self.passages
                ], encoder, cache_dir, seed)
                self.embedding_stats = {"encoded_inputs": encoded_count,
                                        "reused_passages": len(self.passages) - encoded_count}
                if len(self.vectors) != len(self.passages):
                    raise ValueError("Embedding 返回条目数与资料数不一致")
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, vectors=self.vectors)
                    path.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "corpus_hash": store.fingerprint,
                                "encoder": encoder.identity,
                                "count": len(self.passages),
                                "mode": self.mode,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            else:
                cached_embeddings([p.index_text or (p.title + "\n" + p.text) for p in self.passages],
                                  encoder, cache_dir, self.vectors)
                self.embedding_stats["reused_passages"] = len(self.passages)
            if database_path is not None:
                from .vector_database import VectorDatabase

                database_key = hashlib.sha256(encoder.identity.encode()).hexdigest()
                self.database = VectorDatabase(database_path, database_key, self.passages, self.vectors)

    def close(self):
        if self.database is not None:
            self.database.close()

    def search(self, query: str, top_k: int = 6, topic: str | None = None) -> list[dict]:
        indices = [i for i, p in enumerate(self.passages) if topic is None or p.topic == topic]
        lexical_scores = self.lexical.get_scores(tokenize(query))
        lexical = sorted(
            (i for i in indices if lexical_scores[i] > 0), key=lambda i: (-lexical_scores[i], i)
        )[:12 if hasattr(self.store, "parents") else 8]
        semantic_scores = None
        semantic = []
        if self.encoder:
            query_vector = normalized(self.encoder.encode([query], query=True))
            if len(query_vector) != 1 or query_vector.shape[1] != self.vectors.shape[1]:
                raise ValueError("查询向量维度与索引不一致")
            semantic_scores = self.vectors @ query_vector[0]
            if self.database is not None:
                semantic = [i for i, _ in self.database.search(query_vector[0], 12, self.min_score, topic)]
            else:
                semantic = sorted(
                    (i for i in indices if semantic_scores[i] >= self.min_score),
                    key=lambda i: (-semantic_scores[i], i),
                )[:12 if hasattr(self.store, "parents") else 8]
        fused: dict[int, float] = {}
        for ranking in (lexical, semantic):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0) + 1 / (60 + rank)
        hits = []
        parents_seen = set()
        for i in sorted(fused, key=lambda index: (-fused[index], index)):
            p = self.passages[i]
            parent_key = (p.doc_id, p.version_date, p.parent_id or p.chunk_id)
            if parent_key in parents_seen:
                continue
            parents_seen.add(parent_key)
            hits.append(
                {
                    **p.model_dump(exclude={"text", "start", "end", "source_locations", "index_text"}),
                    "source_locations": [loc.model_dump(exclude={"text"}) for loc in p.source_locations],
                    "snippet": p.text[:180],
                    "rrf_score": round(fused[i], 6),
                    "bm25_score": round(float(lexical_scores[i]), 6),
                    "vector_score": round(float(semantic_scores[i]), 6)
                    if semantic_scores is not None
                    else None,
                }
            )
            if len(hits) >= top_k:
                break
        return hits
