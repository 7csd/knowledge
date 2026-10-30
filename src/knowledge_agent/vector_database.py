"""Small, persistent Qdrant local collections; exact local search, no service required."""

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np


class VectorDatabase:
    def __init__(self, path: Path, identity: str, passages, vectors: np.ndarray):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ValueError('Qdrant需要安装 pip install -e ".[ingestion]"') from exc
        self.models = models
        self.path = path / identity
        self.client = QdrantClient(path=str(self.path), force_disable_check_same_thread=True)
        self.collection = "knowledge_children"
        point_ids = [str(uuid5(NAMESPACE_URL, p.key + "\0" + hashlib.sha256(
            json.dumps(p.model_dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
                     for p in passages]
        self.keys = {key: i for i, key in enumerate(point_ids)}
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
            )
        self.added_count = 0
        for offset in range(0, len(passages), 128):
            ids = point_ids[offset:offset + 128]
            existing = {str(p.id) for p in self.client.retrieve(self.collection, ids=ids,
                                                              with_payload=False, with_vectors=False)}
            missing = [i for i in range(offset, min(offset + 128, len(passages)))
                       if point_ids[i] not in existing]
            if missing:
                self.client.upsert(
                    self.collection,
                    points=[
                        models.PointStruct(
                            id=point_ids[i], vector=vectors[i].tolist(),
                            payload={"passage_key": p.key, "doc_id": p.doc_id,
                                     "version_date": p.version_date, "topic": p.topic,
                                     "scope": p.scope, "parent_id": p.parent_id,
                                     "text": p.index_text or p.text,
                                     "source_locations": [s.model_dump() for s in p.source_locations]},
                        )
                        for i in missing for p in [passages[i]]
                    ], wait=True,
                )
                self.added_count += len(missing)

    def search(self, vector: np.ndarray, limit: int, min_score: float, topic: str | None):
        conditions = [self.models.HasIdCondition(has_id=list(self.keys))]
        if topic is not None:
            conditions.append(self.models.FieldCondition(
                key="topic", match=self.models.MatchValue(value=topic)
            ))
        filters = self.models.Filter(must=conditions)
        results = self.client.query_points(
            self.collection, query=vector.tolist(), limit=limit,
            query_filter=filters, score_threshold=min_score,
        ).points
        return [(self.keys[str(p.id)], p.score) for p in results]

    def close(self):
        self.client.close()
