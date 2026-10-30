import hashlib
import json
import re
from pathlib import Path

from pydantic import TypeAdapter

from .schemas import Document, Passage


class DocumentStore:
    """Immutable snapshot; all quoted offsets point into this snapshot's raw content."""

    def __init__(self, documents: list[Document], max_chars: int = 350):
        if not documents:
            raise ValueError("知识库不能为空")
        if max_chars < 1:
            raise ValueError("切分长度必须大于零")
        self.documents = {(d.doc_id, d.version_date.isoformat()): d for d in documents}
        if len(self.documents) != len(documents):
            raise ValueError("文档 ID 和版本组合必须唯一")
        self.passages: dict[str, Passage] = {}
        for doc in documents:
            chunk_number = 0
            # Keep paragraphs/clauses intact; fallback windows for unusually long paragraphs.
            for match in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]+)*", doc.content):
                for start in range(match.start(), match.end(), max_chars):
                    end = min(start + max_chars, match.end())
                    chunk_number += 1
                    passage = Passage(
                        doc_id=doc.doc_id,
                        version_date=doc.version_date.isoformat(),
                        chunk_id=f"p{chunk_number:03}",
                        title=doc.title,
                        topic=doc.topic,
                        scope=doc.scope,
                        status=doc.status,
                        start=start,
                        end=end,
                        text=doc.content[start:end],
                    )
                    self.passages[passage.key] = passage
        encoded = json.dumps(
            [p.model_dump() for p in self.passages.values()], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        self.fingerprint = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            return cls(TypeAdapter(list[Document]).validate_python(raw))
        if raw.get("schema_version") != 2:
            raise ValueError("不支持的知识库快照版本")
        instance = cls(TypeAdapter(list[Document]).validate_python(raw["documents"]))
        instance.passages = {p.key: p for p in TypeAdapter(list[Passage]).validate_python(raw["passages"])}
        instance.parents = {p.key: p for p in TypeAdapter(list[Passage]).validate_python(raw["parents"])}
        instance.ingestion_config = raw["config"]
        instance.source_files = raw.get("files", [])
        for p in [*instance.parents.values(), *instance.passages.values()]:
            doc = instance.documents.get((p.doc_id, p.version_date))
            if doc is None or doc.content[p.start:p.end] != p.text:
                raise ValueError("快照片段与原文偏移不一致")
            for ref in [p.parent_id, *p.related_ids]:
                if ref and f"{p.doc_id}@{p.version_date}#{ref}" not in instance.parents:
                    raise ValueError("快照缺少父块或关联条款")
        instance.fingerprint = hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return instance

    def read(self, doc_id: str, version_date: str, chunk_id: str) -> Passage:
        key = f"{doc_id}@{version_date}#{chunk_id}"
        if key in getattr(self, "parents", {}):
            return self.parents[key]
        if key not in self.passages:
            raise ValueError("未找到该文档版本或片段")
        return self.passages[key]

    def read_bundle(self, doc_id: str, version_date: str, chunk_id: str) -> list[Passage]:
        hit = self.read(doc_id, version_date, chunk_id)
        parent = self.read(doc_id, version_date, hit.parent_id) if hit.parent_id else hit
        result = [parent]
        for ref in parent.related_ids:
            related = self.read(doc_id, version_date, ref)
            if related.key not in {p.key for p in result}:
                result.append(related)
        return result

    def related(self, doc_id: str, version_date: str) -> list[dict]:
        doc = self.documents.get((doc_id, version_date))
        if doc is None:
            raise ValueError("未找到该文档版本")
        return [
            {
                **d.model_dump(mode="json", exclude={"content"}),
                "chunks": [
                    p.chunk_id
                    for p in self.passages.values()
                    if p.doc_id == d.doc_id and p.version_date == d.version_date.isoformat()
                ],
            }
            for d in self.documents.values()
            if d.topic == doc.topic
        ]

    def verify_sources(self):
        """Do not silently serve a snapshot after its source files were replaced."""
        checked = set()
        for record in getattr(self, "source_files", []):
            path = Path(record["source_path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"源文件已变化或缺失：{path}，请保留独立版本并重新ingest")
        for passage in getattr(self, "parents", {}).values():
            for loc in passage.source_locations:
                identity = (loc.source_path, loc.sha256)
                if identity in checked:
                    continue
                path = Path(loc.source_path)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != loc.sha256:
                    raise ValueError(f"源文件已变化或缺失：{loc.source_path}，请更新清单并重新ingest")
                checked.add(identity)
