import hashlib
import json
from datetime import date

import numpy as np
import pytest

from knowledge_agent.incremental import classify_file, source_record
from knowledge_agent.ingestion import ingest
from knowledge_agent.retrieval import HybridRetriever
from knowledge_agent.schemas import Document
from knowledge_agent.store import DocumentStore
from knowledge_agent.vector_database import VectorDatabase


def test_md5_deduplicates_renamed_file_but_preserves_changed_version(tmp_path):
    path = tmp_path / "old.md"
    path.write_text("18点", encoding="utf-8")
    entry = {"doc_id": "D", "version_date": "2026-01-01", "title": "快递", "topic": "快递",
             "scope": "总部", "status": "有效", "file_id": "old",
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    old = source_record(entry, path)
    assert old["md5"] == hashlib.md5(path.read_bytes()).hexdigest()
    copy = tmp_path / "copy.md"
    copy.write_bytes(path.read_bytes())
    assert classify_file(source_record({**entry, "file_id": "copy"}, copy), [old]) == "old"
    path.write_text("17点", encoding="utf-8")
    changed = source_record({**entry, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, path)
    with pytest.raises(ValueError, match="增加版本"):
        classify_file(changed, [old])
    assert classify_file({**changed, "version_date": "2026-10-30"}, [old]) is None


def test_incremental_ingest_skips_duplicate_and_keeps_versions_absent_from_manifest(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from knowledge_agent import parsing

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(parsing, "parser_identity", lambda: {"version": "test"})
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = tmp_path / "manifest.json"

    def entry(name, date, hour):
        path = corpus / name
        path.write_text(f"# 快递\n- 文档 ID：D\n- 版本日期：{date}\n\n## 正文\n快递{hour}点前领取。\n", encoding="utf-8")
        return {"file_id": name, "doc_id": "D", "version_date": date, "title": "快递", "scope": "总部",
                "status": "有效", "topic": "快递", "path": f"corpus/{name}", "canonical": True,
                "representation": "markdown", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    old = entry("old.md", "2026-01-01", 18)
    manifest.write_text(json.dumps({"files": [old]}), encoding="utf-8")
    first = tmp_path / "first"
    ingest(manifest, first, tmp_path / "cache", "test", "test")
    new = entry("new.md", "2026-10-30", 17)
    copy_path = corpus / "copy.md"
    copy_path.write_bytes((corpus / "new.md").read_bytes())
    duplicate = {**new, "file_id": "copy", "path": "corpus/copy.md"}
    manifest.write_text(json.dumps({"files": [new, duplicate]}), encoding="utf-8")
    second = tmp_path / "second"
    report = ingest(manifest, second, tmp_path / "cache", "test", "test", base=first / "corpus.json")
    assert report["added_files"] == 1 and report["skipped_files"] == 1
    store = DocumentStore.load(second / "corpus.json")
    store.verify_sources()
    assert {d["version_date"] for d in store.related("D", "2026-01-01")} == {"2026-01-01", "2026-10-30"}
    assert "18点" in store.read("D", "2026-01-01", "r001").text
    assert "17点" in store.read("D", "2026-10-30", "r001").text
    third = tmp_path / "third"
    report = ingest(manifest, third, tmp_path / "cache", "test", "test", base=second / "corpus.json")
    assert report["added_files"] == 0 and report["children"] == 2
    with pytest.raises(ValueError, match="配置不一致"):
        ingest(manifest, tmp_path / "bad", tmp_path / "cache", "test", "test",
               target=200, base=second / "corpus.json")


def test_embedding_only_encodes_new_inputs(tmp_path):
    class Encoder:
        identity = "test-incremental-not-semantic"

        def __init__(self):
            self.inputs = []

        def encode(self, texts, **kwargs):
            self.inputs.extend(texts)
            return np.ones((len(texts), 2))

    old = Document(doc_id="D", version_date="2026-01-01", title="快递", topic="快递", content="18点领取")
    new = old.model_copy(update={"version_date": date(2026, 10, 30), "content": "17点领取"})
    encoder = Encoder()
    HybridRetriever(DocumentStore([old]), encoder, tmp_path)
    encoder.inputs.clear()
    index = HybridRetriever(DocumentStore([old, new]), encoder, tmp_path)
    assert len(encoder.inputs) == 1 and "17点" in encoder.inputs[0]
    assert index.embedding_stats == {"encoded_inputs": 1, "reused_passages": 1}


def test_qdrant_appends_only_missing_points_and_filters_snapshot(tmp_path):
    pytest.importorskip("qdrant_client")
    docs = [Document(doc_id=f"D{i}", version_date="2026-01-01", title="快递", topic="快递", content=str(i))
            for i in range(2)]
    passages = list(DocumentStore(docs).passages.values())
    vectors = np.ones((2, 2), dtype=np.float32)
    for selected, expected_added, expected_hits in [(passages[:1], 1, 1), (passages, 1, 2),
                                                    (passages, 0, 2), (passages[:1], 0, 1)]:
        db = VectorDatabase(tmp_path, "test-shared-model", selected, vectors[:len(selected)])
        try:
            assert db.added_count == expected_added
            assert len(db.search(vectors[0], 5, 0, None)) == expected_hits
        finally:
            db.close()
