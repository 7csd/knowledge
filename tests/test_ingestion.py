import copy
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from knowledge_agent.agent import KnowledgeAgent
from knowledge_agent.config import Settings
from knowledge_agent.ingestion import TokenChunker, select_files
from knowledge_agent.offline import ScriptedModel, scripted_call
from knowledge_agent.parsing import SourceParser
from knowledge_agent.retrieval import HybridRetriever
from knowledge_agent.schemas import Answer, Document, Passage, SourceLocation
from knowledge_agent.store import DocumentStore
from knowledge_agent.tools import KnowledgeTools, validate_citations


class CharacterTokenizer:
    """Only tests splitting invariants. Real BGE token budgets are verified separately."""

    def encode(self, text, **kwargs):
        return [0, *list(text), 1]


def test_token_windows_keep_text_and_overlap_without_gaps():
    splitter = TokenChunker(CharacterTokenizer(), target=60, max_input=100, overlap=10, parent_limit=120)
    text = "完整的规则和条件必须保留。" * 40
    windows = list(splitter.windows(text, 60, 10))
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    for (a, z), (b, end) in pairwise(windows):
        assert a < b < z < end
        assert splitter.count(text[a:z]) <= 60
    assert set().union(*(set(range(a, z)) for a, z in windows)) == set(range(len(text)))


def test_select_scan_replaces_representation_not_document():
    manifest = json.loads(Path("data/source_materials/manifest.json").read_text(encoding="utf-8"))
    selected = select_files(manifest, ["D05"])
    assert len(selected) == len({(m["doc_id"], m["version_date"]) for m in manifest["files"] if m.get("canonical")})
    assert len([m for m in selected if m["doc_id"] == "D05"]) == 1
    assert next(m for m in selected if m["doc_id"] == "D05")["representation"] == "image_only_pdf"


def test_source_change_rejects_stale_manifest_before_parsing(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "doc.md"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        SourceParser(tmp_path / "cache").parse({"sha256": "0" * 64, "file_id": "D-test"}, path)


def test_real_pdf_table_continuation_and_shared_notes(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    manifest_path = Path("data/source_materials/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(m for m in manifest["files"] if m["doc_id"] == "D04")
    parsed = SourceParser(tmp_path / "cache").parse(entry, manifest_path.parent / entry["path"])
    rows = [b for b in parsed["blocks"] if b["kind"] == "table_row"]
    assert len(rows) == 32
    chengdu = next(r for r in rows if r["row_id"] == "T08")
    assert {s["page"] for s in chengdu["sources"]} == {2, 3}
    assert "350" in chengdu["text"] and "事先取得部门负责人书面批准" in chengdu["text"]
    notes = [b for b in parsed["blocks"] if b["kind"] == "table_note"]
    assert len(notes) == 4
    assert all(n["table_id"] == chengdu["table_id"] for n in notes)
    assert all(n["sources"][0]["page"] == 5 for n in notes)


@pytest.fixture
def structured_store(tmp_path):
    doc = Document(doc_id="T", version_date="2026-01-01", title="表格", topic="住宿", content="成都 | 350\n\n注2：须事先批准。")
    loc = SourceLocation(file_id="T-native", source_path="table.pdf", sha256="f" * 64,
                         method="pdf_table", text="350", page=2, bbox=(1, 2, 3, 4))
    parent = Passage(doc_id="T", version_date="2026-01-01", chunk_id="r1", title="表格", topic="住宿",
                     scope="国内", status="有效", start=0, end=8, text="成都 | 350", kind="table_row",
                     related_ids=["r2"], source_locations=[loc])
    note = parent.model_copy(update={"chunk_id": "r2", "start": 10, "end": len(doc.content),
                                     "text": "注2：须事先批准。", "kind": "table_note", "related_ids": [],
                                     "source_locations": [loc.model_copy(update={"page": 5, "text": "注2：须事先批准。"})]})
    child = parent.model_copy(update={"chunk_id": "c1", "parent_id": "r1"})
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"schema_version": 2, "config": {}, "documents": [doc.model_dump(mode="json")],
                                "parents": [parent.model_dump(), note.model_dump()],
                                "passages": [child.model_dump()]}), encoding="utf-8")
    return DocumentStore.load(path), path


def test_parent_read_expands_notes_and_keeps_source_offsets(structured_store):
    store, _ = structured_store
    result = store.read_bundle("T", "2026-01-01", "c1")
    assert [p.chunk_id for p in result] == ["r1", "r2"]
    assert [p.source_locations[0].page for p in result] == [2, 5]


def test_stitched_table_quote_and_forged_coordinates_rejected(structured_store):
    store, _ = structured_store
    source = store.read("T", "2026-01-01", "r1")
    answer = Answer(status="completed", answer="测试", citations=[
        {"doc_id": "T", "version_date": "2026-01-01", "chunk_id": "r1", "quote": "成都 | 350"}
    ])
    assert validate_citations(answer, {source.key: source})
    answer.citations[0].quote = "350"
    assert not validate_citations(answer, {source.key: source})
    answer.citations[0].source_locations = [source.source_locations[0].model_copy(update={"page": 99})]
    assert validate_citations(answer, {source.key: source})


def test_agent_registers_related_read_evidence_and_attaches_verified_locator(structured_store):
    store, _ = structured_store
    tools = KnowledgeTools(store, HybridRetriever(store))
    try:
        actions = [scripted_call("read_passage", {"doc_id": "T", "version_date": "2026-01-01", "chunk_id": "c1"}),
                   scripted_call("submit_result", {"status": "completed", "answer": "须事先批准。", "citations": [
                       {"doc_id": "T", "version_date": "2026-01-01", "chunk_id": "r2", "quote": "须事先批准。"}
                   ]})]
        output = KnowledgeAgent(ScriptedModel(actions), tools, Settings()).run("超标需要批准吗？")
        assert output.result.status == "completed"
        assert output.result.citations[0].source_locations[0].page == 5
        assert output.result.gaps == []
    finally:
        tools.close()


def test_snapshot_tampering_is_rejected(structured_store):
    _, path = structured_store
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["passages"][0]["text"] = "错误金额"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="偏移"):
        DocumentStore.load(path)


def test_qdrant_persistence_filter_and_reopen(tmp_path):
    pytest.importorskip("qdrant_client")
    from knowledge_agent.vector_database import VectorDatabase

    docs = [Document(doc_id=f"D{i}", version_date="2026-01-01", title="测试", topic=str(i), content="原文")
            for i in range(2)]
    passages = list(DocumentStore(docs).passages.values())
    vectors = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    identity = hashlib.sha256(b"fixture-not-semantic").hexdigest()
    for _ in range(2):
        database = VectorDatabase(tmp_path, identity, passages, vectors)
        try:
            assert database.client.count(database.collection).count == 2
            assert database.search(vectors[0], 2, 0.5, None)[0][0] == 0
            assert database.search(vectors[0], 2, 0.5, "1") == []
        finally:
            database.close()


def test_related_missing_in_snapshot_is_rejected(structured_store):
    _, path = structured_store
    raw = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    changed["parents"][0]["related_ids"] = ["missing"]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="关联"):
        DocumentStore.load(path)


def test_low_quality_ocr_cannot_be_used_as_completed_evidence(structured_store):
    store, _ = structured_store
    source = store.read("T", "2026-01-01", "r1").model_copy(
        update={"quality_warnings": ["blocked: OCR低置信度"]}
    )
    answer = Answer(status="completed", answer="测试", citations=[
        {"doc_id": "T", "version_date": "2026-01-01", "chunk_id": "r1", "quote": "350"}
    ])
    assert any("OCR" in e for e in validate_citations(answer, {source.key: source}))


def test_source_replacement_requires_reingestion(structured_store, tmp_path):
    store, _ = structured_store
    path = tmp_path / "source.txt"
    path.write_text("旧规则", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for p in store.parents.values():
        p.source_locations = [loc.model_copy(update={"source_path": str(path), "sha256": digest})
                              for loc in p.source_locations]
    store.verify_sources()
    path.write_text("新规则", encoding="utf-8")
    with pytest.raises(ValueError, match="重新ingest"):
        store.verify_sources()


def test_children_from_same_parent_do_not_consume_multiple_hits(structured_store):
    store, _ = structured_store
    original = next(iter(store.passages.values()))
    duplicate = original.model_copy(update={"chunk_id": "c2"})
    store.passages[duplicate.key] = duplicate
    class FixedEncoder:
        identity = "unit-test-vectors-not-semantic"

        def encode(self, texts, *, query=False):
            return np.ones((len(texts), 2), dtype=np.float32)

    hits = HybridRetriever(store, FixedEncoder()).search("成都 350")
    assert len(hits) == 1
