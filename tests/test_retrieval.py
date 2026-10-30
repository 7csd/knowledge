import numpy as np
import pytest

from knowledge_agent.retrieval import HybridRetriever, normalized
from knowledge_agent.schemas import Document
from knowledge_agent.store import DocumentStore


class FixtureEncoder:
    """Synthetic vectors test fusion/math/cache only, NOT semantic quality."""

    identity = "test-fixture-encoder-v1"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        return np.array([[1.0, 0.1 if "年假" in text or query else 2.0] for text in texts])


def test_clause_offsets_and_versions(store):
    for passage in store.passages.values():
        doc = store.documents[passage.doc_id, passage.version_date]
        assert passage.text == doc.content[passage.start : passage.end]
    assert len(store.related("D01", "2026-01-01")) == 2
    assert store.read("D03", "2026-02-01", "p001").text.endswith("直属主管审批。")


@pytest.mark.parametrize(
    "query,expected", [("年假 申请 提前", "D03"), ("星云 CSV 导出", "D06"), ("设备申领 笔记本", "D07")]
)
def test_bm25_retrieval(store, query, expected):
    hits = HybridRetriever(store).search(query)
    assert hits[0]["doc_id"] == expected
    assert hits[0]["vector_score"] is None


def test_unknown_lexical_does_not_return_arbitrary_docs(store):
    assert HybridRetriever(store).search("量子纠缠火星生态") == []


def test_conflict_sources_not_hidden(store):
    hits = HybridRetriever(store).search("报销期限")
    assert {"D01", "D02"} <= {h["doc_id"] for h in hits}
    assert all(h["topic"] == "报销期限" for h in HybridRetriever(store).search("报销", topic="报销期限"))


def test_hybrid_cache_reuse_and_content_invalidation(store, tmp_path):
    encoder = FixtureEncoder()
    index = HybridRetriever(store, encoder, tmp_path)
    assert encoder.calls == 1
    assert index.search("申请年假")[0]["vector_score"] is not None
    calls = encoder.calls
    cached = HybridRetriever(store, encoder, tmp_path)
    assert encoder.calls == calls
    documents = [doc.model_copy(deep=True) for doc in store.documents.values()]
    documents[2].content = documents[2].content.replace("3个工作日", "5个工作日")
    changed = HybridRetriever(DocumentStore(documents), encoder, tmp_path)
    assert changed.cache_key != cached.cache_key
    assert encoder.calls == calls + 1


def test_semantic_threshold_and_filter(store):
    index = HybridRetriever(store, FixtureEncoder(), vector_min_score=1.0)
    assert index.search("无关词汇") == []


def test_duplicate_document_version_rejected(store):
    doc = next(iter(store.documents.values()))
    with pytest.raises(ValueError, match="唯一"):
        DocumentStore([doc, doc])


def test_long_paragraph_keeps_exact_offsets():
    doc = Document(doc_id="LONG", version_date="2026-01-01", title="长文", topic="长文", content="甲" * 800)
    store = DocumentStore([doc])
    assert "".join(p.text for p in store.passages.values()) == doc.content


@pytest.mark.parametrize("matrix", [np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]]), np.array([1.0, 2.0])])
def test_invalid_vectors_rejected(matrix):
    with pytest.raises(ValueError):
        normalized(matrix)
