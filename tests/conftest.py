from pathlib import Path

import pytest

from knowledge_agent.retrieval import HybridRetriever
from knowledge_agent.store import DocumentStore
from knowledge_agent.tools import KnowledgeTools

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store():
    return DocumentStore.load(ROOT / "data/documents.json")


@pytest.fixture
def tools(store):
    instance = KnowledgeTools(store, HybridRetriever(store))
    yield instance
    instance.close()
