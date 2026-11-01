import os

import pytest

from finrag.config import Settings
from finrag.embeddings import LocalEmbeddings
from finrag.index import VectorIndex
from finrag.ingest import read_corpus
from finrag.pipeline import RAGPipeline

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv("FINRAG_INTEGRATION") != "1", reason="set FINRAG_INTEGRATION=1 to use the real model/PDF")]


@pytest.fixture(scope="module")
def resources():
    settings = Settings.load()
    return settings, LocalEmbeddings(settings)


def test_real_pdf_and_no_silent_truncation(resources):
    settings, embeddings = resources
    pages, _ = read_corpus(settings.root)
    assert len(pages) == 9 and "30,040" in pages[0].page_content
    for method in ["baseline", "semantic"]:
        index = VectorIndex.load(settings.root / "data/indexes" / method, embeddings.fingerprint)
        assert all(embeddings.token_count("passage: " + d.page_content) + 2 <= 512 for d in index.documents)
    with pytest.raises(ValueError, match="Silent truncation"):
        embeddings.embed_query("This question is deliberately very long. " * 200)


def test_real_bilingual_retrieval_and_cross_page_header(resources):
    settings, embeddings = resources
    pipeline = RAGPipeline(settings, embeddings)
    answer = pipeline.ask("What was revenue for the six months ended July 28, 2024?")
    assert answer["answer"] is None and answer["mode"] == "evidence"
    assert any("56,084" in s["text"] and "Six Months" in s["text"] for s in answer["evidence"])
    answer = pipeline.ask("游戏业务第二季度的收入是多少？")
    assert any("2.9 billion" in s["text"] for s in answer["evidence"])
    assert any(d.metadata.get("header_page") == 3 and d.metadata["page"] == 4 and "Six Months" in d.page_content for d in pipeline.index.documents)
