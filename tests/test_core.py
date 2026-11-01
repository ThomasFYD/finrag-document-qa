import json
from dataclasses import replace

import numpy as np
import pytest
from langchain_core.documents import Document

from finrag.chunking import is_numeric_row, page_blocks, split_pages
from finrag.config import Settings
from finrag.evaluation import evidence_metrics
from finrag.index import VectorIndex
from finrag.pipeline import validate_answer
from finrag.retrieval import Hit, Retriever, pack_context


def doc(text, page=1, **meta):
    return Document(page_content=text, metadata={"doc_id": "test", "source": "test.pdf", "source_sha256": "abc",
        "title": "Annual report", "page": page, "chunk_id": f"c{page}", **meta})


class FakeEmbeddings:
    fingerprint = "fake-v1"
    dimension = 2

    def token_count(self, text):
        return len(text.split())

    def embed_documents(self, texts):
        return [[1.0, 0.0] if "revenue" in t.lower() else [0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_table_header_survives_page_boundary():
    pages = [doc("CONDENSED CONSOLIDATED STATEMENTS OF INCOME\n(In millions)\nThree Months Ended        Six Months Ended\nJuly 28,        July 30,        July 28,        July 30,\n2024           2023           2024           2023\nRevenue          30,040         13,507         56,084         20,699"),
             doc("Net income           16,599         6,188          31,480          8,232", 2)]
    chunks = split_pages(pages, FakeEmbeddings(), Settings(chunk_tokens=100, overlap_tokens=5), "semantic")
    second = next(c for c in chunks if c.metadata["page"] == 2)
    assert "Six Months Ended" in second.page_content
    assert "In millions" in second.page_content
    assert second.metadata["header_page"] == 1
    assert "31,480" in second.page_content


def test_column_date_header_is_not_numeric_data():
    assert not is_numeric_row("July 28,      July 30,      July 28,")
    assert is_numeric_row("Revenue          $30,040       $13,507")


def test_chunk_length_is_bounded():
    paragraph = "Revenue grew. " * 100
    settings = Settings(chunk_tokens=80, overlap_tokens=8)
    for method in ["baseline", "semantic"]:
        chunks = split_pages([doc(paragraph)], FakeEmbeddings(), settings, method)
        assert len(chunks) > 1
        assert all(c.metadata["tokens"] < 100 for c in chunks)


def test_persistence_integrity_and_embedding_version(tmp_path):
    index = VectorIndex([doc("revenue")], [[3.0, 4.0]], {"schema": 1, "embedding": "fake-v1", "dimension": 2})
    index.save(tmp_path)
    restored = VectorIndex.load(tmp_path, "fake-v1")
    assert np.allclose(restored.vectors, [[0.6, 0.8]])
    assert restored.search_vector([3, 4], 1)[0][1] == pytest.approx(1)
    with pytest.raises(ValueError, match="model changed"):
        VectorIndex.load(tmp_path, "different-model")
    chunks = tmp_path / index.manifest["generation"] / "chunks.json"
    chunks.write_text("[]")
    with pytest.raises(ValueError, match="integrity check failed"):
        VectorIndex.load(tmp_path, "fake-v1")


def test_empty_and_filtered_retrieval_and_hyde():
    idx = VectorIndex([doc("revenue"), doc("cash", 2, doc_id="other")], [[1, 0], [0, 1]], {"dimension": 2})
    r = Retriever(idx, FakeEmbeddings())
    with pytest.raises(ValueError):
        r.retrieve(" ")
    with pytest.raises(ValueError):
        r.retrieve("revenue", strategy="hyde")
    assert r.retrieve("revenue", doc_id="missing") == []
    assert r.retrieve("revenue", doc_id="other")[0].document.metadata["doc_id"] == "other"
    assert r.retrieve("revenue", strategy="hybrid")[0].document.page_content == "revenue"


def test_context_budget_never_cuts_rows_and_deduplicates():
    hit = Hit(doc("Gross margin 75.1%\nRevenue $30,040 million"), 0.8, 0.8, "cosine")
    assert pack_context([hit], 5) == ([], 0)
    packed, used = pack_context([hit, hit], 1000)
    assert len(packed) == 1 and used <= 1000
    assert "30,040 million" in packed[0]["text"]


def test_citation_validation_rejects_unknown_and_mismatched_ids():
    evidence = [{"citation": "S1"}]
    good = {"answer": "Revenue was $30,040 million [S1].", "citations": ["S1"], "insufficient_evidence": False}
    assert validate_answer(json.dumps(good), evidence)["citations"] == ["S1"]
    for bad in [dict(good, citations=["S2"]), dict(good, answer="Made up [S2]."), dict(good, citations=[]), dict(good, insufficient_evidence="false")]:
        with pytest.raises(ValueError):
            validate_answer(json.dumps(bad), evidence)


def test_insufficient_evidence_hides_model_claims():
    answer = validate_answer(json.dumps({"answer": "A fabricated claim", "citations": [], "insufficient_evidence": True}), [])
    assert "fabricated" not in answer["answer"] and answer["insufficient_evidence"]


def test_evaluation_requires_units_and_all_evidence_groups():
    def g(terms):
        return {"alternatives": [{"doc_id": "test", "pages": [1], "contains": terms}]}
    question = {"evidence": [g(["Revenue", "30,040", "millions"]), g(["Net income", "16,599", "millions"])]}
    evidence = [{"text": "Revenue 30,040", "metadata": {"doc_id": "test", "page": 1}}]
    assert evidence_metrics(question, evidence)["hit"] == 0
    evidence[0]["text"] += " (in millions)"
    score = evidence_metrics(question, evidence)
    assert score["hit"] == 1 and score["evidence_recall"] == 0.5 and score["all_evidence"] == 0


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1]])
def test_bad_vectors_fail(vector):
    with pytest.raises(ValueError):
        VectorIndex([doc("text")], [vector], {"dimension": 2})
