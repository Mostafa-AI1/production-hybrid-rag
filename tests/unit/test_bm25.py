"""
Unit tests for BM25 retrieval.

These tests verify the semantic properties of BM25:
  - Exact keyword matches are found
  - Results are ranked (higher relevance = higher position)
  - Edge cases (empty corpus, empty query) don't crash
  - Score normalisation works correctly
"""

import pytest

from rag.retrieval.bm25 import BM25Retriever, normalise_bm25_scores


def _make_corpus(texts: list[str]) -> list[dict]:
    """Helper: make a list of BM25 corpus documents from raw strings."""
    return [
        {"text": t, "chunk_id": f"chunk_{i}", "source_file": "test.txt"}
        for i, t in enumerate(texts)
    ]


@pytest.mark.unit
class TestBM25Retriever:
    def test_finds_exact_keyword_match(self) -> None:
        """A query that exactly matches a document's keyword should find it."""
        retriever = BM25Retriever()
        corpus = _make_corpus(
            [
                "Python is a programming language",
                "Elephants are large mammals",
                "Machine learning uses neural networks",
            ]
        )
        retriever.index(corpus)

        results = retriever.search("Python programming", top_k=3)
        assert len(results) > 0
        assert results[0]["text"] == "Python is a programming language"

    def test_no_match_returns_empty(self) -> None:
        """A query with no matching terms should return no results."""
        retriever = BM25Retriever()
        retriever.index(_make_corpus(["cats and dogs", "fish and birds"]))
        results = retriever.search("xyznonsenseword", top_k=5)
        assert results == []

    def test_search_before_index_returns_empty(self) -> None:
        """Calling search before index() should return empty list, not crash."""
        retriever = BM25Retriever()
        results = retriever.search("anything", top_k=5)
        assert results == []

    def test_empty_corpus_returns_empty(self) -> None:
        """Indexing an empty corpus then searching should return empty."""
        retriever = BM25Retriever()
        retriever.index([])
        results = retriever.search("something", top_k=5)
        assert results == []

    def test_results_sorted_by_relevance(self) -> None:
        """Results should be ordered with highest BM25 score first."""
        retriever = BM25Retriever()
        corpus = _make_corpus(
            [
                "Python is great",
                "Python Python Python is the best Python language ever Python",
                "Java is also a language",
            ]
        )
        retriever.index(corpus)
        results = retriever.search("Python", top_k=3)
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self) -> None:
        """Search should return at most top_k results."""
        retriever = BM25Retriever()
        retriever.index(_make_corpus([f"document about topic {i}" for i in range(20)]))
        results = retriever.search("document topic", top_k=5)
        assert len(results) <= 5

    def test_bm25_score_in_results(self) -> None:
        """Every result should have a bm25_score field."""
        retriever = BM25Retriever()
        retriever.index(_make_corpus(["hello world", "test document"]))
        results = retriever.search("hello", top_k=2)
        for r in results:
            assert "bm25_score" in r
            assert r["bm25_score"] > 0

    def test_add_documents_increases_count(self) -> None:
        """add_documents() should increase the document_count."""
        retriever = BM25Retriever()
        retriever.index(_make_corpus(["doc one", "doc two"]))
        assert retriever.document_count == 2
        retriever.add_documents(_make_corpus(["doc three"]))
        assert retriever.document_count == 3


@pytest.mark.unit
class TestNormaliseBM25Scores:
    def test_scores_in_zero_one_range(self) -> None:
        """Normalised scores should all be in [0, 1]."""
        docs = [
            {"bm25_score": 5.0, "text": "a"},
            {"bm25_score": 10.0, "text": "b"},
            {"bm25_score": 2.0, "text": "c"},
        ]
        normalised = normalise_bm25_scores(docs)
        for doc in normalised:
            assert 0.0 <= doc["bm25_score_normalised"] <= 1.0

    def test_highest_score_becomes_one(self) -> None:
        """The document with the highest raw score should get normalised score of 1.0."""
        docs = [{"bm25_score": 3.0}, {"bm25_score": 7.0}, {"bm25_score": 1.0}]
        normalised = normalise_bm25_scores(docs)
        max_normalised = max(d["bm25_score_normalised"] for d in normalised)
        assert max_normalised == pytest.approx(1.0)

    def test_empty_list_returns_empty(self) -> None:
        assert normalise_bm25_scores([]) == []
