"""
Unit tests for Reciprocal Rank Fusion (hybrid retrieval).

These tests are the most "interview-worthy" tests in the project because
RRF has clear mathematical properties that are easy to verify:
  - A document in multiple lists ranks higher than one in a single list
  - Rank matters more than score (RRF is rank-based, not score-based)
  - Documents not in any list don't appear in results
  - Empty input lists are handled gracefully
"""

import pytest

from rag.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion


def _make_docs(ids: list[str]) -> list[dict]:
    """Helper: create a ranked list of docs with the given chunk_ids."""
    return [{"chunk_id": did, "text": f"Text for {did}", "source_file": "test.txt"} for did in ids]


@pytest.mark.unit
class TestReciprocalRankFusion:
    def test_document_in_both_lists_ranks_higher(self) -> None:
        """
        A document appearing in BOTH result lists should outscore one in only one list.

        This is the fundamental property of RRF. If dense retrieval ranks "A" first
        and BM25 also ranks "A" first, it should dominate the fused ranking.
        """
        dense = _make_docs(["A", "B", "C"])
        bm25 = _make_docs(["A", "D", "E"])

        results = reciprocal_rank_fusion([dense, bm25], top_k=5)
        ids = [r["chunk_id"] for r in results]

        assert ids[0] == "A", "A (in both lists at rank 1) should be top result"

    def test_rrf_score_decreases_with_rank(self) -> None:
        """Higher-ranked documents should have higher RRF scores."""
        docs = _make_docs(["X", "Y", "Z"])
        results = reciprocal_rank_fusion([docs], top_k=3)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores should be decreasing"

    def test_returns_at_most_top_k_results(self) -> None:
        """RRF should return exactly top_k results (or fewer if candidates < top_k)."""
        docs = _make_docs([f"doc_{i}" for i in range(20)])
        results = reciprocal_rank_fusion([docs], top_k=5)
        assert len(results) <= 5

    def test_empty_lists_return_empty(self) -> None:
        """Empty input should return an empty list without error."""
        results = reciprocal_rank_fusion([], top_k=10)
        assert results == []

    def test_empty_inner_list_handled(self) -> None:
        """A mix of populated and empty result lists should work fine."""
        dense = _make_docs(["A", "B"])
        results = reciprocal_rank_fusion([dense, []], top_k=5)
        assert len(results) == 2

    def test_rrf_score_field_present(self) -> None:
        """Every result should have an 'rrf_score' field."""
        docs = _make_docs(["A", "B", "C"])
        results = reciprocal_rank_fusion([docs], top_k=3)
        for r in results:
            assert "rrf_score" in r
            assert isinstance(r["rrf_score"], float)
            assert r["rrf_score"] > 0

    def test_deduplication(self) -> None:
        """The same document should appear only once in the fused results."""
        # Both lists contain the same documents
        list1 = _make_docs(["A", "B", "C"])
        list2 = _make_docs(["A", "B", "C"])
        results = reciprocal_rank_fusion([list1, list2], top_k=10)
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids)), "No duplicate documents in fused results"

    def test_missing_id_key_skipped(self) -> None:
        """Documents without the id_key should be skipped gracefully (no crash)."""
        docs = [{"text": "no id here", "source_file": "test.txt"}]
        results = reciprocal_rank_fusion([docs], top_k=5, id_key="chunk_id")
        assert results == []


@pytest.mark.unit
class TestHybridSearch:
    def test_uses_both_sources(self) -> None:
        """hybrid_search should return results from both dense and BM25."""
        dense = _make_docs(["dense_1", "dense_2"])
        bm25 = _make_docs(["bm25_1", "bm25_2"])
        results = hybrid_search(dense, bm25, top_k=10)
        ids = {r["chunk_id"] for r in results}
        assert ids == {"dense_1", "dense_2", "bm25_1", "bm25_2"}

    def test_falls_back_to_dense_when_bm25_empty(self) -> None:
        """With no BM25 results, should still return dense results."""
        dense = _make_docs(["A", "B", "C"])
        results = hybrid_search(dense, [], top_k=5)
        assert len(results) == 3
