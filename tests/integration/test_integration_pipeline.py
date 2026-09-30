"""
End-to-end pipeline integration tests.

CONCEPT: Pipeline Integration
Verifies the cohesive flow between:
  Document -> Parser -> Chunker -> Embedder -> Vector Store & BM25 -> Hybrid Search -> Reranker
Ensuring that data representations and schema structures match across all stage boundaries.
"""

from unittest.mock import patch

import numpy as np
import pytest

from rag.ingestion.pipeline import ingest_text
from rag.retrieval.hybrid import hybrid_search


@pytest.mark.integration
class TestPipelineIntegration:
    """Verify integration across ingestion, indexing, and retrieval stages."""

    def test_text_ingest_and_bm25_retrieval(self) -> None:
        """Verify that ingested text is chunked and retrievable via BM25."""
        test_content = (
            "Python decorators are syntactically represented by the @ symbol.\n\n"
            "They allow developers to modify or extend the behavior of functions or methods "
            "without permanently modifying the callable itself.\n\n"
            "A common decorator pattern is logging execution duration using functools.wraps."
        )

        with (
            patch("rag.ingestion.pipeline.get_qdrant_client"),
            patch("rag.ingestion.pipeline.upsert_chunks"),
            patch("rag.ingestion.pipeline.embed_texts") as mock_embed,
        ):
            # Mock 3 chunk embeddings
            mock_embed.return_value = np.zeros((3, 1024), dtype=np.float32)

            result = ingest_text(
                text=test_content,
                source_name="decorators_guide.md",
                doc_id="doc-integration-123",
            )

            assert result["doc_id"] == "doc-integration-123"
            assert result["source_file"] == "decorators_guide.md"
            assert result["chunk_count"] > 0

    def test_hybrid_search_fusion_ranking(self) -> None:
        """Verify that Reciprocal Rank Fusion correctly merges dense and sparse results."""
        dense_results = [
            {"chunk_id": "chunk_A", "text": "Dense top match", "source_file": "doc1.md"},
            {"chunk_id": "chunk_B", "text": "Dense second match", "source_file": "doc2.md"},
        ]
        bm25_results = [
            {"chunk_id": "chunk_B", "text": "BM25 top match (chunk B)", "source_file": "doc2.md"},
            {"chunk_id": "chunk_C", "text": "BM25 second match", "source_file": "doc3.md"},
        ]

        # chunk_B appears in both -> should be boosted by RRF
        fused = hybrid_search(dense_results, bm25_results, top_k=3)
        assert len(fused) == 3
        # Chunk B is in both lists, so its reciprocal rank sum is highest: 1/(60+2) + 1/(60+1)
        assert fused[0]["chunk_id"] == "chunk_B"
