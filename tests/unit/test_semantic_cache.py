"""
Unit tests for the Qdrant Semantic Query Cache.

Verifies:
  1. Cache hit when vector cosine similarity is above threshold.
  2. Cache miss when no vectors match or similarity is below threshold.
  3. Safe fallback when semantic cache is disabled in settings.
  4. Exception handling without pipeline interruption.
  5. Cache insertion and collection management.
"""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models

from rag.cache.semantic_cache import SemanticCache


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Fixture providing a mock Qdrant client."""
    client = MagicMock()
    return client


def test_semantic_cache_disabled_returns_none(mock_qdrant_client: MagicMock) -> None:
    """When disabled in config, get and set operations return immediately."""
    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_enabled = False
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        assert cache.get([0.1] * 1024) is None

        cache.set("query", [0.1] * 1024, {"answer": "test"})
        mock_qdrant_client.upsert.assert_not_called()


def test_semantic_cache_hit(mock_qdrant_client: MagicMock) -> None:
    """When a similar vector is found in Qdrant, return cached payload."""
    hit = MagicMock()
    hit.score = 0.95
    hit.payload = {
        "original_query": "What is Python asyncio?",
        "answer": "Asyncio is a library to write concurrent code using async/await.",
        "sources": [{"chunk_id": "c1", "content": "Asyncio overview"}],
        "provider": "groq",
    }
    mock_qdrant_client.query_points.return_value = MagicMock(points=[hit])

    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_enabled = True
        settings_mock.semantic_cache_collection = "rag_semantic_cache"
        settings_mock.semantic_cache_threshold = 0.92
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        result = cache.get([0.1] * 1024)

        assert result is not None
        assert result["answer"] == hit.payload["answer"]
        assert result["similarity_score"] == 0.95
        assert result["matched_query"] == "What is Python asyncio?"
        assert result["sources"] == hit.payload["sources"]


def test_semantic_cache_miss_empty_results(mock_qdrant_client: MagicMock) -> None:
    """When no points meet the score threshold, search returns empty list -> miss."""
    mock_qdrant_client.query_points.return_value = MagicMock(points=[])

    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_enabled = True
        settings_mock.semantic_cache_collection = "rag_semantic_cache"
        settings_mock.semantic_cache_threshold = 0.92
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        result = cache.get([0.1] * 1024)
        assert result is None


def test_semantic_cache_exception_graceful_handling(mock_qdrant_client: MagicMock) -> None:
    """If Qdrant fails during get/set, cache catches error and returns None."""
    mock_qdrant_client.query_points.side_effect = RuntimeError("Connection dropped")

    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_enabled = True
        settings_mock.semantic_cache_collection = "rag_semantic_cache"
        settings_mock.semantic_cache_threshold = 0.92
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        result = cache.get([0.1] * 1024)
        assert result is None  # Does not raise!


def test_semantic_cache_set(mock_qdrant_client: MagicMock) -> None:
    """Setting cache entry calls upsert with PointStruct containing embedding vector."""
    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_enabled = True
        settings_mock.semantic_cache_collection = "rag_semantic_cache"
        settings_mock.semantic_cache_threshold = 0.92
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        cache.set(
            query="What is GIL?",
            query_vector=[0.05] * 1024,
            response_data={"answer": "Global Interpreter Lock", "sources": [], "provider": "groq"},
        )

        assert mock_qdrant_client.upsert.called
        call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "rag_semantic_cache"
        points = call_kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["original_query"] == "What is GIL?"
        assert points[0].payload["answer"] == "Global Interpreter Lock"


def test_semantic_cache_ensure_collection(mock_qdrant_client: MagicMock) -> None:
    """Ensure collection creates new collection if it does not exist."""
    collections_desc = MagicMock()
    collections_desc.collections = []  # Empty -> collection does not exist
    mock_qdrant_client.get_collections.return_value = collections_desc

    with patch("rag.cache.semantic_cache.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.semantic_cache_collection = "rag_semantic_cache"
        mock_settings.return_value = settings_mock

        cache = SemanticCache(client=mock_qdrant_client)
        cache.ensure_collection(vector_dim=1024)

        mock_qdrant_client.create_collection.assert_called_once()
        create_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert create_kwargs["collection_name"] == "rag_semantic_cache"
        assert create_kwargs["vectors_config"].size == 1024
        assert create_kwargs["vectors_config"].distance == models.Distance.COSINE
