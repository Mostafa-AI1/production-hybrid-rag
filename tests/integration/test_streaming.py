"""
Integration tests for the Server-Sent Events (SSE) Streaming Query Endpoint.

Verifies:
  1. POST /api/v1/query/stream returns HTTP 200 with text/event-stream content type.
  2. Stream emits token events, sources event, and done event in correct sequence.
  3. L1 exact cache hit streams cached payload without calling LLM.
  4. L2 semantic cache hit streams cached payload without calling LLM.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.api.routes.query import _get_cache, _get_semantic_cache


@pytest.fixture
def app_instance() -> FastAPI:
    """Create a fresh app instance for testing."""
    return create_app()


@pytest.fixture
def client(app_instance: FastAPI) -> Generator[TestClient, None, None]:
    """TestClient fixture with automatic dependency override cleanup."""
    with TestClient(app_instance, raise_server_exceptions=True) as c:
        yield c
    app_instance.dependency_overrides.clear()


def test_query_stream_success(client: TestClient) -> None:
    """Test standard streaming response emitting token, sources, and done events."""
    mock_chunks = [
        {
            "source_file": "doc1.md",
            "chunk_index": 0,
            "text": "Python asyncio handles I/O concurrency.",
            "reranker_score": 0.95,
        }
    ]

    def fake_stream(*args: object, **kwargs: object) -> Generator[str, None, None]:
        yield "Asyncio "
        yield "is "
        yield "great."

    # Override L1 and L2 caches to simulate cold cache
    l1_cache = AsyncMock()
    l1_cache.get.return_value = None
    l2_cache = MagicMock()
    l2_cache.get.return_value = None

    client.app.dependency_overrides[_get_cache] = lambda: l1_cache  # type: ignore[attr-defined]
    client.app.dependency_overrides[_get_semantic_cache] = lambda: l2_cache  # type: ignore[attr-defined]

    with (
        patch("rag.api.routes.query.embed_query", return_value=[0.1] * 1024),
        patch("rag.api.routes.query.search_dense", return_value=mock_chunks),
        patch("rag.api.routes.query.get_bm25_retriever") as mock_bm25_fn,
        patch("rag.api.routes.query.rerank", return_value=mock_chunks),
        patch("rag.api.routes.query.generate_stream", side_effect=fake_stream),
    ):
        mock_bm25 = MagicMock()
        mock_bm25.document_count = 1
        mock_bm25.search.return_value = mock_chunks
        mock_bm25_fn.return_value = mock_bm25

        response = client.post(
            "/api/v1/query/stream",
            json={"query": "Explain asyncio concurrency", "use_cache": False},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert '{"type": "token", "content": "Asyncio "}' in content
        assert '{"type": "token", "content": "is "}' in content
        assert '{"type": "token", "content": "great."}' in content
        assert '"type": "sources"' in content
        assert '"type": "done"' in content


def test_query_stream_l1_cache_hit(client: TestClient) -> None:
    """When exact query exists in L1 Redis cache, stream immediately from cache."""
    cached_payload = {
        "answer": "Cached answer string.",
        "sources": [
            {
                "source_file": "cached.md",
                "chunk_index": 0,
                "text_preview": "Cached preview",
            }
        ],
        "provider": "redis",
    }

    l1_cache = AsyncMock()
    l1_cache.get.return_value = cached_payload
    client.app.dependency_overrides[_get_cache] = lambda: l1_cache  # type: ignore[attr-defined]

    with patch("rag.api.routes.query.generate_stream") as mock_generate_stream:
        response = client.post(
            "/api/v1/query/stream",
            json={"query": "Exact cached question?", "use_cache": True},
        )

        assert response.status_code == 200
        content = response.text
        assert '{"type": "token", "content": "Cached answer string."}' in content
        assert '"cached": true' in content
        mock_generate_stream.assert_not_called()


def test_query_stream_semantic_cache_hit(client: TestClient) -> None:
    """When semantically equivalent query exists in L2 Qdrant cache, stream immediately."""
    semantic_hit = {
        "answer": "Semantic cached answer.",
        "sources": [{"source_file": "sem.md", "chunk_index": 1}],
        "provider": "semantic_cache",
        "similarity_score": 0.96,
    }

    l1_cache = AsyncMock()
    l1_cache.get.return_value = None
    l2_cache = MagicMock()
    l2_cache.get.return_value = semantic_hit

    client.app.dependency_overrides[_get_cache] = lambda: l1_cache  # type: ignore[attr-defined]
    client.app.dependency_overrides[_get_semantic_cache] = lambda: l2_cache  # type: ignore[attr-defined]

    with (
        patch("rag.api.routes.query.embed_query", return_value=[0.2] * 1024),
        patch("rag.api.routes.query.generate_stream") as mock_generate_stream,
    ):
        response = client.post(
            "/api/v1/query/stream",
            json={"query": "Synonymous question?", "use_cache": True},
        )

        assert response.status_code == 200
        content = response.text
        assert '{"type": "token", "content": "Semantic cached answer."}' in content
        assert '"provider": "semantic_cache"' in content
        mock_generate_stream.assert_not_called()
