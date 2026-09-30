"""
Integration tests for FastAPI application routes, authentication, and rate limiting.

CONCEPT: Integration Testing in API Services
Unlike unit tests (which test individual functions in isolation with mocks),
integration tests verify that the entire request-response stack works together:
  Client -> Middleware Stack (Logging, Auth, Rate Limiting) -> Routing ->
  Dependency Injection -> Route Handler -> Response Serialization

WHY TEST CLIENT (httpx / TestClient)?
Starlette's TestClient simulates requests directly against the ASGI application
without needing to bind to a live TCP port or spin up a separate background server.
It provides lightning-fast testing of HTTP status codes, headers, and payload schemas.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.core.config import get_settings


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient with fresh app instance."""
    app = create_app()
    return TestClient(app)


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health check and readiness probe endpoints."""

    def test_liveness_health_endpoint(self, client: TestClient) -> None:
        """GET /health must return 200 OK without requiring authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_readiness_probe_structure(self, client: TestClient) -> None:
        """GET /health/ready returns service check details."""
        response = client.get("/health/ready")
        assert response.status_code in {200, 503}
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "qdrant" in data["checks"]
        assert "redis" in data["checks"]


@pytest.mark.integration
class TestAuthenticationMiddleware:
    """Test API key authentication enforcement."""

    def test_auth_enforced_when_enabled(self, client: TestClient) -> None:
        """When api_auth_enabled is True, protected endpoints reject unauthenticated calls."""
        settings = get_settings()
        settings.api_auth_enabled = True
        settings.api_secret_key = "secret-test-token"

        # Missing key -> 401
        res_missing = client.post("/api/v1/query", json={"query": "What is Python?"})
        assert res_missing.status_code == 401
        assert "Invalid or missing API key" in res_missing.json()["detail"]

        # Wrong key -> 401
        res_wrong = client.post(
            "/api/v1/query",
            json={"query": "What is Python?"},
            headers={"X-API-Key": "wrong-token"},
        )
        assert res_wrong.status_code == 401

        # Valid X-API-Key header -> bypasses 401 (reaches handler)
        # We mock downstream pipeline steps so test focuses on auth
        with (
            patch("rag.api.routes.query.embed_query") as mock_embed,
            patch("rag.api.routes.query.search_dense") as mock_dense,
            patch("rag.api.routes.query.generate") as mock_gen,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_dense.return_value = [
                {"text": "Python is a programming language", "source_file": "doc.md"}
            ]
            mock_gen.return_value = {
                "answer": "Python is a language.",
                "provider": "mock",
                "usage": None,
            }

            res_valid = client.post(
                "/api/v1/query",
                json={"query": "What is Python?", "use_cache": False},
                headers={"X-API-Key": "secret-test-token"},
            )
            assert res_valid.status_code == 200

        # Valid Bearer token in Authorization header -> also succeeds
        with (
            patch("rag.api.routes.query.embed_query") as mock_embed,
            patch("rag.api.routes.query.search_dense") as mock_dense,
            patch("rag.api.routes.query.generate") as mock_gen,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_dense.return_value = [
                {"text": "Python is a programming language", "source_file": "doc.md"}
            ]
            mock_gen.return_value = {
                "answer": "Python is a language.",
                "provider": "mock",
                "usage": None,
            }

            res_bearer = client.post(
                "/api/v1/query",
                json={"query": "What is Python?", "use_cache": False},
                headers={"Authorization": "Bearer secret-test-token"},
            )
            assert res_bearer.status_code == 200


@pytest.mark.integration
class TestIngestionEndpoints:
    """Test document ingestion routes and input validation."""

    def test_ingest_text_valid_payload(self, client: TestClient) -> None:
        """POST /api/v1/ingest/text accepts valid text and returns 202 Accepted."""
        payload = {
            "text": "Decorators in Python allow wrapping functions dynamically.",
            "source_name": "python_decorators.txt",
        }
        response = client.post("/api/v1/ingest/text", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing"
        assert "job_id" in data

    def test_ingest_text_validation_failure(self, client: TestClient) -> None:
        """Short text (< 10 chars) triggers Pydantic 422 Unprocessable Entity."""
        payload = {"text": "short", "source_name": "test"}
        response = client.post("/api/v1/ingest/text", json=payload)
        assert response.status_code == 422

    def test_ingest_file_unsupported_type(self, client: TestClient) -> None:
        """Uploading an unsupported file extension returns 415 Unsupported Media Type."""
        files = {"file": ("malicious.exe", b"binarycontent", "application/octet-stream")}
        response = client.post("/api/v1/ingest/file", files=files)
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.integration
class TestQueryEndpointPipeline:
    """Test full query endpoint orchestration and caching."""

    def test_query_pipeline_success(self, client: TestClient) -> None:
        """Successful query returns grounded answer with sources and trace ID."""
        with (
            patch("rag.api.routes.query.embed_query") as mock_embed,
            patch("rag.api.routes.query.search_dense") as mock_dense,
            patch("rag.api.routes.query.rerank") as mock_rerank,
            patch("rag.api.routes.query.generate") as mock_gen,
        ):
            mock_embed.return_value = [0.0] * 1024
            mock_dense.return_value = [
                {
                    "text": "Python decorators wrap functions.",
                    "source_file": "decorators.md",
                    "chunk_index": 0,
                }
            ]
            mock_rerank.return_value = [
                {
                    "text": "Python decorators wrap functions.",
                    "source_file": "decorators.md",
                    "chunk_index": 0,
                    "reranker_score": 0.95,
                }
            ]
            mock_gen.return_value = {
                "answer": "A decorator wraps another function.",
                "provider": "mock-llm",
                "usage": {"input": 15, "output": 8, "total": 23},
            }

            response = client.post(
                "/api/v1/query",
                json={"query": "What is a decorator in Python?", "use_cache": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "A decorator wraps another function."
            assert len(data["sources"]) == 1
            assert data["sources"][0]["source_file"] == "decorators.md"
            assert data["provider"] == "mock-llm"
            assert data["cached"] is False
            assert "trace_id" in data
            assert data["latency_ms"] >= 0.0

    def test_query_no_results_raises_404(self, client: TestClient) -> None:
        """When no documents exist in vector DB or BM25, query returns 404."""
        with (
            patch("rag.api.routes.query.embed_query") as mock_embed,
            patch("rag.api.routes.query.search_dense") as mock_dense,
            patch("rag.retrieval.bm25.get_bm25_retriever") as mock_bm25,
        ):
            mock_embed.return_value = [0.0] * 1024
            mock_dense.return_value = []
            bm25_instance = MagicMock()
            bm25_instance.document_count = 0
            mock_bm25.return_value = bm25_instance

            response = client.post(
                "/api/v1/query",
                json={"query": "Unindexed obscure question?", "use_cache": False},
            )

            assert response.status_code == 404
            assert "No relevant documents found" in response.json()["detail"]
