"""
Semantic Query Cache — Embedding similarity matching for semantically equivalent queries.

CONCEPT: Semantic Caching
Traditional caching uses exact key matching (e.g., SHA-256 of the normalized query string).
In conversational AI, different users ask the exact same question with completely different wording:
  - User 1: "What is a Python decorator?"
  - User 2: "Can you explain how decorators work in Python?"
  - User 3: "Tell me about Python decorators."

Exact hash caching fails for User 2 and User 3 (0% hit rate).
Semantic caching embeds the query vector and computes cosine similarity
against previously answered queries.
If cosine similarity >= threshold (default 0.92):
  - Cache Hit!
  - Return the cached answer and sources in ~10ms.
  - Bypass reranking and LLM generation completely (saves 1-3 seconds and 100% of LLM cost).

CALIBRATING THE SIMILARITY THRESHOLD:
  - Too low (< 0.88): Risk of "false hits" — returning an answer to a subtly different question.
  - Too high (> 0.96): Negligible hit rate advantage over exact match.
  - Optimal range: 0.91 – 0.94 for BGE-M3 dense embeddings.

INTERVIEW QUESTIONS TO MASTER:
  1. What is the difference between exact-match and semantic caching, and what are the trade-offs?
  2. How do you handle cache invalidation in a semantic cache when underlying documents change?
  3. What is "cache poisoning" in semantic caching, and how do you protect against it?
"""

import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.retrieval.vector_store import get_qdrant_client

log = get_logger(__name__)


class SemanticCache:
    """
    Qdrant-backed semantic query cache.

    Stores past queries and responses indexed by their dense embedding vector.
    """

    def __init__(self, client: QdrantClient | None = None) -> None:
        self.settings = get_settings()
        self.client = client or get_qdrant_client()
        self.collection = self.settings.semantic_cache_collection
        self.threshold = self.settings.semantic_cache_threshold

    def ensure_collection(self, vector_dim: int = 1024) -> None:
        """Ensure the semantic cache collection exists in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                log.info("semantic_cache_collection_created", collection=self.collection)
        except Exception as e:
            log.warning("semantic_cache_init_failed", error=str(e))

    def get(self, query_vector: list[float]) -> dict[str, Any] | None:
        """
        Search for a semantically similar cached query.

        Returns:
            Cached response dictionary if cosine similarity >= threshold, else None.
        """
        if not self.settings.semantic_cache_enabled:
            return None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=1,
                score_threshold=self.threshold,
                with_payload=True,
            )
            points = getattr(response, "points", [])
            if not points:
                log.debug("semantic_cache_miss")
                return None

            hit = points[0]
            score = float(hit.score)
            payload = hit.payload or {}

            log.info(
                "semantic_cache_hit",
                similarity=round(score, 4),
                cached_query=payload.get("original_query", "")[:60],
            )

            return {
                "answer": payload.get("answer", ""),
                "sources": payload.get("sources", []),
                "provider": payload.get("provider", "semantic_cache"),
                "similarity_score": round(score, 4),
                "matched_query": payload.get("original_query", ""),
            }

        except Exception as e:
            # Semantic cache errors must never crash the query pipeline
            log.warning("semantic_cache_lookup_error", error=str(e))
            return None

    def set(
        self,
        query: str,
        query_vector: list[float],
        response_data: dict[str, Any],
    ) -> None:
        """
        Store a query embedding and its generated response into the semantic cache.
        """
        if not self.settings.semantic_cache_enabled:
            return

        try:
            point_id = str(uuid.uuid4())
            payload = {
                "original_query": query,
                "answer": response_data.get("answer", ""),
                "sources": response_data.get("sources", []),
                "provider": response_data.get("provider", ""),
                "created_at": time.time(),
            }

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload=payload,
                    )
                ],
            )
            log.debug("semantic_cache_stored", query=query[:50])

        except Exception as e:
            log.warning("semantic_cache_store_error", error=str(e))

    def clear(self) -> None:
        """Flush all cached entries by recreating the collection."""
        try:
            self.client.delete_collection(self.collection)
            log.info("semantic_cache_cleared")
        except Exception as e:
            log.warning("semantic_cache_clear_error", error=str(e))
