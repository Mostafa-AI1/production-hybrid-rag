"""
Qdrant vector store wrapper.

CONCEPT: A vector database is a specialised database that stores vectors
(embeddings) alongside structured metadata, and efficiently answers
"find the K vectors most similar to this query vector" using Approximate
Nearest Neighbor (ANN) algorithms like HNSW.

WHY NOT use a regular database for this?
  A Postgres table with 1M rows of float[1024] columns would require a full
  table scan for every similarity query — O(N) time. Qdrant's HNSW index
  makes this O(log N) at the cost of some approximation error. At 1M vectors,
  exact search takes minutes; HNSW takes milliseconds.

PAYLOAD SCHEMA (what we store alongside each vector):
  {
    "doc_id": "uuid",
    "chunk_id": "uuid",
    "source_file": "filename.pdf",
    "source_type": "pdf",
    "chunk_index": 3,
    "total_chunks": 47,
    "text": "The actual chunk text...",
    "ingested_at": "2026-09-19T...",
    "char_count": 1024
  }

  This metadata enables filtered search — e.g., "only search chunks from
  documents ingested after 2026-01-01" — without touching the vector index.
"""

import math
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.ingestion.chunker import TextChunk

log = get_logger(__name__)

# Qdrant vector distance metric. Cosine similarity is standard for text embeddings
# because the magnitude of a vector is less important than its direction.
# Since we L2-normalise at embedding time, cosine == dot product (faster).
_DISTANCE = qdrant_models.Distance.COSINE


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client connected to the configured instance."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,  # None for self-hosted
        timeout=30,
    )


def ensure_collection_exists(client: QdrantClient, vector_dim: int) -> None:
    """
    Create the Qdrant collection if it doesn't already exist.

    Called at application startup — idempotent (safe to call multiple times).

    Args:
        client: Qdrant client instance.
        vector_dim: Dimension of the embedding vectors (e.g., 1024 for BGE-M3).
    """
    settings = get_settings()
    collection_name = settings.qdrant.collection

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        log.info("collection_already_exists", collection=collection_name)
        return

    log.info("creating_collection", collection=collection_name, dim=vector_dim)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_dim,
            distance=_DISTANCE,
            # HNSW config — these are the tunable knobs:
            # m: number of connections per node (higher = more accurate, more RAM)
            # ef_construct: search depth during index build (higher = better index quality)
            hnsw_config=qdrant_models.HnswConfigDiff(m=16, ef_construct=100),
        ),
    )
    # Create payload indexes for efficient filtering
    for field_name in ["source_file", "source_type", "ingested_at"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
    log.info("collection_created", collection=collection_name)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    doc_id: str | None = None,
) -> list[str]:
    """
    Upsert chunks and their embeddings into Qdrant.

    UPSERT = update if exists, insert if not. This means ingesting the same
    document twice is idempotent — you won't get duplicate chunks.

    Args:
        client: Qdrant client.
        chunks: TextChunks from the chunker.
        embeddings: Corresponding embeddings (must be same length as chunks).
        doc_id: Optional document-level ID. Auto-generated if not provided.

    Returns:
        List of chunk_ids that were upserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
        )

    settings = get_settings()
    resolved_doc_id = doc_id or str(uuid.uuid4())
    now_iso = datetime.now(UTC).isoformat()

    points = []
    chunk_ids = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_id = str(uuid.uuid4())
        chunk_ids.append(chunk_id)
        points.append(
            qdrant_models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "doc_id": resolved_doc_id,
                    "chunk_id": chunk_id,
                    "source_file": chunk.source_file,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "text": chunk.text,
                    "ingested_at": now_iso,
                    "char_count": chunk.char_count,
                    **chunk.metadata,
                },
            )
        )

    # Qdrant recommends batches of 100-1000 points
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=settings.qdrant.collection,
            points=batch,
            wait=True,  # wait=True ensures the points are indexed before returning
        )

    log.info(
        "chunks_upserted",
        doc_id=resolved_doc_id,
        count=len(chunks),
        collection=settings.qdrant.collection,
    )
    return chunk_ids


def search_dense(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
    filter_conditions: qdrant_models.Filter | None = None,
) -> list[dict[str, Any]]:
    """
    Dense vector search — find the top_k most similar chunks.

    Args:
        client: Qdrant client.
        query_vector: Embedding of the user's query.
        top_k: Number of results to return.
        filter_conditions: Optional Qdrant filter (e.g., by source_file).

    Returns:
        List of dicts containing 'score', 'text', and all payload fields.
    """
    settings = get_settings()
    results = client.search(  # type: ignore[attr-defined]
        collection_name=settings.qdrant.collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=filter_conditions,
        with_payload=True,  # return the stored text alongside the score
        score_threshold=0.3,  # ignore very low-relevance results
    )

    return [
        {
            "score": hit.score,
            "chunk_id": hit.id,
            **(hit.payload or {}),
        }
        for hit in results
    ]


def get_collection_stats(client: QdrantClient) -> dict[str, Any]:
    """Return basic statistics about the collection for health monitoring."""
    settings = get_settings()
    info = client.get_collection(settings.qdrant.collection)
    return {
        "points_count": getattr(info, "points_count", 0),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
        "status": getattr(info.status, "value", str(info.status)),
    }


def compute_sparse_tokens(text: str) -> qdrant_models.SparseVector:
    """
    Generate a sparse vector representation (lexical term-frequency weights).

    Maps word token hashes to term frequency weights.
    Used for native Qdrant BM25-style lexical search.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return qdrant_models.SparseVector(indices=[], values=[])

    counts = Counter(words)
    indices: list[int] = []
    values: list[float] = []

    for word, count in counts.items():
        # Hash to fixed 32-bit positive index space
        idx = abs(hash(word)) % 1_000_000
        # Sublinear term frequency scaling: 1 + ln(count)
        weight = 1.0 + math.log(count)
        indices.append(idx)
        values.append(round(weight, 4))

    return qdrant_models.SparseVector(indices=indices, values=values)


def search_sparse(
    client: QdrantClient,
    query_text: str,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Search Qdrant using native sparse lexical representations.
    """
    sparse_vec = compute_sparse_tokens(query_text)
    if not sparse_vec.indices:
        return []

    settings = get_settings()
    try:
        response = client.query_points(
            collection_name=settings.qdrant.collection,
            query=sparse_vec,
            using="bm25_sparse",
            limit=top_k,
            with_payload=True,
        )
        points = getattr(response, "points", [])
        return [
            {
                "score": hit.score,
                "chunk_id": hit.id,
                **(hit.payload or {}),
            }
            for hit in points
        ]
    except Exception as e:
        log.debug("qdrant_sparse_search_unavailable_using_fallback", error=str(e))
        return []
