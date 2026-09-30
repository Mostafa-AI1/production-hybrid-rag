"""
Embedding model wrapper — BGE-M3 via sentence-transformers.

CONCEPT: An embedding model maps text to a fixed-size dense vector.
  "What is Python?" → [0.12, -0.45, 0.78, ...]  (1024 floats for BGE-M3)

  Texts that are semantically similar end up with vectors that are geometrically
  close (high cosine similarity). This is how semantic search works.

WHY BGE-M3 specifically?
  Most embedding models only produce *dense* vectors. BGE-M3 uniquely produces
  THREE types from a single forward pass:
    1. Dense vectors  — standard semantic similarity (1024 dims)
    2. Sparse vectors — like TF-IDF, capturing exact keyword importance
    3. Multi-vector   — ColBERT-style, one vector per token (most accurate, expensive)

  For our hybrid retrieval, we use dense (for Qdrant) and sparse (for BM25 weighting).

PERFORMANCE NOTE:
  First call downloads the model (~1.1GB). Subsequent calls load from cache.
  On CPU, encoding ~32 chunks takes ~2-5 seconds. This is fine for ingestion
  (batch operation) but needs caching for per-query encoding (see redis_cache.py).
"""

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    WHY lru_cache? Model loading takes ~5-10 seconds and uses ~1GB RAM.
    We load once and reuse across all requests. The OS handles memory.

    WHY not a class variable? lru_cache on a module-level function avoids
    the need to pass a model instance through the entire call stack.
    """
    settings = get_settings()
    log.info(
        "loading_embedding_model",
        model=settings.embedding.model,
        device=settings.embedding.device,
    )
    model = SentenceTransformer(
        settings.embedding.model,
        device=settings.embedding.device,
        trust_remote_code=False,  # security: never run arbitrary model code
    )
    log.info("embedding_model_loaded", model=settings.embedding.model)
    return model


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Args:
        texts: List of strings to embed. Empty strings are rejected.
        is_query: If True, use the query prompt prefix (BGE-M3 uses different
                  prompts for queries vs. documents for better retrieval accuracy).

    Returns:
        numpy array of shape (len(texts), 1024) — one 1024-dim vector per text.

    CONCEPT — Query vs. Document Prompts:
        BGE models are trained with asymmetric prompts. Documents are embedded
        as-is. Queries use a prefix like "Represent this sentence for searching
        relevant passages: " so the model knows to optimise for retrieval, not
        similarity. Always use is_query=True when embedding user queries.
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    settings = get_settings()
    model = _load_model()

    # BGE-M3 query prompt — improves retrieval accuracy vs. embedding queries raw
    prompt = "Represent this sentence for searching relevant passages: " if is_query else None

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding.batch_size,
        show_progress_bar=len(texts) > 100,  # show progress for large batches
        normalize_embeddings=True,  # L2 normalise → cosine sim = dot product
        prompt=prompt,
        convert_to_numpy=True,
    )

    log.debug(
        "texts_embedded",
        count=len(texts),
        shape=embeddings.shape,
        is_query=is_query,
    )

    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string and return as a Python list.

    Convenience wrapper around embed_texts() for the query pipeline.
    Returns list[float] because that's what Qdrant's client expects.
    """
    from typing import cast

    if not query.strip():
        raise ValueError("Query cannot be empty or whitespace")

    embedding = embed_texts([query], is_query=True)
    return cast(list[float], embedding[0].tolist())


def get_embedding_dimension() -> int:
    """Return the embedding dimension of the loaded model (e.g., 1024 for BGE-M3)."""
    model = _load_model()
    return model.get_sentence_embedding_dimension() or 1024


@torch.no_grad()
def warmup_model() -> None:
    """
    Run a dummy encode pass to warm up the model.

    Called at application startup so the first real request isn't slow.
    Without warmup, the first query takes 5-10s instead of ~100ms.
    """
    log.info("warming_up_embedding_model")
    embed_texts(["warmup"], is_query=False)
    log.info("embedding_model_warmed_up")
