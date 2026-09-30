"""
Cross-encoder reranker — BGE-Reranker-v2-m3 via sentence-transformers.

CONCEPT: The Two-Stage Retrieval Pattern

  Stage 1 — Recall (bi-encoder + BM25):
    Retrieve top 50 candidates quickly. The bi-encoder encodes query and
    document *independently*, so retrieval is O(log N) via HNSW index.
    Fast, but less accurate.

  Stage 2 — Precision (cross-encoder):
    Re-score the 50 candidates with a cross-encoder that sees BOTH the query
    AND the document simultaneously. This joint encoding catches subtle
    relevance signals that independent encoding misses.
    Slow (O(candidates)), but very accurate — run only on the small candidate set.

WHY does joint encoding matter?
  Consider: query = "Python decorator syntax"
  Bi-encoder: embeds each separately → vectors might be close for "Python"
              docs even if the doc is about Python snakes, not the language.
  Cross-encoder: sees "Python decorator syntax" + the candidate text together
              → immediately knows "Python" means the programming language here.

LATENCY IMPACT:
  Reranking 50 candidates on CPU takes ~200-500ms for BGE-Reranker-v2-m3.
  This is the biggest latency bottleneck. Acceptable for a portfolio system.
  In production you'd use a GPU, a smaller model, or serve via HuggingFace TEI.

INTERVIEW QUESTION: "Why can't you use a cross-encoder for the initial retrieval?"
  Because it requires O(N × query_time) comparisons — one forward pass per
  document. With 1M documents, that's millions of inference calls per query.
  The bi-encoder is fast because you pre-compute document embeddings offline.
"""

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranker model."""
    settings = get_settings()
    log.info(
        "loading_reranker_model",
        model=settings.reranker.model,
        device=settings.reranker.device,
    )
    reranker = CrossEncoder(
        settings.reranker.model,
        device=settings.reranker.device,
        trust_remote_code=False,
        max_length=512,  # truncate to model's effective context window
    )
    log.info("reranker_model_loaded", model=settings.reranker.model)
    from typing import cast

    return cast(CrossEncoder, reranker)


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank candidate documents using the cross-encoder.

    Args:
        query: The user's original query.
        candidates: List of candidate dicts (from hybrid_search). Each must
                    have a "text" field containing the chunk text.
        top_k: Number of results to return. Defaults to settings.reranker.top_k.

    Returns:
        Top_k candidates reranked by cross-encoder score, each with
        "reranker_score" added. Sorted best-first.
    """
    settings = get_settings()
    resolved_top_k = top_k or settings.reranker.top_k

    if not candidates:
        return []

    if not query.strip():
        raise ValueError("Query cannot be empty")

    reranker = _load_reranker()

    # Prepare (query, document) pairs for the cross-encoder
    pairs = [(query, candidate["text"]) for candidate in candidates]

    log.info(
        "reranking_started",
        query_preview=query[:80],
        candidate_count=len(candidates),
        top_k=resolved_top_k,
    )

    # predict() returns a raw logit score — higher is more relevant
    # activation_fct=None keeps raw logits (slightly faster than sigmoid)
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    scored = [
        {**candidate, "reranker_score": float(score)}
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda x: x["reranker_score"], reverse=True)

    result = scored[:resolved_top_k]

    log.info(
        "reranking_complete",
        top_score=result[0]["reranker_score"] if result else None,
        returned=len(result),
    )

    return result


def warmup_reranker() -> None:
    """Warm up the reranker model at startup."""
    log.info("warming_up_reranker")
    rerank("test query", [{"text": "test document", "chunk_id": "warmup"}], top_k=1)
    log.info("reranker_warmed_up")
