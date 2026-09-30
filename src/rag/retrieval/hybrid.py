"""
Hybrid retrieval — fusing dense vector search and BM25 with Reciprocal Rank Fusion.

CONCEPT: Reciprocal Rank Fusion (RRF)
  RRF is a score-free rank fusion algorithm. Instead of combining raw scores
  (which are on incomparable scales — cosine similarity vs. BM25 score),
  it uses the *rank* of each document in each result list.

  Formula: RRF(d) = Σ 1 / (k + rank_in_list(d))
    where k=60 is a smoothing constant (recommended by the original RRF paper).

  WHY RRF over weighted sum?
    - No normalisation required — ranks are always comparable
    - Robust: a document ranked #1 by one method and #5 by another gets a
      high fused score, even if the raw scores differ by orders of magnitude
    - Simple to implement, hard to get wrong
    - Shown to outperform weighted score combination in most benchmarks

  EXAMPLE:
    Dense results:  [A, B, C, D, ...]    (A ranked #1)
    BM25 results:   [C, A, E, B, ...]    (C ranked #1, A ranked #2)
    RRF scores:     A=1/(60+1)+1/(60+2), C=1/(60+1)+1/(60+3), ...
    Fused ranking:  A first (strong across both), C second (BM25 winner)

INTERVIEW QUESTION: "What is Reciprocal Rank Fusion and why is it preferred
over normalised score combination for hybrid retrieval?"
"""

from typing import Any

from rag.core.logging import get_logger

log = get_logger(__name__)

# RRF smoothing constant — from the original paper (Cormack et al., 2009)
# Higher k = more weight to lower-ranked documents. k=60 is the standard default.
_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    top_k: int,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of ranked result lists. Each list contains dicts
                      that must have an `id_key` field for deduplication.
        top_k: Number of results to return after fusion.
        id_key: The field name to use as a unique document identifier.

    Returns:
        Fused and deduplicated list of results, sorted by RRF score descending.
        Each result dict has a "rrf_score" field added.
    """
    rrf_scores: dict[str, float] = {}
    docs_by_id: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = str(doc.get(id_key, ""))
            if not doc_id:
                log.warning("rrf_document_missing_id_key", id_key=id_key)
                continue

            # RRF formula: 1 / (k + rank)   (rank is 0-indexed, so rank+1 for 1-indexed)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

            # Keep the most complete version of the document (first seen)
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = doc

    # Sort by RRF score descending, take top_k
    sorted_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)[:top_k]

    fused_results = [
        {**docs_by_id[doc_id], "rrf_score": rrf_scores[doc_id]} for doc_id in sorted_ids
    ]

    log.debug(
        "rrf_fusion_complete",
        input_lists=len(result_lists),
        unique_candidates=len(docs_by_id),
        returned=len(fused_results),
    )

    return fused_results


def hybrid_search(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Combine dense vector search and BM25 results using RRF.

    This is the main entry point for hybrid retrieval.

    Args:
        dense_results: Ranked results from Qdrant dense search.
        bm25_results: Ranked results from BM25 search.
        top_k: Number of fused results to return (before reranking).

    Returns:
        Fused result list with "rrf_score" added to each document.
    """
    log.info(
        "hybrid_fusion_started",
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        top_k=top_k,
    )

    # Normalise BM25 chunk_id key to match dense results' chunk_id format
    # BM25 results come from our in-memory corpus — ensure chunk_id is present
    normalised_bm25 = []
    for doc in bm25_results:
        if "chunk_id" not in doc and "id" in doc:
            doc = {**doc, "chunk_id": doc["id"]}
        normalised_bm25.append(doc)

    fused = reciprocal_rank_fusion(
        result_lists=[dense_results, normalised_bm25],
        top_k=top_k,
        id_key="chunk_id",
    )

    log.info("hybrid_fusion_complete", fused_count=len(fused))
    return fused
