"""
BM25 retrieval — keyword-based search using the rank_bm25 library.

CONCEPT: BM25 (Best Match 25) is a probabilistic ranking function.
  For each term in the query, it scores documents based on:
    - Term Frequency (TF): How often does this term appear in the document?
      Penalises very long documents (hence "normalised TF").
    - Inverse Document Frequency (IDF): How rare is this term across all docs?
      "the" appears everywhere → low IDF → low weight
      "HNSW" appears rarely → high IDF → high weight

  BM25 scores documents purely on lexical (surface-form) overlap.

WHY do we need BM25 alongside dense embeddings?
  Dense retrieval excels at semantic understanding but struggles with:
    - Exact keywords: product codes, names, technical terms (e.g., "RFC 9110", "GPT-4o")
    - Rare vocabulary: new terms the embedding model has never seen
    - Queries with very specific, uncommon words

  Hybrid search = Dense (semantic) + BM25 (lexical) → best of both worlds.
  Studies consistently show hybrid outperforms either alone by 5–15% on recall.

IMPLEMENTATION NOTE: This is an in-memory BM25 index. For Phase 2, we'll
migrate to Qdrant's native sparse vector support, which persists the index
and scales to millions of documents.
"""

import math
import re
from typing import Any

from rank_bm25 import BM25Okapi

from rag.core.logging import get_logger

log = get_logger(__name__)


class BM25Retriever:
    """
    BM25 retriever backed by an in-memory index.

    Usage:
        retriever = BM25Retriever()
        retriever.index(corpus)          # call once after ingestion
        results = retriever.search(query, top_k=50)
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []  # parallel to index documents

    def index(self, documents: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of document dicts.

        Args:
            documents: Each dict must have a "text" key. All other keys are
                       returned unchanged in search results.
        """
        if not documents:
            log.warning("bm25_index_called_with_empty_corpus")
            return

        self._corpus = documents
        tokenised = [_tokenise(doc["text"]) for doc in documents]
        self._index = BM25Okapi(tokenised)

        log.info("bm25_index_built", document_count=len(documents))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Retrieve the top_k most relevant documents for a query.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with a "bm25_score" key added to each document dict.
            Sorted by score descending.
        """
        if self._index is None:
            log.warning("bm25_search_called_before_indexing")
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Get indices of top_k highest-scoring documents
        # argsort gives ascending order, so we reverse and take the top
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                # Skip documents with zero relevance — they share no terms with the query
                continue
            results.append(
                {
                    **self._corpus[idx],
                    "bm25_score": score,
                }
            )

        log.debug("bm25_search_complete", query_tokens=query_tokens, results=len(results))
        return results

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Append documents and rebuild the index.

        NOTE: BM25Okapi doesn't support incremental updates — we rebuild.
        For large corpora, this would need a smarter approach (or native Qdrant
        sparse vectors). Rebuilding on every ingestion is acceptable for
        a portfolio system where the corpus is small and ingestion is infrequent.
        """
        all_docs = self._corpus + documents
        self.index(all_docs)

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._corpus)


# ── Module-level singleton ────────────────────────────────────────────────────
# One shared BM25 index for the application lifetime.
# In production with multiple workers, this would be a shared external store.
_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """Return the global BM25Retriever instance, creating it if necessary."""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever


# ── Tokenisation ──────────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    """
    Tokenise text for BM25 indexing.

    Simple but effective:
      - Lowercase (case-insensitive matching)
      - Extract alphanumeric tokens (strips punctuation)
      - Remove very short tokens (< 2 chars) to reduce noise

    PRODUCTION NOTE: For domain-specific systems, stemming (reducing "running"
    to "run") or lemmatisation can improve recall. The trade-off is complexity.
    """
    tokens = re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", text.lower())
    return [t for t in tokens if len(t) >= 2]


def normalise_bm25_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise BM25 scores to [0, 1] range using min-max normalisation.

    WHY normalise? BM25 scores are unbounded (depend on corpus size and term
    frequency). Reciprocal Rank Fusion (our fusion method) doesn't use the raw
    score, so normalisation isn't strictly needed there. But it's useful for
    logging and debugging to compare BM25 and dense scores on the same scale.
    """
    if not results:
        return results
    scores = [r["bm25_score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    if math.isclose(min_s, max_s):
        return [{**r, "bm25_score_normalised": 1.0} for r in results]
    return [
        {**r, "bm25_score_normalised": (r["bm25_score"] - min_s) / (max_s - min_s)} for r in results
    ]
