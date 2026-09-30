"""
Query API route — POST /api/v1/query

This is the core endpoint. It orchestrates the full RAG pipeline:
  1. Cache lookup (Redis)
  2. Query embedding (BGE-M3)
  3. Dense retrieval (Qdrant)
  4. Sparse retrieval (BM25)
  5. Hybrid fusion (RRF)
  6. Reranking (BGE-Reranker)
  7. LLM generation (Groq)
  8. Cache write + Langfuse trace
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.api.middleware import limiter
from rag.cache.redis_cache import QueryCache, get_redis_client
from rag.cache.semantic_cache import SemanticCache
from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.generation.llm import generate, generate_stream
from rag.generation.prompt import build_rag_messages
from rag.observability.tracing import start_trace
from rag.reranking.reranker import rerank
from rag.retrieval.bm25 import get_bm25_retriever
from rag.retrieval.embedder import embed_query
from rag.retrieval.hybrid import hybrid_search
from rag.retrieval.hyde import generate_hypothetical_document
from rag.retrieval.vector_store import get_qdrant_client, search_dense

log = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for a RAG query."""

    query: str = Field(
        min_length=3,
        max_length=1000,
        description="The user's question",
        examples=["What are Python decorators and how do I use them?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of context chunks to use (overrides default setting)",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check/write the query cache",
    )
    use_hyde: bool = Field(
        default=False,
        description="Whether to use Hypothetical Document Embeddings for query expansion",
    )


class SourceChunk(BaseModel):
    """A source chunk returned alongside the answer for transparency."""

    source_file: str
    chunk_index: int
    text_preview: str  # first 200 chars of the chunk
    reranker_score: float | None


class QueryResponse(BaseModel):
    """Response from a RAG query."""

    query: str
    answer: str
    sources: list[SourceChunk]
    trace_id: str
    cached: bool
    provider: str  # which LLM provider answered
    latency_ms: float


async def _get_cache() -> QueryCache:
    """FastAPI dependency — creates a QueryCache backed by Redis."""
    redis = get_redis_client()
    return QueryCache(redis)


def _get_semantic_cache() -> SemanticCache:
    """FastAPI dependency — creates a SemanticCache backed by Qdrant."""
    return SemanticCache()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description=(
        "Submit a question. The system retrieves relevant document chunks, "
        "reranks them, and generates a grounded answer."
    ),
)
@limiter.limit(get_settings().rate_limit.query_limit)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    cache: QueryCache = Depends(_get_cache),
    semantic_cache: SemanticCache = Depends(_get_semantic_cache),
) -> QueryResponse:
    """
    Full RAG pipeline:
    Cache (L1 Redis + L2 Semantic) → Embed (optional HyDE) → Dense Retrieve →
    BM25 → RRF Fuse → Rerank → Generate → Cache Write
    """
    start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    settings = get_settings()

    log.info("query_received", query_preview=body.query[:80], trace_id=trace_id)

    # ── 1. Tier 1: Exact Cache lookup (Redis) ─────────────────────────────────
    if body.use_cache:
        cached_result: dict[str, Any] | None = await cache.get(body.query)
        if cached_result is not None:
            return QueryResponse(
                query=body.query,
                answer=cached_result["answer"],
                sources=[SourceChunk(**s) for s in cached_result["sources"]],
                trace_id=trace_id,
                cached=True,
                provider=cached_result.get("provider", "cache"),
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

    # ── Langfuse trace ────────────────────────────────────────────────────────
    trace = start_trace(query=body.query, trace_id=trace_id)

    # ── 2. HyDE & Embed query ─────────────────────────────────────────────────
    dense_query_text = body.query
    if body.use_hyde or settings.enable_hyde:
        with trace.span("hyde_expansion"):
            hyde_doc = generate_hypothetical_document(body.query, metadata={"trace_id": trace_id})
            dense_query_text = f"{body.query}\n{hyde_doc}"

    with trace.span("embed_query"):
        try:
            query_vector = embed_query(dense_query_text)
        except Exception as e:
            log.error("embed_query_failed", error=str(e), trace_id=trace_id)
            raise HTTPException(status_code=500, detail="Embedding failed") from e

    # ── Tier 2: Semantic Cache lookup (Qdrant) ────────────────────────────────
    if body.use_cache and settings.semantic_cache_enabled:
        with trace.span("semantic_cache_lookup"):
            semantic_result = semantic_cache.get(query_vector)
            if semantic_result is not None:
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                trace.finalise(
                    answer=semantic_result["answer"],
                    metadata={"cached": True, "cache_type": "semantic"},
                )
                return QueryResponse(
                    query=body.query,
                    answer=semantic_result["answer"],
                    sources=[SourceChunk(**s) for s in semantic_result["sources"]],
                    trace_id=trace_id,
                    cached=True,
                    provider=f"semantic_cache (sim: {semantic_result['similarity_score']})",
                    latency_ms=latency_ms,
                )

    # ── 3. Dense retrieval ────────────────────────────────────────────────────
    with trace.span("dense_retrieve") as span:
        try:
            qdrant_client = get_qdrant_client()
            dense_results = search_dense(
                client=qdrant_client,
                query_vector=query_vector,
                top_k=settings.retrieval_top_k,
            )
            span.set_output({"count": len(dense_results)})
        except Exception as e:
            log.error("dense_retrieval_failed", error=str(e), trace_id=trace_id)
            dense_results = []

    # ── 4. BM25 retrieval ─────────────────────────────────────────────────────
    with trace.span("bm25_retrieve") as span:
        bm25 = get_bm25_retriever()
        if bm25.document_count == 0:
            log.warning("bm25_index_empty", trace_id=trace_id)
            bm25_results: list[dict[str, Any]] = []
        else:
            bm25_results = bm25.search(body.query, top_k=settings.retrieval_top_k)
        span.set_output({"count": len(bm25_results)})

    # ── 5. Hybrid fusion (RRF) ────────────────────────────────────────────────
    with trace.span("hybrid_fusion") as span:
        if dense_results and bm25_results:
            fused = hybrid_search(dense_results, bm25_results, top_k=settings.retrieval_top_k)
        elif dense_results:
            fused = dense_results[: settings.retrieval_top_k]
        elif bm25_results:
            fused = bm25_results[: settings.retrieval_top_k]
        else:
            log.warning("no_results_from_retrieval", trace_id=trace_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found. Please ingest documents first.",
            )
        span.set_output({"fused_count": len(fused)})

    # ── 6. Reranking ──────────────────────────────────────────────────────────
    with trace.span("rerank") as span:
        top_k = body.top_k or settings.reranker.top_k
        try:
            reranked = rerank(query=body.query, candidates=fused, top_k=top_k)
        except Exception as e:
            log.error("reranking_failed", error=str(e), trace_id=trace_id)
            reranked = fused[:top_k]
        span.set_output({"top_chunk_preview": reranked[0]["text"][:100] if reranked else ""})

    if not reranked:
        raise HTTPException(status_code=404, detail="No relevant context found after reranking.")

    # ── 7. LLM generation ────────────────────────────────────────────────────
    messages = build_rag_messages(question=body.query, chunks=reranked)
    with trace.generation(
        name="llm_generate",
        model=settings.llm.model,
        model_parameters={
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
        },
        input_data=messages,
    ) as gen_span:
        try:
            gen_result = generate(messages=messages, metadata={"trace_id": trace_id})
        except Exception as e:
            log.error("generation_failed", error=str(e), trace_id=trace_id)
            raise HTTPException(status_code=502, detail="LLM generation failed") from e
        answer = gen_result["answer"]
        provider = gen_result["provider"]
        usage = gen_result.get("usage")
        gen_span.set_output({"answer_preview": answer[:200]}, usage=usage)

    # ── 8. Build sources for response ─────────────────────────────────────────
    sources = [
        SourceChunk(
            source_file=chunk.get("source_file", "unknown"),
            chunk_index=chunk.get("chunk_index", 0),
            text_preview=chunk.get("text", "")[:200],
            reranker_score=chunk.get("reranker_score"),
        )
        for chunk in reranked
    ]

    # ── 9. Cache write (L1 Redis + L2 Semantic) ───────────────────────────────
    if body.use_cache:
        cache_data = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "provider": provider,
        }
        await cache.set(body.query, cache_data)
        if settings.semantic_cache_enabled:
            semantic_cache.set(body.query, query_vector, cache_data)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    trace.finalise(answer=answer, metadata={"latency_ms": latency_ms, "provider": provider})

    log.info(
        "query_complete",
        trace_id=trace_id,
        latency_ms=latency_ms,
        provider=provider,
        chunk_count=len(reranked),
    )

    return QueryResponse(
        query=body.query,
        answer=answer,
        sources=sources,
        trace_id=trace_id,
        cached=False,
        provider=provider,
        latency_ms=latency_ms,
    )


@router.post(
    "/query/stream",
    summary="Query the RAG system with streaming tokens (SSE)",
    description="Submit a question and receive real-time streamed tokens via SSE.",
)
@limiter.limit(get_settings().rate_limit.query_limit)
async def query_stream_endpoint(
    request: Request,
    body: QueryRequest,
    cache: QueryCache = Depends(_get_cache),
    semantic_cache: SemanticCache = Depends(_get_semantic_cache),
) -> StreamingResponse:
    """
    Streaming RAG pipeline:
    Returns Server-Sent Events (SSE) emitting token deltas, followed by source citations.
    """
    start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    settings = get_settings()

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Check L1 Cache
        if body.use_cache:
            cached_result = await cache.get(body.query)
            if cached_result:
                # Stream cached answer in chunks for UX consistency
                token_data = {"type": "token", "content": cached_result["answer"]}
                yield f"data: {json.dumps(token_data)}\n\n"
                sources_data = {"type": "sources", "sources": cached_result["sources"]}
                yield f"data: {json.dumps(sources_data)}\n\n"
                done_data = {
                    "type": "done",
                    "cached": True,
                    "provider": "cache",
                    "trace_id": trace_id,
                }
                yield f"data: {json.dumps(done_data)}\n\n"
                return

        # 2. Embedding + HyDE
        dense_query_text = body.query
        if body.use_hyde or settings.enable_hyde:
            hyde_doc = generate_hypothetical_document(body.query, metadata={"trace_id": trace_id})
            dense_query_text = f"{body.query}\n{hyde_doc}"

        query_vector = embed_query(dense_query_text)

        # Check L2 Semantic Cache
        if body.use_cache and settings.semantic_cache_enabled:
            sem_hit = semantic_cache.get(query_vector)
            if sem_hit:
                sem_token = {"type": "token", "content": sem_hit["answer"]}
                yield f"data: {json.dumps(sem_token)}\n\n"
                sem_sources = {"type": "sources", "sources": sem_hit["sources"]}
                yield f"data: {json.dumps(sem_sources)}\n\n"
                sem_done = {
                    "type": "done",
                    "cached": True,
                    "provider": "semantic_cache",
                    "trace_id": trace_id,
                }
                yield f"data: {json.dumps(sem_done)}\n\n"
                return

        # 3. Retrieval
        try:
            qclient = get_qdrant_client()
            dense_results = search_dense(qclient, query_vector, top_k=settings.retrieval_top_k)
        except Exception:
            dense_results = []

        bm25 = get_bm25_retriever()
        bm25_results = (
            bm25.search(body.query, top_k=settings.retrieval_top_k)
            if bm25.document_count > 0
            else []
        )

        if dense_results and bm25_results:
            fused = hybrid_search(dense_results, bm25_results, top_k=settings.retrieval_top_k)
        elif dense_results:
            fused = dense_results[: settings.retrieval_top_k]
        elif bm25_results:
            fused = bm25_results[: settings.retrieval_top_k]
        else:
            err_data = {"type": "error", "detail": "No relevant documents found."}
            yield f"data: {json.dumps(err_data)}\n\n"
            return

        # 4. Rerank
        top_k = body.top_k or settings.reranker.top_k
        reranked = rerank(query=body.query, candidates=fused, top_k=top_k)

        # 5. Stream LLM Generation
        messages = build_rag_messages(question=body.query, chunks=reranked)
        answer_parts: list[str] = []

        for token in generate_stream(messages=messages, metadata={"trace_id": trace_id}):
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        full_answer = "".join(answer_parts)
        sources_payload = [
            {
                "source_file": chunk.get("source_file", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "text_preview": chunk.get("text", "")[:200],
                "reranker_score": chunk.get("reranker_score"),
            }
            for chunk in reranked
        ]

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

        # 6. Cache write
        if body.use_cache:
            cache_payload = {
                "answer": full_answer,
                "sources": sources_payload,
                "provider": settings.llm.provider,
            }
            await cache.set(body.query, cache_payload)
            if settings.semantic_cache_enabled:
                semantic_cache.set(body.query, query_vector, cache_payload)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        done_payload = {
            "type": "done",
            "cached": False,
            "provider": settings.llm.provider,
            "latency_ms": latency_ms,
            "trace_id": trace_id,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
