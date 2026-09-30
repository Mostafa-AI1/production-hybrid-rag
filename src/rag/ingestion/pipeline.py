"""
Ingestion pipeline orchestrator — parse → chunk → embed → index.

This module ties together the parser, chunker, embedder, vector store,
and BM25 index into a single callable pipeline.

DESIGN DECISION: Synchronous pipeline
  Ingestion is a batch operation, not a latency-sensitive path. We keep it
  synchronous for simplicity. In a production system with many documents,
  you'd offload this to a background task queue (Celery, ARQ, or FastAPI
  BackgroundTasks for small workloads) so the API response returns immediately
  and the user polls for status.

  For this portfolio system, we call the pipeline from a FastAPI endpoint
  using BackgroundTasks — the endpoint returns 202 Accepted immediately and
  the processing happens in the background.
"""

import uuid
from pathlib import Path

import numpy as np

from rag.core.logging import get_logger
from rag.ingestion.chunker import TextChunk, chunk_document
from rag.ingestion.parser import ParsedDocument, parse_file, parse_text_content
from rag.observability.tracing import start_ingestion_trace
from rag.retrieval.bm25 import get_bm25_retriever
from rag.retrieval.embedder import embed_texts
from rag.retrieval.vector_store import get_qdrant_client, upsert_chunks

log = get_logger(__name__)


def ingest_file(file_path: Path, doc_id: str | None = None) -> dict[str, object]:
    """
    Ingest a single file into the RAG index.

    Args:
        file_path: Path to the file to ingest (PDF, MD, or TXT).
        doc_id: Optional document ID. Auto-generated if not provided.

    Returns:
        Dict with ingestion summary: doc_id, chunk_count, source_file.
    """
    resolved_doc_id = doc_id or str(uuid.uuid4())

    log.info("ingestion_started", path=str(file_path), doc_id=resolved_doc_id)

    # Step 1: Parse
    document = parse_file(file_path)

    # Steps 2–4: Chunk → Embed → Index
    return _run_pipeline(document, resolved_doc_id)


def ingest_text(text: str, source_name: str, doc_id: str | None = None) -> dict[str, object]:
    """
    Ingest raw text content directly (e.g., from API upload).

    Useful for testing and for clients that send pre-extracted text.
    """
    resolved_doc_id = doc_id or str(uuid.uuid4())
    document = parse_text_content(text, source_name)
    return _run_pipeline(document, resolved_doc_id)


def _run_pipeline(document: ParsedDocument, doc_id: str) -> dict[str, object]:
    """
    Internal pipeline: parsed document → Qdrant + BM25.

    Steps:
      1. Chunk the document text
      2. Embed all chunks in one batched call (GPU/CPU)
      3. Upsert vectors + payloads into Qdrant
      4. Add chunks to the in-memory BM25 index

    WHY batch embedding?
      Calling the embedding model per-chunk is ~50x slower than batching.
      The model's forward pass has a fixed overhead; batching amortises it.
    """
    trace = start_ingestion_trace(source_file=document.source_file, doc_id=doc_id)

    # ── 1. Chunk ──────────────────────────────────────────────────────────────
    with trace.span("chunk_document") as span:
        chunks: list[TextChunk] = chunk_document(document)
        span.set_output({"chunk_count": len(chunks)})

    if not chunks:
        log.warning("ingestion_produced_no_chunks", doc_id=doc_id)
        trace.finalise(
            answer="No chunks produced",
            metadata={"doc_id": doc_id, "chunk_count": 0, "source_file": document.source_file},
        )
        return {"doc_id": doc_id, "chunk_count": 0, "source_file": document.source_file}

    # ── 2. Embed (batched) ────────────────────────────────────────────────────
    with trace.span("embed_chunks") as span:
        texts = [chunk.text for chunk in chunks]
        log.info("embedding_chunks", doc_id=doc_id, count=len(texts))
        embeddings: np.ndarray = embed_texts(texts, is_query=False)
        span.set_output({"embedded_count": len(embeddings)})

    # ── 3. Upsert to Qdrant ───────────────────────────────────────────────────
    with trace.span("upsert_qdrant") as span:
        client = get_qdrant_client()
        upsert_chunks(
            client=client,
            chunks=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            doc_id=doc_id,
        )
        span.set_output({"upserted_count": len(chunks)})

    # ── 4. Update BM25 index ──────────────────────────────────────────────────
    with trace.span("index_bm25") as span:
        bm25 = get_bm25_retriever()
        bm25_docs = [
            {
                "text": chunk.text,
                "chunk_id": f"{doc_id}_{chunk.chunk_index}",  # stable BM25 identifier
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "char_count": chunk.char_count,
            }
            for chunk in chunks
        ]
        bm25.add_documents(bm25_docs)
        span.set_output({"bm25_doc_count": len(bm25_docs)})

    trace.finalise(
        answer=f"Successfully ingested {len(chunks)} chunks from {document.source_file}",
        metadata={
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "source_file": document.source_file,
        },
    )

    log.info(
        "ingestion_complete",
        doc_id=doc_id,
        chunk_count=len(chunks),
        source_file=document.source_file,
    )

    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "source_file": document.source_file,
        "source_type": document.source_type,
    }
