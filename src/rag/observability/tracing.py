"""
Langfuse observability integration — trace every step of the RAG pipeline.

CONCEPT: Observability in ML systems is harder than in traditional software
because failures are often silent (the system returns an answer, just a bad one).

Langfuse captures:
  - Traces: one trace per query, containing all pipeline steps as spans
  - Spans: individual steps (retrieve, rerank, generate) with timing and metadata
  - Scores: evaluation metrics attached to traces (faithfulness, relevance)
  - Input/output: what went in and what came out of each step

WHY does this matter?
  Without tracing, debugging "why did the system return a wrong answer?" means
  adding print statements and re-running. With Langfuse:
    - Click on a trace → see exactly which chunks were retrieved
    - See how the reranker reordered them
    - See the exact prompt sent to the LLM
    - See the LLM's raw response and token count
    - Compare two queries side by side

  This is how real teams debug RAG in production.

STRUCTURE: We use Langfuse's trace/span model:
  Trace: one top-level object per query (has a unique trace_id)
  ├── Span: "retrieve"  (dense search + BM25 + RRF)
  ├── Span: "rerank"    (cross-encoder scoring)
  └── Span: "generate"  (LLM call)
"""

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)

# Lazy import — Langfuse is optional. If keys are not configured, all tracing
# calls become no-ops, so the pipeline works even without Langfuse running.
_langfuse_client: Any = None


def _get_langfuse() -> Any:
    """Return the Langfuse client, initialising it on first call."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    settings = get_settings()
    if not settings.langfuse.is_enabled:
        return None

    try:
        from langfuse import Langfuse  # lazy import

        _langfuse_client = Langfuse(
            host=settings.langfuse.host,
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
        )
        log.info("langfuse_client_initialised", host=settings.langfuse.host)
    except ImportError:
        log.warning("langfuse_not_installed", hint="pip install langfuse")
    except Exception as e:
        log.warning("langfuse_init_failed", error=str(e))

    return _langfuse_client


class PipelineTrace:
    """
    Context object for a single query's trace.

    Wraps Langfuse trace/span creation and makes all calls no-ops when
    Langfuse is not configured. This means the pipeline code never needs
    to check "is tracing enabled?" — it just calls trace methods.

    Usage:
        trace = start_trace(query="what is Python?")
        with trace.span("retrieve") as span:
            results = do_retrieval()
            span.set_output(results)
        trace.finalise(answer="...", metadata={...})
    """

    def __init__(self, query: str, trace_id: str, name: str = "rag_query") -> None:
        self.query = query
        self.trace_id = trace_id
        self._lf = _get_langfuse()
        self._trace: Any = None

        if self._lf is not None:
            try:
                self._trace = self._lf.trace(
                    id=trace_id,
                    name=name,
                    input={"query": query},
                    metadata={"trace_id": trace_id},
                )
            except Exception as e:
                log.warning("langfuse_trace_create_failed", error=str(e))
                self._trace = None

    @contextmanager
    def span(
        self, name: str, input_data: dict[str, Any] | None = None
    ) -> Generator["SpanContext", None, None]:
        """Context manager that wraps a pipeline step as a Langfuse span."""
        span_ctx = SpanContext(name=name, trace=self._trace, input_data=input_data)
        try:
            yield span_ctx
        except Exception:
            span_ctx.mark_error()
            raise
        finally:
            span_ctx.end()

    @contextmanager
    def generation(
        self,
        name: str = "llm_generate",
        model: str = "",
        model_parameters: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> Generator["GenerationContext", None, None]:
        """
        Context manager that wraps an LLM generation step as a Langfuse generation.

        Captures model name, hyper-parameters (temperature, max_tokens),
        prompt messages, output text, and token usage counts.
        """
        gen_ctx = GenerationContext(
            name=name,
            trace=self._trace,
            model=model,
            model_parameters=model_parameters,
            input_data=input_data,
        )
        try:
            yield gen_ctx
        except Exception:
            gen_ctx.mark_error()
            raise
        finally:
            gen_ctx.end()

    def set_score(self, name: str, value: float, comment: str = "") -> None:
        """Attach an evaluation score to this trace (e.g., faithfulness=0.92)."""
        if self._trace is None:
            return
        try:
            self._lf.score(
                trace_id=self.trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            log.warning("langfuse_score_failed", error=str(e))

    def finalise(self, answer: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark the trace as complete with the final answer."""
        if self._trace is None:
            return
        try:
            self._trace.update(
                output={"answer": answer},
                metadata=metadata or {},
            )
        except Exception as e:
            log.warning("langfuse_trace_finalise_failed", error=str(e))


class SpanContext:
    """Wrapper around a Langfuse span for a single pipeline step."""

    def __init__(
        self,
        name: str,
        trace: Any,
        input_data: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._start = time.perf_counter()
        self._span: Any = None
        self._errored = False

        if trace is not None:
            try:
                self._span = trace.span(name=name, input=input_data or {})
            except Exception:
                pass

    def set_output(self, output: Any) -> None:
        """Record the output of this span."""
        if self._span is not None:
            try:
                self._span.update(output=output)
            except Exception:
                pass

    def mark_error(self) -> None:
        """Mark this span as failed."""
        self._errored = True
        if self._span is not None:
            try:
                self._span.update(level="ERROR")
            except Exception:
                pass

    def end(self) -> None:
        """Finalise the span with duration."""
        latency_ms = (time.perf_counter() - self._start) * 1000
        log.debug("span_complete", name=self._name, latency_ms=round(latency_ms, 2))
        if self._span is not None:
            try:
                self._span.end()
            except Exception:
                pass


class GenerationContext:
    """Wrapper around a Langfuse generation for LLM generation steps."""

    def __init__(
        self,
        name: str,
        trace: Any,
        model: str = "",
        model_parameters: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> None:
        self._name = name
        self._start = time.perf_counter()
        self._generation: Any = None
        self._errored = False

        if trace is not None:
            try:
                self._generation = trace.generation(
                    name=name,
                    model=model,
                    model_parameters=model_parameters or {},
                    input=input_data or {},
                )
            except Exception:
                pass

    def set_output(
        self,
        output: Any,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Record the output and token usage of this generation."""
        if self._generation is not None:
            try:
                update_kwargs: dict[str, Any] = {"output": output}
                if usage:
                    update_kwargs["usage"] = usage
                self._generation.update(**update_kwargs)
            except Exception:
                pass

    def mark_error(self) -> None:
        """Mark this generation as failed."""
        self._errored = True
        if self._generation is not None:
            try:
                self._generation.update(level="ERROR")
            except Exception:
                pass

    def end(self) -> None:
        """Finalise the generation with duration."""
        latency_ms = (time.perf_counter() - self._start) * 1000
        log.debug("generation_complete", name=self._name, latency_ms=round(latency_ms, 2))
        if self._generation is not None:
            try:
                self._generation.end()
            except Exception:
                pass


def start_trace(query: str, trace_id: str | None = None) -> PipelineTrace:
    """
    Start a new trace for a RAG query.

    Args:
        query: The user's query string.
        trace_id: Optional explicit trace ID (useful for correlating with API request IDs).

    Returns:
        A PipelineTrace object. If Langfuse is not configured, all methods are no-ops.
    """
    resolved_id = trace_id or str(uuid.uuid4())
    return PipelineTrace(query=query, trace_id=resolved_id, name="rag_query")


def start_ingestion_trace(source_file: str, doc_id: str | None = None) -> PipelineTrace:
    """
    Start a new trace for document ingestion.

    Args:
        source_file: Name or path of the document being ingested.
        doc_id: Unique document ID.
    """
    resolved_id = doc_id or str(uuid.uuid4())
    return PipelineTrace(
        query=f"Ingest: {source_file}",
        trace_id=resolved_id,
        name="document_ingestion",
    )


def flush_tracing() -> None:
    """Flush pending Langfuse events synchronously on shutdown."""
    global _langfuse_client
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            log.info("langfuse_flushed")
        except Exception as e:
            log.warning("langfuse_flush_failed", error=str(e))
