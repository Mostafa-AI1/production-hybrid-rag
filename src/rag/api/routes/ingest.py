"""
Ingestion API routes — POST /api/v1/ingest/file and POST /api/v1/ingest/text.

DESIGN: The endpoints return 202 Accepted immediately and process in the background
using FastAPI's BackgroundTasks. This is the right pattern for operations that take
several seconds (embedding a large document).

The client flow:
  POST /ingest/file → 202 { "job_id": "...", "status": "processing" }
  (client polls or waits)
  GET  /health → (or a dedicated status endpoint in Phase 2)

PYDANTIC MODELS: Every request/response is a Pydantic model. This gives us:
  - Automatic request validation (FastAPI returns 422 if input is invalid)
  - Auto-generated OpenAPI schema (visible at /docs)
  - Type safety in the route handler
"""

import tempfile
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from rag.api.middleware import limiter
from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.ingestion.pipeline import ingest_file, ingest_text

log = get_logger(__name__)
router = APIRouter()

# Simple in-memory job status store — Phase 2 will use Redis for persistence
_job_status: dict[str, dict[str, object]] = {}


class TextIngestionRequest(BaseModel):
    """Request body for raw text ingestion."""

    text: str = Field(min_length=10, description="The text content to ingest")
    source_name: str = Field(
        default="api_upload",
        description="Human-readable source name for this content",
    )
    doc_id: str | None = Field(
        default=None,
        description="Optional custom document ID (UUID). Auto-generated if omitted.",
    )


class IngestionResponse(BaseModel):
    """Response returned when an ingestion job is accepted."""

    job_id: str
    status: str
    message: str


class IngestionResult(BaseModel):
    """Result stored after ingestion completes."""

    job_id: str
    doc_id: str
    chunk_count: int
    source_file: str
    status: str


@router.post(
    "/ingest/file",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a document file",
    description="Upload a PDF, Markdown, or plain text file to be chunked, embedded, and indexed.",
)
@limiter.limit(get_settings().rate_limit.ingest_limit)
async def ingest_file_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (PDF, .md, .txt)"),
) -> IngestionResponse:
    """
    Ingest a file upload into the RAG index.

    The file is saved to a temp directory, then processed in the background.
    Returns 202 immediately — processing continues asynchronously.
    """
    allowed_suffixes = {".pdf", ".md", ".txt"}
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed_suffixes)}",
        )

    job_id = str(uuid.uuid4())
    _job_status[job_id] = {"status": "processing", "job_id": job_id}

    # Save uploaded file to temp location so background task can read it
    # We read the bytes now because the UploadFile stream may be closed
    # by the time the background task runs.
    content = await file.read()

    async def _background_ingest(job_id: str, content: bytes, filename: str) -> None:
        """Background task that does the actual ingestion work."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            result = ingest_file(tmp_path, doc_id=job_id)
            _job_status[job_id] = {"status": "complete", **result}
            log.info("ingestion_job_complete", job_id=job_id)
        except Exception as e:
            log.error("ingestion_job_failed", job_id=job_id, error=str(e))
            _job_status[job_id] = {"status": "failed", "error": str(e)}
        finally:
            tmp_path.unlink(missing_ok=True)

    background_tasks.add_task(_background_ingest, job_id, content, filename)

    log.info("ingestion_job_accepted", job_id=job_id, filename=filename)
    return IngestionResponse(
        job_id=job_id,
        status="processing",
        message=f"File '{filename}' accepted. Ingestion running in background.",
    )


@router.post(
    "/ingest/text",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest raw text content",
    description="Submit raw text content to be chunked, embedded, and indexed.",
)
@limiter.limit(get_settings().rate_limit.ingest_limit)
async def ingest_text_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    body: TextIngestionRequest,
) -> IngestionResponse:
    """Ingest raw text string into the RAG index."""
    job_id = body.doc_id or str(uuid.uuid4())
    _job_status[job_id] = {"status": "processing", "job_id": job_id}

    async def _background_text_ingest(job_id: str) -> None:
        try:
            result = ingest_text(body.text, body.source_name, doc_id=job_id)
            _job_status[job_id] = {"status": "complete", **result}
        except Exception as e:
            log.error("text_ingestion_failed", job_id=job_id, error=str(e))
            _job_status[job_id] = {"status": "failed", "error": str(e)}

    background_tasks.add_task(_background_text_ingest, job_id)

    return IngestionResponse(
        job_id=job_id,
        status="processing",
        message=f"Text from '{body.source_name}' accepted. Processing in background.",
    )


@router.get(
    "/ingest/status/{job_id}",
    summary="Check ingestion job status",
)
async def ingestion_status(job_id: str) -> dict[str, object]:
    """Poll the status of an ingestion job."""
    if job_id not in _job_status:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_status[job_id]
