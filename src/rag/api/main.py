"""
FastAPI application entrypoint.

CONCEPT: FastAPI is an async Python web framework built on Starlette and Pydantic.
  - Async (ASGI): handles many concurrent requests without blocking
  - Auto-docs: OpenAPI spec + Swagger UI generated from type hints
  - Dependency injection: clean way to share clients (Qdrant, Redis) across routes

APPLICATION LIFECYCLE:
  FastAPI has lifespan events — code that runs once at startup and once at
  shutdown. We use startup to:
    1. Configure logging
    2. Ensure Qdrant collection exists
    3. Warm up ML models (prevents slow first request)

  This is better than initialising models on the first request because:
    - The first request doesn't time out due to model loading
    - Health checks return "ready" only after models are loaded
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.middleware import add_middleware
from rag.api.routes import ingest, query
from rag.core.config import get_settings
from rag.core.logging import configure_logging, get_logger
from rag.retrieval.embedder import get_embedding_dimension, warmup_model
from rag.retrieval.vector_store import ensure_collection_exists, get_qdrant_client

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan — startup and shutdown logic.

    Using @asynccontextmanager means startup code runs before `yield` and
    shutdown code runs after `yield`. This replaced the deprecated
    @app.on_event("startup") pattern in FastAPI 0.95+.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    configure_logging()
    settings = get_settings()
    log.info("application_starting", env=settings.app_env)

    start = time.perf_counter()

    # Ensure Qdrant collection is ready
    try:
        client = get_qdrant_client()
        dim = get_embedding_dimension()
        ensure_collection_exists(client, vector_dim=dim)
    except Exception as e:
        log.error("qdrant_startup_failed", error=str(e))
        # Don't crash — allow health endpoint to report degraded status

    # Warm up ML models (download from HuggingFace Hub on first run)
    try:
        warmup_model()
        # Reranker warmup is slower — only if reranker module is importable
        from rag.reranking.reranker import warmup_reranker

        warmup_reranker()
    except Exception as e:
        log.error("model_warmup_failed", error=str(e))

    elapsed = time.perf_counter() - start
    log.info("application_ready", startup_seconds=round(elapsed, 2))

    yield  # ← application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    log.info("application_shutting_down")
    try:
        from rag.observability.tracing import flush_tracing

        flush_tracing()
    except Exception as e:
        log.warning("shutdown_tracing_flush_failed", error=str(e))


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    WHY a factory function instead of module-level `app = FastAPI()`?
      - Easier to test: create a fresh app instance per test
      - Easier to configure: pass settings as arguments
      - Follows the same pattern used by Flask and other frameworks
    """
    from pathlib import Path

    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from slowapi.errors import RateLimitExceeded

    from rag.api.middleware import limiter, rate_limit_exceeded_handler

    settings = get_settings()

    app = FastAPI(
        title="Production RAG System",
        description=(
            "A production-grade Retrieval-Augmented Generation API with "
            "hybrid search, cross-encoder reranking, and full observability."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,  # hide docs in prod
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Attach SlowAPI rate limiter state and 429 exception handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # CORS — allow the frontend (if any) to call the API from a browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3001"],  # restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (request logging, API key auth)
    add_middleware(app)

    # Mount route modules
    app.include_router(ingest.router, prefix="/api/v1", tags=["ingestion"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    # Mount interactive RAG Studio UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root_ui() -> FileResponse:
            """Serve the Production RAG Studio interface."""
            return FileResponse(static_dir / "index.html")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """
        Liveness probe.

        Container orchestrators (Docker, Kubernetes) call this to know if
        the process is alive. Return 200 = healthy.
        """
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/health/ready", tags=["system"])
    async def health_ready() -> JSONResponse:
        """
        Readiness probe.

        Checks connectivity to Qdrant vector DB and Redis cache.
        Returns 200 if dependencies are operational, 503 Service Unavailable if degraded.
        """
        checks: dict[str, str] = {}
        is_ready = True

        # Check Qdrant
        try:
            client = get_qdrant_client()
            client.get_collections()
            checks["qdrant"] = "ok"
        except Exception as e:
            checks["qdrant"] = f"unhealthy: {e}"
            is_ready = False

        # Check Redis
        try:
            from rag.cache.redis_cache import get_redis_client

            redis_client = get_redis_client()
            pong = await redis_client.ping()
            await redis_client.close()
            if pong:
                checks["redis"] = "ok"
            else:
                checks["redis"] = "unresponsive"
                is_ready = False
        except Exception as e:
            checks["redis"] = f"unhealthy: {e}"
            # Redis is non-fatal if caching is optional, but marked in checks
            is_ready = False

        status_code = 200 if is_ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if is_ready else "degraded",
                "version": "0.1.0",
                "checks": checks,
            },
        )

    return app


# Module-level app instance — used by uvicorn: `uvicorn rag.api.main:app`
app = create_app()
