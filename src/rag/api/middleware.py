"""
API middleware — request logging, API key authentication, and rate limiting.

CONCEPT: Middleware runs on every request/response cycle, before the route
handler sees the request. Starlette (FastAPI's foundation) uses a middleware
stack — each middleware wraps the next.

Request flow:
  client → [LoggingMiddleware] → [API key check] → route handler
                                                          ↓
  client ← [LoggingMiddleware] ← [API key check] ← route handler

WHY API key auth here (middleware) vs. FastAPI dependency?
  FastAPI dependencies (via Depends()) are per-route. Middleware applies to
  ALL routes. For an API where everything requires auth, middleware is cleaner
  and ensures we never accidentally forget to add the dependency to a new route.

SECURITY NOTE: API key auth in a header (X-API-Key) is sufficient for a
portfolio project. In production you'd want:
  - JWT tokens with expiry (OAuth 2.0 / OIDC)
  - Rate limiting per user identity (not just per IP)
  - HTTPS enforced (TLS termination at load balancer)
"""

import json
import secrets
import time
import uuid

from fastapi import FastAPI, Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)

# Public endpoints that don't require an API key
_PUBLIC_PATHS = {"/health", "/health/ready", "/docs", "/redoc", "/openapi.json", "/"}


def _get_storage_uri() -> str:
    """Determine the storage URI for rate limits."""
    settings = get_settings()
    return settings.rate_limit.storage_url or "memory://"


def create_limiter() -> Limiter:
    """
    Create a SlowAPI limiter instance.

    Uses client IP as rate limiting key by default.
    Storage backend is configurable (memory:// for local/test, redis:// for distributed prod).
    """
    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit.default_limit] if settings.rate_limit.enabled else [],
        storage_uri=_get_storage_uri(),
        enabled=settings.rate_limit.enabled,
    )


# Module-level limiter instance used across route decorators
limiter = create_limiter()


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom 429 handler returning structured JSON and Retry-After header.

    CONCEPT: RFC 6585 specifies that 429 Too Many Requests responses SHOULD
    include a Retry-After header indicating how long the client must wait.
    """
    retry_after = getattr(exc, "retry_after", 60)
    log.warning(
        "rate_limit_exceeded",
        path=request.url.path,
        ip=request.client.host if request.client else "unknown",
        retry_after=retry_after,
    )
    return Response(
        content=json.dumps(
            {
                "detail": f"Rate limit exceeded: {exc.detail}",
                "retry_after": retry_after,
            }
        ),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": str(retry_after)},
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request with method, path, status code, and latency.

    This gives us an access log in structured JSON format, which a log
    aggregator (Grafana Loki, Datadog) can query and alert on.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach request_id to structlog's context — it appears in all log lines
        # emitted during this request, making it easy to trace a single request
        # through many log entries.
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        log.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        # Attach request ID to response headers so clients can report it in bug reports
        response.headers["X-Request-Id"] = request_id
        return response


def _extract_api_key(request: Request) -> str | None:
    """Extract API key from X-API-Key header or Authorization: Bearer <key>."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()

    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    return None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    API key authentication middleware using constant-time comparison.

    Supports:
      - X-API-Key: <key>
      - Authorization: Bearer <key>

    SECURITY NOTE: Uses secrets.compare_digest to prevent timing attacks.
    Standard string equality (key == secret) terminates on the first mismatch,
    allowing attackers to infer key characters by measuring latency.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()

        # Skip auth if not required (development without explicit flag) or for public paths
        if (
            not settings.is_auth_required
            or request.url.path in _PUBLIC_PATHS
            or request.url.path.startswith("/ui")
            or request.url.path.startswith("/static")
        ):
            return await call_next(request)

        api_key = _extract_api_key(request)
        if not api_key or not secrets.compare_digest(api_key, settings.api_secret_key):
            log.warning(
                "auth_failed",
                path=request.url.path,
                ip=request.client.host if request.client else "unknown",
            )
            return Response(
                content='{"detail": "Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


def add_middleware(app: FastAPI) -> None:
    """Register all custom middleware on the app. Called from create_app()."""
    # IMPORTANT: middleware is applied in reverse order of registration.
    # The last added is the outermost (runs first on request, last on response).
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
