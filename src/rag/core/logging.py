"""
Structured JSON logging with structlog.

CONCEPT: Structured logging means log entries are JSON objects, not plain strings.
Instead of:
  "2026-09-19 INFO Retrieved 5 chunks in 0.23s"
We get:
  {"timestamp": "...", "level": "INFO", "event": "chunks_retrieved", "count": 5, "latency_ms": 230}

WHY does this matter in production?
  - Log aggregators (Datadog, Grafana Loki, CloudWatch) can filter/query JSON fields
  - You can alert on: 'latency_ms > 1000 AND event = "llm_generation"'
  - Impossible to do with free-form strings

WHY structlog over Python's built-in logging?
  - The built-in logging module has a complex, legacy API
  - structlog provides a clean, chainable API: log.info("event", key=value)
  - It can render as JSON (production) or pretty-printed color (development)
  - It integrates with the stdlib logging so third-party libraries (uvicorn, etc.) flow through
"""

import logging
import sys

import structlog

from rag.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structlog for the application.

    In development: rich, colourised, human-readable output.
    In production:  compact JSON, one object per line (for log aggregators).

    This should be called once at application startup.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Shared processors (run in both dev and prod) ──────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # attach request-scoped context
        structlog.stdlib.add_log_level,  # add "level" field
        structlog.stdlib.add_logger_name,  # add "logger" field
        structlog.processors.TimeStamper(fmt="iso"),  # add ISO-8601 timestamp
        structlog.processors.StackInfoRenderer(),  # render stack trace as string
    ]

    if settings.is_production:
        # ── Production: JSON output ───────────────────────────────────────────
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # ── Development: pretty colourised output ─────────────────────────────
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Wire structlog into Python's stdlib logging so uvicorn/httpx logs flow through
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=renderer,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for a module.

    Usage:
        log = get_logger(__name__)
        log.info("chunks_retrieved", count=5, latency_ms=230, query_id="abc")
    """
    from typing import cast

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
