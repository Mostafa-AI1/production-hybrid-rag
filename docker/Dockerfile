# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile
#
# CONCEPT: Multi-stage builds produce a small final image by using a heavy
# "builder" stage (with compilers, dev tools) and a lean "runtime" stage
# that only copies the compiled artifacts. This reduces attack surface and
# image size significantly.
#
# Stage 1 (builder): Install all deps including build tools
# Stage 2 (runtime): Copy only the installed packages + source code
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv — the fastest Python package installer (written in Rust)
# uv is 10-100x faster than pip for dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency files first — Docker layer caching means this layer
# only rebuilds when pyproject.toml changes, not when source code changes.
COPY pyproject.toml .

# Install production dependencies into /app/.venv
# --no-dev: exclude dev/test dependencies from the production image
RUN uv sync --no-dev --frozen

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# SECURITY: Run as non-root user. Root in a container can be root on the host
# if there's a container escape vulnerability. This is a production requirement.
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from builder — all compiled packages
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ src/

# Model cache directory — writable by appuser for first-run model downloads
RUN mkdir -p /app/.cache && chown -R appuser:appgroup /app

# Use the venv's Python interpreter
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
# Don't buffer Python output — important for log visibility in Docker
ENV PYTHONUNBUFFERED=1
# Don't write .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Point HuggingFace models to our writable cache dir
ENV HF_HOME="/app/.cache"

USER appuser

# Health check — Docker and orchestrators use this to know if the app is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

EXPOSE 8080

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
