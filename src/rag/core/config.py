"""
Core application configuration using Pydantic Settings.

CONCEPT: Pydantic Settings reads configuration from environment variables
(and .env files) and validates them against a typed schema. This means:
  - Type errors in config are caught at startup, not at runtime
  - Config is self-documenting (the class IS the schema)
  - The same class works locally (.env file) and in production (env vars)

WHY NOT os.environ.get()? Because that returns str | None with no validation.
You'd find out about a missing QDRANT_URL only when the first query fails.
With Pydantic Settings, the app refuses to start with a clear error message.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: str = Field(default="groq", description="LLM provider name")
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="OpenAI-compatible base URL",
    )
    api_key: str = Field(
        default="",
        description="API key for the LLM provider. Validated at call time, not startup.",
    )
    model: str = Field(default="llama-3.1-70b-versatile")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class FallbackLLMSettings(BaseSettings):
    """Fallback LLM (used when primary provider is rate-limited)."""

    model_config = SettingsConfigDict(env_prefix="FALLBACK_LLM_", extra="ignore")

    base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai")
    api_key: str = Field(default="")
    model: str = Field(default="gemini-2.0-flash")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration (local inference)."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    model: str = Field(default="BAAI/bge-m3")
    device: str = Field(default="cpu", description="cpu | cuda | mps")
    batch_size: int = Field(default=32, ge=1, le=512)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and normalize device selection."""
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker configuration (local inference)."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", extra="ignore")

    model: str = Field(default="BAAI/bge-reranker-v2-m3")
    device: str = Field(default="cpu")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to pass to LLM after reranking",
    )


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_", extra="ignore")

    url: str = Field(default="http://localhost:6333")
    collection: str = Field(default="rag_documents")
    api_key: str | None = Field(default=None, description="Only needed for Qdrant Cloud")


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    url: str = Field(default="redis://localhost:6379")


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", extra="ignore")

    host: str = Field(default="http://localhost:3000")
    public_key: str = Field(default="")
    secret_key: str = Field(default="")

    @property
    def is_enabled(self) -> bool:
        """Langfuse is optional — disabled if keys are not set."""
        return bool(self.public_key and self.secret_key)


class RateLimitSettings(BaseSettings):
    """API rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_", extra="ignore")

    enabled: bool = Field(default=True, description="Enable API rate limiting")
    default_limit: str = Field(
        default="60/minute",
        description="Default rate limit applied to routes",
    )
    query_limit: str = Field(
        default="30/minute",
        description="Rate limit for /api/v1/query to protect LLM quota",
    )
    ingest_limit: str = Field(
        default="10/minute",
        description="Rate limit for /api/v1/ingest endpoints (compute-heavy)",
    )
    storage_url: str | None = Field(
        default=None,
        description="Redis URI for distributed limits (e.g. redis://host:6379/1).",
    )


class Settings(BaseSettings):
    """
    Root application settings.

    Loads from environment variables and .env file.
    Nested settings classes are instantiated automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # don't fail on unrecognised env vars
    )

    # Application
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_secret_key: str = Field(default="dev-secret-change-me")
    api_auth_enabled: bool = Field(
        default=False,
        description="Explicitly enforce API key authentication (always enforced in production)",
    )

    # RAG pipeline tuning
    retrieval_top_k: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Candidates to retrieve (before reranking)",
    )
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=256)
    cache_ttl_seconds: int = Field(default=3600)

    # Phase 3: Semantic Cache & Advanced Retrieval
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable embedding-based semantic caching",
    )
    semantic_cache_threshold: float = Field(
        default=0.92,
        ge=0.5,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hits",
    )
    semantic_cache_collection: str = Field(default="rag_semantic_cache")
    enable_hyde: bool = Field(default=False, description="Enable HyDE query rewriting")

    # Nested settings — each reads its own env prefix
    llm: LLMSettings = Field(default_factory=LLMSettings)
    fallback_llm: FallbackLLMSettings = Field(default_factory=FallbackLLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.app_env == "production"

    @property
    def is_auth_required(self) -> bool:
        """True when API authentication is active (in production or explicitly enabled)."""
        return self.is_production or self.api_auth_enabled


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    WHY lru_cache? Settings reads from disk (.env file) and validates.
    We don't want to re-read and re-validate on every request.
    lru_cache(maxsize=1) creates a singleton — one instance for the app lifetime.

    In tests, call get_settings.cache_clear() to reset between test cases.
    """
    return Settings()
