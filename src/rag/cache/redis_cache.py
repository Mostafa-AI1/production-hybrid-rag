"""
Redis query cache — stores query→answer mappings with TTL expiry.

CONCEPT: Caching is the single highest-leverage optimisation in a production
RAG system. Why?

  Without cache:  Every query → embed → retrieve → rerank → LLM call → response
                  Cost: ~1-2 API calls, 1-3 seconds
  With cache hit: query → hash lookup → response
                  Cost: ~1ms, 0 API calls

  For real-world workloads, 20-40% of queries are repeats or near-repeats.
  Caching those saves both latency and API quota.

IMPLEMENTATION: Exact-match cache (Phase 1)
  Key: SHA-256 hash of the lowercased, stripped query
  Value: JSON-serialised response dict
  TTL: Configurable (default 1 hour)

  WHY hash the query? Keys must be strings. Hashing also normalises the query
  ("what is python" and "What is Python?" produce the same hash after cleaning).

PHASE 3 UPGRADE: Semantic cache
  Instead of exact hash matching, embed the query and find semantically similar
  cached queries using a small Qdrant collection. A "Python functions" query
  would hit a cache entry for "What are Python functions?" — different words,
  same meaning.

INTERVIEW QUESTION: "What is the difference between exact and semantic caching
in a RAG system, and what are the trade-offs?"
"""

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)

_CACHE_PREFIX = "rag:query:"


def _make_cache_key(query: str) -> str:
    """
    Generate a deterministic cache key from a query string.

    Normalise before hashing: lowercase + strip whitespace → same hash for
    "What is Python?" and "what is python?".
    """
    normalised = query.lower().strip()
    digest = hashlib.sha256(normalised.encode()).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def get_redis_client() -> aioredis.Redis:  # type: ignore[type-arg]
    """Create a Redis client from settings."""
    settings = get_settings()
    return aioredis.from_url(
        settings.redis.url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
    )


class QueryCache:
    """
    Async Redis-backed query cache.

    Usage in the query pipeline:
        cache = QueryCache(redis_client)
        cached = await cache.get(query)
        if cached:
            return cached
        # ... run pipeline ...
        await cache.set(query, result)
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._redis = redis_client
        self._ttl = get_settings().cache_ttl_seconds

    async def get(self, query: str) -> dict[str, Any] | None:
        """
        Look up a cached response for the query.

        Returns None on cache miss, Redis connection error, or deserialisation error.
        We swallow errors rather than letting cache failures break the pipeline.
        """
        key = _make_cache_key(query)
        try:
            value = await self._redis.get(key)
            if value is None:
                log.debug("cache_miss", key_prefix=key[:20])
                return None
            result: dict[str, Any] = json.loads(value)
            log.info("cache_hit", key_prefix=key[:20])
            return result
        except (aioredis.RedisError, json.JSONDecodeError) as e:
            # Cache errors must NOT break the main query flow
            log.warning("cache_get_error", error=str(e))
            return None

    async def set(self, query: str, value: dict[str, Any]) -> None:
        """
        Store a query response in the cache with TTL expiry.

        Redis EX parameter sets the key to expire after TTL seconds.
        When a key expires, Redis removes it automatically — no manual cleanup.
        """
        key = _make_cache_key(query)
        try:
            await self._redis.set(key, json.dumps(value), ex=self._ttl)
            log.debug("cache_set", key_prefix=key[:20], ttl=self._ttl)
        except aioredis.RedisError as e:
            log.warning("cache_set_error", error=str(e))

    async def invalidate(self, query: str) -> None:
        """Manually invalidate a cached response (e.g., after document update)."""
        key = _make_cache_key(query)
        await self._redis.delete(key)

    async def health_check(self) -> bool:
        """Ping Redis to confirm the connection is alive."""
        try:
            return await self._redis.ping()
        except aioredis.RedisError:
            return False
