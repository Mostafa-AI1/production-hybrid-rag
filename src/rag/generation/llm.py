"""
LLM client — OpenAI-compatible client for Groq/Gemini with retry and fallback.

CONCEPT: We use the `openai` Python SDK with a custom `base_url` to call
different providers. This works because Groq, Gemini (via AI Studio), and
Ollama all implement the OpenAI chat completions API format.

BENEFIT: We can swap providers by changing a single environment variable
(LLM_BASE_URL) with zero code changes. This is the "open standard" advantage.

RELIABILITY PATTERNS implemented here:
  1. Tenacity retry: Automatically retries on transient failures (rate limits,
     timeouts, 5xx errors) with exponential backoff + jitter.
  2. Provider fallback: If Groq returns a 429 (rate limit), we automatically
     retry with the Gemini client. This keeps the system responsive under load.
  3. Timeout: Hard limit on generation time. Without this, a slow LLM response
     blocks the request indefinitely.

INTERVIEW QUESTION: "How do you handle LLM API rate limits in production?"
  - Implement exponential backoff with jitter (not fixed intervals — avoids
    the "thundering herd" problem where all retries hit at the same time)
  - Cache frequent queries (see redis_cache.py)
  - Use a fallback provider
  - Monitor x-ratelimit-remaining headers to proactively shed load
"""

from collections.abc import Generator
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.core.config import get_settings
from rag.core.logging import get_logger

log = get_logger(__name__)


def _make_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client for a given provider."""
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,  # generation timeout in seconds
        max_retries=0,  # we handle retries ourselves via tenacity
    )


@retry(
    retry=retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )
    ),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, int] | None]:
    """
    Make a single LLM API call with automatic retry.

    The @retry decorator (from tenacity) handles:
      - Retrying on rate limit (429), timeout, and connection errors
      - wait_exponential_jitter: waits 1s, 2s, 4s... (+ random jitter) between retries
      - stop_after_attempt(3): gives up after 3 total attempts
      - reraise=True: propagates the final exception if all retries fail
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content")

    usage: dict[str, int] | None = None
    if response.usage:
        usage = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }

    return content, usage


def generate(
    messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate an answer from the LLM with automatic fallback.

    Tries the primary provider (Groq) first. If it fails after retries,
    falls back to the secondary provider (Gemini).

    Args:
        messages: OpenAI-format messages list (system + user).
        metadata: Optional metadata for logging (query_id, etc.).

    Returns:
        Dict with "answer" (str), "provider" (str), and "usage" (dict | None) fields.
    """
    settings = get_settings()
    meta = metadata or {}

    # ── Validate API key at call time (not at startup) ────────────────────────
    if not settings.llm.api_key:
        raise RuntimeError(
            "LLM_API_KEY is not configured. "
            "Add it to your .env file. Get a free key at https://console.groq.com"
        )

    # ── Primary provider ──────────────────────────────────────────────────────
    primary_client = _make_client(settings.llm.base_url, settings.llm.api_key)
    try:
        log.info(
            "llm_generation_started",
            provider=settings.llm.provider,
            model=settings.llm.model,
            **meta,
        )
        answer, usage = _call_llm(
            client=primary_client,
            model=settings.llm.model,
            messages=messages,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        log.info(
            "llm_generation_complete",
            provider=settings.llm.provider,
            response_length=len(answer),
            usage=usage,
            **meta,
        )
        return {"answer": answer, "provider": settings.llm.provider, "usage": usage}

    except (openai.RateLimitError, openai.APIConnectionError) as primary_error:
        log.warning(
            "llm_primary_failed_using_fallback",
            error=str(primary_error),
            fallback_model=settings.fallback_llm.model,
            **meta,
        )

    # ── Fallback provider ─────────────────────────────────────────────────────
    if not settings.fallback_llm.api_key:
        raise RuntimeError(
            "Primary LLM failed and no fallback API key is configured. "
            "Set FALLBACK_LLM_API_KEY in your .env file."
        )

    fallback_client = _make_client(settings.fallback_llm.base_url, settings.fallback_llm.api_key)
    answer, usage = _call_llm(
        client=fallback_client,
        model=settings.fallback_llm.model,
        messages=messages,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    log.info(
        "llm_fallback_generation_complete",
        provider="fallback",
        response_length=len(answer),
        usage=usage,
        **meta,
    )
    return {"answer": answer, "provider": "fallback", "usage": usage}


def generate_stream(
    messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Generate an answer as a real-time token stream.

    Yields:
        Individual string tokens as they arrive from the LLM.
    """
    settings = get_settings()
    meta = metadata or {}

    if not settings.llm.api_key:
        raise RuntimeError("LLM_API_KEY is not configured. Add it to your .env file.")

    primary_client = _make_client(settings.llm.base_url, settings.llm.api_key)
    try:
        log.info(
            "llm_streaming_started",
            provider=settings.llm.provider,
            model=settings.llm.model,
            **meta,
        )
        stream = primary_client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices and choices[0].delta and choices[0].delta.content:
                yield str(choices[0].delta.content)
        return

    except (openai.RateLimitError, openai.APIConnectionError) as primary_error:
        log.warning(
            "llm_stream_primary_failed_using_fallback",
            error=str(primary_error),
            fallback_model=settings.fallback_llm.model,
            **meta,
        )

    if not settings.fallback_llm.api_key:
        raise RuntimeError("Primary LLM stream failed and no fallback API key is configured.")

    fallback_client = _make_client(settings.fallback_llm.base_url, settings.fallback_llm.api_key)
    fallback_stream = fallback_client.chat.completions.create(
        model=settings.fallback_llm.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        stream=True,
    )
    for chunk in fallback_stream:
        choices = getattr(chunk, "choices", None)
        if choices and choices[0].delta and choices[0].delta.content:
            yield str(choices[0].delta.content)
