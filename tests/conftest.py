"""
Pytest configuration and shared fixtures for the test suite.

CONCEPT: conftest.py is automatically loaded by pytest for all tests in
the same directory and subdirectories. It's where you put:
  - Shared fixtures (reusable test helpers)
  - Hooks (modify pytest behaviour)
  - Global test configuration

WHY do we need this file?
  1. Settings cache: get_settings() uses @lru_cache, which means the same
     Settings instance is reused across tests. If one test modifies env vars,
     it won't see the change unless we clear the cache. The autouse fixture
     below ensures every test starts with a fresh Settings instance.

  2. Environment vars: Unit tests run without a .env file. We set minimal
     required environment variables here so Settings can construct without
     failing on missing values.
"""

import os

import pytest

# ── Set minimal environment variables for tests ───────────────────────────────
# These must be set BEFORE any rag module is imported, so we set them at
# module level (not inside a fixture).
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """
    Clear the lru_cache on get_settings() before every test.

    WHY: @lru_cache(maxsize=1) means get_settings() returns the same object
    for the lifetime of the process. Without this fixture, settings loaded by
    the first test bleed into subsequent tests, making tests order-dependent
    and hard to debug.

    autouse=True means this fixture runs for EVERY test automatically —
    no need to declare it as a parameter in each test function.
    """
    from rag.core.config import get_settings

    get_settings.cache_clear()
