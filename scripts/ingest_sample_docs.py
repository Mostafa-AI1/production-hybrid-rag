"""
Sample ingestion script — ingest Python documentation into the RAG index.

Run this script to populate the system with real content before querying.

Usage:
    # Start the services first:
    docker compose -f docker/docker-compose.yml up -d qdrant redis

    # Then run this script:
    python scripts/ingest_sample_docs.py

What this script does:
    1. Downloads a subset of the Python 3.12 documentation as Markdown
    2. Ingests each document through the full pipeline (parse → chunk → embed → index)
    3. Prints a summary of what was ingested

WHY Python docs?
    - Freely available, well-structured, domain-specific enough to be interesting
    - Easy to write evaluation questions for (e.g., "What is a decorator?")
    - Large enough to make retrieval non-trivial (thousands of chunks)
"""

import sys
import time
from pathlib import Path

# Add src to path so we can import rag modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.core.config import get_settings
from rag.core.logging import configure_logging
from rag.ingestion.pipeline import ingest_text
from rag.retrieval.embedder import get_embedding_dimension, warmup_model
from rag.retrieval.vector_store import ensure_collection_exists, get_qdrant_client

# Sample content — Python concepts as markdown documents
# In production you'd download these from the Python docs website or your own S3 bucket
SAMPLE_DOCUMENTS = [
    {
        "source_name": "python_functions.md",
        "content": """
# Python Functions

## Defining Functions
A function is defined using the `def` keyword. Functions allow you to encapsulate
reusable pieces of code.

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Default Arguments
Functions can have default argument values that are used when an argument is not provided.

```python
def power(base: float, exponent: float = 2.0) -> float:
    return base ** exponent
```

## *args and **kwargs
Use `*args` to accept a variable number of positional arguments,
and `**kwargs` for variable keyword arguments.

```python
def sum_all(*args: float) -> float:
    return sum(args)

def display_info(**kwargs: str) -> None:
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

## Lambda Functions
Anonymous functions defined with `lambda`. Best for short, simple operations.

```python
square = lambda x: x ** 2
```
""",
    },
    {
        "source_name": "python_decorators.md",
        "content": """
# Python Decorators

## What is a Decorator?
A decorator is a function that takes another function as input, adds some behaviour,
and returns a new function. They use the `@` syntax.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")
```

## functools.wraps
Always use `@functools.wraps(func)` in your wrapper to preserve the original
function's metadata (name, docstring, etc.).

```python
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
```

## Class-Based Decorators
Decorators can also be classes that implement `__call__`.

```python
class retry:
    def __init__(self, times: int = 3):
        self.times = times

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == self.times - 1:
                        raise
        return wrapper
```

## Common Built-in Decorators
- `@property`: makes a method accessible as an attribute
- `@staticmethod`: method that doesn't receive self or cls
- `@classmethod`: method that receives the class as the first argument
- `@functools.lru_cache`: memoises function results
""",
    },
    {
        "source_name": "python_context_managers.md",
        "content": """
# Python Context Managers

## The with Statement
Context managers manage resources automatically, ensuring cleanup even on exceptions.

```python
with open("file.txt") as f:
    content = f.read()
# File is automatically closed here, even if an exception occurred
```

## Creating Context Managers with __enter__ and __exit__
```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        return False  # don't suppress exceptions
```

## Using contextlib.contextmanager
The decorator approach is simpler for most cases.

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
```

## asynccontextmanager
For async context managers used with `async with`.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_db_session():
    session = await create_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```
""",
    },
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging()
    settings = get_settings()

    print(f"[*] Starting sample ingestion (Qdrant: {settings.qdrant.url})")

    # Ensure Qdrant collection exists
    client = get_qdrant_client()
    dim = get_embedding_dimension()
    ensure_collection_exists(client, vector_dim=dim)

    # Warm up embedding model
    print("[-] Loading embedding model (first run downloads ~1GB)...")
    warmup_model()
    print("[OK] Embedding model ready")

    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        print(f"\n[-] Ingesting: {doc['source_name']}")
        start = time.perf_counter()
        result = ingest_text(doc["content"], source_name=doc["source_name"])
        elapsed = time.perf_counter() - start
        chunks = result["chunk_count"]
        total_chunks += chunks
        print(f"   + {chunks} chunks in {elapsed:.1f}s")

    print(f"\n[OK] Ingestion complete! Total chunks indexed: {total_chunks}")
    print("\nNow try a query:")
    print("  curl -X POST http://localhost:8080/api/v1/query \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What is a Python decorator?"}\'')


if __name__ == "__main__":
    main()
