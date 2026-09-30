"""
Unit tests for the text chunker.

TESTING PHILOSOPHY:
  Unit tests should be fast, deterministic, and have zero external dependencies
  (no network, no database, no ML model). This makes them safe to run on every
  commit in CI.

  We test behaviour, not implementation. The chunker's job is to:
    1. Split text into chunks below the size limit
    2. Maintain overlap between consecutive chunks
    3. Preserve all content (no text is lost)
    4. Reject degenerate inputs gracefully

  We don't test internal helpers directly — only the public `chunk_document()`
  function. If the internals change but the behaviour stays correct, tests pass.

MARKERS: `@pytest.mark.unit` marks this as a fast unit test.
  Run only unit tests with: pytest -m unit
"""

import pytest

from rag.ingestion.chunker import TextChunk, chunk_document
from rag.ingestion.parser import ParsedDocument


def _make_document(text: str, source: str = "test.txt") -> ParsedDocument:
    """Helper: create a minimal ParsedDocument for testing."""
    return ParsedDocument(text=text, source_file=source, source_type="txt")


@pytest.mark.unit
class TestChunkDocument:
    def test_short_document_produces_single_chunk(self) -> None:
        """A document shorter than chunk_size should produce exactly one chunk."""
        doc = _make_document("This is a short document with very little content.")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short document with very little content."

    def test_chunks_respect_size_limit(self) -> None:
        """No chunk should exceed the character size limit (with small tolerance)."""
        long_text = "word " * 5000  # ~25,000 characters
        doc = _make_document(long_text)
        # chunk_size=100 tokens ≈ 400 chars
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        max_allowed = 100 * 4 * 1.2  # 20% tolerance for edge cases
        for chunk in chunks:
            assert chunk.char_count <= max_allowed, (
                f"Chunk {chunk.chunk_index} has {chunk.char_count} chars, "
                f"exceeds limit {max_allowed}"
            )

    def test_chunks_are_ordered(self) -> None:
        """Chunks should be numbered sequentially starting from 0."""
        doc = _make_document("paragraph one.\n\n" * 100)
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_is_consistent(self) -> None:
        """Every chunk should report the same total_chunks value."""
        doc = _make_document("paragraph.\n\n" * 50)
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_source_metadata_preserved(self) -> None:
        """Source file name and type should propagate to every chunk."""
        doc = _make_document("some text " * 200, source="myfile.pdf")
        doc.source_type = "pdf"
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        for chunk in chunks:
            assert chunk.source_file == "myfile.pdf"
            assert chunk.source_type == "pdf"

    def test_empty_document_returns_no_chunks(self) -> None:
        """An empty or whitespace-only document should produce no chunks."""
        doc = _make_document("   \n\n  \t  ")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        """Setting overlap >= chunk_size should raise a clear ValueError."""
        doc = _make_document("some text")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=100, chunk_overlap=100)

    def test_char_count_matches_text_length(self) -> None:
        """chunk.char_count should always equal len(chunk.text)."""
        doc = _make_document("hello world " * 500)
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.char_count == len(chunk.text), (
                f"chunk_index={chunk.chunk_index}: "
                f"char_count={chunk.char_count} != len={len(chunk.text)}"
            )

    def test_returns_text_chunk_instances(self) -> None:
        """chunk_document() should return a list of TextChunk objects."""
        doc = _make_document("test content " * 50)
        chunks = chunk_document(doc)
        assert all(isinstance(c, TextChunk) for c in chunks)
