"""
Text chunker — splits documents into overlapping chunks for embedding.

CONCEPT: Why do we chunk at all?
  Embedding models have a token limit (typically 512–8192 tokens). A 50-page
  PDF won't fit. More importantly, embedding an entire document produces a
  single vector that represents "everything" — which is too coarse for precise
  retrieval. We want vectors that represent specific *concepts*, so the LLM
  gets focused context, not an entire document.

THE CHUNKING TRILEMMA:
  1. Too small (< 100 tokens): Chunks lack context. "The function returns None"
     — returns None *from what*? The LLM has no idea.
  2. Too large (> 1000 tokens): Chunks dilute relevance. The embedding becomes
     a "confused average" of multiple topics.
  3. Poor boundaries: Splitting mid-sentence breaks coherence.

Our strategy: Recursive Character splitting with configurable overlap.
  - Start by splitting on paragraph breaks (\n\n)
  - If a chunk is still too large, split on newlines (\n)
  - If still too large, split on sentences (.) then spaces
  - Overlap ensures context continuity across chunk boundaries

INTERVIEW QUESTION: "What is chunk overlap and why does it matter?"
  Without overlap, a key fact at the end of chunk N and a key fact at the start
  of chunk N+1 are artificially separated. Overlap of ~64 tokens means both
  chunks contain enough context to be independently meaningful.
"""

from dataclasses import dataclass, field

from rag.core.config import get_settings
from rag.core.logging import get_logger
from rag.ingestion.parser import ParsedDocument

log = get_logger(__name__)

# Separators tried in order — coarser to finer granularity
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


@dataclass
class TextChunk:
    """A single chunk of text ready for embedding."""

    text: str
    chunk_index: int
    total_chunks: int  # filled in after all chunks are created
    source_file: str
    source_type: str
    char_count: int = field(init=False)
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def chunk_document(
    document: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split a ParsedDocument into overlapping TextChunks.

    Args:
        document: The parsed document to chunk.
        chunk_size: Max characters per chunk. Defaults to settings value.
        chunk_overlap: Character overlap between consecutive chunks. Defaults to settings value.

    Returns:
        List of TextChunks ordered by position in the document.

    NOTE: We use *character* counts rather than token counts for simplicity and
    speed (no tokenizer needed here). At ~4 chars/token, a 512-token limit ≈
    2048 chars. We default to 2048 chars (≈ 512 tokens) for BGE-M3's 8192-token
    limit, giving us headroom for long words and Unicode.
    """
    settings = get_settings()
    # Character-based sizing: ~4 chars per token
    chars_per_chunk = (chunk_size or settings.chunk_size) * 4
    chars_overlap = (chunk_overlap or settings.chunk_overlap) * 4

    if chars_overlap >= chars_per_chunk:
        raise ValueError(
            f"chunk_overlap ({chars_overlap}) must be less than chunk_size ({chars_per_chunk})"
        )

    raw_chunks = _recursive_split(document.text, chars_per_chunk, chars_overlap)

    # Filter out chunks that are too short to be meaningful (< 50 chars)
    raw_chunks = [c for c in raw_chunks if len(c.strip()) >= 50]

    total = len(raw_chunks)
    chunks = [
        TextChunk(
            text=chunk.strip(),
            chunk_index=i,
            total_chunks=total,
            source_file=document.source_file,
            source_type=document.source_type,
            metadata=dict(document.metadata),
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    log.info(
        "document_chunked",
        source=document.source_file,
        total_chunks=total,
        chunk_size_chars=chars_per_chunk,
        overlap_chars=chars_overlap,
        avg_chunk_chars=sum(c.char_count for c in chunks) // max(total, 1),
    )

    return chunks


# ── Private helpers ───────────────────────────────────────────────────────────


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.

    This is the core algorithm. It works like this:
      1. Find the "best" (coarsest) separator that exists in the text
      2. Split on that separator
      3. Merge short splits back together until they approach chunk_size
      4. For splits that are still too large, recurse with finer separators
    """
    if separators is None:
        separators = _RECURSIVE_SEPARATORS

    # Find the first separator that exists in the text
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    splits = text.split(separator) if separator else [text]
    final_chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split)

        if split_len > chunk_size:
            # This split is too large — recurse with finer separators
            if current_parts:
                merged = _merge_with_overlap(current_parts, separator, overlap)
                final_chunks.extend(merged)
                current_parts = []
                current_len = 0
            if remaining_separators:
                sub_chunks = _recursive_split(split, chunk_size, overlap, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # No finer separator — hard split (last resort)
                final_chunks.extend(_hard_split(split, chunk_size, overlap))
        elif current_len + split_len + len(separator) > chunk_size and current_parts:
            # Adding this split would exceed the limit — flush current batch
            merged = _merge_with_overlap(current_parts, separator, overlap)
            final_chunks.extend(merged)
            # Keep overlap: retain last parts that fit within overlap size
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part)
                else:
                    break
            current_parts = [*overlap_parts, split]
            current_len = sum(len(p) for p in current_parts)
        else:
            current_parts.append(split)
            current_len += split_len + len(separator)

    if current_parts:
        final_chunks.extend(_merge_with_overlap(current_parts, separator, overlap))

    return [c for c in final_chunks if c.strip()]


def _merge_with_overlap(parts: list[str], separator: str, overlap: int) -> list[str]:
    """Merge a list of parts into a single chunk string."""
    return [separator.join(parts)]


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort hard split by character count when no separator works."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
