"""
Document parser — converts raw files into plain text.

Supported formats:
  - PDF  (.pdf)  via pypdf
  - Markdown (.md) — kept as-is, stripped of HTML
  - Plain text (.txt) — trivial passthrough

WHY a separate parser module?
  The rest of the pipeline works on plain strings. By isolating format-specific
  logic here, we can add DOCX, HTML, or even audio transcripts later without
  touching the chunker or embedder.

PRODUCTION NOTE: For a real system handling thousands of documents, you'd want
to run parsing in a background task queue (Celery + Redis, or ARQ). For this
project, we keep it synchronous and note where to add async processing.
"""

import re
from pathlib import Path

import pypdf

from rag.core.logging import get_logger

log = get_logger(__name__)


class ParsedDocument:
    """Result of parsing a document file."""

    __slots__ = ("metadata", "source_file", "source_type", "text")

    def __init__(
        self,
        text: str,
        source_file: str,
        source_type: str,
        metadata: dict[str, str | int | float] | None = None,
    ) -> None:
        self.text = text
        self.source_file = source_file
        self.source_type = source_type
        self.metadata = metadata or {}


def parse_file(file_path: Path) -> ParsedDocument:
    """
    Parse a document file into a ParsedDocument.

    Dispatches to the appropriate parser based on file extension.
    Raises ValueError for unsupported formats.
    """
    suffix = file_path.suffix.lower()

    log.info("parsing_document", path=str(file_path), format=suffix)

    parsers = {
        ".pdf": _parse_pdf,
        ".md": _parse_markdown,
        ".txt": _parse_text,
    }

    if suffix not in parsers:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. Supported formats: {list(parsers.keys())}"
        )

    text = parsers[suffix](file_path)
    text = _clean_text(text)

    log.info(
        "document_parsed",
        path=str(file_path),
        format=suffix,
        char_count=len(text),
    )

    return ParsedDocument(
        text=text,
        source_file=file_path.name,
        source_type=suffix.lstrip("."),
        metadata={"original_path": str(file_path)},
    )


def parse_text_content(text: str, source_name: str) -> ParsedDocument:
    """
    Parse a raw text string directly (for API uploads or tests).

    Useful when the client sends text content directly rather than a file.
    """
    cleaned = _clean_text(text)
    return ParsedDocument(
        text=cleaned,
        source_file=source_name,
        source_type="text",
    )


# ── Private parsers ───────────────────────────────────────────────────────────


def _parse_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pypdf.

    LIMITATION: pypdf is pure-Python and works well for text-based PDFs.
    For scanned PDFs (images), you'd need OCR (pytesseract, AWS Textract).
    This is a known limitation we document rather than silently fail on.
    """
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            # Prepend page number as context — useful in citations
            pages.append(f"[Page {page_num + 1}]\n{page_text}")

    if not pages:
        log.warning(
            "pdf_no_text_extracted",
            path=str(path),
            hint="PDF may be scanned/image-based — OCR not implemented",
        )

    return "\n\n".join(pages)


def _parse_markdown(path: Path) -> str:
    """Read Markdown file. We preserve the raw text — chunker handles structure."""
    return path.read_text(encoding="utf-8")


def _parse_text(path: Path) -> str:
    """Read plain text file."""
    return path.read_text(encoding="utf-8")


def _clean_text(text: str) -> str:
    """
    Clean extracted text.

    - Collapse multiple blank lines into at most two (preserve paragraph breaks)
    - Strip leading/trailing whitespace per line
    - Remove null bytes and other control characters
    """
    # Remove null bytes and control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
