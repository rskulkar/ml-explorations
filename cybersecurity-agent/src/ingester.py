"""Document ingestion and chunking for the cybersecurity-agent RAG pipeline."""
from __future__ import annotations

import pathlib

from pypdf import PdfReader


def load_pdf(pdf_path: pathlib.Path) -> str:
    """Load a PDF and return all extracted text as a single string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages, separated by newlines.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    pdf_path = pathlib.Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks approximating a token budget.

    Token approximation: 1 token ≈ 4 characters.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size in tokens (default 500 → 2000 chars).
        overlap: Overlap between adjacent chunks in tokens (default 50 → 200 chars).

    Returns:
        List of non-empty string chunks.
    """
    if not text:
        return []

    chars_per_token = 4
    chunk_chars = chunk_size * chars_per_token   # 2000
    overlap_chars = overlap * chars_per_token    # 200
    step = chunk_chars - overlap_chars           # 1800

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks
