"""Utilities for extracting text from PDFs and splitting it into chunks."""

from __future__ import annotations

from typing import List

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read a PDF file and return all text content as a single string."""
    reader = PdfReader(pdf_path)
    text_parts: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text.strip())

    return "\n\n".join(part for part in text_parts if part)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks to help RAG retrieval.

    This keeps the chunk size relatively small and allows context to overlap
    between neighboring chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be zero or greater")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks
