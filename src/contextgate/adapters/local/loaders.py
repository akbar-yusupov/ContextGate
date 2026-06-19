from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from contextgate.domain.documents import TextSection

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}


def validate_document_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    with path.open("rb") as handle:
        prefix = handle.read(4096)
    if suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise ValueError("PDF signature is invalid")
    if suffix != ".pdf":
        if b"\x00" in prefix:
            raise ValueError("Text document contains binary data")
        prefix.decode("utf-8")


def iter_sections(
    path: Path,
    *,
    max_pdf_pages: int = 500,
    max_extracted_chars: int = 10_000_000,
) -> Iterator[TextSection]:
    suffix = path.suffix.lower()
    validate_document_file(path)
    if suffix == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        if len(reader.pages) > max_pdf_pages:
            raise ValueError(f"PDF exceeds the {max_pdf_pages}-page limit")
        extracted_chars = 0
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_chars += len(text)
            if extracted_chars > max_extracted_chars:
                raise ValueError("PDF extracted text exceeds the configured limit")
            if not text.strip():
                continue
            yield TextSection(
                text=text,
                source=path.name,
                metadata={"page": index + 1},
            )
        return

    text = path.read_text(encoding="utf-8")
    if len(text) > max_extracted_chars:
        raise ValueError("Document text exceeds the configured extraction limit")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style", "nav"]):
            element.decompose()
        text = soup.get_text("\n")
    yield TextSection(text=text, source=path.name)


def load_sections(
    path: Path,
    *,
    max_pdf_pages: int = 500,
    max_extracted_chars: int = 10_000_000,
) -> list[TextSection]:
    return list(
        iter_sections(
            path,
            max_pdf_pages=max_pdf_pages,
            max_extracted_chars=max_extracted_chars,
        )
    )


def iter_document_paths(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            yield candidate
