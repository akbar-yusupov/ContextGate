from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from contextgate.domain.documents import TextSection

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}


def iter_sections(path: Path) -> Iterator[TextSection]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            yield TextSection(
                text=text,
                source=path.name,
                metadata={"page": index + 1},
            )
        return

    text = path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style", "nav"]):
            element.decompose()
        text = soup.get_text("\n")
    yield TextSection(text=text, source=path.name)


def load_sections(path: Path) -> list[TextSection]:
    return list(iter_sections(path))


def iter_document_paths(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            yield candidate
