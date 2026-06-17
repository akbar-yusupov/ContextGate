from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextSection:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


def iter_chunks(
    sections: Iterable[TextSection],
    *,
    document_id: str,
    language: str,
    target_chars: int = 1_200,
    overlap_chars: int = 180,
) -> Iterator[Chunk]:
    chunk_index = 0
    buffer = ""
    metadata: dict[str, Any] = {}
    source = document_id

    def flush() -> Chunk | None:
        nonlocal buffer, chunk_index
        text = re.sub(r"\s+", " ", buffer).strip()
        if not text:
            return None
        chunk = Chunk(
            chunk_id=f"{document_id}:{chunk_index}",
            document_id=document_id,
            text=text,
            source=source,
            language=language,
            metadata=dict(metadata),
        )
        chunk_index += 1
        buffer = text[-overlap_chars:] if overlap_chars else ""
        return chunk

    for section in sections:
        source = section.source
        metadata = section.metadata
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section.text) if part.strip()]
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 2 > target_chars and buffer and (chunk := flush()):
                yield chunk
            if len(paragraph) > target_chars * 2:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if (
                        len(buffer) + len(sentence) + 1 > target_chars
                        and buffer
                        and (chunk := flush())
                    ):
                        yield chunk
                    buffer = f"{buffer} {sentence}".strip()
            else:
                buffer = f"{buffer}\n\n{paragraph}".strip()
    if chunk := flush():
        yield chunk


def chunk_sections(
    sections: list[TextSection],
    *,
    document_id: str,
    language: str,
    target_chars: int = 1_200,
    overlap_chars: int = 180,
) -> list[Chunk]:
    return list(
        iter_chunks(
            sections,
            document_id=document_id,
            language=language,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
    )
