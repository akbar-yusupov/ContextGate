from __future__ import annotations

import hashlib
import re
from contextlib import suppress
from itertools import chain
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from contextgate.adapters.local.loaders import iter_document_paths, iter_sections
from contextgate.adapters.qdrant.vector_index import VectorStore, get_vector_store
from contextgate.adapters.sqlalchemy import Document, Job, KnowledgeBase
from contextgate.adapters.sqlalchemy.lookup import get_knowledge_base
from contextgate.config import Settings, get_settings
from contextgate.domain.documents import Chunk, iter_chunks
from contextgate.domain.language import detect_language
from contextgate.observability.metrics import INGESTED_CHUNKS

DOCUMENT_EXTERNAL_ID_MAX_LENGTH = 240


def _bounded_identifier(value: str, *, max_length: int = DOCUMENT_EXTERNAL_ID_MAX_LENGTH) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{value[: max_length - len(digest) - 1].rstrip('-')}-{digest}"


def _ingestion_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "dimension" in lowered or "incompatible qdrant collection" in lowered:
        return (
            f"{message}. Embedding dimensions are fixed by the selected models and by the "
            "existing Qdrant collection. Restore the matching model/dimension settings, or use "
            "a new knowledge base (or reset only disposable demo volumes) before re-ingesting."
        )
    if "value too long" in lowered:
        return (
            f"{message}. A configured identifier exceeds its database limit; check pipeline and "
            "idempotency values. Document paths are bounded automatically."
        )
    return message


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def document_external_id(document_path: Path, root: Path) -> str:
    relative = document_path.name if root.is_file() else document_path.relative_to(root).as_posix()
    without_suffix = str(Path(relative).with_suffix("")).replace("\\", "/").lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", without_suffix).strip("-")
    if "/" not in without_suffix and normalized:
        return _bounded_identifier(normalized)
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
    return _bounded_identifier(f"{normalized or 'document'}-{digest}")


def _lock_knowledge_base(session: Session, identifier: str) -> KnowledgeBase:
    knowledge_base = session.scalar(
        select(KnowledgeBase)
        .where((KnowledgeBase.slug == identifier) | (KnowledgeBase.id == identifier))
        .with_for_update()
    )
    if knowledge_base is None:
        raise ValueError(f"Knowledge base not found: {identifier}")
    return knowledge_base


class IngestionService:
    def __init__(
        self,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()

    def ingest_path(
        self,
        session: Session,
        knowledge_base: str,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
        job: Job | None = None,
    ) -> dict[str, Any]:
        kb = get_knowledge_base(session, knowledge_base)
        paths = list(iter_document_paths(path))
        total_chunks = 0
        ingested = 0
        skipped = 0
        failures: list[dict[str, str]] = []
        cancelled = False

        for index, document_path in enumerate(paths):
            if job:
                session.refresh(job)
                if job.status == "cancelled":
                    cancelled = True
                    break
            content_hash = ""
            external_id = ""
            kb = _lock_knowledge_base(session, knowledge_base)
            try:
                content_hash = file_hash(document_path)
                duplicate = session.scalar(
                    select(Document).where(
                        Document.knowledge_base_id == kb.id,
                        Document.content_hash == content_hash,
                        Document.pipeline_version == self.settings.pipeline_version,
                        Document.status == "ready",
                    )
                )
                if duplicate:
                    skipped += 1
                    continue

                section_iterator = iter_sections(
                    document_path,
                    max_pdf_pages=self.settings.max_pdf_pages,
                    max_extracted_chars=self.settings.max_extracted_chars,
                )
                sampled_sections = []
                sample_size = 0
                while sample_size < 2_000:
                    try:
                        section = next(section_iterator)
                    except StopIteration:
                        break
                    sampled_sections.append(section)
                    sample_size += len(section.text)
                combined = " ".join(section.text for section in sampled_sections)[:2_000]
                language = detect_language(combined)
                external_id = document_external_id(document_path, path)
                previous_documents = session.scalars(
                    select(Document).where(
                        Document.knowledge_base_id == kb.id,
                        Document.external_id == external_id,
                        Document.status == "ready",
                    )
                ).all()
                document = Document(
                    knowledge_base_id=kb.id,
                    source=document_path.name,
                    external_id=external_id,
                    content_hash=content_hash,
                    pipeline_version=self.settings.pipeline_version,
                    status="processing",
                    metadata_json=metadata or {},
                )
                session.add(document)
                session.flush()
                chunks = iter_chunks(
                    chain(sampled_sections, section_iterator),
                    document_id=external_id,
                    language=language,
                )
                batch: list[Chunk] = []
                count = 0
                for chunk in chunks:
                    chunk.metadata.update(metadata or {})
                    chunk.metadata["db_document_id"] = document.id
                    batch.append(chunk)
                    if len(batch) >= self.settings.embedding_batch_size:
                        count += self.store.upsert_chunks(
                            kb.collection_name,
                            batch,
                            content_hash=content_hash,
                            pipeline_version=self.settings.pipeline_version,
                        )
                        batch.clear()
                if batch:
                    count += self.store.upsert_chunks(
                        kb.collection_name,
                        batch,
                        content_hash=content_hash,
                        pipeline_version=self.settings.pipeline_version,
                    )
                if count == 0:
                    raise ValueError("No extractable text found in document")
                self.store.delete_document_versions(
                    kb.collection_name,
                    external_id,
                    keep_content_hash=content_hash,
                )
                for previous in previous_documents:
                    previous.status = "superseded"
                document.status = "ready"
                kb.corpus_version += 1
                session.add(kb)
                session.commit()
                INGESTED_CHUNKS.inc(count)
                total_chunks += count
                ingested += 1
            except Exception as exc:
                session.rollback()
                if content_hash and external_id:
                    with suppress(Exception):
                        self.store.delete_document_version(
                            kb.collection_name,
                            external_id,
                            content_hash,
                        )
                failures.append({"source": document_path.name, "error": _ingestion_error(exc)})
            finally:
                if job:
                    job.progress = (index + 1) / max(len(paths), 1)
                    session.add(job)
                    session.commit()

        result = {
            "documents": len(paths),
            "ingested": ingested,
            "skipped": skipped,
            "chunks": total_chunks,
            "failures": failures,
            "outcome": (
                "cancelled"
                if cancelled
                else "failed"
                if failures and not ingested and not skipped
                else "succeeded_with_errors"
                if failures
                else "succeeded"
            ),
        }
        if job:
            job.result = result
            job.progress = 1
            session.add(job)
            session.commit()
        return result

    def sync_collection(
        self,
        session: Session,
        knowledge_base: str,
        source_collection: str,
        *,
        job: Job | None = None,
    ) -> dict[str, Any]:
        kb = _lock_knowledge_base(session, knowledge_base)
        target = f"{kb.collection_name}-sync"
        copied = self.store.copy_collection(source_collection, target)
        kb.collection_name = target
        kb.corpus_version += 1
        session.add(kb)
        result = {
            "source_collection": source_collection,
            "target_collection": target,
            "copied": copied,
        }
        if job:
            job.status = "succeeded"
            job.progress = 1
            job.result = result
            session.add(job)
        session.commit()
        return result
