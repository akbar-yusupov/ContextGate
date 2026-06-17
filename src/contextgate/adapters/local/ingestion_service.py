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
from contextgate.adapters.sqlalchemy import Document, Job
from contextgate.adapters.sqlalchemy.lookup import get_knowledge_base
from contextgate.config import Settings, get_settings
from contextgate.domain.documents import Chunk, iter_chunks
from contextgate.domain.language import detect_language
from contextgate.observability.metrics import INGESTED_CHUNKS


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
        return normalized
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
    return f"{normalized or 'document'}-{digest}"


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

        for index, document_path in enumerate(paths):
            content_hash = ""
            external_id = ""
            kb = get_knowledge_base(session, knowledge_base)
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

                section_iterator = iter_sections(document_path)
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
                failures.append({"source": document_path.name, "error": str(exc)})
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
        }
        if job:
            job.status = "failed" if failures and not ingested else "succeeded"
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
        kb = get_knowledge_base(session, knowledge_base)
        target = f"{kb.collection_name}-sync"
        copied = self.store.copy_collection(source_collection, target)
        kb.collection_name = target
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
