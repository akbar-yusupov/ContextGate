from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.local.ingestion_service import IngestionService, document_external_id
from contextgate.adapters.sqlalchemy.models import Base, Document, Job, KnowledgeBase
from contextgate.config import Settings


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[Any], str, str]] = []
        self.deleted_versions: list[tuple[str, str, str]] = []

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Any],
        *,
        content_hash: str,
        pipeline_version: str,
    ) -> int:
        self.upserts.append((collection, list(chunks), content_hash, pipeline_version))
        return len(chunks)

    def delete_document_versions(
        self, collection: str, external_id: str, *, keep_content_hash: str
    ) -> None:
        return None

    def delete_document_version(self, collection: str, external_id: str, content_hash: str) -> None:
        self.deleted_versions.append((collection, external_id, content_hash))

    def copy_collection(self, source: str, target: str) -> int:
        assert source == "source"
        return 7


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ingestion.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    session = sessions()
    session.add(
        KnowledgeBase(
            name="Demo",
            slug="demo",
            collection_name="contextgate-demo",
        )
    )
    session.commit()
    return session


def test_ingestion_versions_documents_and_skips_identical_content(tmp_path) -> None:
    session = _session(tmp_path)
    store = FakeVectorStore()
    service = IngestionService(
        store=store,  # type: ignore[arg-type]
        settings=Settings(embedding_batch_size=1, pipeline_version="test-v1"),
    )
    document = tmp_path / "policy.md"
    document.write_text("Orders may be cancelled before courier handoff.", encoding="utf-8")

    first = service.ingest_path(session, "demo", document, metadata={"team": "support"})
    duplicate = service.ingest_path(session, "demo", document)
    document.write_text("Orders may now be cancelled within one hour.", encoding="utf-8")
    replacement = service.ingest_path(session, "demo", document)

    rows = list(session.scalars(select(Document).order_by(Document.created_at)).all())
    kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.slug == "demo"))
    assert first["outcome"] == "succeeded" and first["chunks"] > 0
    assert duplicate["skipped"] == 1
    assert replacement["ingested"] == 1
    assert [row.status for row in rows] == ["superseded", "ready"]
    assert rows[0].metadata_json == {"team": "support"}
    assert kb is not None and kb.corpus_version == 2
    assert len(store.upserts) == 2


def test_ingestion_records_validation_failure_and_cooperative_cancellation(tmp_path) -> None:
    session = _session(tmp_path)
    service = IngestionService(
        store=FakeVectorStore(),  # type: ignore[arg-type]
        settings=Settings(),
    )
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not-pdf")
    failed = service.ingest_path(session, "demo", bad_pdf)
    assert failed["outcome"] == "failed"
    assert "signature" in failed["failures"][0]["error"]

    text = tmp_path / "cancel.md"
    text.write_text("cancel policy", encoding="utf-8")
    job = Job(kind="ingest", status="cancelled", payload={})
    session.add(job)
    session.commit()
    cancelled = service.ingest_path(session, "demo", text, job=job)
    assert cancelled["outcome"] == "cancelled"
    assert job.progress == 1


def test_sync_collection_updates_authoritative_corpus_version(tmp_path) -> None:
    session = _session(tmp_path)
    service = IngestionService(
        store=FakeVectorStore(),  # type: ignore[arg-type]
        settings=Settings(),
    )
    job = Job(kind="sync_qdrant", payload={})
    session.add(job)
    session.commit()

    result = service.sync_collection(session, "demo", "source", job=job)

    kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.slug == "demo"))
    assert result == {
        "source_collection": "source",
        "target_collection": "contextgate-demo-sync",
        "copied": 7,
    }
    assert kb is not None and kb.corpus_version == 1
    assert job.status == "succeeded" and job.progress == 1


def test_document_external_id_is_stable_and_bounded_for_deep_paths() -> None:
    root = Path("documents")
    document = root / ("section-" + "a" * 180) / ("policy-" + "b" * 180 + ".md")

    first = document_external_id(document, root)
    second = document_external_id(document, root)

    assert first == second
    assert len(first) <= 240
    suffix = first.rsplit("-", 1)[-1]
    assert len(suffix) == 16
    assert all(character in "0123456789abcdef" for character in suffix)


def test_ingestion_dimension_failure_has_actionable_remediation(tmp_path) -> None:
    class IncompatibleStore(FakeVectorStore):
        def upsert_chunks(self, *args, **kwargs) -> int:
            raise ValueError(
                "Incompatible Qdrant collection contextgate-demo: dense dimension is 384, "
                "expected 64"
            )

    session = _session(tmp_path)
    document = tmp_path / "policy.md"
    document.write_text("A supported policy statement.", encoding="utf-8")
    result = IngestionService(
        store=IncompatibleStore(),  # type: ignore[arg-type]
        settings=Settings(),
    ).ingest_path(session, "demo", document)

    assert result["outcome"] == "failed"
    assert "dimensions are fixed by the selected models" in result["failures"][0]["error"]
    assert "new knowledge base" in result["failures"][0]["error"]
