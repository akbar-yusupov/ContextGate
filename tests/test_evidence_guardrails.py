from __future__ import annotations

from contextgate.adapters.fastembed.embeddings import DeterministicEmbeddingProvider
from contextgate.adapters.local.guardrails import SemanticClaimVerifier
from contextgate.domain.evidence import build_evidence_report, score_evidence, validate_citations
from contextgate.domain.gateway import Citation
from contextgate.domain.retrieval import RetrievalHit


def test_unsupported_claim_detector_flags_absent_terms() -> None:
    evidence = score_evidence(
        query="Can I cancel my order?",
        answer="You can cancel the order and receive an instant teleport refund.",
        contexts=["Orders can be cancelled before courier handoff."],
        abstained=False,
    )

    assert evidence.unsupported_claims
    assert "teleport" in evidence.unsupported_claims


def _hit(rank: int, chunk_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc",
        source=f"{chunk_id}.md",
        text=text,
        language="en",
        score=0.9,
        rank=rank,
    )


def test_citation_index_must_resolve_to_the_same_chunk_and_source() -> None:
    hits = [
        _hit(1, "first", "Orders can be cancelled."),
        _hit(2, "second", "Refunds take five days."),
    ]
    validation = validate_citations(
        [Citation(index=1, chunk_id="second", source="second.md")],
        hits,
        require_citation=True,
    )

    assert validation.valid is False


def test_claim_report_rejects_unsupported_cited_claim() -> None:
    hits = [_hit(1, "orders", "Orders can be cancelled before courier handoff.")]
    report = build_evidence_report(
        answer="Orders can be cancelled and receive teleport refunds. [1]",
        citations=[Citation(index=1, chunk_id="orders", source="orders.md")],
        hits=hits,
    )

    assert report.passed is False
    assert report.reason == "unsupported_claim"


def test_local_semantic_verifier_records_its_version_and_support() -> None:
    hits = [_hit(1, "orders", "Orders can be cancelled before courier handoff.")]
    report = SemanticClaimVerifier(DeterministicEmbeddingProvider()).verify(
        answer="Orders can be cancelled before courier handoff. [1]",
        citations=[Citation(index=1, chunk_id="orders", source="orders.md")],
        hits=hits,
    )

    assert report.passed is True
    assert report.verifier == "local-semantic-claim-verifier"
    assert report.verifier_version == "v1"
