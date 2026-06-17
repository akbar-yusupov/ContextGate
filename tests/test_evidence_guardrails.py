from __future__ import annotations

from contextgate.domain.evidence import score_evidence


def test_unsupported_claim_detector_flags_absent_terms() -> None:
    evidence = score_evidence(
        query="Can I cancel my order?",
        answer="You can cancel the order and receive an instant teleport refund.",
        contexts=["Orders can be cancelled before courier handoff."],
        abstained=False,
    )

    assert evidence.unsupported_claims
    assert "teleport" in evidence.unsupported_claims
