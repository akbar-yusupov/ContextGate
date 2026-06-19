from __future__ import annotations

import math
from dataclasses import replace

from contextgate.adapters.fastembed.embeddings import EmbeddingProvider
from contextgate.domain.evidence import build_evidence_report
from contextgate.domain.gateway import AbstentionReason, Citation, EvidenceReport
from contextgate.domain.retrieval import RetrievalHit


class DeterministicClaimVerifier:
    """Offline verifier with deterministic claim/citation behavior.

    This is the no-download baseline. The verifier port allows a multilingual NLI or
    structured judge implementation to replace it without changing admission logic.
    """

    name = "deterministic-claim-verifier"
    version = "v1"

    def verify(
        self,
        *,
        answer: str,
        citations: list[Citation],
        hits: list[RetrievalHit],
    ) -> EvidenceReport:
        return build_evidence_report(
            answer=answer,
            citations=citations,
            hits=hits,
            require_citations=True,
        )


class SemanticClaimVerifier:
    """Local multilingual semantic verifier backed by the configured embedding model."""

    name = "local-semantic-claim-verifier"
    version = "v1"

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder

    def verify(
        self,
        *,
        answer: str,
        citations: list[Citation],
        hits: list[RetrievalHit],
    ) -> EvidenceReport:
        baseline = build_evidence_report(
            answer=answer,
            citations=citations,
            hits=hits,
            require_citations=True,
        )
        if not baseline.claims:
            return replace(
                baseline,
                verifier=self.name,
                verifier_version=self.version,
            )
        hits_by_id = {hit.chunk_id: hit for hit in hits}
        verified_claims = []
        for claim in baseline.claims:
            evidence = " ".join(
                hits_by_id[chunk_id].text
                for chunk_id in claim.supporting_chunk_ids
                if chunk_id in hits_by_id
            )
            semantic = self._cosine(
                self.embedder.dense_query(claim.claim),
                self.embedder.dense_documents([evidence])[0] if evidence else [],
            )
            entailment = 0.5 * claim.entailment_score + 0.5 * max(0.0, semantic)
            supported = (
                bool(claim.citation_indices)
                and entailment >= 0.55
                and claim.contradiction_score < 0.5
            )
            verified_claims.append(
                replace(
                    claim,
                    entailment_score=entailment,
                    status="supported" if supported else "unsupported",
                )
            )
        passed = all(claim.status == "supported" for claim in verified_claims)
        return replace(
            baseline,
            verifier=self.name,
            verifier_version=self.version,
            claims=tuple(verified_claims),
            passed=passed,
            score=min((claim.entailment_score for claim in verified_claims), default=0.0),
            reason=None if passed else baseline.reason or AbstentionReason.UNSUPPORTED_CLAIM,
        )

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        return numerator / denominator if denominator else 0.0
