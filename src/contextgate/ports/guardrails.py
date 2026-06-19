from __future__ import annotations

from typing import Protocol

from contextgate.domain.gateway import Citation, EvidenceReport, RiskReport
from contextgate.domain.retrieval import RetrievalHit


class ClaimVerifier(Protocol):
    def verify(
        self,
        *,
        answer: str,
        citations: list[Citation],
        hits: list[RetrievalHit],
    ) -> EvidenceReport: ...


class RiskPolicy(Protocol):
    version: str

    def assess_query(self, text: str) -> RiskReport: ...

    def assess_contexts(self, contexts: list[str]) -> RiskReport: ...
