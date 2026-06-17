from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from contextgate.domain.retrieval import RetrievalResult


class AbstentionReason(StrEnum):
    RETRIEVAL_EMPTY = "retrieval_empty"
    LOW_SUPPORT = "low_support"
    LOW_COVERAGE = "low_coverage"
    INVALID_CITATIONS = "invalid_citations"
    BUDGET_EXCEEDED = "budget_exceeded"


class EvidenceDecision(StrEnum):
    ALLOW_GENERATION = "allow_generation"
    ABSTAIN = "abstain"


@dataclass(slots=True, frozen=True)
class Citation:
    index: int
    chunk_id: str
    source: str


@dataclass(slots=True, frozen=True)
class CitationValidation:
    valid: bool
    reason: AbstentionReason | None = None


@dataclass(slots=True, frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieval: RetrievalResult
    provider: str
    grounded: bool
    run_id: str | None = None
    selected_provider: str = "extractive"
    evidence_score: float = 0.0
    answerability_score: float = 0.0
    coverage_score: float = 0.0
    support_score: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    rejected_claims: list[str] = field(default_factory=list)
    abstention_reason: AbstentionReason | None = None
    cost: dict[str, float] = field(default_factory=lambda: {"estimated_usd": 0.0})
