from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from contextgate.domain.retrieval import RetrievalResult


class AbstentionReason(StrEnum):
    RETRIEVAL_EMPTY = "retrieval_empty"
    LOW_ANSWERABILITY = "low_answerability"
    LOW_SUPPORT = "low_support"
    LOW_COVERAGE = "low_coverage"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    INVALID_CITATIONS = "invalid_citations"
    UNSAFE_QUERY = "unsafe_query"
    UNSAFE_CONTEXT = "unsafe_context"
    BUDGET_EXCEEDED = "budget_exceeded"
    LATENCY_BUDGET_EXCEEDED = "latency_budget_exceeded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


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
class ClaimEvidence:
    claim: str
    citation_indices: tuple[int, ...]
    supporting_chunk_ids: tuple[str, ...]
    entailment_score: float
    contradiction_score: float
    status: str


@dataclass(slots=True, frozen=True)
class EvidenceReport:
    verifier: str
    verifier_version: str
    claims: tuple[ClaimEvidence, ...] = ()
    passed: bool = False
    score: float = 0.0
    reason: AbstentionReason | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False


@dataclass(slots=True, frozen=True)
class RiskReport:
    score: float
    blocked: bool
    reason: AbstentionReason | None = None
    matched_rules: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieval: RetrievalResult
    provider: str
    grounded: bool
    status: AnswerStatus = AnswerStatus.ANSWERED
    run_id: str | None = None
    selected_provider: str = "extractive"
    evidence_score: float = 0.0
    answerability_score: float = 0.0
    coverage_score: float = 0.0
    support_score: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    rejected_claims: list[str] = field(default_factory=list)
    abstention_reason: AbstentionReason | None = None
    evidence_report: EvidenceReport | None = None
    risk_report: RiskReport | None = None
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=lambda: {"estimated_usd": 0.0})
