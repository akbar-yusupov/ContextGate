from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class RetrievalPolicyName(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    AUTO = "auto"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_ERRORS = "succeeded_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class KnowledgeBase:
    id: str
    slug: str
    name: str
    collection_name: str
    description: str = ""
    corpus_version: int = 0


@dataclass(slots=True, frozen=True)
class Document:
    id: str
    knowledge_base_id: str
    source: str
    external_id: str
    content_hash: str
    pipeline_version: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Query:
    text: str
    knowledge_base: str
    policy: RetrievalPolicyName = RetrievalPolicyName.AUTO
    latency_budget_ms: float = 1000
    cost_budget_usd: float | None = None
    max_context_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetrievalPolicy:
    name: RetrievalPolicyName
    output_limit: int
    use_late_interaction: bool
    use_cross_encoder: bool
    abstention_threshold: float


@dataclass(slots=True, frozen=True)
class EvidenceSet:
    answerability_score: float
    coverage_score: float
    support_score: float
    unsupported_claims: tuple[str, ...] = ()
    rejected_claims: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return min(self.answerability_score, self.coverage_score, self.support_score)


@dataclass(slots=True, frozen=True)
class AnswerDraft:
    text: str
    citations: tuple[str, ...]
    provider: str
    grounded: bool


@dataclass(slots=True, frozen=True)
class GatewayRun:
    id: str
    trace_id: str
    knowledge_base: str
    query: str
    selected_retrieval_policy: str
    selected_provider: str
    evidence: EvidenceSet
    abstained: bool
    created_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class ProviderCall:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    estimated_cost_usd: float = 0


@dataclass(slots=True, frozen=True)
class CostRecord:
    request_id: str
    run_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    embedding_units: int
    rerank_units: int
    estimated_cost_usd: float


@dataclass(slots=True, frozen=True)
class EvaluationRun:
    id: str
    dataset_path: str
    metrics: dict[str, float]
    report_path: str


@dataclass(slots=True, frozen=True)
class RouterVersion:
    run_id: str
    artifact_path: str
    status: str
    metrics: dict[str, Any]
