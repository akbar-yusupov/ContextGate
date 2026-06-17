from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from contextgate.application import dto
from contextgate.domain.gateway import AnswerResult
from contextgate.domain.gateway import Citation as DomainCitation
from contextgate.domain.retrieval import (
    RetrievalFilter as DomainRetrievalFilter,
)
from contextgate.domain.retrieval import (
    RetrievalHit as DomainRetrievalHit,
)
from contextgate.domain.retrieval import (
    RetrievalResult,
)
from contextgate.domain.retrieval import (
    RouteDecision as DomainRouteDecision,
)

PolicyName = Literal["fast", "balanced", "accurate", "auto"]
FixedPolicyName = Literal["fast", "balanced", "accurate"]


def default_benchmark_policies() -> list[FixedPolicyName]:
    return ["fast", "balanced", "accurate"]


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    description: str = ""

    def to_command(self) -> dto.KnowledgeBaseCreate:
        return dto.KnowledgeBaseCreate(
            name=self.name,
            slug=self.slug,
            description=self.description,
        )


class KnowledgeBaseRead(KnowledgeBaseCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    collection_name: str
    created_at: datetime


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    progress: float
    error: str | None
    error_json: dict[str, Any] | None = None
    result: dict[str, Any] | None
    retry_count: int = 0
    idempotency_key: str | None = None


class RetrievalFilter(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def to_domain(self) -> DomainRetrievalFilter:
        return DomainRetrievalFilter(
            document_ids=self.document_ids,
            languages=self.languages,
            sources=self.sources,
            metadata=self.metadata,
        )


class RetrieveRequest(BaseModel):
    knowledge_base: str
    query: str = Field(min_length=1)
    policy: PolicyName = "auto"
    latency_budget_ms: float = Field(default=1000, gt=0)
    cost_budget_usd: float | None = Field(default=None, ge=0)
    max_context_tokens: int = Field(default=4096, ge=256, le=128_000)
    allowed_providers: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    filters: RetrievalFilter | None = None
    debug: bool = False
    request_id: str | None = None

    def to_command(self) -> dto.RetrieveCommand:
        return dto.RetrieveCommand(
            knowledge_base=self.knowledge_base,
            query=self.query,
            policy=self.policy,
            latency_budget_ms=self.latency_budget_ms,
            cost_budget_usd=self.cost_budget_usd,
            max_context_tokens=self.max_context_tokens,
            allowed_providers=self.allowed_providers,
            limit=self.limit,
            filters=self.filters.to_domain() if self.filters else None,
            debug=self.debug,
            request_id=self.request_id,
        )


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    text: str
    language: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, hit: DomainRetrievalHit) -> RetrievalHit:
        return cls(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            source=hit.source,
            text=hit.text,
            language=hit.language,
            score=hit.score,
            rank=hit.rank,
            metadata=hit.metadata,
        )


class RouteDecision(BaseModel):
    requested_policy: PolicyName
    selected_policy: Literal["fast", "balanced", "accurate"]
    reason: str
    predicted_quality: dict[str, float] = Field(default_factory=dict)
    latency_budget_ms: float
    router_version: str | None = None
    out_of_distribution: bool = False

    @classmethod
    def from_domain(cls, decision: DomainRouteDecision) -> RouteDecision:
        return cls(
            requested_policy=decision.requested_policy,
            selected_policy=decision.selected_policy,
            reason=decision.reason,
            predicted_quality=decision.predicted_quality,
            latency_budget_ms=decision.latency_budget_ms,
            router_version=decision.router_version,
            out_of_distribution=decision.out_of_distribution,
        )


class RetrieveResponse(BaseModel):
    query: str
    policy: str
    abstained: bool
    hits: list[RetrievalHit]
    route: RouteDecision
    timings_ms: dict[str, float]
    features: dict[str, float | int | str]
    trace_id: str
    raw_top_score: float | None = None
    abstention_threshold: float

    @classmethod
    def from_domain(cls, result: RetrievalResult) -> RetrieveResponse:
        return cls(
            query=result.query,
            policy=result.policy,
            abstained=result.abstained,
            hits=[RetrievalHit.from_domain(hit) for hit in result.hits],
            route=RouteDecision.from_domain(result.route),
            timings_ms=result.timings_ms,
            features=result.features,
            trace_id=result.trace_id,
            raw_top_score=result.raw_top_score,
            abstention_threshold=result.abstention_threshold,
        )


class AnswerRequest(RetrieveRequest):
    system_prompt: str | None = None
    llm_provider: str | None = None

    def to_command(self) -> dto.AnswerCommand:
        base = super().to_command()
        return dto.AnswerCommand(
            knowledge_base=base.knowledge_base,
            query=base.query,
            policy=base.policy,
            latency_budget_ms=base.latency_budget_ms,
            cost_budget_usd=base.cost_budget_usd,
            max_context_tokens=base.max_context_tokens,
            allowed_providers=base.allowed_providers,
            limit=base.limit,
            filters=base.filters,
            debug=base.debug,
            request_id=base.request_id,
            system_prompt=self.system_prompt,
            llm_provider=self.llm_provider,
        )


class Citation(BaseModel):
    index: int
    chunk_id: str
    source: str

    @classmethod
    def from_domain(cls, citation: DomainCitation) -> Citation:
        return cls(index=citation.index, chunk_id=citation.chunk_id, source=citation.source)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval: RetrieveResponse
    provider: str
    grounded: bool
    run_id: str | None = None
    selected_provider: str = "extractive"
    evidence_score: float = 0.0
    answerability_score: float = 0.0
    coverage_score: float = 0.0
    support_score: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    cost: dict[str, float] = Field(default_factory=lambda: {"estimated_usd": 0.0})

    @classmethod
    def from_domain(cls, result: AnswerResult) -> AnswerResponse:
        return cls(
            answer=result.answer,
            citations=[Citation.from_domain(citation) for citation in result.citations],
            retrieval=RetrieveResponse.from_domain(result.retrieval),
            provider=result.provider,
            grounded=result.grounded,
            run_id=result.run_id,
            selected_provider=result.selected_provider,
            evidence_score=result.evidence_score,
            answerability_score=result.answerability_score,
            coverage_score=result.coverage_score,
            support_score=result.support_score,
            unsupported_claims=result.unsupported_claims,
            rejected_claims=result.rejected_claims,
            abstention_reason=result.abstention_reason,
            cost=result.cost,
        )


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ProviderTestRequest(BaseModel):
    provider: str | None = None


class PolicyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    retrieval_policy: FixedPolicyName = "balanced"
    provider_policy: str = "extractive"
    latency_budget_ms: float = Field(default=1000, gt=0)
    cost_budget_usd: float | None = Field(default=None, ge=0)

    def to_command(self) -> dto.PolicyCreateCommand:
        return dto.PolicyCreateCommand(
            name=self.name,
            description=self.description,
            retrieval_policy=self.retrieval_policy,
            provider_policy=self.provider_policy,
            latency_budget_ms=self.latency_budget_ms,
            cost_budget_usd=self.cost_budget_usd,
        )


class PolicyRead(PolicyCreateRequest):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str = "draft"


class BenchmarkQuery(BaseModel):
    id: str
    group_id: str | None = None
    query: str
    language: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    answerable: bool = True
    tags: list[str] = Field(default_factory=list)


class BenchmarkRequest(BaseModel):
    knowledge_base: str
    dataset_path: str
    policies: list[FixedPolicyName] = Field(default_factory=default_benchmark_policies)
    evaluate_answers: bool = False

    def to_command(self) -> dto.BenchmarkCommand:
        return dto.BenchmarkCommand(
            knowledge_base=self.knowledge_base,
            dataset_path=self.dataset_path,
            policies=self.policies,
            evaluate_answers=self.evaluate_answers,
        )


class RouterTrainRequest(BaseModel):
    benchmark_run_id: str
    knowledge_base: str

    def to_command(self) -> dto.RouterTrainCommand:
        return dto.RouterTrainCommand(
            benchmark_run_id=self.benchmark_run_id,
            knowledge_base=self.knowledge_base,
        )


class RouterPromoteRequest(BaseModel):
    run_id: str
    knowledge_base: str

    def to_command(self) -> dto.RouterPromoteCommand:
        return dto.RouterPromoteCommand(run_id=self.run_id, knowledge_base=self.knowledge_base)


class SyncQdrantRequest(BaseModel):
    source_collection: str

    def to_command(self) -> dto.SyncQdrantCommand:
        return dto.SyncQdrantCommand(source_collection=self.source_collection)
