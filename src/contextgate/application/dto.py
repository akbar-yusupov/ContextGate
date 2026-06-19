from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextgate.domain.retrieval import FixedPolicyName, PolicyName, RetrievalFilter


@dataclass(slots=True, frozen=True)
class KnowledgeBaseCreate:
    name: str
    slug: str
    description: str = ""


@dataclass(slots=True, frozen=True)
class RetrieveCommand:
    knowledge_base: str
    query: str
    policy: PolicyName = "auto"
    latency_budget_ms: float = 1000
    cost_budget_usd: float | None = None
    max_context_tokens: int = 4096
    allowed_providers: list[str] = field(default_factory=list)
    limit: int = 10
    filters: RetrievalFilter | None = None
    debug: bool = False
    request_id: str | None = None


@dataclass(slots=True, frozen=True)
class AnswerCommand(RetrieveCommand):
    gateway_policy_id: str | None = None
    system_prompt: str | None = None
    llm_provider: str | None = None
    stream_mode: str = "none"
    deadline_monotonic: float | None = None


@dataclass(slots=True, frozen=True)
class BenchmarkCommand:
    knowledge_base: str
    dataset_path: str
    policies: list[FixedPolicyName] = field(
        default_factory=lambda: ["fast", "balanced", "accurate"]
    )
    evaluate_answers: bool = False


@dataclass(slots=True, frozen=True)
class RouterTrainCommand:
    benchmark_run_id: str
    knowledge_base: str


@dataclass(slots=True, frozen=True)
class RouterPromoteCommand:
    run_id: str
    knowledge_base: str


@dataclass(slots=True, frozen=True)
class SyncQdrantCommand:
    source_collection: str


@dataclass(slots=True, frozen=True)
class PolicyCreateCommand:
    name: str
    description: str = ""
    retrieval_policy: FixedPolicyName = "balanced"
    provider_policy: str = "extractive"
    latency_budget_ms: float = 1000
    cost_budget_usd: float | None = None


@dataclass(slots=True, frozen=True)
class IngestDocumentCommand:
    knowledge_base: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
