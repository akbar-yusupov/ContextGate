from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PolicyName = Literal["fast", "balanced", "accurate", "auto"]
FixedPolicyName = Literal["fast", "balanced", "accurate"]


def default_benchmark_policies() -> list[FixedPolicyName]:
    return ["fast", "balanced", "accurate"]


@dataclass(slots=True, frozen=True)
class RetrievalFilter:
    document_ids: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    source: str
    text: str
    language: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RouteDecision:
    requested_policy: PolicyName
    selected_policy: FixedPolicyName
    reason: str
    predicted_quality: dict[str, float] = field(default_factory=dict)
    latency_budget_ms: float = 1000
    router_version: str | None = None
    out_of_distribution: bool = False


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    query: str
    policy: str
    abstained: bool
    hits: list[RetrievalHit]
    route: RouteDecision
    timings_ms: dict[str, float]
    features: dict[str, float | int | str]
    trace_id: str
    raw_top_score: float | None
    abstention_threshold: float
