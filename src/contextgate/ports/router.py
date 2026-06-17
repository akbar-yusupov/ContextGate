from __future__ import annotations

from typing import Protocol

from contextgate.domain.retrieval import RouteDecision


class RouterRepository(Protocol):
    def decide(
        self,
        knowledge_base: str,
        features: dict[str, float | int | str],
        latency_budget_ms: float,
    ) -> RouteDecision: ...

    def abstention_threshold(
        self,
        knowledge_base: str,
        policy: str,
        *,
        fallback: float,
    ) -> float: ...

    def train(self, benchmark_run_id: str, knowledge_base: str) -> dict: ...

    def promote(self, benchmark_run_id: str, knowledge_base: str): ...
