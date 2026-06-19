from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from contextgate.domain.errors import ContextGateError


class ProviderRegistry:
    def __init__(
        self,
        default_model: str | None = None,
        *,
        input_cost_per_1m_tokens: float | None = None,
        output_cost_per_1m_tokens: float | None = None,
        max_output_tokens: int = 512,
    ) -> None:
        self.default_model = default_model
        self.input_cost_per_1m_tokens = input_cost_per_1m_tokens
        self.output_cost_per_1m_tokens = output_cost_per_1m_tokens
        self.max_output_tokens = max_output_tokens

    def list(self) -> list[dict[str, Any]]:
        providers = [
            {
                "id": "extractive",
                "kind": "fallback",
                "available": True,
                "cost_per_1k_tokens": 0.0,
            }
        ]
        if self.default_model:
            providers.append(
                {
                    "id": self.default_model,
                    "kind": "litellm",
                    "available": True,
                    "input_cost_per_1m_tokens": self.input_cost_per_1m_tokens,
                    "output_cost_per_1m_tokens": self.output_cost_per_1m_tokens,
                }
            )
        if os.getenv("OLLAMA_HOST"):
            providers.append(
                {
                    "id": "ollama",
                    "kind": "local",
                    "available": True,
                    "cost_per_1k_tokens": 0.0,
                }
            )
        return providers

    def test(self, provider: str | None = None) -> dict[str, Any]:
        available = {item["id"]: item for item in self.list()}
        selected = provider or next(iter(available))
        return {
            "provider": selected,
            "available": selected in available,
            "details": available.get(selected, {"reason": "unknown_provider"}),
        }

    def choose(
        self,
        *,
        cost_budget_usd: float | None,
        latency_budget_ms: float,
        allowed_providers: Sequence[str] | None = None,
        requested_provider: str | None = None,
        max_context_tokens: int = 4096,
    ) -> str:
        available = {item["id"]: item for item in self.list()}
        allowed = set(allowed_providers or available)

        if requested_provider:
            if requested_provider in available and requested_provider in allowed:
                self._enforce_budget(requested_provider, cost_budget_usd, max_context_tokens)
                return requested_provider
            raise ContextGateError(
                "provider_unavailable",
                f"Provider is not available or allowed: {requested_provider}",
                {"allowed_providers": sorted(allowed), "available_providers": sorted(available)},
            )

        if cost_budget_usd is not None and cost_budget_usd <= 0 and "extractive" in allowed:
            return "extractive"

        if (
            self.default_model
            and latency_budget_ms >= 250
            and self.default_model in allowed
            and self._fits_budget(cost_budget_usd, max_context_tokens)
        ):
            return self.default_model

        if "extractive" in allowed:
            return "extractive"

        candidates = [provider for provider in available if provider in allowed]
        if candidates:
            return candidates[0]

        raise ContextGateError(
            "budget_exceeded" if cost_budget_usd is not None else "provider_unavailable",
            "No configured provider satisfies the request policy and hard budget.",
            {"allowed_providers": sorted(allowed), "available_providers": sorted(available)},
        )

    def projected_cost(self, max_context_tokens: int) -> float | None:
        if self.input_cost_per_1m_tokens is None or self.output_cost_per_1m_tokens is None:
            return None
        return (
            max_context_tokens * self.input_cost_per_1m_tokens
            + self.max_output_tokens * self.output_cost_per_1m_tokens
        ) / 1_000_000

    def _fits_budget(self, budget: float | None, max_context_tokens: int) -> bool:
        if budget is None:
            return True
        projected = self.projected_cost(max_context_tokens)
        return projected is not None and projected <= budget

    def _enforce_budget(self, provider: str, budget: float | None, max_context_tokens: int) -> None:
        if provider == "extractive" or budget is None:
            return
        projected = self.projected_cost(max_context_tokens)
        if projected is None or projected > budget:
            raise ContextGateError(
                "budget_exceeded",
                "Requested provider has unknown pricing or exceeds the hard cost budget.",
                {"provider": provider, "projected_usd": projected, "budget_usd": budget},
            )
