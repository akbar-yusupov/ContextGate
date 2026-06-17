from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from contextgate.domain.errors import ContextGateError


class ProviderRegistry:
    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model

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
                    "cost_per_1k_tokens": None,
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
    ) -> str:
        available = {item["id"]: item for item in self.list()}
        allowed = set(allowed_providers or available)

        if requested_provider:
            if requested_provider in available and requested_provider in allowed:
                return requested_provider
            raise ContextGateError(
                "provider_unavailable",
                f"Provider is not available or allowed: {requested_provider}",
                {"allowed_providers": sorted(allowed), "available_providers": sorted(available)},
            )

        if cost_budget_usd is not None and cost_budget_usd <= 0 and "extractive" in allowed:
            return "extractive"

        if self.default_model and latency_budget_ms >= 250 and self.default_model in allowed:
            return self.default_model

        if "extractive" in allowed:
            return "extractive"

        candidates = [provider for provider in available if provider in allowed]
        if candidates:
            return candidates[0]

        raise ContextGateError(
            "provider_unavailable",
            "No configured provider satisfies the request policy.",
            {"allowed_providers": sorted(allowed), "available_providers": sorted(available)},
        )
        return "extractive"
