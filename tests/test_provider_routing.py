from __future__ import annotations

import pytest

from contextgate.adapters.litellm.providers import ProviderRegistry
from contextgate.domain.errors import ContextGateError


def test_requested_provider_must_be_available_and_allowed() -> None:
    registry = ProviderRegistry(default_model="openai/gpt-4o-mini")

    selected = registry.choose(
        cost_budget_usd=1.0,
        latency_budget_ms=1000,
        allowed_providers=["openai/gpt-4o-mini"],
        requested_provider="openai/gpt-4o-mini",
    )

    assert selected == "openai/gpt-4o-mini"


def test_zero_cost_budget_uses_extractive_when_allowed() -> None:
    registry = ProviderRegistry(default_model="openai/gpt-4o-mini")

    selected = registry.choose(
        cost_budget_usd=0,
        latency_budget_ms=1000,
        allowed_providers=["extractive"],
    )

    assert selected == "extractive"


def test_unavailable_provider_returns_typed_error() -> None:
    registry = ProviderRegistry(default_model=None)

    with pytest.raises(ContextGateError) as exc:
        registry.choose(
            cost_budget_usd=1.0,
            latency_budget_ms=1000,
            allowed_providers=["extractive"],
            requested_provider="openai/gpt-4o-mini",
        )

    assert exc.value.code == "provider_unavailable"
