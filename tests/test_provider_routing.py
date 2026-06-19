from __future__ import annotations

import pytest

from contextgate.adapters.litellm.providers import ProviderRegistry
from contextgate.domain.errors import ContextGateError


def test_requested_provider_must_be_available_and_allowed() -> None:
    registry = ProviderRegistry(
        default_model="openai/gpt-4o-mini",
        input_cost_per_1m_tokens=0.15,
        output_cost_per_1m_tokens=0.60,
    )

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


def test_unknown_provider_pricing_is_rejected_under_hard_budget() -> None:
    registry = ProviderRegistry(default_model="openai/gpt-4o-mini")

    with pytest.raises(ContextGateError) as exc:
        registry.choose(
            cost_budget_usd=1.0,
            latency_budget_ms=1000,
            allowed_providers=["openai/gpt-4o-mini"],
            requested_provider="openai/gpt-4o-mini",
        )

    assert exc.value.code == "budget_exceeded"


def test_registry_lists_and_tests_configured_and_local_providers(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    registry = ProviderRegistry(
        default_model="openai/test",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
    )

    assert [item["id"] for item in registry.list()] == ["extractive", "openai/test", "ollama"]
    assert registry.test("openai/test")["available"] is True
    assert registry.test("missing")["details"]["reason"] == "unknown_provider"
    assert registry.projected_cost(1000) == pytest.approx(0.002024)


def test_registry_falls_back_by_latency_and_rejects_empty_allow_list() -> None:
    registry = ProviderRegistry(
        default_model="openai/test",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=1.0,
    )

    assert (
        registry.choose(
            cost_budget_usd=None,
            latency_budget_ms=100,
            allowed_providers=["extractive"],
        )
        == "extractive"
    )
    with pytest.raises(ContextGateError, match="No configured provider") as exc:
        registry.choose(
            cost_budget_usd=None,
            latency_budget_ms=100,
            allowed_providers=["not-configured"],
        )
    assert exc.value.code == "provider_unavailable"


def test_registry_selects_configured_candidates_and_budget_paths() -> None:
    registry = ProviderRegistry(
        default_model="openai/test",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=1.0,
    )

    assert (
        registry.choose(
            cost_budget_usd=None,
            latency_budget_ms=1000,
            allowed_providers=["openai/test"],
        )
        == "openai/test"
    )
    assert (
        registry.choose(
            cost_budget_usd=1.0,
            latency_budget_ms=1000,
            allowed_providers=["openai/test"],
        )
        == "openai/test"
    )
    assert (
        registry.choose(
            cost_budget_usd=None,
            latency_budget_ms=100,
            allowed_providers=["openai/test"],
        )
        == "openai/test"
    )
    assert (
        registry.choose(
            cost_budget_usd=0,
            latency_budget_ms=100,
            allowed_providers=["extractive"],
            requested_provider="extractive",
        )
        == "extractive"
    )
