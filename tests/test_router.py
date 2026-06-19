import json
from pathlib import Path

import numpy as np
import pytest

from contextgate.adapters.mlflow.router_registry import (
    RouterManager,
    _registry_version,
    calibrate_abstention_threshold,
)
from contextgate.config import Settings


def test_registry_client_uses_dedicated_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, *, tracking_uri: str, registry_uri: str) -> None:
            captured["tracking_uri"] = tracking_uri
            captured["registry_uri"] = registry_uri

    monkeypatch.setattr(
        "contextgate.adapters.mlflow.router_registry.MlflowClient",
        FakeClient,
    )
    settings = Settings(
        environment="test",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_registry_store_uri="postgresql+psycopg://user:pass@postgres:5432/mlflow",
        report_dir=tmp_path / "reports",
        router_dir=tmp_path / "routers",
    )

    client = RouterManager(settings)._registry_client()

    assert isinstance(client, FakeClient)
    assert captured == {
        "tracking_uri": "http://mlflow:5000",
        "registry_uri": "postgresql+psycopg://user:pass@postgres:5432/mlflow",
    }
    assert _registry_version("2") == 2


def _features(index: int) -> dict:
    return {
        "query_token_count": 4 + index,
        "dense_margin": 0.1 + index / 100,
        "sparse_margin": 0.05,
        "dense_entropy": 0.8,
        "sparse_entropy": 0.7,
        "retriever_overlap": index / 20,
        "top1_agreement": float(index % 2),
        "first_stage_latency_ms": 5 + index,
        "language": ("en", "ru", "uz")[index % 3],
    }


def test_router_train_promote_and_select(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        report_dir=tmp_path / "reports",
        router_dir=tmp_path / "routers",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        router_min_release_cases=0,
        router_min_unanswerable_cases=0,
        router_required_languages="",
        router_max_false_answer_upper_95=1,
        router_min_citation_lower_95=0,
        router_min_claim_support_lower_95=0,
    )
    settings.prepare_directories()
    benchmark_id = "benchmark-1"
    result_dir = settings.report_dir / benchmark_id
    result_dir.mkdir(parents=True)
    rows = []
    for index in range(20):
        rows.append(
            {
                "id": f"query-{index}",
                "group_id": f"group-{index // 2}",
                "answerable": index % 5 != 0,
                "probe_latency_ms": 10 + index,
                "features": _features(index),
                "policies": {
                    "fast": {
                        "ndcg_at_10": 0.85,
                        "latency_ms": 40 + index,
                        "top_score": 0.9 if index % 5 else 0.1,
                    },
                    "balanced": {
                        "ndcg_at_10": 0.85,
                        "latency_ms": 100 + index,
                        "top_score": 0.9 if index % 5 else 0.1,
                    },
                    "accurate": {
                        "ndcg_at_10": 0.85,
                        "latency_ms": 400 + index,
                        "top_score": 0.9 if index % 5 else 0.1,
                    },
                },
            }
        )
    payload = {
        "queries": rows,
        "summary": {
            "fast": {"latency_p95_ms": 50},
            "balanced": {"latency_p95_ms": 150},
            "accurate": {"latency_p95_ms": 400},
        },
        "gateway_evaluation": {"cases": [], "summary": {"overall": {}, "by_policy": {}}},
    }
    (result_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    manager = RouterManager(settings)

    trained = manager.train(benchmark_id, "demo")
    manager.promote(benchmark_id, "demo")
    features = _features(5)
    decision = manager.decide("demo", features, latency_budget_ms=200)

    assert trained["artifact_path"].endswith("router.skops")
    assert trained["eligible_for_promotion"] is True
    assert decision.selected_policy == "fast"
    assert decision.reason == "quality_equivalent_fastest_within_slo"
    assert float(features["router_overhead_ms"]) >= 0


def test_abstention_threshold_is_calibrated_per_score_domain() -> None:
    threshold = calibrate_abstention_threshold(
        np.asarray([0.05, 0.10, 0.70, 0.90]),
        np.asarray([False, False, True, True]),
        fallback=0.2,
    )

    assert 0.10 < threshold < 0.70
