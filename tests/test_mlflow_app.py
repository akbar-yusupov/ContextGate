from __future__ import annotations

import contextgate.apps.mlflow.main as mlflow_app
from contextgate.apps.mlflow.main import build_mlflow_server_args, validate_mlflow_schema
from contextgate.config import Settings


def test_mlflow_server_args_are_explicit_for_local_compose() -> None:
    settings = Settings(
        database_backend="postgres",
        database_host="postgres",
        database_name="contextgate",
        database_user="contextgate",
        database_password="secret",
        mlflow_database_name="mlflow",
        mlflow_server_host="0.0.0.0",
        mlflow_server_port=5000,
        mlflow_workers=1,
        mlflow_allowed_hosts="localhost,127.0.0.1,mlflow",
        mlflow_cors_allowed_origins="http://localhost:5000",
    )

    args = build_mlflow_server_args(settings)

    assert args[0:2] == ["mlflow", "server"]
    assert args[args.index("--workers") + 1] == "1"
    assert args[args.index("--backend-store-uri") + 1] == (
        "postgresql+psycopg://contextgate:secret@postgres:5432/mlflow"
    )
    assert args[args.index("--registry-store-uri") + 1] == (
        "postgresql+psycopg://contextgate:secret@postgres:5432/mlflow"
    )
    assert args[args.index("--allowed-hosts") + 1] == "localhost,127.0.0.1,mlflow"
    assert args[args.index("--cors-allowed-origins") + 1] == "http://localhost:5000"


def test_mlflow_database_wait_retries_transient_dns_failure(monkeypatch) -> None:
    attempts = 0

    def check(_settings) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary DNS failure")

    monkeypatch.setattr(mlflow_app, "check_mlflow_database", check)
    monkeypatch.setattr(mlflow_app.time, "sleep", lambda _: None)

    mlflow_app.wait_for_mlflow_database(Settings(), attempts=3, interval_seconds=0)

    assert attempts == 3


def test_mlflow_schema_requires_tracking_registry_and_scheduler_tables() -> None:
    validate_mlflow_schema(
        {"alembic_version", "experiments", "online_scoring_configs", "registered_models"}
    )

    try:
        validate_mlflow_schema({"alembic_version"})
    except RuntimeError as exc:
        assert "experiments" in str(exc)
        assert "online_scoring_configs" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Incomplete MLflow schema must fail readiness")
