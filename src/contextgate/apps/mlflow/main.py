from __future__ import annotations

import os
import time
from urllib.request import urlopen

from sqlalchemy import create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect

from contextgate.config import Settings, get_settings

REQUIRED_MLFLOW_TABLES = frozenset(
    {"alembic_version", "experiments", "online_scoring_configs", "registered_models"}
)


def build_mlflow_server_args(settings: Settings) -> list[str]:
    return [
        "mlflow",
        "server",
        "--host",
        settings.mlflow_server_host,
        "--port",
        str(settings.mlflow_server_port),
        "--backend-store-uri",
        settings.resolved_mlflow_backend_store_uri,
        "--registry-store-uri",
        settings.resolved_mlflow_registry_store_uri,
        "--default-artifact-root",
        settings.mlflow_artifact_root,
        "--workers",
        str(settings.mlflow_workers),
        "--allowed-hosts",
        settings.mlflow_allowed_hosts,
        "--cors-allowed-origins",
        settings.mlflow_cors_allowed_origins,
    ]


def configure_warning_filters(settings: Settings) -> None:
    if settings.mlflow_suppress_upstream_warnings:
        existing_warnings = os.environ.get("PYTHONWARNINGS")
        upstream_filter = "ignore:starlette.middleware.wsgi is deprecated"
        os.environ["PYTHONWARNINGS"] = (
            f"{existing_warnings},{upstream_filter}" if existing_warnings else upstream_filter
        )


def check_mlflow_database(settings: Settings) -> None:
    engine = create_engine(settings.resolved_mlflow_backend_store_uri, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        engine.dispose()


def validate_mlflow_schema(table_names: set[str]) -> None:
    missing = sorted(REQUIRED_MLFLOW_TABLES - table_names)
    if missing:
        raise RuntimeError(
            "MLflow database schema is not initialized; missing tables: " + ", ".join(missing)
        )


def check_mlflow_schema(settings: Settings) -> None:
    engine = create_engine(settings.resolved_mlflow_backend_store_uri, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            validate_mlflow_schema(set(sqlalchemy_inspect(connection).get_table_names()))
    finally:
        engine.dispose()


def wait_for_mlflow_database(
    settings: Settings,
    *,
    attempts: int = 30,
    interval_seconds: float = 2,
) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            check_mlflow_database(settings)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(interval_seconds)
    raise RuntimeError(
        "MLflow cannot reach PostgreSQL at "
        f"{settings.database_host}:{settings.database_port}/{settings.mlflow_database_name}. "
        "Start MLflow and PostgreSQL with the same Docker Compose project/profile, or correct "
        "CONTEXTGATE_DATABASE_HOST and network settings."
    ) from last_error


def healthcheck(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    check_mlflow_database(settings)
    check_mlflow_schema(settings)
    tracking_uri = settings.resolved_mlflow_tracking_uri.rstrip("/")
    if not tracking_uri.startswith(("http://", "https://")):
        raise RuntimeError(f"MLflow HTTP tracking URI is not configured: {tracking_uri}")
    with urlopen(
        f"{tracking_uri}/health",
        timeout=5,
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"MLflow HTTP health returned {response.status}")


def main() -> None:
    settings = get_settings()
    configure_warning_filters(settings)
    wait_for_mlflow_database(settings)
    args = build_mlflow_server_args(settings)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
