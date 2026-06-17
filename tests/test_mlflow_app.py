from __future__ import annotations

from contextgate.apps.mlflow.main import build_mlflow_server_args
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
