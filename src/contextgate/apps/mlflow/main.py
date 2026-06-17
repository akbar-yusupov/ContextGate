from __future__ import annotations

import os

from contextgate.config import Settings, get_settings


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


def main() -> None:
    settings = get_settings()
    configure_warning_filters(settings)
    args = build_mlflow_server_args(settings)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
