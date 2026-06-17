from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_mlflow_service_or_database_uri(value: str) -> bool:
    return "://" in value and not value.startswith("file:")


def _mlflow_local_path(value: str) -> Path:
    if value.startswith("file:"):
        return Path(value.removeprefix("file:"))
    return Path(value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTEXTGATE_",
        extra="ignore",
    )

    environment: str = "development"

    database_url: str | None = None
    database_backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: Path = Path("./.contextgate/contextgate.db")
    database_driver: str = "postgresql+psycopg"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "contextgate"
    database_user: str = "contextgate"
    database_password: str = "contextgate-dev-password"

    qdrant_url: str | None = None
    qdrant_host: str | None = None
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_local_path: Path = Path("./.contextgate/qdrant")

    redis_url: str | None = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    mlflow_tracking_uri: str | None = None
    mlflow_tracking_path: Path = Path("./mlruns")
    mlflow_tracking_scheme: Literal["http", "https"] = "http"
    mlflow_tracking_host: str | None = None
    mlflow_tracking_port: int = 5000
    mlflow_database_name: str = "mlflow"
    mlflow_registry_store_uri: str | None = None
    mlflow_artifact_root: str = "/mlartifacts"
    mlflow_server_host: str = "0.0.0.0"
    mlflow_server_port: int = 5000
    mlflow_workers: int = 1
    mlflow_allowed_hosts: str = (
        "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,"
        "mlflow,mlflow:5000,host.docker.internal,host.docker.internal:5000"
    )
    mlflow_cors_allowed_origins: str = (
        "http://localhost:5000,http://127.0.0.1:5000,"
        "http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:8001,http://127.0.0.1:8001"
    )
    mlflow_suppress_upstream_warnings: bool = True

    ui_api_url: str = "http://localhost:8000"
    api_key: str = "contextgate-dev-key"
    auth_enabled: bool = False
    rate_limit_per_minute: int = 120

    policies_path: Path = Path("configs/policies.yaml")
    upload_dir: Path = Path("./data/uploads")
    report_dir: Path = Path("./reports")
    router_dir: Path = Path("./data/routers")
    pipeline_version: str = "v1"

    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dense_dimension: int = 384
    sparse_model: str = "Qdrant/bm25"
    late_model: str = "answerdotai/answerai-colbert-small-v1"
    late_dimension: int = 96
    late_interaction_languages: str = "en"
    cross_encoder_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 16
    embedding_backend: Literal["fastembed", "deterministic"] = "fastembed"
    qdrant_strict_mode: bool = True
    indexed_metadata_fields: dict[str, Literal["keyword", "integer", "float", "bool"]] = Field(
        default_factory=dict
    )

    llm_model: str | None = None
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    default_knowledge_base: str = "demo"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.database_backend == "sqlite":
            return f"sqlite:///{self.sqlite_path.as_posix()}"
        user = quote(self.database_user, safe="")
        password = quote(self.database_password, safe="")
        return (
            f"{self.database_driver}://{user}:{password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def resolved_redis_url(self) -> str:
        if self.redis_url:
            return self.redis_url
        auth = f":{quote(self.redis_password, safe='')}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def resolved_qdrant_url(self) -> str | None:
        if self.qdrant_url:
            return self.qdrant_url
        if self.qdrant_host:
            return f"http://{self.qdrant_host}:{self.qdrant_port}"
        return None

    @property
    def resolved_mlflow_backend_store_uri(self) -> str:
        user = quote(self.database_user, safe="")
        password = quote(self.database_password, safe="")
        return (
            f"{self.database_driver}://{user}:{password}"
            f"@{self.database_host}:{self.database_port}/{self.mlflow_database_name}"
        )

    @property
    def resolved_mlflow_registry_store_uri(self) -> str:
        return self.mlflow_registry_store_uri or self.resolved_mlflow_backend_store_uri

    @property
    def resolved_mlflow_tracking_uri(self) -> str:
        if self.mlflow_tracking_uri and _is_mlflow_service_or_database_uri(
            self.mlflow_tracking_uri
        ):
            return self.mlflow_tracking_uri
        if self.mlflow_tracking_host:
            return (
                f"{self.mlflow_tracking_scheme}://"
                f"{self.mlflow_tracking_host}:{self.mlflow_tracking_port}"
            )
        directory = _mlflow_local_path(
            self.mlflow_tracking_uri or str(self.mlflow_tracking_path)
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(directory / 'mlflow.db').as_posix()}"

    def supports_late_interaction(self, language: str) -> bool:
        languages = {
            item.strip() for item in self.late_interaction_languages.split(",") if item.strip()
        }
        return "*" in languages or language in languages

    @property
    def indexed_filter_fields(self) -> set[str]:
        return {
            "document_id",
            "source",
            "language",
            "content_hash",
            "pipeline_version",
            *(f"metadata.{name}" for name in self.indexed_metadata_fields),
        }

    def prepare_directories(self) -> None:
        for path in (
            self.qdrant_local_path.parent,
            self.sqlite_path.parent,
            self.upload_dir,
            self.report_dir,
            self.router_dir,
            _mlflow_local_path(self.mlflow_tracking_uri or str(self.mlflow_tracking_path))
            if not self.mlflow_tracking_host
            and not (
                self.mlflow_tracking_uri
                and _is_mlflow_service_or_database_uri(self.mlflow_tracking_uri)
            )
            else Path("."),
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings
