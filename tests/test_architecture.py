from __future__ import annotations

import ast
from pathlib import Path

from contextgate.adapters.langgraph.checkpointing import create_postgres_checkpointer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "contextgate"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_domain_and_application_do_not_import_infrastructure() -> None:
    forbidden = (
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "qdrant_client",
        "mlflow",
        "celery",
        "chainlit",
        "redis",
        "langgraph",
    )
    violations: list[str] = []
    for package in ("domain", "application", "ports"):
        for path in (SRC / package).rglob("*.py"):
            for module in _imports(path):
                if module.startswith(forbidden):
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []


def test_sqlite_runtime_does_not_create_postgres_checkpointer() -> None:
    assert create_postgres_checkpointer("sqlite:///./contextgate.adapters.sqlalchemy") is None


def test_contextgate_package_root_contains_only_subpackages() -> None:
    root_files = sorted(path.name for path in SRC.iterdir() if path.is_file())

    assert root_files == ["__init__.py"]


def test_alembic_lives_with_sqlalchemy_adapter() -> None:
    sqlalchemy_adapter = SRC / "adapters" / "sqlalchemy"

    assert (sqlalchemy_adapter / "models.py").exists()
    assert (sqlalchemy_adapter / "alembic.ini").exists()
    assert (sqlalchemy_adapter / "migrations" / "env.py").exists()
    assert not (ROOT / "alembic.ini").exists()
    assert not (ROOT / "migrations").exists()


def test_api_main_and_celery_tasks_do_not_reach_behind_application_boundary() -> None:
    forbidden_by_file = {
        SRC / "apps" / "api" / "main.py": (
            "contextgate.adapters.sqlalchemy",
            "contextgate.adapters.qdrant.vector_index",
            "qdrant_client",
            "sqlalchemy",
            "mlflow",
        ),
        SRC / "adapters" / "celery" / "tasks.py": (
            "contextgate.adapters.sqlalchemy",
            "contextgate.adapters.local.ingestion_service",
            "contextgate.adapters.mlflow.evaluation_store",
            "contextgate.adapters.mlflow.router_registry",
            "sqlalchemy",
        ),
    }
    violations: list[str] = []
    for path, forbidden in forbidden_by_file.items():
        for module in _imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []
