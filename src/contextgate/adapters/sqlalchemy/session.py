from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from contextgate.adapters.sqlalchemy.models import Base
from contextgate.config import get_settings


def _build_engine() -> Engine:
    settings = get_settings()
    database_url = settings.resolved_database_url
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    return create_engine(database_url, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
