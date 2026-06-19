from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy.models import ApiKey, Base
from contextgate.adapters.sqlalchemy.repositories import SqlAlchemyApiKeyRepository
from contextgate.apps.api.dependencies import ensure_bootstrap_api_key, hash_api_key
from contextgate.config import Settings


def _sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_bootstrap_key_rotation_updates_one_record(tmp_path) -> None:
    sessions = _sessions(tmp_path)
    with sessions() as session:
        ensure_bootstrap_api_key(session, Settings(api_key="first"))
        ensure_bootstrap_api_key(session, Settings(api_key="second"))
        records = list(session.scalars(select(ApiKey)).all())

    assert len(records) == 1
    assert records[0].key_hash == hash_api_key("second")
    assert records[0].scopes_json == ["read", "write", "admin"]


def test_scoped_key_is_returned_once_and_can_be_rotated_and_disabled(tmp_path) -> None:
    sessions = _sessions(tmp_path)
    repository = SqlAlchemyApiKeyRepository(sessions)

    created, secret = repository.create("writer", ["read", "write"])
    rotated, rotated_secret = repository.rotate(created.id)
    disabled = repository.disable(created.id)

    assert secret.startswith("ctxg_")
    assert rotated_secret.startswith("ctxg_")
    assert secret != rotated_secret
    assert rotated.scopes_json == ["read", "write"]
    assert disabled.enabled is False
