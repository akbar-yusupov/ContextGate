from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

import contextgate.apps.api.dependencies as dependencies
from contextgate.adapters.sqlalchemy.models import ApiKey, Base
from contextgate.apps.api.dependencies import (
    RateLimiter,
    hash_api_key,
    require_api_key,
    require_scope,
)
from contextgate.config import Settings


def _request(headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers or [],
            "client": ("127.0.0.1", 1),
        }
    )


def test_api_key_authentication_and_scope_enforcement(monkeypatch, tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'keys.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as session:
        session.add(
            ApiKey(
                name="reader",
                key_hash=hash_api_key("secret"),
                scopes_json=["read"],
            )
        )
        session.commit()

    monkeypatch.setattr(dependencies, "SessionLocal", sessions)
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: Settings(auth_enabled=True, rate_limit_enabled=False),
    )
    with pytest.raises(HTTPException) as missing:
        require_api_key(_request(), None)
    assert missing.value.status_code == 401

    request = _request()
    require_api_key(request, "secret")
    require_scope("read")(request)
    with pytest.raises(HTTPException) as forbidden:
        require_scope("write")(request)
    assert forbidden.value.status_code == 403


class FakeRedis:
    def __init__(self, *, count: int = 1, error: Exception | None = None) -> None:
        self.count = count
        self.error = error
        self.expired = False

    async def incr(self, key: str) -> int:
        if self.error:
            raise self.error
        return self.count

    async def expire(self, key: str, seconds: int) -> None:
        self.expired = True


def test_rate_limiter_disabled_limit_and_dependency_failure_modes() -> None:
    request = SimpleNamespace(headers={}, client=None)

    disabled = RateLimiter(Settings(rate_limit_enabled=False))
    disabled.redis = FakeRedis(error=RuntimeError("must not be called"))
    asyncio.run(disabled.check(request))  # type: ignore[arg-type]

    limited = RateLimiter(Settings(rate_limit_enabled=True, rate_limit_per_minute=1))
    limited.redis = FakeRedis(count=2)
    with pytest.raises(HTTPException) as too_many:
        asyncio.run(limited.check(request))  # type: ignore[arg-type]
    assert too_many.value.status_code == 429

    fail_open = RateLimiter(Settings(rate_limit_enabled=True, rate_limit_fail_open=True))
    fail_open.redis = FakeRedis(error=ConnectionError())
    asyncio.run(fail_open.check(request))  # type: ignore[arg-type]

    fail_closed = RateLimiter(Settings(rate_limit_enabled=True, rate_limit_fail_open=False))
    fail_closed.redis = FakeRedis(error=ConnectionError())
    with pytest.raises(HTTPException) as unavailable:
        asyncio.run(fail_closed.check(request))  # type: ignore[arg-type]
    assert unavailable.value.status_code == 503
