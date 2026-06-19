from __future__ import annotations

import hashlib
import secrets
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from contextgate.adapters.sqlalchemy import ApiKey, SessionLocal
from contextgate.config import Settings, get_settings


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_bootstrap_api_key(session: Session, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    digest = hash_api_key(settings.api_key)
    existing = session.scalar(select(ApiKey).where(ApiKey.name == "bootstrap"))
    if existing is None:
        existing = ApiKey(name="bootstrap", key_hash=digest)
    else:
        existing.key_hash = digest
        existing.enabled = True
    existing.scopes_json = ["read", "write", "admin"]
    session.add(existing)
    session.commit()


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    settings = get_settings()
    if not settings.auth_enabled:
        request.state.api_key_scopes = {"read", "write", "admin"}
        return
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    digest = hash_api_key(x_api_key)
    if secrets.compare_digest(digest, hash_api_key(settings.api_key)):
        request.state.api_key_scopes = {"read", "write", "admin"}
        return
    with SessionLocal() as session:
        record = session.scalar(
            select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.enabled.is_(True))
        )
        if record is None or not secrets.compare_digest(record.key_hash, digest):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        request.state.api_key_scopes = set(record.scopes_json)
        now = datetime.now(UTC)
        last_used = record.last_used_at
        if (
            last_used is None
            or (
                now - (last_used if last_used.tzinfo else last_used.replace(tzinfo=UTC))
            ).total_seconds()
            >= 300
        ):
            record.last_used_at = now
            session.add(record)
            session.commit()


def require_scope(scope: str):
    def dependency(request: Request) -> None:
        scopes: set[str] = getattr(request.state, "api_key_scopes", set())
        if scope not in scopes and "admin" not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key requires the {scope} scope",
            )

    return dependency


class RateLimiter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = Redis.from_url(self.settings.resolved_redis_url, decode_responses=True)

    async def check(self, request: Request) -> None:
        if not self.settings.rate_limit_enabled:
            return
        identity = request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )
        minute = int(time.time() // 60)
        key = f"contextgate:rate:{hash_api_key(identity)}:{minute}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, 70)
        except Exception as exc:
            if self.settings.rate_limit_fail_open:
                return
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter unavailable",
            ) from exc
        if count > self.settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
