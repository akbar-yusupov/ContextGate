from __future__ import annotations

import hashlib
import secrets
import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from contextgate.adapters.sqlalchemy import ApiKey, get_db
from contextgate.config import Settings, get_settings


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_bootstrap_api_key(session: Session, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    digest = hash_api_key(settings.api_key)
    existing = session.scalar(select(ApiKey).where(ApiKey.key_hash == digest))
    if existing is None:
        session.add(ApiKey(name="bootstrap", key_hash=digest))
        session.commit()


def require_api_key(
    session: Annotated[Session, Depends(get_db)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    digest = hash_api_key(x_api_key)
    record = session.scalar(
        select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.enabled.is_(True))
    )
    if record is None or not secrets.compare_digest(record.key_hash, digest):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class RateLimiter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = Redis.from_url(self.settings.resolved_redis_url, decode_responses=True)

    async def check(self, request: Request) -> None:
        identity = request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )
        minute = int(time.time() // 60)
        key = f"contextgate:rate:{hash_api_key(identity)}:{minute}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, 70)
        except Exception:
            return
        if count > self.settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
