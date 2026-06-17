from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ErrorCode = Literal[
    "validation_error",
    "not_found",
    "policy_rejected",
    "provider_unavailable",
    "retrieval_insufficient",
    "budget_exceeded",
    "internal_error",
]


@dataclass(slots=True)
class ContextGateError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message
