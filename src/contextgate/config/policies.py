from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from contextgate.config.settings import get_settings


class PolicyConfig(BaseModel):
    dense_limit: int = Field(ge=0)
    sparse_limit: int = Field(ge=0)
    prefetch_limit: int = Field(gt=0)
    output_limit: int = Field(gt=0)
    use_late_interaction: bool
    use_cross_encoder: bool
    abstention_threshold: float = Field(ge=0, le=1)


class PoliciesConfig(BaseModel):
    policies: dict[Literal["fast", "balanced", "accurate"], PolicyConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> PoliciesConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache
def get_policies() -> PoliciesConfig:
    settings = get_settings()
    path = settings.policies_path
    if not path.is_absolute() and not path.exists():
        package_root = Path(__file__).resolve().parents[3]
        path = package_root / path
    return PoliciesConfig.from_yaml(path)
