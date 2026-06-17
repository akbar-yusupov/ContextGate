from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class BenchmarkQuery:
    id: str
    query: str
    language: str
    group_id: str | None = None
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    answerable: bool = True
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, value: str) -> BenchmarkQuery:
        payload = json.loads(value)
        return cls(**payload)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "language": self.language,
            "group_id": self.group_id,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "expected_facts": self.expected_facts,
            "answerable": self.answerable,
            "tags": self.tags,
        }


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    dcg = sum(
        1 / math.log2(index + 2) for index, item in enumerate(retrieved[:k]) if item in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))
