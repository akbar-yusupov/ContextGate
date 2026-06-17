from __future__ import annotations

from typing import Any, Protocol

from contextgate.config import PolicyConfig
from contextgate.domain.retrieval import RetrievalFilter


class VectorIndex(Protocol):
    settings: Any

    def embed_query(self, query: str) -> Any: ...

    def probe_search(
        self,
        collection_name: str,
        embeddings: Any,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
    ) -> tuple[list[Any], list[Any]]: ...

    def policy_search(
        self,
        collection_name: str,
        query: str,
        policy: PolicyConfig,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
        language: str,
        embeddings: Any | None = None,
    ) -> list[Any]: ...
