from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from contextgate.config import Settings, get_settings


@dataclass(slots=True)
class SparseEmbeddingValue:
    indices: list[int]
    values: list[float]


class EmbeddingProvider(Protocol):
    dense_dimension: int
    late_dimension: int

    def dense_documents(self, texts: list[str]) -> list[list[float]]: ...

    def dense_query(self, text: str) -> list[float]: ...

    def sparse_documents(self, texts: list[str]) -> list[SparseEmbeddingValue]: ...

    def sparse_query(self, text: str) -> SparseEmbeddingValue: ...

    def late_documents(self, texts: list[str]) -> list[list[list[float]]]: ...

    def late_query(self, text: str) -> list[list[float]]: ...

    def cross_scores(self, query: str, documents: list[str]) -> list[float] | None: ...


class FastEmbedProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dense_dimension = self.settings.dense_dimension
        self.late_dimension = self.settings.late_dimension
        self._validate_model_contracts()

    @staticmethod
    def _model_metadata(models: list[dict], model_name: str) -> dict:
        for model in models:
            if model.get("model") == model_name:
                return model
        raise ValueError(f"FastEmbed model is not supported by this installation: {model_name}")

    def _validate_model_contracts(self) -> None:
        from fastembed import (
            LateInteractionTextEmbedding,
            SparseTextEmbedding,
            TextEmbedding,
        )
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        dense = self._model_metadata(
            TextEmbedding.list_supported_models(),
            self.settings.dense_model,
        )
        if int(dense["dim"]) != self.dense_dimension:
            raise ValueError(
                "Dense vector dimension mismatch: "
                f"{self.settings.dense_model} emits {dense['dim']}, "
                f"but CONTEXTGATE_DENSE_DIMENSION={self.dense_dimension}"
            )
        late = self._model_metadata(
            LateInteractionTextEmbedding.list_supported_models(),
            self.settings.late_model,
        )
        if int(late["dim"]) != self.late_dimension:
            raise ValueError(
                "Late-interaction vector dimension mismatch: "
                f"{self.settings.late_model} emits {late['dim']}, "
                f"but CONTEXTGATE_LATE_DIMENSION={self.late_dimension}"
            )
        self._model_metadata(
            SparseTextEmbedding.list_supported_models(),
            self.settings.sparse_model,
        )
        self._model_metadata(
            TextCrossEncoder.list_supported_models(),
            self.settings.cross_encoder_model,
        )

    @cached_property
    def _dense(self):
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.settings.dense_model, lazy_load=True)

    @cached_property
    def _sparse(self):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=self.settings.sparse_model, lazy_load=True)

    @cached_property
    def _late(self):
        from fastembed import LateInteractionTextEmbedding

        return LateInteractionTextEmbedding(model_name=self.settings.late_model, lazy_load=True)

    @cached_property
    def _cross_encoder(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self.settings.cross_encoder_model, lazy_load=True)

    def dense_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            item.astype(np.float32).tolist()
            for item in self._dense.embed(texts, batch_size=self.settings.embedding_batch_size)
        ]

    def dense_query(self, text: str) -> list[float]:
        return next(iter(self._dense.query_embed(text))).astype(np.float32).tolist()

    @staticmethod
    def _sparse_value(item) -> SparseEmbeddingValue:
        return SparseEmbeddingValue(
            indices=item.indices.astype(np.int64).tolist(),
            values=item.values.astype(np.float32).tolist(),
        )

    def sparse_documents(self, texts: list[str]) -> list[SparseEmbeddingValue]:
        return [
            self._sparse_value(item)
            for item in self._sparse.embed(texts, batch_size=self.settings.embedding_batch_size)
        ]

    def sparse_query(self, text: str) -> SparseEmbeddingValue:
        return self._sparse_value(next(iter(self._sparse.query_embed(text))))

    def late_documents(self, texts: list[str]) -> list[list[list[float]]]:
        return [
            item.astype(np.float32).tolist()
            for item in self._late.embed(texts, batch_size=self.settings.embedding_batch_size)
        ]

    def late_query(self, text: str) -> list[list[float]]:
        return next(iter(self._late.query_embed(text))).astype(np.float32).tolist()

    def cross_scores(self, query: str, documents: list[str]) -> list[float] | None:
        return [float(score) for score in self._cross_encoder.rerank(query, documents)]


class DeterministicEmbeddingProvider:
    """Small dependency-free embedding provider for tests and smoke demos."""

    def __init__(self, dense_dimension: int = 64, late_dimension: int = 32) -> None:
        self.dense_dimension = dense_dimension
        self.late_dimension = late_dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _index(token: str, size: int) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        return raw % size, 1.0 if raw & 1 else -1.0

    def _dense_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dense_dimension, dtype=np.float32)
        for token in self._tokens(text):
            index, sign = self._index(token, self.dense_dimension)
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm else vector).tolist()

    def dense_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._dense_one(text) for text in texts]

    def dense_query(self, text: str) -> list[float]:
        return self._dense_one(text)

    def _sparse_one(self, text: str) -> SparseEmbeddingValue:
        counts: dict[int, float] = {}
        for token in self._tokens(text):
            index, _ = self._index(token, 2**20)
            counts[index] = counts.get(index, 0) + 1
        indices = sorted(counts)
        return SparseEmbeddingValue(
            indices=indices,
            values=[1 + math.log(counts[index]) for index in indices],
        )

    def sparse_documents(self, texts: list[str]) -> list[SparseEmbeddingValue]:
        return [self._sparse_one(text) for text in texts]

    def sparse_query(self, text: str) -> SparseEmbeddingValue:
        return self._sparse_one(text)

    def _late_one(self, text: str) -> list[list[float]]:
        result: list[list[float]] = []
        for token in self._tokens(text)[:128]:
            vector = np.zeros(self.late_dimension, dtype=np.float32)
            index, sign = self._index(token, self.late_dimension)
            vector[index] = sign
            result.append(vector.tolist())
        return result or [np.zeros(self.late_dimension, dtype=np.float32).tolist()]

    def late_documents(self, texts: list[str]) -> list[list[list[float]]]:
        return [self._late_one(text) for text in texts]

    def late_query(self, text: str) -> list[list[float]]:
        return self._late_one(text)

    def cross_scores(self, query: str, documents: list[str]) -> list[float] | None:
        query_tokens = set(self._tokens(query))
        return [
            len(query_tokens & set(self._tokens(document))) / max(len(query_tokens), 1)
            for document in documents
        ]


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.embedding_backend == "deterministic":
        return DeterministicEmbeddingProvider(
            dense_dimension=settings.dense_dimension,
            late_dimension=settings.late_dimension,
        )
    return FastEmbedProvider(settings)
