from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from contextgate.adapters.fastembed.embeddings import (
    EmbeddingProvider,
    SparseEmbeddingValue,
    create_embedding_provider,
)
from contextgate.config import PolicyConfig, Settings, get_settings
from contextgate.domain.documents import Chunk
from contextgate.domain.retrieval import RetrievalFilter


@dataclass(slots=True)
class StoreHit:
    chunk_id: str
    document_id: str
    source: str
    text: str
    language: str
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class QueryEmbeddings:
    dense: list[float]
    sparse: SparseEmbeddingValue
    late: list[list[float]] | None = None


class VectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embedding_provider(self.settings)
        if client is not None:
            self.client = client
        elif self.settings.resolved_qdrant_url:
            self.client = QdrantClient(
                url=self.settings.resolved_qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=120,
            )
        else:
            self.client = QdrantClient(path=str(self.settings.qdrant_local_path))

    def close(self) -> None:
        self.client.close()

    def _validate_collection(self, collection_name: str) -> None:
        info = self.client.get_collection(collection_name)
        params = info.config.params
        vectors = params.vectors
        if not isinstance(vectors, dict):
            raise ValueError(f"Collection {collection_name} must use named vectors")
        dense = vectors.get("dense")
        late = vectors.get("late")
        sparse = (params.sparse_vectors or {}).get("sparse")
        errors: list[str] = []
        if dense is None:
            errors.append("missing dense vector")
        elif dense.size != self.embedder.dense_dimension:
            errors.append(
                f"dense dimension is {dense.size}, expected {self.embedder.dense_dimension}"
            )
        elif dense.distance != models.Distance.COSINE:
            errors.append("dense distance must be cosine")
        if late is None:
            errors.append("missing late vector")
        else:
            if late.size != self.embedder.late_dimension:
                errors.append(
                    f"late dimension is {late.size}, expected {self.embedder.late_dimension}"
                )
            if late.distance != models.Distance.COSINE:
                errors.append("late distance must be cosine")
            if (
                late.multivector_config is None
                or late.multivector_config.comparator != models.MultiVectorComparator.MAX_SIM
            ):
                errors.append("late vector must use MAX_SIM multivector scoring")
            if late.hnsw_config is None or late.hnsw_config.m != 0:
                errors.append("late vector HNSW must be disabled with m=0")
        if sparse is None:
            errors.append("missing sparse vector")
        elif sparse.modifier != models.Modifier.IDF:
            errors.append("sparse vector must use the IDF modifier")
        if errors:
            details = "; ".join(errors)
            raise ValueError(f"Incompatible Qdrant collection {collection_name}: {details}")

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        schema_types = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "bool": models.PayloadSchemaType.BOOL,
        }
        fields: dict[str, models.PayloadSchemaType] = {
            "document_id": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "language": models.PayloadSchemaType.KEYWORD,
            "content_hash": models.PayloadSchemaType.KEYWORD,
            "pipeline_version": models.PayloadSchemaType.KEYWORD,
        }
        fields.update(
            {
                f"metadata.{name}": schema_types[field_type]
                for name, field_type in self.settings.indexed_metadata_fields.items()
            }
        )
        existing = set(self.client.get_collection(collection_name).payload_schema)
        for field, field_type in fields.items():
            if field in existing:
                continue
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=field_type,
                wait=True,
            )

    def ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            self._validate_collection(collection_name)
            self._ensure_payload_indexes(collection_name)
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedder.dense_dimension,
                    distance=models.Distance.COSINE,
                ),
                "late": models.VectorParams(
                    size=self.embedder.late_dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            strict_mode_config=models.StrictModeConfig(
                enabled=self.settings.qdrant_strict_mode,
                unindexed_filtering_retrieve=False,
                unindexed_filtering_update=False,
                max_query_limit=1_000,
                upsert_max_batchsize=max(self.settings.embedding_batch_size, 64),
            ),
            metadata={
                "managed_by": "contextgate",
                "pipeline_version": self.settings.pipeline_version,
                "dense_model": self.settings.dense_model,
                "sparse_model": self.settings.sparse_model,
                "late_model": self.settings.late_model,
            },
        )
        self._ensure_payload_indexes(collection_name)

    @staticmethod
    def _sparse(value: SparseEmbeddingValue) -> models.SparseVector:
        return models.SparseVector(indices=value.indices, values=value.values)

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        *,
        content_hash: str = "",
        pipeline_version: str | None = None,
    ) -> int:
        self.ensure_collection(collection_name)
        total = 0
        batch_size = self.settings.embedding_batch_size
        version = pipeline_version or self.settings.pipeline_version
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            texts = [chunk.text for chunk in batch]
            dense = self.embedder.dense_documents(texts)
            sparse = self.embedder.sparse_documents(texts)
            late_indices = [
                index
                for index, chunk in enumerate(batch)
                if self.settings.supports_late_interaction(chunk.language)
            ]
            late_by_index: dict[int, list[list[float]]] = {}
            if late_indices:
                late_values = self.embedder.late_documents([texts[index] for index in late_indices])
                late_by_index = dict(zip(late_indices, late_values, strict=True))
            points = []
            for index, chunk in enumerate(batch):
                vectors: dict[str, Any] = {
                    "dense": dense[index],
                    "sparse": self._sparse(sparse[index]),
                    # Qdrant server skips absent named vectors, while embedded Qdrant
                    # currently includes them as empty candidates during nested prefetch.
                    "late": late_by_index.get(index) or [[0.0] * self.embedder.late_dimension],
                }
                point_key = f"{chunk.chunk_id}:{content_hash}:{version}"
                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, point_key)),
                        vector=vectors,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "source": chunk.source,
                            "text": chunk.text,
                            "language": chunk.language,
                            "content_hash": content_hash,
                            "pipeline_version": version,
                            "metadata": chunk.metadata,
                        },
                    )
                )
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            total += len(points)
        return total

    def delete_document_versions(
        self,
        collection_name: str,
        document_id: str,
        *,
        keep_content_hash: str | None = None,
    ) -> None:
        must = [
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
        must_not = []
        if keep_content_hash is not None:
            must_not.append(
                models.FieldCondition(
                    key="content_hash",
                    match=models.MatchValue(value=keep_content_hash),
                )
            )
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=must, must_not=must_not)
            ),
            wait=True,
        )

    def delete_document_version(
        self,
        collection_name: str,
        document_id: str,
        content_hash: str,
    ) -> None:
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="content_hash",
                            match=models.MatchValue(value=content_hash),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def build_filter(self, filters: RetrievalFilter | None) -> models.Filter | None:
        if filters is None:
            return None
        must: list[models.Condition] = []
        fields = {
            "document_id": filters.document_ids,
            "language": filters.languages,
            "source": filters.sources,
        }
        for field, values in fields.items():
            if values:
                must.append(models.FieldCondition(key=field, match=models.MatchAny(any=values)))
        for key, value in filters.metadata.items():
            field = f"metadata.{key}"
            if field not in self.settings.indexed_filter_fields:
                raise ValueError(
                    f"Metadata filter {key!r} is not indexed. Add it to "
                    "CONTEXTGATE_INDEXED_METADATA_FIELDS before collection creation."
                )
            must.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=must) if must else None

    @staticmethod
    def _with_language_filter(
        query_filter: models.Filter | None,
        language: str,
    ) -> models.Filter:
        language_condition = models.FieldCondition(
            key="language",
            match=models.MatchValue(value=language),
        )
        if query_filter is None:
            return models.Filter(must=[language_condition])
        return models.Filter(
            should=query_filter.should,
            min_should=query_filter.min_should,
            must=[*(query_filter.must or []), language_condition],
            must_not=query_filter.must_not,
        )

    def embed_query(self, query: str, *, include_late: bool = False) -> QueryEmbeddings:
        return QueryEmbeddings(
            dense=self.embedder.dense_query(query),
            sparse=self.embedder.sparse_query(query),
            late=self.embedder.late_query(query) if include_late else None,
        )

    @staticmethod
    def _hits(response: models.QueryResponse) -> list[StoreHit]:
        hits: list[StoreHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                StoreHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    source=str(payload.get("source", "")),
                    text=str(payload.get("text", "")),
                    language=str(payload.get("language", "unknown")),
                    score=float(point.score),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return hits

    def dense_search(
        self,
        collection_name: str,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
        embeddings: QueryEmbeddings | None = None,
    ) -> list[StoreHit]:
        embeddings = embeddings or self.embed_query(query)
        response = self.client.query_points(
            collection_name=collection_name,
            query=embeddings.dense,
            using="dense",
            query_filter=self.build_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return self._hits(response)

    def sparse_search(
        self,
        collection_name: str,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
        embeddings: QueryEmbeddings | None = None,
    ) -> list[StoreHit]:
        embeddings = embeddings or self.embed_query(query)
        response = self.client.query_points(
            collection_name=collection_name,
            query=self._sparse(embeddings.sparse),
            using="sparse",
            query_filter=self.build_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return self._hits(response)

    def probe_search(
        self,
        collection_name: str,
        embeddings: QueryEmbeddings,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
    ) -> tuple[list[StoreHit], list[StoreHit]]:
        query_filter = self.build_filter(filters)
        responses = self.client.query_batch_points(
            collection_name=collection_name,
            requests=[
                models.QueryRequest(
                    query=embeddings.dense,
                    using="dense",
                    filter=query_filter,
                    limit=limit,
                    with_payload=True,
                ),
                models.QueryRequest(
                    query=self._sparse(embeddings.sparse),
                    using="sparse",
                    filter=query_filter,
                    limit=limit,
                    with_payload=True,
                ),
            ],
        )
        return self._hits(responses[0]), self._hits(responses[1])

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        policy: PolicyConfig,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
        embeddings: QueryEmbeddings | None = None,
    ) -> list[StoreHit]:
        query_filter = self.build_filter(filters)
        embeddings = embeddings or self.embed_query(query)
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=embeddings.dense,
                    using="dense",
                    filter=query_filter,
                    limit=policy.dense_limit,
                ),
                models.Prefetch(
                    query=self._sparse(embeddings.sparse),
                    using="sparse",
                    filter=query_filter,
                    limit=policy.sparse_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=min(limit, policy.output_limit),
            with_payload=True,
        )
        return self._hits(response)

    def policy_search(
        self,
        collection_name: str,
        query: str,
        policy: PolicyConfig,
        *,
        limit: int,
        filters: RetrievalFilter | None = None,
        language: str = "unknown",
        embeddings: QueryEmbeddings | None = None,
    ) -> list[StoreHit]:
        query_filter = self.build_filter(filters)
        use_late = policy.use_late_interaction and self.settings.supports_late_interaction(language)
        embeddings = embeddings or self.embed_query(query, include_late=use_late)
        if use_late and embeddings.late is None:
            embeddings.late = self.embedder.late_query(query)
        if not policy.use_late_interaction:
            return self.dense_search(
                collection_name,
                query,
                limit=min(limit, policy.output_limit),
                filters=filters,
                embeddings=embeddings,
            )
        if not use_late:
            return self.hybrid_search(
                collection_name,
                query,
                policy,
                limit=limit,
                filters=filters,
                embeddings=embeddings,
            )

        query_filter = self._with_language_filter(query_filter, language)
        dense_prefetch = models.Prefetch(
            query=embeddings.dense,
            using="dense",
            filter=query_filter,
            limit=policy.dense_limit,
        )
        sparse_prefetch = models.Prefetch(
            query=self._sparse(embeddings.sparse),
            using="sparse",
            filter=query_filter,
            limit=policy.sparse_limit,
        )
        fusion_prefetch = models.Prefetch(
            prefetch=[dense_prefetch, sparse_prefetch],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=policy.prefetch_limit,
        )
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=fusion_prefetch,
            query=embeddings.late,
            using="late",
            query_filter=query_filter,
            limit=min(limit, policy.output_limit),
            with_payload=True,
        )
        hits = self._hits(response)
        if policy.use_cross_encoder and language == "en" and hits:
            scores = self.embedder.cross_scores(query, [hit.text for hit in hits])
            if scores:
                for hit, score in zip(hits, scores, strict=True):
                    hit.score = float(score)
                hits.sort(key=lambda item: item.score, reverse=True)
        return hits

    def copy_collection(self, source: str, target: str) -> int:
        if not self.client.collection_exists(source):
            raise ValueError(f"Source collection does not exist: {source}")
        source_info = self.client.get_collection(source)
        params = source_info.config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors
        names = set(vectors if isinstance(vectors, dict) else [])
        if not {"dense", "late"}.issubset(names) or "sparse" not in (sparse_vectors or {}):
            raise ValueError("Source collection must expose named vectors: dense, sparse, and late")
        if self.client.collection_exists(target):
            raise ValueError(f"Target collection already exists: {target}")
        self.client.create_collection(
            collection_name=target,
            vectors_config=vectors,
            sparse_vectors_config=sparse_vectors,
            strict_mode_config=(
                models.StrictModeConfig.model_validate(
                    source_info.config.strict_mode_config.model_dump()
                )
                if source_info.config.strict_mode_config
                else None
            ),
            hnsw_config=models.HnswConfigDiff.model_validate(
                source_info.config.hnsw_config.model_dump()
            ),
            optimizers_config=models.OptimizersConfigDiff.model_validate(
                source_info.config.optimizer_config.model_dump()
            ),
            quantization_config=source_info.config.quantization_config,
            metadata=source_info.config.metadata,
        )
        copied = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=source,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if points:
                point_structs = [
                    models.PointStruct(
                        id=point.id,
                        vector=point.vector or {},
                        payload=point.payload or {},
                    )
                    for point in points
                ]
                self.client.upsert(
                    collection_name=target,
                    points=point_structs,
                    wait=True,
                )
                copied += len(points)
            if offset is None:
                break
        self._validate_collection(target)
        self._ensure_payload_indexes(target)
        return copied


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
