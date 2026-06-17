from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import numpy as np

from contextgate.application.dto import RetrieveCommand
from contextgate.config import PoliciesConfig
from contextgate.domain.language import detect_language
from contextgate.domain.retrieval import (
    RetrievalHit,
    RetrievalResult,
    RouteDecision,
)
from contextgate.ports.repositories import KnowledgeBaseRepository
from contextgate.ports.router import RouterRepository
from contextgate.ports.vector_index import VectorIndex


@dataclass(slots=True)
class ProbeResult:
    features: dict[str, float | int | str]
    hits: dict[str, list]
    embeddings: object


class _ScoredHit(Protocol):
    score: float


def _margin(hits: Sequence[_ScoredHit]) -> float:
    if not hits:
        return 0.0
    if len(hits) == 1:
        return hits[0].score
    return hits[0].score - hits[1].score


def _entropy(hits: Sequence[_ScoredHit]) -> float:
    if len(hits) < 2:
        return 0.0
    scores = np.asarray([hit.score for hit in hits], dtype=np.float64)
    scores = np.exp(scores - np.max(scores))
    probabilities = scores / np.sum(scores)
    value = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return value / math.log(len(hits))


class RetrievalService:
    def __init__(
        self,
        *,
        vector_index: VectorIndex,
        policies: PoliciesConfig,
        router: RouterRepository,
        knowledge_bases: KnowledgeBaseRepository,
    ) -> None:
        self.vector_index = vector_index
        self.policies = policies
        self.router = router
        self.knowledge_bases = knowledge_bases

    def probe(
        self,
        collection_name: str,
        query: str,
        *,
        filters=None,
        limit: int = 20,
    ) -> ProbeResult:
        started = perf_counter()
        embeddings = self.vector_index.embed_query(query)
        dense, sparse = self.vector_index.probe_search(
            collection_name,
            embeddings,
            limit=limit,
            filters=filters,
        )
        elapsed = (perf_counter() - started) * 1000
        dense_ids = {hit.chunk_id for hit in dense[:10]}
        sparse_ids = {hit.chunk_id for hit in sparse[:10]}
        union = dense_ids | sparse_ids
        overlap = len(dense_ids & sparse_ids) / len(union) if union else 0.0
        language = detect_language(query)
        features: dict[str, float | int | str] = {
            "query_token_count": len(query.split()),
            "language": language,
            "dense_margin": _margin(dense),
            "sparse_margin": _margin(sparse),
            "dense_entropy": _entropy(dense),
            "sparse_entropy": _entropy(sparse),
            "retriever_overlap": overlap,
            "top1_agreement": float(
                bool(dense and sparse and dense[0].chunk_id == sparse[0].chunk_id)
            ),
            "first_stage_latency_ms": elapsed,
        }
        return ProbeResult(
            features=features,
            hits={"dense": dense, "sparse": sparse},
            embeddings=embeddings,
        )

    def retrieve(self, request: RetrieveCommand) -> RetrievalResult:
        trace_id = str(uuid4())
        kb = self.knowledge_bases.get(request.knowledge_base)
        overall_started = perf_counter()
        probe: ProbeResult | None = None
        language = detect_language(request.query)
        features: dict[str, float | int | str] = {
            "query_token_count": len(request.query.split()),
            "language": language,
            "first_stage_latency_ms": 0.0,
        }
        if request.policy == "auto":
            probe = self.probe(
                kb.collection_name,
                request.query,
                filters=request.filters,
                limit=max(20, request.limit),
            )
            features = probe.features
            route = self.router.decide(
                kb.slug,
                features,
                request.latency_budget_ms,
            )
        else:
            route = RouteDecision(
                requested_policy=request.policy,
                selected_policy=request.policy,
                reason="explicit_policy",
                latency_budget_ms=request.latency_budget_ms,
            )
        selected = route.selected_policy
        policy = self.policies.policies[selected]
        features["late_interaction_enabled"] = float(
            policy.use_late_interaction
            and self.vector_index.settings.supports_late_interaction(language)
        )
        retrieval_started = perf_counter()
        if probe is not None and selected == "fast":
            hits = probe.hits["dense"][: min(request.limit, policy.output_limit)]
        else:
            hits = self.vector_index.policy_search(
                kb.collection_name,
                request.query,
                policy,
                limit=request.limit,
                filters=request.filters,
                language=language,
                embeddings=probe.embeddings if probe is not None else None,
            )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        total_ms = (perf_counter() - overall_started) * 1000
        raw_top_score = hits[0].score if hits else None
        threshold = self.router.abstention_threshold(
            kb.slug,
            selected,
            fallback=policy.abstention_threshold,
        )
        abstained = raw_top_score is None or raw_top_score < threshold
        response_hits = [
            RetrievalHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source=hit.source,
                text=hit.text,
                language=hit.language,
                score=hit.score,
                rank=index + 1,
                metadata=hit.metadata,
            )
            for index, hit in enumerate(hits)
        ]
        return RetrievalResult(
            query=request.query,
            policy=selected,
            abstained=abstained,
            hits=response_hits,
            route=route,
            timings_ms={
                "first_stage": float(features["first_stage_latency_ms"]),
                "selected_policy": retrieval_ms,
                "total": total_ms,
            },
            features=features,
            trace_id=trace_id,
            raw_top_score=raw_top_score,
            abstention_threshold=threshold,
        )
