from __future__ import annotations

import re

from contextgate.domain.gateway import AbstentionReason, Citation, CitationValidation
from contextgate.domain.models import EvidenceSet
from contextgate.domain.retrieval import RetrievalHit


def score_evidence(
    *,
    query: str,
    answer: str,
    contexts: list[str],
    abstained: bool,
) -> EvidenceSet:
    if abstained or not contexts:
        return EvidenceSet(
            answerability_score=0.0,
            coverage_score=0.0,
            support_score=0.0,
            rejected_claims=("retrieval_insufficient",),
        )
    query_tokens = _tokens(query)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts))
    coverage = len(query_tokens & context_tokens) / max(len(query_tokens), 1)
    support = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    answerability = min(1.0, (coverage + support) / 2)
    unsupported = tuple(sorted(answer_tokens - context_tokens))[:8]
    return EvidenceSet(
        answerability_score=answerability,
        coverage_score=coverage,
        support_score=support,
        unsupported_claims=unsupported if support < 0.5 else (),
    )


def abstention_reason(evidence: EvidenceSet, *, retrieval_empty: bool) -> AbstentionReason | None:
    if retrieval_empty:
        return AbstentionReason.RETRIEVAL_EMPTY
    if evidence.coverage_score < 0.35:
        return AbstentionReason.LOW_COVERAGE
    if evidence.support_score < 0.35:
        return AbstentionReason.LOW_SUPPORT
    return None


def generation_allowed(evidence: EvidenceSet, *, retrieval_empty: bool) -> bool:
    return abstention_reason(evidence, retrieval_empty=retrieval_empty) is None


def validate_citations(
    citations: list[Citation],
    hits: list[RetrievalHit],
    *,
    require_citation: bool,
) -> CitationValidation:
    if require_citation and not citations:
        return CitationValidation(valid=False, reason=AbstentionReason.INVALID_CITATIONS)
    valid_chunks = {hit.chunk_id for hit in hits}
    valid = all(
        citation.chunk_id in valid_chunks and 1 <= citation.index <= len(hits)
        for citation in citations
    )
    return CitationValidation(
        valid=valid,
        reason=None if valid else AbstentionReason.INVALID_CITATIONS,
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(token) > 2}
