from __future__ import annotations

import re

from contextgate.domain.gateway import (
    AbstentionReason,
    Citation,
    CitationValidation,
    ClaimEvidence,
    EvidenceReport,
)
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


def score_answerability(*, query: str, contexts: list[str], abstained: bool) -> EvidenceSet:
    if abstained or not contexts:
        return EvidenceSet(
            answerability_score=0.0,
            coverage_score=0.0,
            support_score=0.0,
            rejected_claims=("retrieval_insufficient",),
        )
    query_tokens = _tokens(query)
    context_tokens = _tokens(" ".join(contexts))
    coverage = len(query_tokens & context_tokens) / max(len(query_tokens), 1)
    return EvidenceSet(
        answerability_score=coverage,
        coverage_score=coverage,
        support_score=1.0,
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
    hits_by_rank = {hit.rank: hit for hit in hits}
    valid = all(
        citation.index in hits_by_rank
        and citation.chunk_id == hits_by_rank[citation.index].chunk_id
        and citation.source == hits_by_rank[citation.index].source
        for citation in citations
    )
    return CitationValidation(
        valid=valid,
        reason=None if valid else AbstentionReason.INVALID_CITATIONS,
    )


def build_evidence_report(
    *,
    answer: str,
    citations: list[Citation],
    hits: list[RetrievalHit],
    require_citations: bool = True,
) -> EvidenceReport:
    validation = validate_citations(citations, hits, require_citation=require_citations)
    if not validation.valid:
        return EvidenceReport(
            verifier="deterministic-claim-verifier",
            verifier_version="v1",
            passed=False,
            reason=validation.reason,
        )

    hits_by_rank = {hit.rank: hit for hit in hits}
    claims: list[ClaimEvidence] = []
    normalized_answer = re.sub(
        r"([.!?])\s+((?:\[\d+]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)} ",
        answer.strip(),
    )
    for paragraph in re.split(r"\n+", normalized_answer):
        paragraph_indices = tuple(
            dict.fromkeys(int(value) for value in re.findall(r"\[(\d+)]", paragraph))
        )
        for raw_claim in re.split(r"(?<=[.!?])\s+", paragraph):
            claim = raw_claim.strip()
            if not claim or not _tokens(claim):
                continue
            claim_indices = tuple(
                dict.fromkeys(int(value) for value in re.findall(r"\[(\d+)]", claim))
            )
            indices = claim_indices or paragraph_indices
            plain_claim = re.sub(r"\[(\d+)]", "", claim).strip()
            claim_tokens = _tokens(plain_claim)
            cited_hits = [hits_by_rank[index] for index in indices if index in hits_by_rank]
            evidence_tokens = _tokens(" ".join(hit.text for hit in cited_hits))
            entailment = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
            contradiction = _contradiction_score(plain_claim, [hit.text for hit in cited_hits])
            supported = bool(indices) and entailment >= 0.55 and contradiction < 0.5
            claims.append(
                ClaimEvidence(
                    claim=plain_claim,
                    citation_indices=indices,
                    supporting_chunk_ids=tuple(hit.chunk_id for hit in cited_hits),
                    entailment_score=entailment,
                    contradiction_score=contradiction,
                    status="supported" if supported else "unsupported",
                )
            )

    passed = bool(claims) and all(claim.status == "supported" for claim in claims)
    score = min((claim.entailment_score for claim in claims), default=0.0)
    contradiction = any(claim.contradiction_score >= 0.5 for claim in claims)
    reason = (
        None
        if passed
        else (
            AbstentionReason.CONTRADICTION if contradiction else AbstentionReason.UNSUPPORTED_CLAIM
        )
    )
    return EvidenceReport(
        verifier="deterministic-claim-verifier",
        verifier_version="v1",
        claims=tuple(claims),
        passed=passed,
        score=score,
        reason=reason,
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(token) > 2}


def _contradiction_score(claim: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    claim_lower = f" {claim.lower()} "
    context_lower = f" {' '.join(contexts).lower()} "
    negations = (" not ", " never ", " no ", " cannot ", " can't ", " mustn't ")
    claim_negated = any(value in claim_lower for value in negations)
    context_negated = any(value in context_lower for value in negations)
    shared = _tokens(claim) & _tokens(context_lower)
    return 1.0 if shared and claim_negated != context_negated else 0.0
