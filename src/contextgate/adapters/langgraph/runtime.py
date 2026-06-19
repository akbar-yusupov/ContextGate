from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from contextgate.adapters.litellm.generation import AnswerGenerator
from contextgate.adapters.local.guardrails import DeterministicClaimVerifier
from contextgate.application.dto import AnswerCommand
from contextgate.application.retrieval import RetrievalService
from contextgate.domain.errors import ContextGateError
from contextgate.domain.evidence import (
    abstention_reason,
    generation_allowed,
    score_answerability,
    validate_citations,
)
from contextgate.domain.gateway import (
    AbstentionReason,
    AnswerResult,
    AnswerStatus,
    Citation,
    ClaimEvidence,
    EvidenceReport,
    RiskReport,
)
from contextgate.domain.retrieval import (
    RetrievalFilter,
    RetrievalHit,
    RetrievalResult,
    RouteDecision,
)
from contextgate.domain.risk import RuleBasedRiskPolicy
from contextgate.ports.guardrails import ClaimVerifier, RiskPolicy
from contextgate.ports.repositories import ProviderRegistry


class GatewayState(TypedDict, total=False):
    request: dict[str, Any]
    query_analysis: dict[str, Any]
    risk: dict[str, Any]
    retrieval_plan: dict[str, Any]
    retrieval: dict[str, Any]
    evidence: dict[str, Any]
    provider: dict[str, Any]
    response: dict[str, Any]
    citation_validation: dict[str, Any]
    final: dict[str, Any]


@dataclass(slots=True)
class GatewayContext:
    request_id: str
    token_callback: Callable[[str], None] | None = None


def _command_to_payload(request: AnswerCommand) -> dict[str, Any]:
    return asdict(request)


def _command_from_payload(payload: dict[str, Any]) -> AnswerCommand:
    filters = payload.get("filters")
    if isinstance(filters, dict):
        filters = RetrievalFilter(**filters)
    return AnswerCommand(**{**payload, "filters": filters})


def _retrieval_to_payload(retrieval: RetrievalResult) -> dict[str, Any]:
    return asdict(retrieval)


def _retrieval_from_payload(payload: dict[str, Any]) -> RetrievalResult:
    route = RouteDecision(**payload["route"])
    hits = [RetrievalHit(**hit) for hit in payload.get("hits", [])]
    return RetrievalResult(**{**payload, "route": route, "hits": hits})


def _answer_to_payload(answer: AnswerResult) -> dict[str, Any]:
    return asdict(answer)


def _answer_from_payload(payload: dict[str, Any]) -> AnswerResult:
    retrieval = _retrieval_from_payload(payload["retrieval"])
    citations = [Citation(**citation) for citation in payload.get("citations", [])]
    evidence_report = payload.get("evidence_report")
    if isinstance(evidence_report, dict):
        evidence_report = EvidenceReport(
            **{
                **evidence_report,
                "claims": tuple(
                    ClaimEvidence(**claim) for claim in evidence_report.get("claims", [])
                ),
            }
        )
    risk_report = payload.get("risk_report")
    if isinstance(risk_report, dict):
        risk_report = RiskReport(**risk_report)
    return AnswerResult(
        **{
            **payload,
            "retrieval": retrieval,
            "citations": citations,
            "evidence_report": evidence_report,
            "risk_report": risk_report,
        }
    )


class GatewayGraph:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator | None = None,
        risk_policy: RiskPolicy | None = None,
        claim_verifier: ClaimVerifier | None = None,
        provider_registry: ProviderRegistry | None = None,
        checkpointer: Any | None = None,
        evidence_threshold: float = 0.35,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator or AnswerGenerator()
        self.risk_policy = risk_policy or RuleBasedRiskPolicy()
        self.claim_verifier = claim_verifier or DeterministicClaimVerifier()
        self.provider_registry = provider_registry
        self.checkpointer = checkpointer
        self.evidence_threshold = evidence_threshold
        self.graph = self._compile()

    def _compile(self):
        def normalize_request(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            return {"request": _command_to_payload(request)}

        def analyze_query(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            tokens = request.query.split()
            return {
                "query_analysis": {
                    "token_count": len(tokens),
                    "has_question_mark": "?" in request.query,
                    "multi_intent": any(term in request.query.lower() for term in (" and ", " и ")),
                }
            }

        def risk_policy_check(state: GatewayState) -> GatewayState:
            report = self.risk_policy.assess_query(str(state["request"]["query"]))
            return {"risk": {"stage": "query", **asdict(report)}}

        def route_query_risk(state: GatewayState) -> Literal["continue", "block"]:
            return "block" if state["risk"]["blocked"] else "continue"

        def plan_retrieval(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            return {
                "retrieval_plan": {
                    "policy": request.policy,
                    "limit": request.limit,
                    "latency_budget_ms": request.latency_budget_ms,
                }
            }

        def retrieve_context(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            retrieval = self.retrieval.retrieve(request)
            return {"retrieval": _retrieval_to_payload(retrieval)}

        def context_risk_check(state: GatewayState) -> GatewayState:
            retrieval = _retrieval_from_payload(state["retrieval"])
            report = self.risk_policy.assess_contexts([hit.text for hit in retrieval.hits])
            return {"risk": {"stage": "context", **asdict(report)}}

        def route_context_risk(state: GatewayState) -> Literal["continue", "block"]:
            return "block" if state["risk"]["blocked"] else "continue"

        def score_context(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            retrieval = _retrieval_from_payload(state["retrieval"])
            evidence = score_answerability(
                query=request.query,
                contexts=[hit.text for hit in retrieval.hits],
                abstained=retrieval.abstained,
            )
            deadline_exceeded = (
                request.deadline_monotonic is not None
                and perf_counter() >= request.deadline_monotonic
            )
            reason = (
                AbstentionReason.LATENCY_BUDGET_EXCEEDED
                if deadline_exceeded
                else abstention_reason(
                    evidence,
                    retrieval_empty=retrieval.abstained or not retrieval.hits,
                )
            )
            return {
                "evidence": {
                    "evidence_score": evidence.score,
                    "answerability_score": evidence.answerability_score,
                    "coverage_score": evidence.coverage_score,
                    "support_score": evidence.support_score,
                    "unsupported_claims": list(evidence.unsupported_claims),
                    "rejected_claims": list(evidence.rejected_claims),
                    "abstention_reason": reason,
                    "generation_allowed": not deadline_exceeded
                    and evidence.score >= self.evidence_threshold
                    and generation_allowed(
                        evidence,
                        retrieval_empty=retrieval.abstained or not retrieval.hits,
                    ),
                }
            }

        def route_evidence(state: GatewayState) -> Literal["generate", "abstain"]:
            return "generate" if state["evidence"]["generation_allowed"] else "abstain"

        def select_provider(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            try:
                provider = (
                    self.provider_registry.choose(
                        cost_budget_usd=request.cost_budget_usd,
                        latency_budget_ms=request.latency_budget_ms,
                        allowed_providers=request.allowed_providers,
                        requested_provider=request.llm_provider,
                        max_context_tokens=request.max_context_tokens,
                    )
                    if self.provider_registry is not None
                    else request.llm_provider or "extractive"
                )
                return {"provider": {"selected_provider": provider}}
            except ContextGateError as exc:
                if exc.code not in {"budget_exceeded", "provider_unavailable"}:
                    raise
                return {
                    "provider": {
                        "selected_provider": "abstention",
                        "error": exc.code,
                    }
                }

        def route_provider(state: GatewayState) -> Literal["generate", "abstain"]:
            return "abstain" if state["provider"].get("error") else "generate"

        def generate_answer(state: GatewayState, runtime: Runtime[GatewayContext]) -> GatewayState:
            request = _command_from_payload(state["request"])
            retrieval = _retrieval_from_payload(state["retrieval"])
            try:
                try:
                    response = self.generator.generate(
                        retrieval,
                        system_prompt=request.system_prompt,
                        provider=state["provider"]["selected_provider"],
                        max_context_tokens=request.max_context_tokens,
                        on_token=runtime.context.token_callback,
                        deadline_monotonic=request.deadline_monotonic,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    response = self.generator.generate(
                        retrieval,
                        system_prompt=request.system_prompt,
                    )
            except ContextGateError as exc:
                if exc.code != "provider_unavailable":
                    raise
                response = replace(
                    self.generator.abstain(retrieval),
                    status=AnswerStatus.ABSTAINED,
                    abstention_reason=AbstentionReason.PROVIDER_UNAVAILABLE,
                )
            return {"response": _answer_to_payload(response)}

        def abstain(state: GatewayState) -> GatewayState:
            retrieval = _retrieval_from_payload(state["retrieval"])
            response = self.generator.abstain(retrieval)
            reason = state["evidence"].get("abstention_reason")
            if reason:
                response = replace(response, abstention_reason=reason)
            return {"response": _answer_to_payload(response)}

        def provider_abstain(state: GatewayState) -> GatewayState:
            retrieval = _retrieval_from_payload(state["retrieval"])
            response = self.generator.abstain(retrieval)
            error = state["provider"].get("error")
            reason = (
                AbstentionReason.BUDGET_EXCEEDED
                if error == "budget_exceeded"
                else AbstentionReason.PROVIDER_UNAVAILABLE
            )
            return {
                "response": _answer_to_payload(
                    replace(
                        response,
                        status=AnswerStatus.ABSTAINED,
                        abstention_reason=reason,
                    )
                )
            }

        def block(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            risk = RiskReport(
                score=float(state["risk"]["score"]),
                blocked=True,
                reason=state["risk"].get("reason"),
                matched_rules=tuple(state["risk"].get("matched_rules", [])),
            )
            retrieval_payload = state.get("retrieval")
            if retrieval_payload is None:
                selected = "balanced" if request.policy == "auto" else request.policy
                retrieval = RetrievalResult(
                    query=request.query,
                    policy=selected,
                    abstained=True,
                    hits=[],
                    route=RouteDecision(
                        requested_policy=request.policy,
                        selected_policy=selected,
                        reason="risk_blocked_before_retrieval",
                        latency_budget_ms=request.latency_budget_ms,
                    ),
                    timings_ms={"total": 0.0},
                    features={},
                    trace_id=str(uuid4()),
                    raw_top_score=None,
                    abstention_threshold=0.0,
                )
            else:
                retrieval = _retrieval_from_payload(retrieval_payload)
            response = AnswerResult(
                answer="",
                citations=[],
                retrieval=retrieval,
                provider="blocked",
                selected_provider="blocked",
                grounded=False,
                status=AnswerStatus.BLOCKED,
                abstention_reason=risk.reason or AbstentionReason.UNSAFE_QUERY,
                risk_report=risk,
            )
            return {"response": _answer_to_payload(response)}

        def verify_citations(state: GatewayState) -> GatewayState:
            response = _answer_from_payload(state["response"])
            validation = validate_citations(
                response.citations,
                response.retrieval.hits,
                require_citation=response.provider not in {"abstention", "extractive"},
            )
            reason = validation.reason or response.abstention_reason
            report = None
            if response.provider not in {"abstention", "blocked"}:
                report = self.claim_verifier.verify(
                    answer=response.answer,
                    citations=response.citations,
                    hits=response.retrieval.hits,
                )
                repair_attempted = bool(
                    state.get("citation_validation", {}).get("repair_attempted", False)
                )
                report = replace(
                    report,
                    repair_attempted=repair_attempted,
                    repair_succeeded=repair_attempted and validation.valid,
                )
                reason = report.reason or reason
            risk_payload = state.get("risk", {})
            risk_report = RiskReport(
                score=float(risk_payload.get("score", 0.0)),
                blocked=bool(risk_payload.get("blocked", False)),
                reason=risk_payload.get("reason"),
                matched_rules=tuple(risk_payload.get("matched_rules", [])),
            )
            response = replace(
                response,
                grounded=response.grounded
                and validation.valid
                and reason is None
                and response.provider != "abstention",
                selected_provider=state.get("provider", {}).get(
                    "selected_provider", response.selected_provider
                ),
                evidence_score=state["evidence"]["evidence_score"],
                answerability_score=state["evidence"]["answerability_score"],
                coverage_score=state["evidence"]["coverage_score"],
                support_score=state["evidence"]["support_score"],
                unsupported_claims=state["evidence"]["unsupported_claims"],
                rejected_claims=state["evidence"]["rejected_claims"],
                abstention_reason=reason,
                evidence_report=report,
                risk_report=risk_report,
                status=(
                    AnswerStatus.ANSWERED
                    if reason is None and response.provider not in {"abstention", "blocked"}
                    else AnswerStatus.ABSTAINED
                ),
            )
            if reason is not None:
                response = replace(
                    response,
                    answer="",
                    citations=[],
                    provider="abstention",
                    selected_provider="abstention",
                    grounded=False,
                )
            return {
                "citation_validation": {
                    **state.get("citation_validation", {}),
                    "valid": validation.valid,
                },
                "response": _answer_to_payload(response),
            }

        def repair_citations(state: GatewayState) -> GatewayState:
            """Perform one bounded structural repair before semantic verification."""
            response = _answer_from_payload(state["response"])
            if response.provider in {"abstention", "blocked"}:
                return {"citation_validation": {"repair_attempted": False}}
            validation = validate_citations(
                response.citations,
                response.retrieval.hits,
                require_citation=True,
            )
            if validation.valid:
                return {"citation_validation": {"repair_attempted": False}}
            hits_by_rank = {hit.rank: hit for hit in response.retrieval.hits}
            cited_indices = tuple(
                dict.fromkeys(int(value) for value in re.findall(r"\[(\d+)]", response.answer))
            )
            if not cited_indices or any(index not in hits_by_rank for index in cited_indices):
                return {"citation_validation": {"repair_attempted": True}}
            repaired = [
                Citation(
                    index=index,
                    chunk_id=hits_by_rank[index].chunk_id,
                    source=hits_by_rank[index].source,
                )
                for index in cited_indices
            ]
            return {
                "citation_validation": {"repair_attempted": True},
                "response": _answer_to_payload(replace(response, citations=repaired)),
            }

        def finalize(state: GatewayState) -> GatewayState:
            response = _answer_from_payload(state["response"])
            return {"final": _answer_to_payload(response)}

        builder = StateGraph(GatewayState, context_schema=GatewayContext)
        builder.add_node("normalize_request", normalize_request)
        builder.add_node("analyze_query", analyze_query)
        builder.add_node("risk_policy_check", risk_policy_check)
        builder.add_node("plan_retrieval", plan_retrieval)
        builder.add_node("retrieve_context", retrieve_context)
        builder.add_node("context_risk_check", context_risk_check)
        builder.add_node("score_context", score_context)
        builder.add_node("select_provider", select_provider)
        builder.add_node("generate_answer", generate_answer)
        builder.add_node("abstain", abstain)
        builder.add_node("provider_abstain", provider_abstain)
        builder.add_node("block", block)
        builder.add_node("repair_citations", repair_citations)
        builder.add_node("verify_citations", verify_citations)
        builder.add_node("finalize", finalize)
        builder.add_edge(START, "normalize_request")
        builder.add_edge("normalize_request", "analyze_query")
        builder.add_edge("analyze_query", "risk_policy_check")
        builder.add_conditional_edges(
            "risk_policy_check",
            route_query_risk,
            {"continue": "plan_retrieval", "block": "block"},
        )
        builder.add_edge("plan_retrieval", "retrieve_context")
        builder.add_edge("retrieve_context", "context_risk_check")
        builder.add_conditional_edges(
            "context_risk_check",
            route_context_risk,
            {"continue": "score_context", "block": "block"},
        )
        builder.add_conditional_edges(
            "score_context",
            route_evidence,
            {"generate": "select_provider", "abstain": "abstain"},
        )
        builder.add_conditional_edges(
            "select_provider",
            route_provider,
            {"generate": "generate_answer", "abstain": "provider_abstain"},
        )
        builder.add_edge("generate_answer", "repair_citations")
        builder.add_edge("abstain", "repair_citations")
        builder.add_edge("provider_abstain", "repair_citations")
        builder.add_edge("repair_citations", "verify_citations")
        builder.add_edge("block", "finalize")
        builder.add_edge("verify_citations", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    def answer(
        self,
        request: AnswerCommand,
        *,
        token_callback: Callable[[str], None] | None = None,
    ) -> AnswerResult:
        thread_id = request.request_id or str(uuid4())
        state = self.graph.invoke(
            {"request": _command_to_payload(request)},
            config={"configurable": {"thread_id": thread_id}},
            context=GatewayContext(request_id=thread_id, token_callback=token_callback),
        )
        return _answer_from_payload(state["final"])

    def stream_events(self, request: AnswerCommand) -> list[dict[str, Any]]:
        response = self.answer(request)
        return [
            {"event": "query_analyzed", "data": {"query": request.query}},
            {"event": "retrieval_started", "data": {"policy": request.policy}},
            {
                "event": "evidence_scored",
                "data": {
                    "evidence_score": response.evidence_score,
                    "answerability_score": response.answerability_score,
                    "coverage_score": response.coverage_score,
                    "support_score": response.support_score,
                },
            },
            {"event": "provider_selected", "data": {"provider": response.selected_provider}},
            {"event": "citation_verified", "data": {"grounded": response.grounded}},
            {"event": "final", "data": _answer_to_payload(response)},
        ]


@lru_cache
def get_gateway_graph() -> GatewayGraph:
    raise RuntimeError("GatewayGraph must be built by the application container")
