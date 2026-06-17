from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from contextgate.adapters.litellm.generation import AnswerGenerator
from contextgate.application.dto import AnswerCommand
from contextgate.application.retrieval import RetrievalService
from contextgate.domain.evidence import (
    abstention_reason,
    generation_allowed,
    score_evidence,
    validate_citations,
)
from contextgate.domain.gateway import AnswerResult, Citation
from contextgate.domain.retrieval import (
    RetrievalFilter,
    RetrievalHit,
    RetrievalResult,
    RouteDecision,
)


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
    return AnswerResult(**{**payload, "retrieval": retrieval, "citations": citations})


class GatewayGraph:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator | None = None,
        checkpointer: Any | None = None,
        evidence_threshold: float = 0.35,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator or AnswerGenerator()
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
            query = str(state["request"]["query"]).lower()
            risky_terms = ("ignore previous", "system prompt", "developer message")
            risk_score = 1.0 if any(term in query for term in risky_terms) else 0.0
            return {"risk": {"risk_score": risk_score, "blocked": False}}

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

        def score_context(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            retrieval = _retrieval_from_payload(state["retrieval"])
            evidence = score_evidence(
                query=request.query,
                answer=" ".join(hit.text for hit in retrieval.hits[:3]),
                contexts=[hit.text for hit in retrieval.hits],
                abstained=retrieval.abstained,
            )
            return {
                "evidence": {
                    "evidence_score": evidence.score,
                    "answerability_score": evidence.answerability_score,
                    "coverage_score": evidence.coverage_score,
                    "support_score": evidence.support_score,
                    "unsupported_claims": list(evidence.unsupported_claims),
                    "rejected_claims": list(evidence.rejected_claims),
                    "abstention_reason": abstention_reason(
                        evidence,
                        retrieval_empty=retrieval.abstained or not retrieval.hits,
                    ),
                    "generation_allowed": evidence.score >= self.evidence_threshold
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
            provider = request.llm_provider or (
                "litellm" if request.system_prompt else "extractive"
            )
            return {"provider": {"selected_provider": provider}}

        def generate_answer(state: GatewayState) -> GatewayState:
            request = _command_from_payload(state["request"])
            retrieval = _retrieval_from_payload(state["retrieval"])
            response = self.generator.generate(retrieval, system_prompt=request.system_prompt)
            return {"response": _answer_to_payload(response)}

        def abstain(state: GatewayState) -> GatewayState:
            retrieval = _retrieval_from_payload(state["retrieval"])
            response = self.generator.abstain(retrieval)
            reason = state["evidence"].get("abstention_reason")
            if reason:
                response = replace(response, abstention_reason=reason)
            return {"response": _answer_to_payload(response)}

        def verify_citations(state: GatewayState) -> GatewayState:
            response = _answer_from_payload(state["response"])
            validation = validate_citations(
                response.citations,
                response.retrieval.hits,
                require_citation=response.provider not in {"abstention", "extractive"},
            )
            reason = validation.reason or response.abstention_reason
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
            )
            return {
                "citation_validation": {"valid": validation.valid},
                "response": _answer_to_payload(response),
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
        builder.add_node("score_context", score_context)
        builder.add_node("select_provider", select_provider)
        builder.add_node("generate_answer", generate_answer)
        builder.add_node("abstain", abstain)
        builder.add_node("verify_citations", verify_citations)
        builder.add_node("finalize", finalize)
        builder.add_edge(START, "normalize_request")
        builder.add_edge("normalize_request", "analyze_query")
        builder.add_edge("analyze_query", "risk_policy_check")
        builder.add_edge("risk_policy_check", "plan_retrieval")
        builder.add_edge("plan_retrieval", "retrieve_context")
        builder.add_edge("retrieve_context", "score_context")
        builder.add_conditional_edges(
            "score_context",
            route_evidence,
            {"generate": "select_provider", "abstain": "abstain"},
        )
        builder.add_edge("select_provider", "generate_answer")
        builder.add_edge("generate_answer", "verify_citations")
        builder.add_edge("abstain", "verify_citations")
        builder.add_edge("verify_citations", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    def answer(self, request: AnswerCommand) -> AnswerResult:
        thread_id = request.request_id or str(uuid4())
        state = self.graph.invoke(
            {"request": _command_to_payload(request)},
            config={"configurable": {"thread_id": thread_id}},
            context=GatewayContext(request_id=thread_id),
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
