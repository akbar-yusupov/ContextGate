from __future__ import annotations

import json
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from contextgate.apps.api.dependencies import RateLimiter, require_api_key
from contextgate.apps.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    BenchmarkRequest,
    ErrorResponse,
    JobRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    PolicyCreateRequest,
    PolicyRead,
    ProviderTestRequest,
    RetrieveRequest,
    RetrieveResponse,
    RouterPromoteRequest,
    RouterTrainRequest,
    SyncQdrantRequest,
)
from contextgate.apps.container import get_container
from contextgate.config import get_settings
from contextgate.domain.errors import ContextGateError
from contextgate.observability.metrics import REQUEST_LATENCY, REQUESTS

settings = get_settings()
rate_limiter = RateLimiter()
OPENAPI_TAGS = [
    {"name": "System", "description": "Health, metrics and service-level checks."},
    {"name": "Knowledge Bases", "description": "Create and inspect knowledge-base containers."},
    {"name": "Ingestion", "description": "Upload documents and sync existing Qdrant collections."},
    {"name": "Jobs", "description": "Durable background job status."},
    {"name": "Retrieval", "description": "Run retrieval policies without answer generation."},
    {"name": "Answer Runtime", "description": "Evidence-gated LangGraph answer runs."},
    {"name": "Runs/Traces", "description": "Inspect answer traces, events and cost."},
    {"name": "Evaluations", "description": "Run benchmarks and inspect evaluation reports."},
    {"name": "Routers", "description": "Train and promote learned retrieval routers."},
    {"name": "Policies", "description": "Persisted gateway policy configuration."},
    {"name": "Providers", "description": "List and test generation providers."},
    {"name": "OpenAI Compatibility", "description": "OpenAI-compatible model and chat endpoints."},
]


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, dict):
        return value
    return dict(value)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_container().startup()
    yield
    await rate_limiter.redis.aclose()
    get_container().shutdown()


app = FastAPI(
    title="ContextGate",
    version="0.1.0",
    description="Evidence-Gated RAG Gateway for grounded answers, citations, traces and provider policies.",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@app.exception_handler(ContextGateError)
async def contextgate_error_handler(_: Request, exc: ContextGateError):
    status = {
        "validation_error": 422,
        "not_found": 404,
        "policy_rejected": 409,
        "provider_unavailable": 503,
        "retrieval_insufficient": 422,
        "budget_exceeded": 402,
        "internal_error": 500,
    }.get(exc.code, 500)
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(
        status_code=404 if "not found" in str(exc).lower() else 409,
        content=ErrorResponse(code="not_found", message=str(exc)).model_dump(),
    )


@app.middleware("http")
async def metrics_and_limits(request: Request, call_next):
    started = time.perf_counter()
    request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
    if request.url.path not in {"/health", "/metrics"}:
        await rate_limiter.check(request)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    duration = time.perf_counter() - started
    REQUESTS.labels(
        method=request.method,
        path=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(duration)
    return response


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0", "service": "contextgate"}


@app.get("/metrics", tags=["System"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseRead,
    status_code=201,
    tags=["Knowledge Bases"],
)
def create_knowledge_base(payload: KnowledgeBaseCreate) -> Any:
    return get_container().knowledge_bases.create(payload.to_command())


@api.post(
    "/knowledge-bases/{identifier}/documents",
    response_model=JobRead,
    status_code=202,
    tags=["Ingestion"],
)
def upload_document(identifier: str, request: Request, file: Annotated[UploadFile, File()]) -> Any:
    container = get_container()
    destination = (
        container.settings.upload_dir
        / identifier
        / str(uuid4())
        / Path(file.filename or "file").name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return container.ingest_documents.enqueue(
        knowledge_base=identifier,
        path=destination,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


@api.post(
    "/knowledge-bases/{identifier}/sync-qdrant",
    response_model=JobRead,
    status_code=202,
    tags=["Ingestion"],
)
def sync_qdrant(identifier: str, payload: SyncQdrantRequest, request: Request) -> Any:
    return get_container().sync_qdrant.enqueue(
        knowledge_base=identifier,
        request=payload.to_command(),
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


@api.get("/jobs/{job_id}", response_model=JobRead, tags=["Jobs"])
def get_job(job_id: str) -> Any:
    return get_container().knowledge_bases.get_job(job_id)


@api.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    return RetrieveResponse.from_domain(
        get_container().retrieve_context.execute(payload.to_command())
    )


@api.post("/answer", response_model=AnswerResponse, tags=["Answer Runtime"])
def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
    return AnswerResponse.from_domain(
        get_container().answer_with_evidence.execute(
            payload.to_command(),
            request_id=payload.request_id or request.state.request_id,
        )
    )


@api.post("/runs/answer", response_model=AnswerResponse, tags=["Answer Runtime"])
def run_answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
    return answer(payload, request)


@api.get("/runs/{run_id}", tags=["Runs/Traces"])
def get_run(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.run(run_id)


@api.get("/runs/{run_id}/events", tags=["Runs/Traces"])
def get_run_events(run_id: str) -> StreamingResponse:
    events = get_container().inspect_trace.events(run_id)

    def stream():
        for event in events:
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['payload'])}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@api.get("/runs/{run_id}/trace", tags=["Runs/Traces"])
def get_run_trace(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.run(run_id)


@api.get("/runs/{run_id}/cost", tags=["Runs/Traces"])
def get_run_cost(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.cost(run_id)


@api.post("/benchmarks", response_model=JobRead, status_code=202, tags=["Evaluations"])
def benchmark(payload: BenchmarkRequest, request: Request) -> Any:
    return get_container().run_benchmark.enqueue(
        payload.to_command(),
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


@api.post("/evaluations", response_model=JobRead, status_code=202, tags=["Evaluations"])
def create_evaluation(payload: BenchmarkRequest, request: Request) -> Any:
    return benchmark(payload, request)


@api.get("/evaluations/{run_id}", tags=["Evaluations"])
def get_evaluation(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.run(run_id)


@api.get("/evaluations/{run_id}/report", tags=["Evaluations"])
def get_evaluation_report(run_id: str) -> dict[str, Any]:
    report = get_container().settings.report_dir / run_id / "report.html"
    return {"run_id": run_id, "report_path": str(report), "exists": report.exists()}


@api.post("/routers/train", response_model=JobRead, status_code=202, tags=["Routers"])
def train_router(payload: RouterTrainRequest, request: Request) -> Any:
    return get_container().train_router.enqueue(
        payload.to_command(),
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


@api.post("/routers/promote", tags=["Routers"])
def promote_router(payload: RouterPromoteRequest) -> dict[str, str]:
    return get_container().promote_policy.execute(payload.to_command())


@api.post("/policies", response_model=PolicyRead, status_code=201, tags=["Policies"])
def create_policy(payload: PolicyCreateRequest) -> PolicyRead:
    return get_container().policies.create(payload.to_command())


@api.get("/policies/{policy_id}", response_model=PolicyRead, tags=["Policies"])
def get_policy(policy_id: str) -> PolicyRead:
    return get_container().policies.get(policy_id)


@api.post("/policies/{policy_id}/promote", response_model=PolicyRead, tags=["Policies"])
def promote_configured_policy(policy_id: str) -> PolicyRead:
    return get_container().policies.promote(policy_id)


@api.get("/providers", tags=["Providers"])
def list_providers() -> dict[str, Any]:
    return {"providers": get_container().provider_registry.list()}


@api.post("/providers/test", tags=["Providers"])
def test_provider(payload: ProviderTestRequest) -> dict[str, Any]:
    return get_container().provider_registry.test(payload.provider)


app.include_router(api)


@app.get(
    "/v1/models",
    dependencies=[Depends(require_api_key)],
    tags=["OpenAI Compatibility"],
)
def openai_models() -> dict[str, Any]:
    return {"object": "list", "data": get_container().knowledge_bases.list_openai_models()}


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_api_key)],
    tags=["OpenAI Compatibility"],
)
def chat_completions(payload: dict[str, Any], request: Request) -> Any:
    model = str(payload.get("model", ""))
    parts = model.split(":")
    if len(parts) != 3 or parts[0] != "kb":
        raise HTTPException(status_code=400, detail="Model must be kb:<slug>:<policy>")
    messages = payload.get("messages") or []
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="A user message is required")
    response = get_container().answer_with_evidence.execute(
        AnswerRequest(
            knowledge_base=parts[1],
            policy=parts[2],
            query=str(user_messages[-1].get("content", "")),
            latency_budget_ms=float(payload.get("latency_budget_ms", 1000)),
            cost_budget_usd=payload.get("cost_budget_usd"),
        ).to_command(),
        request_id=request.state.request_id,
    )
    completion_id = f"chatcmpl-{uuid4().hex}"
    if payload.get("stream"):

        def stream():
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": response.answer}}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "contextgate": {
            "run_id": response.run_id,
            "trace_id": response.retrieval.trace_id,
            "selected_retrieval_policy": response.retrieval.policy,
            "selected_provider": response.selected_provider,
            "evidence_score": response.evidence_score,
            "abstained": response.retrieval.abstained,
            "abstention_reason": response.abstention_reason,
            "grounded": response.grounded,
            "cost": response.cost,
            "citations": [_to_plain_dict(citation) for citation in response.citations],
        },
    }
