from __future__ import annotations

import json
import logging
import queue
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Thread
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from contextgate.adapters.local.loaders import SUPPORTED_SUFFIXES, validate_document_file
from contextgate.apps.api.dependencies import RateLimiter, require_api_key, require_scope
from contextgate.apps.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    ApiKeyCreateRequest,
    ApiKeyResponse,
    BenchmarkRequest,
    DocumentRead,
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
    RouterVersionRead,
    SyncQdrantRequest,
)
from contextgate.apps.container import get_container
from contextgate.config import get_settings
from contextgate.domain.errors import ContextGateError
from contextgate.domain.gateway import AnswerStatus
from contextgate.observability.metrics import REQUEST_LATENCY, REQUESTS

settings = get_settings()
rate_limiter = RateLimiter()
logger = logging.getLogger("contextgate.api")
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
    {"name": "API Keys", "description": "Create, rotate and disable scoped API keys."},
    {"name": "OpenAI Compatibility", "description": "OpenAI-compatible model and chat endpoints."},
]


def _operation_id(route: APIRoute) -> str:
    return route.name


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, dict):
        return value
    return dict(value)


def _idempotency_key(request: Request) -> str | None:
    value = request.headers.get("Idempotency-Key")
    if value is not None and len(value) > 128:
        raise ContextGateError(
            "validation_error",
            "Idempotency-Key exceeds the 128-character limit.",
            {"max_length": 128},
        )
    return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_container().startup()
    yield
    await rate_limiter.redis.aclose()
    get_container().shutdown()


app = FastAPI(
    title="ContextGate",
    version="0.2.0",
    description="Evidence-Gated RAG Gateway for grounded answers, citations, traces and provider policies.",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    generate_unique_id_function=_operation_id,
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
    route = request.scope.get("route")
    metric_path = getattr(route, "path", request.url.path)
    REQUESTS.labels(
        method=request.method,
        path=metric_path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, path=metric_path).observe(duration)
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request.state.request_id,
                "method": request.method,
                "path": metric_path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 3),
            },
            separators=(",", ":"),
        )
    )
    return response


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0", "service": "contextgate"}


@app.get("/ready", tags=["System"])
async def ready() -> JSONResponse:
    try:
        checks = get_container().readiness()
        await rate_limiter.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": exc.__class__.__name__},
        )
    return JSONResponse(content={"status": "ready", "checks": checks})


@app.get("/metrics", tags=["System"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@api.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseRead,
    status_code=201,
    tags=["Knowledge Bases"],
    dependencies=[Depends(require_scope("admin"))],
)
def create_knowledge_base(payload: KnowledgeBaseCreate) -> Any:
    return get_container().knowledge_bases.create(payload.to_command())


@api.get(
    "/knowledge-bases",
    response_model=list[KnowledgeBaseRead],
    tags=["Knowledge Bases"],
    dependencies=[Depends(require_scope("read"))],
)
def list_knowledge_bases() -> list[Any]:
    """List knowledge bases available to retrieval and OpenAI-compatible models."""
    return get_container().knowledge_bases.list()


@api.get(
    "/knowledge-bases/{identifier}",
    response_model=KnowledgeBaseRead,
    tags=["Knowledge Bases"],
    dependencies=[Depends(require_scope("read"))],
)
def get_knowledge_base(identifier: str) -> Any:
    """Get one knowledge base by UUID or slug."""
    return get_container().knowledge_bases.get(identifier)


@api.get(
    "/knowledge-bases/{identifier}/documents",
    response_model=list[DocumentRead],
    tags=["Knowledge Bases"],
    dependencies=[Depends(require_scope("read"))],
)
def list_knowledge_base_documents(identifier: str) -> list[Any]:
    """List persisted document versions for a knowledge base."""
    return get_container().knowledge_bases.list_documents(identifier)


@api.post(
    "/knowledge-bases/{identifier}/documents",
    response_model=JobRead,
    status_code=202,
    tags=["Ingestion"],
    dependencies=[Depends(require_scope("admin"))],
)
def upload_document(identifier: str, request: Request, file: Annotated[UploadFile, File()]) -> Any:
    container = get_container()
    idempotency_key = _idempotency_key(request)
    knowledge_base = container.knowledge_bases.get(identifier)
    try:
        container.ingestion_service.store.validate_collection_if_exists(
            knowledge_base.collection_name
        )
    except ValueError as exc:
        raise ContextGateError(
            "validation_error",
            "This knowledge base uses an embedding schema that does not match the running service.",
            {
                "error": str(exc),
                "remedy": (
                    "Restore the model and dimension settings used to create this collection, "
                    "or create a new knowledge base. For the disposable demo only, reset its "
                    "volumes and ingest again."
                ),
            },
        ) from exc
    filename = Path(file.filename or "file").name
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ContextGateError(
            "validation_error",
            "Unsupported document type.",
            {"supported_suffixes": sorted(SUPPORTED_SUFFIXES)},
        )
    destination = container.settings.upload_dir / knowledge_base.id / str(uuid4()) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("xb") as output:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > container.settings.max_upload_bytes:
                    raise ContextGateError(
                        "validation_error",
                        "Uploaded document exceeds the configured size limit.",
                        {"max_upload_bytes": container.settings.max_upload_bytes},
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    try:
        validate_document_file(destination)
    except (UnicodeDecodeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise ContextGateError(
            "validation_error",
            "Uploaded document content does not match its supported format.",
            {"error": str(exc)},
        ) from exc
    return container.ingest_documents.enqueue(
        knowledge_base=identifier,
        path=destination,
        idempotency_key=idempotency_key,
    )


@api.post(
    "/knowledge-bases/{identifier}/sync-qdrant",
    response_model=JobRead,
    status_code=202,
    tags=["Ingestion"],
    dependencies=[Depends(require_scope("admin"))],
)
def sync_qdrant(identifier: str, payload: SyncQdrantRequest, request: Request) -> Any:
    return get_container().sync_qdrant.enqueue(
        knowledge_base=identifier,
        request=payload.to_command(),
        idempotency_key=_idempotency_key(request),
    )


@api.get(
    "/jobs/{job_id}",
    response_model=JobRead,
    tags=["Jobs"],
    dependencies=[Depends(require_scope("read"))],
)
def get_job(job_id: str) -> Any:
    return get_container().knowledge_bases.get_job(job_id)


@api.post(
    "/jobs/{job_id}/cancel",
    response_model=JobRead,
    tags=["Jobs"],
    dependencies=[Depends(require_scope("admin"))],
)
def cancel_job(job_id: str) -> Any:
    return get_container().cancel_job.execute(job_id)


@api.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Retrieval"],
    dependencies=[Depends(require_scope("write"))],
)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    return RetrieveResponse.from_domain(
        get_container().retrieve_context.execute(payload.to_command())
    )


@api.post(
    "/runs/answer",
    response_model=AnswerResponse,
    tags=["Answer Runtime"],
    dependencies=[Depends(require_scope("write"))],
)
def run_answer(payload: AnswerRequest, request: Request) -> Any:
    container = get_container()
    if payload.stream_mode == "provisional" and not container.settings.allow_provisional_streaming:
        raise ContextGateError(
            "policy_rejected",
            "Provisional token streaming is disabled by the active strict policy.",
        )
    if payload.stream_mode == "provisional":
        return StreamingResponse(
            _native_provisional_stream(
                container,
                payload.to_command(),
                request.state.request_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = container.answer_with_evidence.execute(
        payload.to_command(),
        request_id=request.state.request_id,
    )
    response = AnswerResponse.from_domain(result)
    if payload.stream_mode == "none":
        return JSONResponse(content=response.model_dump(mode="json"))

    def stream_answer():
        yield f"event: decision\ndata: {json.dumps({'run_id': response.run_id, 'status': response.status})}\n\n"
        if response.status == AnswerStatus.ANSWERED:
            for token in response.answer.split():
                yield f"event: token_delta\ndata: {json.dumps({'text': token + ' ', 'provisional': False})}\n\n"
        yield f"event: final\ndata: {response.model_dump_json()}\n\n"

    return StreamingResponse(
        stream_answer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get(
    "/runs/{run_id}",
    tags=["Runs/Traces"],
    dependencies=[Depends(require_scope("read"))],
)
def get_run(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.run(run_id)


@api.get(
    "/runs/{run_id}/events",
    tags=["Runs/Traces"],
    dependencies=[Depends(require_scope("read"))],
)
def get_run_events(
    run_id: str,
    after_sequence: int = Query(default=-1, ge=-1),
    follow: bool = True,
) -> StreamingResponse:
    def stream():
        sequence = after_sequence
        deadline = time.monotonic() + 120
        while True:
            events = [
                event
                for event in get_container().inspect_trace.events(run_id)
                if event["sequence"] > sequence
            ]
            for event in events:
                sequence = event["sequence"]
                yield f"id: {sequence}\nevent: {event['event']}\n"
                yield f"data: {json.dumps(event['payload'])}\n\n"
                if event["event"] == "final":
                    return
            if not follow or time.monotonic() >= deadline:
                return
            time.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get(
    "/runs/{run_id}/trace",
    tags=["Runs/Traces"],
    dependencies=[Depends(require_scope("read"))],
)
def get_run_trace(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.run(run_id)


@api.get(
    "/runs/{run_id}/cost",
    tags=["Runs/Traces"],
    dependencies=[Depends(require_scope("read"))],
)
def get_run_cost(run_id: str) -> dict[str, Any]:
    return get_container().inspect_trace.cost(run_id)


@api.post(
    "/benchmarks",
    response_model=JobRead,
    status_code=202,
    tags=["Evaluations"],
    dependencies=[Depends(require_scope("admin"))],
)
def benchmark(payload: BenchmarkRequest, request: Request) -> Any:
    settings = get_container().settings
    dataset_name = _safe_identifier(payload.dataset_path)
    dataset_path = settings.evaluation_dataset_dir / dataset_name
    if not dataset_path.is_file():
        raise ContextGateError(
            "not_found",
            "Evaluation dataset was not uploaded.",
            {"dataset_id": dataset_name},
        )
    command = payload.to_command()
    command = type(command)(
        knowledge_base=command.knowledge_base,
        dataset_path=str(dataset_path),
        policies=command.policies,
        evaluate_answers=command.evaluate_answers,
    )
    return get_container().run_benchmark.enqueue(
        command,
        idempotency_key=_idempotency_key(request),
    )


@api.post(
    "/evaluations/datasets",
    status_code=201,
    tags=["Evaluations"],
    dependencies=[Depends(require_scope("admin"))],
)
def upload_evaluation_dataset(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    if Path(file.filename or "").suffix.lower() != ".jsonl":
        raise ContextGateError("validation_error", "Evaluation datasets must be JSONL files.")
    settings = get_container().settings
    dataset_id = f"dataset-{uuid4().hex}.jsonl"
    target = settings.evaluation_dataset_dir / dataset_id
    written = 0
    try:
        with target.open("xb") as output:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise ContextGateError(
                        "validation_error",
                        "Evaluation dataset exceeds the configured size limit.",
                    )
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {"dataset_id": dataset_id, "size_bytes": written}


@api.post(
    "/evaluations",
    response_model=JobRead,
    status_code=202,
    tags=["Evaluations"],
    dependencies=[Depends(require_scope("admin"))],
)
def create_evaluation(payload: BenchmarkRequest, request: Request) -> Any:
    return benchmark(payload, request)


@api.get(
    "/evaluations/{run_id}",
    tags=["Evaluations"],
    dependencies=[Depends(require_scope("read"))],
)
def get_evaluation(run_id: str) -> dict[str, Any]:
    safe_run_id = _safe_identifier(run_id)
    results = get_container().settings.report_dir / safe_run_id / "results.json"
    if not results.is_file():
        raise ContextGateError("not_found", "Evaluation results not found.")
    return json.loads(results.read_text(encoding="utf-8"))


@api.get(
    "/evaluations/{run_id}/report",
    tags=["Evaluations"],
    dependencies=[Depends(require_scope("read"))],
)
def get_evaluation_report(run_id: str) -> FileResponse:
    safe_run_id = _safe_identifier(run_id)
    report = get_container().settings.report_dir / safe_run_id / "report.html"
    if not report.is_file():
        raise ContextGateError("not_found", "Evaluation report not found.")
    return FileResponse(report, media_type="text/html", filename=f"contextgate-{safe_run_id}.html")


@api.post(
    "/routers/train",
    response_model=JobRead,
    status_code=202,
    tags=["Routers"],
    dependencies=[Depends(require_scope("admin"))],
)
def train_router(payload: RouterTrainRequest, request: Request) -> Any:
    return get_container().train_router.enqueue(
        payload.to_command(),
        idempotency_key=_idempotency_key(request),
    )


@api.post(
    "/routers/promote",
    tags=["Routers"],
    dependencies=[Depends(require_scope("admin"))],
)
def promote_router(payload: RouterPromoteRequest) -> dict[str, str]:
    return get_container().promote_policy.execute(payload.to_command())


@api.post(
    "/routers/rollback",
    tags=["Routers"],
    dependencies=[Depends(require_scope("admin"))],
)
def rollback_router(payload: RouterPromoteRequest) -> dict[str, str]:
    result = get_container().promote_policy.execute(payload.to_command())
    return {**result, "status": "rolled_back"}


@api.get(
    "/routers/{knowledge_base}/versions",
    response_model=list[RouterVersionRead],
    tags=["Routers"],
    dependencies=[Depends(require_scope("read"))],
)
def list_router_versions(knowledge_base: str) -> list[Any]:
    """List router candidates and the active version for a knowledge base."""
    return get_container().router_versions.list(knowledge_base)


@api.post(
    "/policies",
    response_model=PolicyRead,
    status_code=201,
    tags=["Policies"],
    dependencies=[Depends(require_scope("admin"))],
)
def create_policy(payload: PolicyCreateRequest) -> PolicyRead:
    return get_container().policies.create(payload.to_command())


@api.get(
    "/policies",
    response_model=list[PolicyRead],
    tags=["Policies"],
    dependencies=[Depends(require_scope("read"))],
)
def list_policies() -> list[Any]:
    """List immutable gateway policy definitions and lifecycle status."""
    return get_container().policies.list()


@api.get(
    "/policies/{policy_id}",
    response_model=PolicyRead,
    tags=["Policies"],
    dependencies=[Depends(require_scope("read"))],
)
def get_policy(policy_id: str) -> PolicyRead:
    return get_container().policies.get(policy_id)


@api.post(
    "/policies/{policy_id}/promote",
    response_model=PolicyRead,
    tags=["Policies"],
    dependencies=[Depends(require_scope("admin"))],
)
def promote_configured_policy(policy_id: str) -> PolicyRead:
    return get_container().policies.promote(policy_id)


@api.get(
    "/providers",
    tags=["Providers"],
    dependencies=[Depends(require_scope("read"))],
)
def list_providers() -> dict[str, Any]:
    return {"providers": get_container().provider_registry.list()}


@api.post(
    "/providers/test",
    tags=["Providers"],
    dependencies=[Depends(require_scope("admin"))],
)
def test_provider(payload: ProviderTestRequest) -> dict[str, Any]:
    return get_container().provider_registry.test(payload.provider)


@api.post(
    "/api-keys",
    response_model=ApiKeyResponse,
    status_code=201,
    tags=["API Keys"],
    dependencies=[Depends(require_scope("admin"))],
)
def create_api_key(payload: ApiKeyCreateRequest) -> ApiKeyResponse:
    record, secret = get_container().api_keys.create(payload.name, list(payload.scopes))
    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        scopes=record.scopes_json,
        key=secret,
    )


@api.get(
    "/api-keys",
    response_model=list[ApiKeyResponse],
    tags=["API Keys"],
    dependencies=[Depends(require_scope("admin"))],
)
def list_api_keys() -> list[ApiKeyResponse]:
    """List key metadata; secret values are never persisted or returned."""
    return [
        ApiKeyResponse(
            id=record.id,
            name=record.name,
            enabled=record.enabled,
            scopes=record.scopes_json,
        )
        for record in get_container().api_keys.list()
    ]


@api.post(
    "/api-keys/{key_id}/rotate",
    response_model=ApiKeyResponse,
    tags=["API Keys"],
    dependencies=[Depends(require_scope("admin"))],
)
def rotate_api_key(key_id: str) -> ApiKeyResponse:
    record, secret = get_container().api_keys.rotate(key_id)
    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        scopes=record.scopes_json,
        key=secret,
    )


@api.delete(
    "/api-keys/{key_id}",
    response_model=ApiKeyResponse,
    tags=["API Keys"],
    dependencies=[Depends(require_scope("admin"))],
)
def disable_api_key(key_id: str) -> ApiKeyResponse:
    record = get_container().api_keys.disable(key_id)
    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        enabled=record.enabled,
        scopes=record.scopes_json,
    )


app.include_router(api)


@app.get(
    "/v1/models",
    dependencies=[Depends(require_api_key), Depends(require_scope("read"))],
    tags=["OpenAI Compatibility"],
)
def openai_models() -> dict[str, Any]:
    return {"object": "list", "data": get_container().knowledge_bases.list_openai_models()}


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_api_key), Depends(require_scope("write"))],
    tags=["OpenAI Compatibility"],
)
def chat_completions(payload: dict[str, Any], request: Request) -> Any:
    model = str(payload.get("model", ""))
    parts = model.split(":")
    if len(parts) != 3 or parts[0] != "kb":
        raise HTTPException(status_code=400, detail="Model must be kb:<slug>:<policy>")
    if parts[2] not in {"auto", "fast", "balanced", "accurate"}:
        raise HTTPException(status_code=400, detail="Unknown retrieval policy")
    messages = payload.get("messages") or []
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="A user message is required")
    user_content = user_messages[-1].get("content", "")
    if not isinstance(user_content, str) or not user_content.strip():
        raise HTTPException(status_code=400, detail="The final user message must be text")
    system_messages = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") in {"system", "developer"}
    ]
    container = get_container()
    requested_stream_mode = str((payload.get("contextgate") or {}).get("stream_mode", "verified"))
    if (
        payload.get("stream")
        and requested_stream_mode == "provisional"
        and not container.settings.allow_provisional_streaming
    ):
        raise HTTPException(status_code=409, detail="Provisional streaming is disabled")
    command = AnswerRequest(
        knowledge_base=parts[1],
        policy=parts[2],
        query=user_content,
        latency_budget_ms=float(payload.get("latency_budget_ms", 1000)),
        cost_budget_usd=payload.get("cost_budget_usd"),
        system_prompt="\n".join(system_messages) or None,
    ).to_command()
    completion_id = f"chatcmpl-{uuid4().hex}"
    if payload.get("stream") and requested_stream_mode == "provisional":
        return StreamingResponse(
            _openai_provisional_stream(
                container,
                command,
                request.state.request_id,
                completion_id,
                model,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    response = container.answer_with_evidence.execute(
        command,
        request_id=request.state.request_id,
    )
    if payload.get("stream"):

        def stream():
            if response.status == AnswerStatus.ANSWERED:
                for token in response.answer.split():
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": token + " "}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "contextgate": _contextgate_metadata(response),
            }
            yield f"data: {json.dumps(final_chunk, default=str)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": _openai_usage(response),
        "contextgate": _contextgate_metadata(response),
    }


def _openai_usage(response: Any) -> dict[str, int]:
    prompt = int(response.cost.get("input_tokens", 0))
    completion = int(response.cost.get("output_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _contextgate_metadata(response: Any) -> dict[str, Any]:
    return {
        "run_id": response.run_id,
        "trace_id": response.retrieval.trace_id,
        "status": response.status,
        "selected_retrieval_policy": response.retrieval.policy,
        "selected_provider": response.selected_provider,
        "evidence_score": response.evidence_score,
        "abstained": response.status != AnswerStatus.ANSWERED,
        "abstention_reason": response.abstention_reason,
        "grounded": response.grounded,
        "cost": response.cost,
        "citations": [_to_plain_dict(citation) for citation in response.citations],
        "evidence_report": _to_plain_dict(response.evidence_report)
        if response.evidence_report
        else None,
    }


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or value in {".", ".."}:
        raise ContextGateError("validation_error", "Identifier contains unsafe path characters.")
    return value


def _provisional_execution(container: Any, command: Any, correlation_id: str):
    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def execute() -> None:
        try:
            result = container.answer_with_evidence.execute(
                command,
                request_id=correlation_id,
                token_callback=lambda value: events.put(("token", value)),
            )
            events.put(("final", result))
        except Exception as exc:
            events.put(("error", exc))

    Thread(target=execute, daemon=True).start()
    while True:
        kind, value = events.get()
        yield kind, value
        if kind in {"final", "error"}:
            return


def _native_provisional_stream(container: Any, command: Any, correlation_id: str):
    yield f"event: provisional\ndata: {json.dumps({'provisional': True})}\n\n"
    for kind, value in _provisional_execution(container, command, correlation_id):
        if kind == "token":
            yield f"event: token_delta\ndata: {json.dumps({'text': value, 'provisional': True})}\n\n"
        elif kind == "error":
            yield f"event: error\ndata: {json.dumps({'type': value.__class__.__name__})}\n\n"
        else:
            response = AnswerResponse.from_domain(value)
            if response.status != AnswerStatus.ANSWERED:
                yield f"event: retraction\ndata: {json.dumps({'reason': response.abstention_reason})}\n\n"
            yield f"event: final\ndata: {response.model_dump_json()}\n\n"


def _openai_provisional_stream(
    container: Any,
    command: Any,
    correlation_id: str,
    completion_id: str,
    model: str,
):
    for kind, value in _provisional_execution(container, command, correlation_id):
        if kind == "token":
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": value}}],
                "contextgate": {"provisional": True},
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        elif kind == "error":
            yield f"data: {json.dumps({'error': {'type': value.__class__.__name__}})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            retracted = value.status != AnswerStatus.ANSWERED
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "contextgate": {
                    **_contextgate_metadata(value),
                    "provisional": False,
                    "retracted": retracted,
                },
            }
            yield f"data: {json.dumps(final_chunk, default=str)}\n\n"
            yield "data: [DONE]\n\n"
