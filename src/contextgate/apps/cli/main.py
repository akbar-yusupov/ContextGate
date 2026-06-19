from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from redis import Redis
from sqlalchemy import select, text

from contextgate.adapters.qdrant.vector_index import get_vector_store
from contextgate.adapters.sqlalchemy import KnowledgeBase, SessionLocal, init_db
from contextgate.application.dto import AnswerCommand, RouterPromoteCommand
from contextgate.apps.container import get_container
from contextgate.apps.mlflow.main import healthcheck as mlflow_healthcheck
from contextgate.config import Settings, get_settings
from contextgate.domain.gateway import AnswerStatus

app = typer.Typer(help="ContextGate LLMOps gateway CLI.", no_args_is_help=True)
router_app = typer.Typer(help="Train and promote adaptive routers.")
app.add_typer(router_app, name="router")


def _percent(value: Any) -> str:
    return f"{float(value or 0):.1%}"


def _promotion_failure_messages(
    training: dict[str, Any],
    benchmark: dict[str, Any],
    settings: Settings,
) -> list[str]:
    overall = (benchmark.get("gateway_summary") or {}).get("overall", {})
    metrics = training.get("metrics", {})
    thresholds = training.get("promotion_thresholds", {})
    query_count = int((benchmark.get("metadata") or {}).get("query_count", 0))
    messages: list[str] = []
    for reason in training.get("promotion_failures", []):
        if reason == "insufficient_release_cases":
            messages.append(
                f"release set has {query_count} queries; at least "
                f"{settings.router_min_release_cases} are required"
            )
        elif reason == "insufficient_unanswerable_cases":
            messages.append(
                "release set has too few unanswerable queries; at least "
                f"{settings.router_min_unanswerable_cases} are required"
            )
        elif reason.startswith("insufficient_language_cases:"):
            language = reason.partition(":")[2]
            messages.append(
                f"language '{language}' has fewer than "
                f"{settings.router_min_cases_per_language} queries"
            )
        elif reason == "false_answer_confidence_gate_failed":
            messages.append(
                "false-answer 95% upper bound is "
                f"{_percent(overall.get('false_answer_upper_95'))}; maximum is "
                f"{_percent(settings.router_max_false_answer_upper_95)}"
            )
        elif reason == "citation_confidence_gate_failed":
            messages.append(
                "citation-correctness 95% lower bound is "
                f"{_percent(overall.get('citation_validity_lower_95'))}; minimum is "
                f"{_percent(settings.router_min_citation_lower_95)}"
            )
        elif reason == "claim_support_confidence_gate_failed":
            messages.append(
                "claim-support 95% lower bound is "
                f"{_percent(overall.get('claim_support_lower_95'))}; minimum is "
                f"{_percent(settings.router_min_claim_support_lower_95)}"
            )
        elif reason == "critical_adversarial_false_answer":
            messages.append("at least one adversarial case produced a false answer")
        elif reason == "router_quality_ratio_below_threshold":
            messages.append(
                f"router quality ratio is {_percent(metrics.get('quality_ratio'))}; minimum is "
                f"{_percent(thresholds.get('quality_ratio_min', 0.95))}"
            )
        elif reason == "router_latency_reduction_below_threshold":
            messages.append(
                "router p95 latency reduction is "
                f"{_percent(metrics.get('latency_reduction'))}; minimum is "
                f"{_percent(thresholds.get('latency_reduction_min', 0.15))}"
            )
        else:
            messages.append(reason.replace("_", " "))
    return messages


@app.command("init")
def initialize(
    force: Annotated[bool, typer.Option("--force", help="Replace an existing .env file.")] = False,
) -> None:
    target = Path(".env")
    source = Path(".env.example")
    if target.exists() and not force:
        typer.echo(".env already exists; no changes made")
    else:
        if not source.is_file():
            raise typer.BadParameter(".env.example was not found in the current directory")
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        typer.echo(f"Created {target.resolve()}")
    get_settings.cache_clear()
    get_settings().prepare_directories()
    typer.echo("Initialized ContextGate directories. Start services, then run `ctxgate doctor`.")


@app.command()
def doctor() -> None:
    container = get_container()
    checks: dict[str, dict[str, str | bool]] = {}

    def check(name: str, operation, remedy: str, *, required: bool = True) -> None:
        try:
            detail = operation()
            checks[name] = {"status": "ok", "detail": str(detail), "required": required}
        except Exception as exc:
            checks[name] = {
                "status": "failed",
                "detail": f"{exc.__class__.__name__}: {exc}",
                "remedy": remedy,
                "required": required,
            }

    def docker_check() -> str:
        executable = shutil.which("docker")
        if executable is None:
            raise RuntimeError("docker executable not found")
        result = subprocess.run(
            [executable, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    check(
        "docker",
        docker_check,
        "Install Docker Desktop/Engine and ensure the daemon is running.",
        required=False,
    )

    def security_check() -> str:
        container.settings.validate_runtime_security()
        return "valid"

    check(
        "security",
        security_check,
        "Set production auth, a non-default API key, and a Redis password.",
    )

    def database_check() -> int:
        with SessionLocal() as session:
            return int(session.execute(text("SELECT 1")).scalar_one())

    check("database", database_check, "Start PostgreSQL or correct CONTEXTGATE_DATABASE_*.")
    check(
        "migrations",
        lambda: database_version(),
        "Run `alembic -c src/contextgate/adapters/sqlalchemy/alembic.ini upgrade head`.",
    )
    check(
        "qdrant",
        lambda: len(get_vector_store().client.get_collections().collections),
        "Start Qdrant or correct CONTEXTGATE_QDRANT_*.",
    )

    def embedding_schema_check() -> str:
        store = get_vector_store()
        incompatible: list[str] = []
        knowledge_bases = container.knowledge_bases.list()
        for knowledge_base in knowledge_bases:
            try:
                store.validate_collection_if_exists(knowledge_base.collection_name)
            except ValueError as exc:
                incompatible.append(f"{knowledge_base.slug}: {exc}")
        if incompatible:
            raise RuntimeError(" | ".join(incompatible))
        return f"{len(knowledge_bases)} knowledge base schema(s) compatible"

    check(
        "embedding_schemas",
        embedding_schema_check,
        (
            "Restore each collection's original embedding model/dimensions or create a new "
            "knowledge base and re-ingest. Reset volumes only for the disposable demo."
        ),
    )
    check(
        "redis",
        lambda: Redis.from_url(container.settings.resolved_redis_url).ping(),
        "Start Redis and ensure CONTEXTGATE_REDIS_PASSWORD matches the server.",
    )
    check(
        "policies",
        lambda: sorted(get_container().retrieve_context.retrieval_gateway.policies.policies),
        "Restore configs/policies.yaml or correct CONTEXTGATE_POLICIES_PATH.",
    )
    check(
        "providers",
        lambda: [item["id"] for item in container.provider_registry.list()],
        "Configure a LiteLLM model or use the built-in extractive provider.",
    )
    check(
        "provider_pricing",
        lambda: pricing_check(container.settings),
        "Set both LLM input and output prices before using hard cost budgets.",
    )
    check(
        "storage",
        lambda: storage_check(container.settings.upload_dir, container.settings.report_dir),
        "Grant the ContextGate process write access to upload and report volumes.",
    )
    typer.echo(json.dumps(checks, indent=2))
    if any(item["status"] == "failed" and item["required"] for item in checks.values()):
        raise typer.Exit(code=1)


def database_version() -> str:
    with SessionLocal() as session:
        value = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    return str(value)


def pricing_check(settings) -> str:
    if not settings.llm_model:
        return "extractive provider is local zero-cost"
    if (
        settings.llm_input_cost_per_1m_tokens is None
        or settings.llm_output_cost_per_1m_tokens is None
    ):
        raise RuntimeError("configured LLM pricing is incomplete")
    return "configured"


def storage_check(*directories: Path) -> str:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".contextgate-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    return ", ".join(str(directory.resolve()) for directory in directories)


def _ensure_kb(slug: str, name: str | None = None) -> KnowledgeBase:
    init_db()
    with SessionLocal() as session:
        knowledge_base = session.scalar(select(KnowledgeBase).where(KnowledgeBase.slug == slug))
        if knowledge_base:
            return knowledge_base
        knowledge_base = KnowledgeBase(
            name=name or slug.title(),
            slug=slug,
            description="Created by ctxgate.",
            collection_name=f"contextgate-{slug}",
        )
        session.add(knowledge_base)
        session.commit()
        session.refresh(knowledge_base)
        return knowledge_base


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
) -> None:
    _ensure_kb(knowledge_base)
    with SessionLocal() as session:
        result = get_container().ingestion_service.ingest_path(session, knowledge_base, path)
    get_vector_store().close()
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("sync-qdrant")
def sync_qdrant(
    source_collection: str,
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
) -> None:
    _ensure_kb(knowledge_base)
    with SessionLocal() as session:
        result = get_container().ingestion_service.sync_collection(
            session,
            knowledge_base,
            source_collection,
        )
    get_vector_store().close()
    typer.echo(json.dumps(result, indent=2))


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Argument(exists=True, readable=True)],
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
    evaluate_answers: Annotated[bool, typer.Option("--evaluate-answers")] = False,
) -> None:
    _ensure_kb(knowledge_base)
    with SessionLocal() as session:
        result = get_container().benchmark_service.run(
            session,
            knowledge_base,
            dataset,
            evaluate_answers=evaluate_answers,
        )
    get_vector_store().close()
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@router_app.command("train")
def train_router(
    benchmark_run_id: str,
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
) -> None:
    result = get_container().router_manager.train(benchmark_run_id, knowledge_base)
    typer.echo(json.dumps(result, indent=2))


@router_app.command("promote")
def promote_router(
    benchmark_run_id: str,
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
) -> None:
    result = get_container().promote_policy.execute(
        RouterPromoteCommand(run_id=benchmark_run_id, knowledge_base=knowledge_base)
    )
    typer.echo(f"Promoted {result['path']}")


@router_app.command("rollback")
def rollback_router(
    benchmark_run_id: str,
    knowledge_base: Annotated[str, typer.Option("--knowledge-base", "-k")] = "demo",
) -> None:
    result = get_container().promote_policy.execute(
        RouterPromoteCommand(run_id=benchmark_run_id, knowledge_base=knowledge_base)
    )
    typer.echo(f"Rolled back to {result['path']}")


@app.command()
def demo(
    documents: Annotated[Path, typer.Option(exists=True)] = Path("demo/documents"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("demo/benchmark.jsonl"),
    with_evaluation: Annotated[
        bool,
        typer.Option(
            "--with-evaluation",
            help="Run the optional MLflow benchmark and router-training stage.",
        ),
    ] = False,
) -> None:
    container = get_container()
    typer.echo(
        f"Embedding backend: {container.settings.embedding_backend} "
        f"(dense={container.settings.dense_dimension}, "
        f"late={container.settings.late_dimension})"
    )
    if with_evaluation:
        try:
            mlflow_healthcheck(container.settings)
        except Exception as exc:
            typer.echo(
                "MLflow evaluation is not ready: "
                f"{exc}. Start or recreate it with "
                "`docker compose --profile evaluation up -d --build --wait mlflow`, "
                "then rerun with `ctxgate demo --with-evaluation`.",
                err=True,
            )
            raise typer.Exit(code=2) from None
    try:
        _ensure_kb("demo", "ContextGate multilingual demo")
        with SessionLocal() as session:
            ingest_result = container.ingestion_service.ingest_path(session, "demo", documents)
            typer.echo(f"Ingested: {json.dumps(ingest_result, ensure_ascii=False)}")
            grounded = container.answer_with_evidence.execute(
                AnswerCommand(
                    knowledge_base="demo",
                    query="How can I cancel an order?",
                    policy="auto",
                    limit=5,
                    debug=True,
                )
            )
            typer.echo(
                "Admission check (answerable): "
                f"status={grounded.status.value}, "
                f"evidence_verified={grounded.grounded}, "
                f"citations={len(grounded.citations)}, "
                f"run_id={grounded.run_id}"
            )
            abstention = container.answer_with_evidence.execute(
                AnswerCommand(
                    knowledge_base="demo",
                    query="Do you accept cryptocurrency?",
                    policy="auto",
                    limit=5,
                    debug=True,
                )
            )
            typer.echo(
                "Admission check (unanswerable): "
                f"status={abstention.status.value}, "
                f"reason={abstention.abstention_reason}, answer_empty={not abstention.answer}, "
                f"run_id={abstention.run_id}"
            )
            if not with_evaluation:
                typer.echo(
                    "Optional MLflow evaluation skipped. Use `ctxgate demo --with-evaluation` "
                    "after starting the Compose evaluation profile."
                )
                return
            benchmark_result = container.benchmark_service.run(
                session,
                "demo",
                dataset,
                evaluate_answers=True,
            )
        run_id = benchmark_result["run_id"]
        gateway_summary = (benchmark_result.get("gateway_summary") or {}).get("overall", {})
        if gateway_summary:
            typer.echo(
                "QA gate summary: "
                f"answered={_percent(gateway_summary['answer_rate'])}, "
                f"abstained={_percent(gateway_summary['abstention_rate'])}, "
                f"false_answers={_percent(gateway_summary['false_answer_rate'])}, "
                f"false_abstentions={_percent(gateway_summary['false_abstention_rate'])}, "
                f"citations_valid={_percent(gateway_summary['citation_validity_rate'])}"
            )
            if float(gateway_summary["false_abstention_rate"]) > 0.25:
                typer.echo(
                    "Evaluation warning: the demo is conservative and abstains on many "
                    "answerable cases; inspect failed cases before tuning thresholds."
                )
        public_api = container.settings.api_public_url.rstrip("/")
        public_mlflow = container.settings.mlflow_public_url.rstrip("/")
        typer.echo(f"MLflow UI: {public_mlflow}")
        typer.echo(
            f"QA report API: {public_api}/api/v1/evaluations/{run_id}/report (requires X-API-Key)"
        )
        report_path = Path(benchmark_result["report_path"])
        if Path("/.dockerenv").exists():
            container_report = (
                report_path if report_path.is_absolute() else Path("/app") / report_path
            )
            typer.echo(
                "Export QA report: "
                f"docker compose cp api:{container_report.as_posix()} "
                f"./contextgate-{run_id}.html"
            )
        else:
            typer.echo(f"QA report file: {report_path.resolve()}")
        training = container.router_manager.train(run_id, "demo")
        if training["eligible_for_promotion"]:
            container.router_manager.promote(run_id, "demo")
            typer.echo(f"Router {run_id} promoted for knowledge base demo")
        else:
            typer.echo("Router candidate was not promoted; balanced remains active.")
            for message in _promotion_failure_messages(
                training, benchmark_result, container.settings
            ):
                typer.echo(f"  - {message}")
    finally:
        get_vector_store().close()


@app.command("seed-demo")
def seed_demo(
    documents: Annotated[Path, typer.Option(exists=True)] = Path("demo/documents"),
) -> None:
    """Idempotently seed and verify the no-paid-provider Compose demo."""
    container = get_container()
    try:
        _ensure_kb("demo", "ContextGate multilingual demo")
        with SessionLocal() as session:
            ingestion = container.ingestion_service.ingest_path(session, "demo", documents)
        if ingestion.get("outcome") == "failed":
            typer.echo(json.dumps(ingestion, indent=2, ensure_ascii=False))
            raise typer.Exit(code=1)
        answered = container.answer_with_evidence.execute(
            AnswerCommand(
                knowledge_base="demo",
                query="How can I cancel an order?",
                policy="balanced",
                limit=5,
            )
        )
        abstained = container.answer_with_evidence.execute(
            AnswerCommand(
                knowledge_base="demo",
                query="Do you accept cryptocurrency?",
                policy="balanced",
                limit=5,
            )
        )
        result = {
            "ingestion": ingestion,
            "answered_status": answered.status,
            "answered_run_id": answered.run_id,
            "abstained_status": abstained.status,
            "abstention_reason": abstained.abstention_reason,
            "api": "http://localhost:8000/docs",
            "ui": "http://localhost:8001",
            "grounded_question": "How can I cancel an order?",
            "unanswerable_question": "Do you accept cryptocurrency?",
        }
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if answered.status != AnswerStatus.ANSWERED or abstained.status != AnswerStatus.ABSTAINED:
            raise typer.Exit(code=1)
    finally:
        get_vector_store().close()


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    import uvicorn

    uvicorn.run("contextgate.apps.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
