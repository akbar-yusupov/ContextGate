from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from contextgate.adapters.qdrant.vector_index import get_vector_store
from contextgate.adapters.sqlalchemy import KnowledgeBase, SessionLocal, init_db
from contextgate.application.dto import AnswerCommand
from contextgate.apps.container import get_container

app = typer.Typer(help="ContextGate LLMOps gateway CLI.", no_args_is_help=True)
router_app = typer.Typer(help="Train and promote adaptive routers.")
app.add_typer(router_app, name="router")


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
    path = get_container().router_manager.promote(benchmark_run_id, knowledge_base)
    typer.echo(f"Promoted {path}")


@app.command()
def demo(
    documents: Annotated[Path, typer.Option(exists=True)] = Path("demo/documents"),
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("demo/benchmark.jsonl"),
) -> None:
    container = get_container()
    typer.echo(f"Embedding backend: {container.settings.embedding_backend}")
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
                "Grounded demo answer: "
                f"grounded={grounded.grounded}, "
                f"abstention_reason={grounded.abstention_reason}, "
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
                "Unanswerable demo answer: "
                f"grounded={abstention.grounded}, "
                f"abstention_reason={abstention.abstention_reason}, "
                f"citations={len(abstention.citations)}, "
                f"run_id={abstention.run_id}"
            )
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
                "QA gate report summary: "
                f"answer_rate={gateway_summary['answer_rate']:.3f}, "
                f"abstention_rate={gateway_summary['abstention_rate']:.3f}, "
                f"false_answer_rate={gateway_summary['false_answer_rate']:.3f}, "
                f"false_abstention_rate={gateway_summary['false_abstention_rate']:.3f}"
            )
        typer.echo(f"QA gate report: {benchmark_result['report_path']}")
        training = container.router_manager.train(run_id, "demo")
        if training["eligible_for_promotion"]:
            container.router_manager.promote(run_id, "demo")
            typer.echo(f"Router {run_id} promoted for knowledge base demo")
        else:
            typer.echo("Router failed release gates; balanced remains the fallback")
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
