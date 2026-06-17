"""Initial ContextGate application schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("collection_name", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_name"),
    )
    op.create_index("ix_knowledge_bases_slug", "knowledge_bases", ["slug"], unique=True)
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=False),
        sa.Column("external_id", sa.String(length=256), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_external_id", "documents", ["external_id"])
    op.create_index("ix_documents_knowledge_base_id", "documents", ["knowledge_base_id"])
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_json", sa.JSON(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kind", "idempotency_key", name="uq_jobs_kind_idempotency_key"),
    )
    op.create_index("ix_jobs_idempotency_key", "jobs", ["idempotency_key"])
    op.create_table(
        "router_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "knowledge_base_id",
            "run_id",
            name="uq_router_version_kb_run",
        ),
    )
    op.create_index(
        "ix_router_versions_knowledge_base_id",
        "router_versions",
        ["knowledge_base_id"],
    )
    op.create_index("ix_router_versions_run_id", "router_versions", ["run_id"])
    op.create_table(
        "gateway_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("knowledge_base", sa.String(length=128), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("selected_retrieval_policy", sa.String(length=32), nullable=False),
        sa.Column("selected_provider", sa.String(length=128), nullable=False),
        sa.Column("evidence_score", sa.Float(), nullable=False),
        sa.Column("answerability_score", sa.Float(), nullable=False),
        sa.Column("coverage_score", sa.Float(), nullable=False),
        sa.Column("support_score", sa.Float(), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_gateway_runs_knowledge_base", "gateway_runs", ["knowledge_base"])
    op.create_index("ix_gateway_runs_trace_id", "gateway_runs", ["trace_id"])
    op.create_table(
        "run_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
    )
    op.create_index("ix_run_events_event_type", "run_events", ["event_type"])
    op.create_index("ix_run_events_run_id_sequence", "run_events", ["run_id", "sequence"])
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])
    op.create_table(
        "cost_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("embedding_units", sa.Integer(), nullable=False),
        sa.Column("rerank_units", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_id",
            "run_id",
            "provider",
            "model",
            name="uq_cost_records_provider_call",
        ),
    )
    op.create_index("ix_cost_records_request_id", "cost_records", ["request_id"])
    op.create_index("ix_cost_records_run_id", "cost_records", ["run_id"])
    op.create_table(
        "gateway_policies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("retrieval_policy", sa.String(length=32), nullable=False),
        sa.Column("provider_policy", sa.String(length=128), nullable=False),
        sa.Column("latency_budget_ms", sa.Float(), nullable=False),
        sa.Column("cost_budget_usd", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
        sa.UniqueConstraint("name"),
    )


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("gateway_policies")
    op.drop_index("ix_cost_records_run_id", table_name="cost_records")
    op.drop_index("ix_cost_records_request_id", table_name="cost_records")
    op.drop_table("cost_records")
    op.drop_index("ix_run_events_run_id", table_name="run_events")
    op.drop_index("ix_run_events_run_id_sequence", table_name="run_events")
    op.drop_index("ix_run_events_event_type", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("ix_gateway_runs_trace_id", table_name="gateway_runs")
    op.drop_index("ix_gateway_runs_knowledge_base", table_name="gateway_runs")
    op.drop_table("gateway_runs")
    op.drop_index("ix_router_versions_run_id", table_name="router_versions")
    op.drop_index(
        "ix_router_versions_knowledge_base_id",
        table_name="router_versions",
    )
    op.drop_table("router_versions")
    op.drop_index("ix_jobs_idempotency_key", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_documents_knowledge_base_id", table_name="documents")
    op.drop_index("ix_documents_external_id", table_name="documents")
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_knowledge_bases_slug", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
