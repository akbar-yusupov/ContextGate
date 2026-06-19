"""Add run lifecycle fields and parent integrity.

Revision ID: 0002_run_integrity
Revises: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_run_integrity"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("gateway_runs") as batch:
        batch.add_column(sa.Column("correlation_id", sa.String(length=128), nullable=True))
        batch.add_column(
            sa.Column("status", sa.String(length=32), nullable=False, server_default="running")
        )
    op.execute("UPDATE gateway_runs SET correlation_id = id WHERE correlation_id IS NULL")
    with op.batch_alter_table("gateway_runs") as batch:
        batch.alter_column("correlation_id", nullable=False)
        batch.create_index("ix_gateway_runs_correlation_id", ["correlation_id"])
        batch.create_index("ix_gateway_runs_status", ["status"])
    with op.batch_alter_table("documents") as batch:
        batch.create_foreign_key(
            "fk_documents_knowledge_base_id",
            "knowledge_bases",
            ["knowledge_base_id"],
            ["id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("router_versions") as batch:
        batch.create_foreign_key(
            "fk_router_versions_knowledge_base_id",
            "knowledge_bases",
            ["knowledge_base_id"],
            ["id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("run_events") as batch:
        batch.create_foreign_key(
            "fk_run_events_run_id",
            "gateway_runs",
            ["run_id"],
            ["id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("cost_records") as batch:
        batch.create_foreign_key(
            "fk_cost_records_run_id",
            "gateway_runs",
            ["run_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("cost_records") as batch:
        batch.drop_constraint("fk_cost_records_run_id", type_="foreignkey")
    with op.batch_alter_table("run_events") as batch:
        batch.drop_constraint("fk_run_events_run_id", type_="foreignkey")
    with op.batch_alter_table("router_versions") as batch:
        batch.drop_constraint("fk_router_versions_knowledge_base_id", type_="foreignkey")
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("fk_documents_knowledge_base_id", type_="foreignkey")
    with op.batch_alter_table("gateway_runs") as batch:
        batch.drop_index("ix_gateway_runs_status")
        batch.drop_index("ix_gateway_runs_correlation_id")
        batch.drop_column("status")
        batch.drop_column("correlation_id")
