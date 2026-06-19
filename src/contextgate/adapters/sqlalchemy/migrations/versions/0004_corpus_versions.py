"""Add corpus and document version integrity.

Revision ID: 0004_corpus_versions
Revises: 0003_job_outbox
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_corpus_versions"
down_revision: str | None = "0003_job_outbox"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.add_column(
            sa.Column("corpus_version", sa.Integer(), nullable=False, server_default="0")
        )
    with op.batch_alter_table("documents") as batch:
        batch.create_unique_constraint(
            "uq_document_version",
            ["knowledge_base_id", "external_id", "content_hash", "pipeline_version"],
        )


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("uq_document_version", type_="unique")
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_column("corpus_version")
