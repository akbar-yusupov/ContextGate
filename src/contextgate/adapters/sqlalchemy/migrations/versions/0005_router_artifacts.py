"""Add router artifact integrity metadata.

Revision ID: 0005_router_artifacts
Revises: 0004_corpus_versions
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_router_artifacts"
down_revision: str | None = "0004_corpus_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("router_versions") as batch:
        batch.add_column(
            sa.Column("artifact_checksum", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("schema_version", sa.String(length=32), nullable=False, server_default="v1")
        )


def downgrade() -> None:
    with op.batch_alter_table("router_versions") as batch:
        batch.drop_column("schema_version")
        batch.drop_column("artifact_checksum")
