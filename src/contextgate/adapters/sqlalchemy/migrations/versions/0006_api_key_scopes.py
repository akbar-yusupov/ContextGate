"""Add API key scopes and usage metadata.

Revision ID: 0006_api_key_scopes
Revises: 0005_router_artifacts
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_api_key_scopes"
down_revision: str | None = "0005_router_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.add_column(sa.Column("scopes_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        'UPDATE api_keys SET scopes_json = \'["read", "write", "admin"]\' WHERE scopes_json IS NULL'
    )
    with op.batch_alter_table("api_keys") as batch:
        batch.alter_column("scopes_json", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("last_used_at")
        batch.drop_column("scopes_json")
