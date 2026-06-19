"""Align cost correlation IDs with gateway runs.

Revision ID: 0008_cost_request_id_length
Revises: 0007_outbox_unique_index
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_cost_request_id_length"
down_revision: str | None = "0007_outbox_unique_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("cost_records") as batch_op:
        batch_op.alter_column(
            "request_id",
            existing_type=sa.String(length=64),
            type_=sa.String(length=128),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("cost_records") as batch_op:
        batch_op.alter_column(
            "request_id",
            existing_type=sa.String(length=128),
            type_=sa.String(length=64),
            existing_nullable=False,
        )
