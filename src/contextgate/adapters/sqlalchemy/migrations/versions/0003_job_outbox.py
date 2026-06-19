"""Add transactional job outbox.

Revision ID: 0003_job_outbox
Revises: 0002_run_integrity
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_job_outbox"
down_revision: str | None = "0002_run_integrity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "job_outbox",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_job_outbox_job_id", "job_outbox", ["job_id"], unique=True)
    op.create_index("ix_job_outbox_status", "job_outbox", ["status"])


def downgrade() -> None:
    op.drop_index("ix_job_outbox_status", table_name="job_outbox")
    op.drop_index("ix_job_outbox_job_id", table_name="job_outbox")
    op.drop_table("job_outbox")
