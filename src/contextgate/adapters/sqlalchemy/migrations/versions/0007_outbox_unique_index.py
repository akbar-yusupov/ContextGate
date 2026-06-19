"""Remove the redundant outbox unique constraint.

Revision ID: 0007_outbox_unique_index
Revises: 0006_api_key_scopes
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_outbox_unique_index"
down_revision: str | None = "0006_api_key_scopes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        op.drop_constraint("job_outbox_job_id_key", "job_outbox", type_="unique")


def downgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        op.create_unique_constraint(
            "job_outbox_job_id_key",
            "job_outbox",
            ["job_id"],
        )
