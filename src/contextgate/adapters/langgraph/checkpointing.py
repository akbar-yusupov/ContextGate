from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(slots=True)
class CheckpointerResource:
    saver: Any
    pool: Any

    def close(self) -> None:
        self.pool.close()


def create_postgres_checkpointer(database_url: str) -> CheckpointerResource | None:
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        return None

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    pool = ConnectionPool(
        conninfo,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    saver = PostgresSaver(cast(Any, pool))
    saver.setup()
    return CheckpointerResource(saver=saver, pool=pool)
