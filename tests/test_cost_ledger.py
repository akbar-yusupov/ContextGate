from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy import Base, CostRecordModel, GatewayRun
from contextgate.adapters.sqlalchemy.ledger import (
    InMemoryResponseCache,
    SqlAlchemyCostLedger,
    SqlAlchemyTraceStore,
)
from contextgate.application.use_cases import InspectTrace, _bounded_correlation_id
from contextgate.domain.models import CostRecord


class EmptyTraceStore:
    def start_run(self, run_id, *, correlation_id, knowledge_base, query):  # pragma: no cover
        raise NotImplementedError

    def append_event(self, run_id, event_type, payload):  # pragma: no cover
        raise NotImplementedError

    def list_events(self, run_id):
        return []

    def get_trace(self, run_id):
        return {"run_id": run_id, "events": []}

    def purge_older_than(self, days):
        return 0


def test_cost_ledger_records_provider_call_once(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'cost.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    ledger = SqlAlchemyCostLedger(session_factory)
    record = CostRecord(
        request_id="req-1",
        run_id="run-1",
        provider="extractive",
        model="extractive",
        input_tokens=4,
        output_tokens=8,
        embedding_units=3,
        rerank_units=0,
        estimated_cost_usd=0.42,
    )

    ledger.record(record)
    ledger.record(record)
    cost = InspectTrace(EmptyTraceStore(), ledger).cost("run-1")

    assert len(cost["records"]) == 1
    assert cost["records"][0]["input_tokens"] == 4
    assert cost["estimated_usd"] == 0.42


def test_cost_ledger_accepts_evaluation_correlation_ids(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation-cost.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    ledger = SqlAlchemyCostLedger(session_factory)
    request_id = "eval-4946edbe292a4f2fbd31aabed5221f59-en-cancel-order-01-balanced"

    assert len(request_id) > 64
    request_id_type = cast(String, CostRecordModel.__table__.c.request_id.type)
    assert request_id_type.length == 128

    ledger.record(
        CostRecord(
            request_id=request_id,
            run_id="run-1",
            provider="abstention",
            model="abstention",
            input_tokens=0,
            output_tokens=0,
            embedding_units=10,
            rerank_units=0,
            estimated_cost_usd=0.0,
        )
    )

    cost = InspectTrace(EmptyTraceStore(), ledger).cost("run-1")
    assert cost["records"][0]["request_id"] == request_id


def test_correlation_ids_are_bounded_without_truncation_collisions() -> None:
    shared_prefix = "eval-" + "x" * 200
    first = _bounded_correlation_id(f"{shared_prefix}-first")
    second = _bounded_correlation_id(f"{shared_prefix}-second")

    assert len(first) == 128
    assert len(second) == 128
    assert first != second
    assert _bounded_correlation_id("short-id") == "short-id"


def test_trace_store_preserves_node_event_order(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'trace.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    trace_store = SqlAlchemyTraceStore(session_factory)

    trace_store.start_run(
        "run-1",
        correlation_id="request-1",
        knowledge_base="demo",
        query="test",
    )
    trace_store.append_event("run-1", "query_analyzed", {})
    trace_store.append_event("run-1", "retrieval_started", {})
    trace_store.append_event("run-1", "final", {"trace_id": "trace-1"})

    events = trace_store.list_events("run-1")

    assert [event["event"] for event in events] == [
        "query_analyzed",
        "retrieval_started",
        "final",
    ]
    assert [event["sequence"] for event in events] == [0, 1, 2]

    trace = trace_store.get_trace("run-1")
    assert trace["run"]["status"] == "abstained"
    assert trace_store.get_trace("missing")["run"] is None
    with pytest.raises(ValueError, match="already exists"):
        trace_store.start_run(
            "run-1", correlation_id="again", knowledge_base="demo", query="duplicate"
        )


def test_trace_retention_and_bounded_in_memory_cache(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'retention.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    trace_store = SqlAlchemyTraceStore(session_factory)
    trace_store.start_run("old-run", correlation_id="old", knowledge_base="demo", query="old")
    with session_factory() as session:
        run = session.scalar(select(GatewayRun).where(GatewayRun.id == "old-run"))
        assert run is not None
        run.created_at = datetime.now(UTC) - timedelta(days=10)
        session.commit()

    assert trace_store.purge_older_than(5) == 1

    cache = InMemoryResponseCache(max_size=1)
    cache.set("first", 1)
    assert cache.get("first") == 1
    cache.set("second", 2)
    assert cache.get("first") is None
    assert cache.get("second") == 2
