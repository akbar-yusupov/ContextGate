from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy import Base
from contextgate.adapters.sqlalchemy.ledger import SqlAlchemyCostLedger, SqlAlchemyTraceStore
from contextgate.application.use_cases import InspectTrace
from contextgate.domain.models import CostRecord


class EmptyTraceStore:
    def append_event(self, run_id, event_type, payload):  # pragma: no cover
        raise NotImplementedError

    def list_events(self, run_id):
        return []

    def get_trace(self, run_id):
        return {"run_id": run_id, "events": []}


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


def test_trace_store_preserves_node_event_order(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'trace.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    trace_store = SqlAlchemyTraceStore(session_factory)

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
