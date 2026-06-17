from __future__ import annotations

from collections import OrderedDict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from contextgate.adapters.sqlalchemy import CostRecordModel, GatewayRun, RunEvent
from contextgate.domain.models import CostRecord


class SqlAlchemyCostLedger:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def record(self, record: CostRecord) -> None:
        with self.session_factory() as session:
            exists = session.scalar(
                select(CostRecordModel).where(
                    CostRecordModel.request_id == record.request_id,
                    CostRecordModel.run_id == record.run_id,
                    CostRecordModel.provider == record.provider,
                    CostRecordModel.model == record.model,
                )
            )
            if exists:
                return
            session.add(
                CostRecordModel(
                    request_id=record.request_id,
                    run_id=record.run_id,
                    provider=record.provider,
                    model=record.model,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    embedding_units=record.embedding_units,
                    rerank_units=record.rerank_units,
                    estimated_cost_usd=record.estimated_cost_usd,
                )
            )
            session.commit()

    def list_for_run(self, run_id: str) -> list[CostRecord]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(CostRecordModel).where(CostRecordModel.run_id == run_id)
            ).all()
            return [
                CostRecord(
                    request_id=row.request_id,
                    run_id=row.run_id,
                    provider=row.provider,
                    model=row.model,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    embedding_units=row.embedding_units,
                    rerank_units=row.rerank_units,
                    estimated_cost_usd=row.estimated_cost_usd,
                )
                for row in rows
            ]


class SqlAlchemyTraceStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self.session_factory() as session:
            last_sequence = session.scalar(
                select(func.max(RunEvent.sequence)).where(RunEvent.run_id == run_id)
            )
            session.add(
                RunEvent(
                    run_id=run_id,
                    sequence=0 if last_sequence is None else last_sequence + 1,
                    event_type=event_type,
                    payload=payload,
                )
            )
            if event_type == "final":
                existing = session.get(GatewayRun, run_id)
                if existing is None:
                    session.add(
                        GatewayRun(
                            id=run_id,
                            trace_id=str(payload.get("trace_id", run_id)),
                            knowledge_base=str(payload.get("knowledge_base", "")),
                            query=str(payload.get("query", "")),
                            selected_retrieval_policy=str(payload.get("retrieval_policy", "")),
                            selected_provider=str(payload.get("provider", "")),
                            evidence_score=float(payload.get("evidence_score", 0)),
                            answerability_score=float(payload.get("answerability_score", 0)),
                            coverage_score=float(payload.get("coverage_score", 0)),
                            support_score=float(payload.get("support_score", 0)),
                            abstained=bool(payload.get("abstained", False)),
                            metadata_json=payload,
                        )
                    )
            session.commit()

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            events = session.scalars(
                select(RunEvent)
                .where(RunEvent.run_id == run_id)
                .order_by(RunEvent.sequence, RunEvent.created_at)
            ).all()
            return [
                {
                    "id": event.id,
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "event": event.event_type,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ]

    def get_trace(self, run_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            run = session.get(GatewayRun, run_id)
        return {
            "run_id": run_id,
            "run": None
            if run is None
            else {
                "trace_id": run.trace_id,
                "knowledge_base": run.knowledge_base,
                "query": run.query,
                "selected_retrieval_policy": run.selected_retrieval_policy,
                "selected_provider": run.selected_provider,
                "evidence_score": run.evidence_score,
                "abstained": run.abstained,
                "metadata": run.metadata_json,
            },
            "events": self.list_events(run_id),
        }


class InMemoryResponseCache:
    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max_size
        self._items: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
