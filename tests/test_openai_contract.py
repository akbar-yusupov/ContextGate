from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI

import contextgate.apps.api.main as api_module
from contextgate.apps.api.main import app
from contextgate.domain.gateway import AnswerResult, AnswerStatus, Citation
from contextgate.domain.retrieval import RetrievalHit, RetrievalResult, RouteDecision


def _answer() -> AnswerResult:
    hit = RetrievalHit(
        chunk_id="orders:0",
        document_id="orders",
        source="orders.md",
        text="Orders can be cancelled before courier handoff.",
        language="en",
        score=0.9,
        rank=1,
        metadata={},
    )
    retrieval = RetrievalResult(
        query="Can I cancel?",
        policy="balanced",
        abstained=False,
        hits=[hit],
        route=RouteDecision(
            requested_policy="balanced",
            selected_policy="balanced",
            reason="explicit_policy",
            latency_budget_ms=1000,
        ),
        timings_ms={"total": 1.0},
        features={},
        trace_id="trace-openai",
        raw_top_score=0.9,
        abstention_threshold=0.2,
    )
    return AnswerResult(
        answer="Orders can be cancelled before courier handoff. [1]",
        citations=[Citation(index=1, chunk_id=hit.chunk_id, source=hit.source)],
        retrieval=retrieval,
        provider="extractive",
        selected_provider="extractive",
        grounded=True,
        status=AnswerStatus.ANSWERED,
        run_id="run-openai",
        evidence_score=0.9,
        cost={
            "estimated_usd": 0.0,
            "actual_usd": 0.0,
            "input_tokens": 12,
            "output_tokens": 8,
        },
    )


class FakeAnswerGateway:
    def execute(self, request, *, request_id=None):
        return _answer()


class FakeContainer:
    settings = SimpleNamespace(allow_provisional_streaming=False)
    answer_with_evidence = FakeAnswerGateway()


async def _exercise_official_client() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://contextgate.test") as http:
        client = AsyncOpenAI(
            base_url="http://contextgate.test/v1",
            api_key="test-key",
            http_client=http,
        )
        completion = await client.chat.completions.create(
            model="kb:demo:balanced",
            messages=[{"role": "user", "content": "Can I cancel?"}],
        )
        assert completion.usage is not None
        assert completion.usage.total_tokens == 20
        assert completion.choices[0].finish_reason == "stop"
        assert completion.model_extra is not None
        assert completion.model_extra["contextgate"]["status"] == "answered"

        stream = await client.chat.completions.create(
            model="kb:demo:balanced",
            messages=[{"role": "user", "content": "Can I cancel?"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert text.strip() == "Orders can be cancelled before courier handoff. [1]"
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].model_extra is not None
        assert chunks[-1].model_extra["contextgate"]["run_id"] == "run-openai"


def test_official_openai_client_non_streamed_and_streamed(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "get_container", lambda: FakeContainer())
    asyncio.run(_exercise_official_client())
