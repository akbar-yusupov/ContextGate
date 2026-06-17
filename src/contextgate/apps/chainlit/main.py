from __future__ import annotations

import json

import chainlit as cl
import httpx
from chainlit.input_widget import Select, Slider, TextInput

from contextgate.config import get_settings

settings = get_settings()
API_URL = settings.ui_api_url
API_KEY = settings.api_key


async def _settings():
    return await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Mode",
                values=[
                    "Chat",
                    "Retrieval Inspector",
                    "Evidence Inspector",
                    "Cost/Provider Inspector",
                    "Policy Compare",
                    "Evaluation Report Viewer",
                ],
                initial_index=0,
            ),
            TextInput(id="knowledge_base", label="Knowledge base", initial="demo"),
            TextInput(id="run_id", label="Run ID for inspectors", initial=""),
            Select(
                id="policy",
                label="Retrieval policy",
                values=["auto", "fast", "balanced", "accurate"],
                initial_index=0,
            ),
            Slider(
                id="latency_budget_ms",
                label="Latency budget (ms)",
                initial=1000,
                min=100,
                max=5000,
                step=100,
            ),
        ]
    ).send()


@cl.on_chat_start
async def on_chat_start():
    settings = await _settings()
    cl.user_session.set("settings", settings)
    await cl.Message(
        content=(
            "ContextGate operator console is ready. The **Why this answer?** panel shows "
            "route, evidence, provider, citations, rejected claims and estimated cost."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)


async def _get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.get(f"{API_URL}{path}", headers={"X-API-Key": API_KEY})
        response.raise_for_status()
        return response.json()


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{API_URL}{path}",
            json=payload,
            headers={"X-API-Key": API_KEY},
        )
        response.raise_for_status()
        return response.json()


def _why_panel(data: dict) -> str:
    retrieval = data["retrieval"]
    route = retrieval["route"]
    timings = retrieval["timings_ms"]
    return (
        "### Why this answer?\n"
        f"- Retrieval policy: `{route['selected_policy']}` ({route['reason']})\n"
        f"- Provider: `{data.get('selected_provider', data.get('provider'))}`\n"
        f"- Evidence score: `{data.get('evidence_score', 0):.3f}`\n"
        f"- Answerability/Coverage/Support: "
        f"`{data.get('answerability_score', 0):.3f}` / "
        f"`{data.get('coverage_score', 0):.3f}` / "
        f"`{data.get('support_score', 0):.3f}`\n"
        f"- Latency total: `{timings['total']:.1f} ms`\n"
        f"- Cost estimate: `${data.get('cost', {}).get('estimated_usd', 0):.6f}`\n"
        f"- Unsupported claims: `{', '.join(data.get('unsupported_claims') or []) or 'none'}`\n"
        f"- Rejected claims: `{', '.join(data.get('rejected_claims') or []) or 'none'}`\n"
        f"- Trace: `{retrieval['trace_id']}`"
    )


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    mode = settings.get("mode", "Chat")
    run_id = settings.get("run_id", "")
    common = {
        "knowledge_base": settings.get("knowledge_base", "demo"),
        "query": message.content,
        "policy": settings.get("policy", "auto"),
        "latency_budget_ms": settings.get("latency_budget_ms", 1000),
        "debug": True,
    }

    if mode == "Policy Compare":
        outputs = []
        for policy in ("fast", "balanced", "accurate"):
            data = await _post("/api/v1/retrieve", dict(common, policy=policy))
            top = data["hits"][0]["source"] if data["hits"] else "abstained"
            outputs.append(
                f"| {policy} | {data['timings_ms']['total']:.1f} ms | {top} | {len(data['hits'])} |"
            )
        await cl.Message(
            content=(
                "| Policy | Total latency | Top source | Hits |\n"
                "|---|---:|---|---:|\n" + "\n".join(outputs)
            )
        ).send()
        return

    if mode == "Cost/Provider Inspector" and run_id:
        providers = await _get("/api/v1/providers")
        cost = await _get(f"/api/v1/runs/{run_id}/cost")
        await cl.Message(
            content=f"```json\n{json.dumps(providers | {'cost': cost}, indent=2)}\n```"
        ).send()
        return

    if mode == "Evaluation Report Viewer" and run_id:
        report = await _get(f"/api/v1/evaluations/{run_id}/report")
        await cl.Message(content=f"Report: `{report}`").send()
        return

    if mode == "Retrieval Inspector":
        data = await _post("/api/v1/retrieve", common)
        hits = "\n".join(
            f"{hit['rank']}. `{hit['chunk_id']}` score={hit['score']:.3f} source={hit['source']}"
            for hit in data["hits"]
        )
        await cl.Message(content=f"{hits}\n\nTrace: `{data['trace_id']}`").send()
        return

    data = await _post("/api/v1/runs/answer", common)
    if mode == "Evidence Inspector":
        await cl.Message(content=_why_panel(data)).send()
    else:
        await cl.Message(content=f"{data['answer']}\n\n{_why_panel(data)}").send()
