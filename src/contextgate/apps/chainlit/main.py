from __future__ import annotations

import asyncio
import json
from pathlib import Path

import chainlit as cl
import httpx
from chainlit.input_widget import Select, Slider, TextInput

from contextgate.config import get_settings

settings = get_settings()
API_URL = settings.ui_api_url
API_KEY = settings.api_key


async def _settings():
    knowledge_bases = ["demo"]
    try:
        models = await _get("/v1/models")
        knowledge_bases = (
            sorted(
                {
                    item["id"].split(":")[1]
                    for item in models.get("data", [])
                    if item.get("id", "").startswith("kb:")
                }
            )
            or knowledge_bases
        )
    except Exception:
        pass
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
            Select(
                id="knowledge_base",
                label="Knowledge base",
                values=knowledge_bases,
                initial_index=0,
            ),
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
    try:
        readiness = await _get("/ready")
        service_status = ", ".join(
            f"{name}={status}" for name, status in readiness.get("checks", {}).items()
        )
    except Exception as exc:
        service_status = f"not ready ({exc.__class__.__name__})"
    await cl.Message(
        content=(
            f"ContextGate service status: `{service_status or 'ready'}`. Attach a document to "
            "ingest it into the selected knowledge base, or use `/kb create <slug> <name>`. "
            "The **Why this answer?** panel shows route, evidence, provider, citations, rejected "
            "claims and estimated cost."
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


async def _post_file(path: str, file_path: Path, filename: str) -> dict:
    async with httpx.AsyncClient(timeout=180) as client:
        with file_path.open("rb") as handle:
            response = await client.post(
                f"{API_URL}{path}",
                files={"file": (filename, handle, "application/octet-stream")},
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

    if message.content.startswith("/kb create "):
        parts = message.content.split(maxsplit=3)
        if len(parts) < 4:
            await cl.Message(content="Usage: `/kb create <slug> <display name>`").send()
            return
        created = await _post(
            "/api/v1/knowledge-bases",
            {"slug": parts[2], "name": parts[3], "description": "Created from Chainlit"},
        )
        await cl.Message(
            content=f"Created knowledge base `{created['slug']}`. Reopen settings to select it."
        ).send()
        return

    attachments = [
        element
        for element in message.elements
        if getattr(element, "path", None) and getattr(element, "name", None)
    ]
    if attachments:
        knowledge_base = common["knowledge_base"]
        for attachment in attachments:
            job = await _post_file(
                f"/api/v1/knowledge-bases/{knowledge_base}/documents",
                Path(attachment.path),
                attachment.name,
            )
            progress = cl.Message(content=f"Ingestion job `{job['id']}` queued.")
            await progress.send()
            for _ in range(180):
                job = await _get(f"/api/v1/jobs/{job['id']}")
                progress.content = (
                    f"Ingestion job `{job['id']}`: `{job['status']}` "
                    f"({float(job['progress']) * 100:.0f}%)."
                )
                if job["status"] in {"failed", "succeeded_with_errors"}:
                    details = job.get("error_json") or (job.get("result") or {}).get("failures")
                    if details:
                        rendered = json.dumps(details, ensure_ascii=False, indent=2)[:4000]
                        progress.content += f"\n\nFailure details:\n```json\n{rendered}\n```"
                await progress.update()
                if job["status"] in {
                    "succeeded",
                    "succeeded_with_errors",
                    "failed",
                    "cancelled",
                }:
                    break
                await asyncio.sleep(1)
        return

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
        await cl.Message(content=f"Report: {API_URL}/api/v1/evaluations/{run_id}/report").send()
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
        answer = data["answer"] or (
            f"ContextGate {data.get('status', 'abstained')}: "
            f"`{data.get('abstention_reason') or 'policy_rejected'}`"
        )
        await cl.Message(content=f"{answer}\n\n{_why_panel(data)}").send()
