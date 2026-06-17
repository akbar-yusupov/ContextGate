from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

COLORS = {"fast": "#38bdf8", "balanced": "#a78bfa", "accurate": "#f59e0b"}


def _pareto_svg(summary: dict[str, dict[str, float]]) -> str:
    width, height, padding = 680, 360, 54
    latencies = [values["latency_p95_ms"] for values in summary.values()]
    qualities = [values["ndcg_at_10"] for values in summary.values()]
    max_latency = max(latencies + [1])
    min_quality = min(qualities + [0])
    quality_span = max(max(qualities) - min_quality, 0.05)
    circles: list[str] = []
    for policy, values in summary.items():
        x = padding + (values["latency_p95_ms"] / max_latency) * (width - 2 * padding)
        y = (
            height
            - padding
            - ((values["ndcg_at_10"] - min_quality) / quality_span) * (height - 2 * padding)
        )
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{COLORS.get(policy, "#22c55e")}"/>'
            f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" fill="#e2e8f0">{policy}</text>'
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Quality latency Pareto chart">
      <rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>
      <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"
            stroke="#64748b"/>
      <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"
            stroke="#64748b"/>
      <text x="{width / 2}" y="{height - 12}" fill="#94a3b8" text-anchor="middle">p95 latency (ms)</text>
      <text x="16" y="{height / 2}" fill="#94a3b8" transform="rotate(-90 16 {height / 2})"
            text-anchor="middle">nDCG@10</text>
      {"".join(circles)}
    </svg>
    """


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _gateway_summary_section(payload: dict[str, Any]) -> str:
    gateway = payload.get("gateway_evaluation")
    if not gateway:
        return ""
    overall = gateway["summary"]["overall"]
    cards = [
        ("Answer rate", _percent(overall["answer_rate"])),
        ("Abstention rate", _percent(overall["abstention_rate"])),
        ("Correct abstentions", _percent(overall["correct_abstention_rate"])),
        ("False answers", _percent(overall["false_answer_rate"])),
        ("False abstentions", _percent(overall["false_abstention_rate"])),
        ("Grounded answers", _percent(overall["grounded_answer_rate"])),
        ("Citation validity", _percent(overall["citation_validity_rate"])),
        ("p95 latency", f"{overall['latency_p95_ms']:.1f} ms"),
        ("Cost / answer", f"${overall['estimated_cost_per_answer']:.6f}"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    failures = [
        case
        for case in gateway["cases"]
        if case.get("failure_type") and case["failure_type"] != "ok"
    ][:30]
    failure_rows = "".join(
        "<tr>"
        f"<td>{html.escape(case['failure_type'])}</td>"
        f"<td>{html.escape(case['id'])}</td>"
        f"<td>{html.escape(case['policy'])}</td>"
        f"<td>{html.escape(case['query'])}</td>"
        f"<td>{html.escape(str(case.get('abstention_reason') or ''))}</td>"
        f"<td>{case['evidence_score']:.3f}</td>"
        f"<td>{case['fact_coverage']:.3f}</td>"
        "</tr>"
        for case in failures
    )
    if not failure_rows:
        failure_rows = '<tr><td colspan="7">No QA gate failures detected in this run.</td></tr>'
    return f"""
  <section>
    <h2>QA Gate Summary</h2>
    <p>
      This section evaluates the production decision ContextGate makes for each query:
      answer with grounded citations, or abstain with a machine-readable reason.
      Retrieval metrics below explain the search behavior that fed the gate.
    </p>
    <div class="cards">{card_html}</div>
  </section>
  <section>
    <h2>QA Gate Failures</h2>
    <table>
      <thead><tr><th>Failure</th><th>ID</th><th>Policy</th><th>Query</th>
      <th>Reason</th><th>Evidence</th><th>Fact coverage</th></tr></thead>
      <tbody>{failure_rows}</tbody>
    </table>
  </section>
    """


def write_html_report(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(policy)}</td>"
        f"<td>{values['recall_at_10']:.3f}</td>"
        f"<td>{values['mrr']:.3f}</td>"
        f"<td>{values['ndcg_at_10']:.3f}</td>"
        f"<td>{values['latency_p50_ms']:.1f}</td>"
        f"<td>{values['latency_p95_ms']:.1f}</td>"
        "</tr>"
        for policy, values in summary.items()
    )
    failures = sorted(
        payload["queries"],
        key=lambda row: max(row["policies"][policy]["ndcg_at_10"] for policy in summary),
    )[:20]
    failure_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['id'])}</td>"
        f"<td>{html.escape(row['query'])}</td>"
        f"<td>{html.escape(row['language'])}</td>"
        f"<td>{max(row['policies'][p]['ndcg_at_10'] for p in summary):.3f}</td>"
        "</tr>"
        for row in failures
    )
    raw_json = html.escape(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ContextGate benchmark {html.escape(payload["run_id"])}</title>
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; background:#020617; color:#e2e8f0;
            max-width:1080px; margin:0 auto; padding:32px; }}
    h1,h2 {{ color:#f8fafc; }} .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
    .card {{ background:#0f172a; border:1px solid #334155; border-radius:12px; padding:14px; }}
    .card span {{ display:block; color:#94a3b8; font-size:13px; }}
    .card strong {{ display:block; color:#f8fafc; font-size:22px; margin-top:6px; }}
    table {{ width:100%; border-collapse:collapse; background:#0f172a; }}
    th,td {{ text-align:left; padding:10px; border-bottom:1px solid #334155; }}
    th {{ color:#93c5fd; }} section {{ margin-bottom:32px; }}
    code,pre {{ background:#0f172a; padding:16px; border-radius:10px; overflow:auto; }}
    @media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <h1>ContextGate benchmark</h1>
  <p>Run <code>{html.escape(payload["run_id"])}</code>, {payload["metadata"]["query_count"]} queries.</p>
  {_gateway_summary_section(payload)}
  <h2>Retrieval Policy Metrics</h2>
  <section class="grid">
    <div>{_pareto_svg(summary)}</div>
    <table>
      <thead><tr><th>Policy</th><th>Recall@10</th><th>MRR</th><th>nDCG@10</th>
      <th>p50 ms</th><th>p95 ms</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Lowest-scoring queries</h2>
    <table><thead><tr><th>ID</th><th>Query</th><th>Language</th><th>Best nDCG</th></tr></thead>
    <tbody>{failure_rows}</tbody></table>
  </section>
  <section><h2>Run metadata</h2><pre>{raw_json}</pre></section>
</body>
</html>"""
    path.write_text(document, encoding="utf-8")
