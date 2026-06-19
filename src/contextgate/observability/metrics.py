from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "contextgate_http_requests_total",
    "HTTP requests handled by ContextGate.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "contextgate_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
)
RETRIEVAL_LATENCY = Histogram(
    "contextgate_retrieval_duration_seconds",
    "Retrieval latency by policy.",
    ["policy"],
)
ROUTE_DECISIONS = Counter(
    "contextgate_route_decisions_total",
    "Adaptive route decisions.",
    ["policy", "reason"],
)
GATE_DECISIONS = Counter(
    "contextgate_gate_decisions_total",
    "Final answer admission decisions.",
    ["status", "reason", "retrieval_policy", "provider"],
)
PROVIDER_LATENCY = Histogram(
    "contextgate_provider_duration_seconds",
    "Generation provider latency.",
    ["provider"],
)
PROVIDER_FAILURES = Counter(
    "contextgate_provider_failures_total",
    "Generation provider failures.",
    ["provider", "error_type"],
)
PROVIDER_CIRCUIT_OPENINGS = Counter(
    "contextgate_provider_circuit_openings_total",
    "Generation provider circuit breaker openings.",
    ["provider"],
)
ESTIMATED_COST = Histogram(
    "contextgate_request_cost_usd",
    "Recorded request cost in USD.",
    ["provider"],
)
INGESTED_CHUNKS = Counter(
    "contextgate_ingested_chunks_total",
    "Chunks successfully written to the vector store.",
)
