from contextgate.apps.cli.main import _promotion_failure_messages
from contextgate.config import Settings


def test_promotion_failures_are_explained_with_measured_values() -> None:
    settings = Settings(
        router_min_release_cases=200,
        router_min_unanswerable_cases=50,
        router_min_cases_per_language=30,
        router_max_false_answer_upper_95=0.02,
        router_min_citation_lower_95=0.98,
    )
    benchmark = {
        "metadata": {"query_count": 150},
        "gateway_summary": {
            "overall": {
                "false_answer_upper_95": 0.0787,
                "citation_validity_lower_95": 0.9771,
            }
        },
    }
    training = {
        "promotion_failures": [
            "insufficient_release_cases",
            "insufficient_unanswerable_cases",
            "insufficient_language_cases:uz",
            "false_answer_confidence_gate_failed",
            "citation_confidence_gate_failed",
            "router_latency_reduction_below_threshold",
        ],
        "metrics": {"latency_reduction": 0.12},
        "promotion_thresholds": {"latency_reduction_min": 0.15},
    }

    messages = _promotion_failure_messages(training, benchmark, settings)

    assert messages == [
        "release set has 150 queries; at least 200 are required",
        "release set has too few unanswerable queries; at least 50 are required",
        "language 'uz' has fewer than 30 queries",
        "false-answer 95% upper bound is 7.9%; maximum is 2.0%",
        "citation-correctness 95% lower bound is 97.7%; minimum is 98.0%",
        "router p95 latency reduction is 12.0%; minimum is 15.0%",
    ]
