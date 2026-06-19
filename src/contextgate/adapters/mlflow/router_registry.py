from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import skops.io as skops_io
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from contextgate.config import Settings, get_settings
from contextgate.domain.retrieval import FixedPolicyName, RouteDecision

FIXED_POLICIES: tuple[FixedPolicyName, ...] = ("fast", "balanced", "accurate")
FEATURE_NAMES = (
    "query_token_count",
    "dense_margin",
    "sparse_margin",
    "dense_entropy",
    "sparse_entropy",
    "retriever_overlap",
    "top1_agreement",
    "first_stage_latency_ms",
    "language_en",
    "language_ru",
    "language_uz",
)
_PREDICT_EXECUTOR = ThreadPoolExecutor(max_workers=len(FIXED_POLICIES))


def _registry_version(value: str | int) -> Any:
    """Bridge MLflow 3.13's string API annotation and integer SQL schema."""
    return int(value)


def _predict_one(
    item: tuple[HistGradientBoostingRegressor, np.ndarray],
) -> float:
    model, vector = item
    return float(np.clip(model.predict(vector)[0], 0, 1))


def feature_vector(features: dict[str, float | int | str]) -> list[float]:
    language = str(features.get("language", "unknown"))
    normalized = dict(features)
    normalized.update(
        {
            "language_en": float(language == "en"),
            "language_ru": float(language == "ru"),
            "language_uz": float(language == "uz"),
        }
    )
    return [float(normalized.get(name, 0)) for name in FEATURE_NAMES]


@dataclass
class RouterBundle:
    models: dict[str, HistGradientBoostingRegressor]
    latency_p95_ms: dict[str, float]
    feature_mean: list[float]
    feature_std: list[float]
    version: str
    abstention_thresholds: dict[str, float]
    quality_tolerance: float = 0.02
    eligible_for_promotion: bool = False
    validation_metrics: dict[str, float] | None = None
    promotion_failures: tuple[str, ...] = ()

    def choose_policy(
        self,
        predicted: dict[str, float],
        latency_budget_ms: float,
    ) -> tuple[FixedPolicyName, str]:
        candidates = [
            policy
            for policy in FIXED_POLICIES
            if self.latency_p95_ms.get(policy, float("inf")) <= latency_budget_ms
        ]
        if not candidates:
            return "fast", "budget_below_profile"
        best_quality = max(predicted.get(policy, 0) for policy in candidates)
        equivalent = [
            policy
            for policy in candidates
            if predicted.get(policy, 0) >= best_quality - self.quality_tolerance
        ]
        selected = min(
            equivalent,
            key=lambda policy: self.latency_p95_ms.get(policy, float("inf")),
        )
        return selected, "quality_equivalent_fastest_within_slo"

    def decide(
        self,
        features: dict[str, float | int | str],
        latency_budget_ms: float,
    ) -> RouteDecision:
        started = perf_counter()
        vector = np.asarray([feature_vector(features)], dtype=np.float64)
        mean = np.asarray(self.feature_mean)
        std = np.asarray(self.feature_std)
        z_score = np.abs((vector[0] - mean) / np.where(std < 1e-6, 1, std))
        out_of_distribution = bool(np.any(z_score > 6))
        policies = list(self.models)
        values = _PREDICT_EXECUTOR.map(
            _predict_one,
            [(self.models[policy], vector) for policy in policies],
        )
        predicted = dict(zip(policies, values, strict=True))
        if out_of_distribution:
            selected: FixedPolicyName = "balanced"
            reason = "ood_fallback"
        else:
            selected, reason = self.choose_policy(predicted, latency_budget_ms)
        features["router_overhead_ms"] = (perf_counter() - started) * 1000
        return RouteDecision(
            requested_policy="auto",
            selected_policy=selected,
            reason=reason,
            predicted_quality=predicted,
            latency_budget_ms=latency_budget_ms,
            router_version=self.version,
            out_of_distribution=out_of_distribution,
        )


class RouterManager:
    def __init__(
        self,
        settings: Settings | None = None,
        active_path_resolver: Callable[[str], Path | tuple[Path, str] | None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.active_path_resolver = active_path_resolver
        self._cache: dict[str, tuple[float, RouterBundle]] = {}

    def _registry_client(self) -> MlflowClient:
        # MLflow 3.13 REST serializes model versions as strings while its PostgreSQL
        # registry schema stores integers. Use the configured registry store directly.
        return MlflowClient(
            tracking_uri=self.settings.resolved_mlflow_tracking_uri,
            registry_uri=self.settings.resolved_mlflow_registry_store_uri,
        )

    def _active_path(self, knowledge_base: str) -> Path:
        return self.settings.router_dir / _safe_segment(knowledge_base) / "active.skops"

    @staticmethod
    def _load_bundle(path: Path) -> RouterBundle:
        untrusted = skops_io.get_untrusted_types(file=path)
        allowed_prefixes = (
            "numpy.",
            "sklearn.",
            "contextgate.adapters.mlflow.router_registry.RouterBundle",
        )
        unexpected = [
            type_name for type_name in untrusted if not type_name.startswith(allowed_prefixes)
        ]
        if unexpected:
            raise ValueError(f"Router artifact contains unexpected types: {unexpected}")
        bundle = skops_io.load(path, trusted=untrusted)
        if not isinstance(bundle, RouterBundle):
            raise ValueError("Router artifact does not contain a RouterBundle")
        return bundle

    @staticmethod
    def _dump_bundle(bundle: RouterBundle, path: Path) -> None:
        skops_io.dump(bundle, path)

    def load(self, knowledge_base: str) -> RouterBundle | None:
        resolved = (
            self.active_path_resolver(knowledge_base)
            if self.active_path_resolver is not None
            else self._active_path(knowledge_base)
        )
        if resolved is None:
            return None
        path, expected_checksum = resolved if isinstance(resolved, tuple) else (resolved, "")
        if not path.exists():
            return None
        if expected_checksum:
            actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError("Active router artifact checksum mismatch")
        modified = path.stat().st_mtime
        cached = self._cache.get(knowledge_base)
        if cached and cached[0] == modified:
            return cached[1]
        bundle = self._load_bundle(path)
        self._cache[knowledge_base] = (modified, bundle)
        return bundle

    def abstention_threshold(
        self,
        knowledge_base: str,
        policy: str,
        *,
        fallback: float,
    ) -> float:
        bundle = self.load(knowledge_base)
        if bundle is None:
            return fallback
        return float(bundle.abstention_thresholds.get(policy, fallback))

    def decide(
        self,
        knowledge_base: str,
        features: dict[str, float | int | str],
        latency_budget_ms: float,
    ) -> RouteDecision:
        bundle = self.load(knowledge_base)
        if bundle is None:
            return RouteDecision(
                requested_policy="auto",
                selected_policy="balanced",
                reason="no_promoted_router",
                latency_budget_ms=latency_budget_ms,
            )
        return bundle.decide(features, latency_budget_ms)

    def train(self, benchmark_run_id: str, knowledge_base: str) -> dict[str, Any]:
        benchmark_run_id = _safe_segment(benchmark_run_id)
        knowledge_base = _safe_segment(knowledge_base)
        results_path = self.settings.report_dir / benchmark_run_id / "results.json"
        if not results_path.exists():
            raise ValueError(f"Benchmark results not found: {benchmark_run_id}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        rows = payload["queries"]
        x = np.asarray([feature_vector(row["features"]) for row in rows], dtype=np.float64)
        if len(x) < 10:
            raise ValueError("At least 10 benchmark queries are required to train the router")
        groups = np.asarray(
            [
                row.get("group_id") or row.get("id") or f"row-{index}"
                for index, row in enumerate(rows)
            ]
        )
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        training_indices, validation_indices = next(splitter.split(x, groups=groups))
        if set(groups[training_indices]) & set(groups[validation_indices]):
            raise RuntimeError("Group-aware split leaked query groups into validation")
        models: dict[str, HistGradientBoostingRegressor] = {}
        latency: dict[str, float] = {}
        metrics: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        for policy in FIXED_POLICIES:
            y = np.asarray([row["policies"][policy]["ndcg_at_10"] for row in rows])
            model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.08,
                max_iter=80,
                l2_regularization=0.1,
                min_samples_leaf=5,
                random_state=42,
            )
            model.fit(x[training_indices], y[training_indices])
            models[policy] = model
            predictions = model.predict(x[validation_indices])
            metrics[f"{policy}_validation_mae"] = float(
                np.mean(np.abs(predictions - y[validation_indices]))
            )
            policy_latencies = [
                (
                    float(row["probe_latency_ms"])
                    if policy == "fast"
                    else float(row["probe_latency_ms"])
                    + float(row["policies"][policy]["latency_ms"])
                )
                for row in rows
            ]
            latency[policy] = float(np.percentile(policy_latencies, 95))
            scores = np.asarray(
                [
                    (
                        row["policies"][policy].get("top_score")
                        if row["policies"][policy].get("top_score") is not None
                        else float("-inf")
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            answerable = np.asarray(
                [bool(row.get("answerable", True)) for row in rows],
                dtype=np.bool_,
            )
            thresholds[policy] = calibrate_abstention_threshold(
                scores[training_indices],
                answerable[training_indices],
                fallback=0.0,
            )
            validation_predictions = scores[validation_indices] >= thresholds[policy]
            metrics[f"{policy}_abstention_balanced_accuracy"] = balanced_accuracy(
                answerable[validation_indices],
                validation_predictions,
            )

        provisional = RouterBundle(
            models=models,
            latency_p95_ms=latency,
            feature_mean=np.mean(x[training_indices], axis=0).tolist(),
            feature_std=np.std(x[training_indices], axis=0).tolist(),
            version=benchmark_run_id,
            abstention_thresholds=thresholds,
        )
        validation_slo_ms = latency["accurate"] * 0.85
        auto_quality: list[float] = []
        auto_latency: list[float] = []
        fixed_quality: dict[str, list[float]] = {policy: [] for policy in FIXED_POLICIES}
        accurate_latency: list[float] = []
        for index in validation_indices:
            row = rows[int(index)]
            vector = x[int(index)].reshape(1, -1)
            predicted = {
                policy: float(np.clip(model.predict(vector)[0], 0, 1))
                for policy, model in models.items()
            }
            selected, _ = provisional.choose_policy(predicted, validation_slo_ms)
            auto_quality.append(float(row["policies"][selected]["ndcg_at_10"]))
            probe_latency = float(row["probe_latency_ms"])
            auto_latency.append(
                probe_latency
                if selected == "fast"
                else probe_latency + float(row["policies"][selected]["latency_ms"])
            )
            accurate_latency.append(
                probe_latency + float(row["policies"]["accurate"]["latency_ms"])
            )
            for policy in FIXED_POLICIES:
                fixed_quality[policy].append(float(row["policies"][policy]["ndcg_at_10"]))
        best_fixed_quality = max(float(np.mean(values)) for values in fixed_quality.values())
        auto_ndcg = float(np.mean(auto_quality))
        auto_latency_p95 = float(np.percentile(auto_latency, 95))
        accurate_latency_p95 = float(np.percentile(accurate_latency, 95))
        quality_ratio = auto_ndcg / max(best_fixed_quality, 1e-9)
        latency_reduction = 1 - auto_latency_p95 / max(accurate_latency_p95, 1e-9)
        router_regret = best_fixed_quality - auto_ndcg
        slo_violation_rate = float(np.mean(np.asarray(auto_latency) > validation_slo_ms))
        promotion_failures = self._gateway_promotion_failures(payload, rows)
        if quality_ratio < 0.95:
            promotion_failures.append("router_quality_ratio_below_threshold")
        if latency_reduction < 0.15:
            promotion_failures.append("router_latency_reduction_below_threshold")
        eligible = not promotion_failures
        validation_metrics = {
            "auto_ndcg_at_10": auto_ndcg,
            "best_fixed_ndcg_at_10": best_fixed_quality,
            "quality_ratio": quality_ratio,
            "auto_latency_p95_ms": auto_latency_p95,
            "accurate_latency_p95_ms": accurate_latency_p95,
            "latency_reduction": latency_reduction,
            "router_regret": router_regret,
            "validation_slo_ms": validation_slo_ms,
            "slo_violation_rate": slo_violation_rate,
            "eligible_for_promotion": float(eligible),
        }
        metrics.update(validation_metrics)

        for model_policy, model in models.items():
            y = np.asarray([row["policies"][model_policy]["ndcg_at_10"] for row in rows])
            model.fit(x, y)
            scores = np.asarray(
                [
                    (
                        row["policies"][model_policy].get("top_score")
                        if row["policies"][model_policy].get("top_score") is not None
                        else float("-inf")
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            thresholds[model_policy] = calibrate_abstention_threshold(
                scores,
                np.asarray(
                    [bool(row.get("answerable", True)) for row in rows],
                    dtype=np.bool_,
                ),
                fallback=thresholds[model_policy],
            )
        version = benchmark_run_id
        bundle = RouterBundle(
            models=models,
            latency_p95_ms=latency,
            feature_mean=np.mean(x, axis=0).tolist(),
            feature_std=np.std(x, axis=0).tolist(),
            version=version,
            abstention_thresholds=thresholds,
            eligible_for_promotion=eligible,
            validation_metrics=validation_metrics,
            promotion_failures=tuple(promotion_failures),
        )
        candidate_dir = self.settings.router_dir / knowledge_base / benchmark_run_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = candidate_dir / "router.skops"
        self._dump_bundle(bundle, artifact_path)
        (candidate_dir / "feature_schema.json").write_text(
            json.dumps(
                {
                    "features": FEATURE_NAMES,
                    "abstention_thresholds": thresholds,
                    "split": "GroupShuffleSplit(test_size=0.2, random_state=42)",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.set_tracking_uri(self.settings.resolved_mlflow_tracking_uri)
        mlflow.set_experiment("contextgate-router")
        with mlflow.start_run(run_name=f"{knowledge_base}-{benchmark_run_id}") as run:
            mlflow.log_params(
                {"knowledge_base": knowledge_base, "benchmark_run_id": benchmark_run_id}
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact_path), artifact_path="router")
            mlflow.log_artifact(
                str(candidate_dir / "feature_schema.json"),
                artifact_path="router",
            )
            registry_manifest: dict[str, dict[str, str]] = {}
            if self.settings.environment != "test":
                client = self._registry_client()
                for model_policy, model in models.items():
                    model_name = f"contextgate-{knowledge_base}-{model_policy}-quality"
                    mlflow.sklearn.log_model(
                        model,
                        name=f"quality-{model_policy}",
                        registered_model_name=model_name,
                        input_example=x[:1],
                        await_registration_for=60,
                        serialization_format="skops",
                    )
                    versions = client.search_model_versions(
                        f"name = '{model_name}' AND run_id = '{run.info.run_id}'"
                    )
                    if not versions:
                        raise RuntimeError(f"MLflow did not register model {model_name}")
                    model_version = max(versions, key=lambda item: int(item.version))
                    model_version_number = _registry_version(model_version.version)
                    client.set_registered_model_alias(
                        model_name,
                        "candidate",
                        model_version_number,
                    )
                    client.set_model_version_tag(
                        model_name,
                        model_version_number,
                        "benchmark_run_id",
                        benchmark_run_id,
                    )
                    registry_manifest[model_policy] = {
                        "name": model_name,
                        "version": str(model_version_number),
                    }
                manifest_path = candidate_dir / "mlflow_registry.json"
                manifest_path.write_text(
                    json.dumps(registry_manifest, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(manifest_path), artifact_path="router")
            training_run_id = run.info.run_id
        return {
            "run_id": benchmark_run_id,
            "mlflow_run_id": training_run_id,
            "benchmark_run_id": benchmark_run_id,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "eligible_for_promotion": eligible,
            "promotion_failures": promotion_failures,
            "promotion_thresholds": {
                "quality_ratio_min": 0.95,
                "latency_reduction_min": 0.15,
            },
        }

    def _gateway_promotion_failures(
        self, payload: dict[str, Any], rows: list[dict[str, Any]]
    ) -> list[str]:
        failures: list[str] = []
        gateway = payload.get("gateway_evaluation")
        if not gateway:
            return ["gateway_answer_evaluation_required"]
        cases = gateway.get("cases", [])
        if len(rows) < self.settings.router_min_release_cases:
            failures.append("insufficient_release_cases")
        unanswerable_rows = sum(1 for row in rows if not row.get("answerable", True))
        if unanswerable_rows < self.settings.router_min_unanswerable_cases:
            failures.append("insufficient_unanswerable_cases")
        required_languages = {
            value.strip()
            for value in self.settings.router_required_languages.split(",")
            if value.strip()
        }
        for language in sorted(required_languages):
            if (
                sum(1 for row in rows if row.get("language") == language)
                < self.settings.router_min_cases_per_language
            ):
                failures.append(f"insufficient_language_cases:{language}")

        total_unanswerable = sum(1 for case in cases if not case.get("answerable", True))
        false_answers = sum(1 for case in cases if case.get("failure_type") == "false_answer")
        false_answer_upper = _wilson_bound(false_answers, total_unanswerable, upper=True)
        answered = [case for case in cases if case.get("answered")]
        citation_passes = sum(float(case.get("citation_validity", 0)) >= 1 for case in answered)
        supported = sum(
            case.get("unsupported_claim_count", 0) == 0 and case.get("grounded", False)
            for case in answered
        )
        citation_lower = _wilson_bound(citation_passes, len(answered), upper=False)
        claim_support_lower = _wilson_bound(supported, len(answered), upper=False)
        if false_answer_upper > self.settings.router_max_false_answer_upper_95:
            failures.append("false_answer_confidence_gate_failed")
        if citation_lower < self.settings.router_min_citation_lower_95:
            failures.append("citation_confidence_gate_failed")
        if claim_support_lower < self.settings.router_min_claim_support_lower_95:
            failures.append("claim_support_confidence_gate_failed")
        if any(
            case.get("failure_type") == "false_answer" and "adversarial" in case.get("tags", [])
            for case in cases
        ):
            failures.append("critical_adversarial_false_answer")
        return failures

    def promote(self, benchmark_run_id: str, knowledge_base: str) -> Path:
        benchmark_run_id = _safe_segment(benchmark_run_id)
        knowledge_base = _safe_segment(knowledge_base)
        candidate_dir = self.settings.router_dir / knowledge_base / benchmark_run_id
        source = candidate_dir / "router.skops"
        if not source.exists():
            raise ValueError(f"Router candidate not found: {benchmark_run_id}")
        bundle = self._load_bundle(source)
        if not bundle.eligible_for_promotion:
            raise ValueError("Router failed promotion gates; balanced remains the runtime fallback")
        manifest_path = candidate_dir / "mlflow_registry.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mlflow.set_tracking_uri(self.settings.resolved_mlflow_tracking_uri)
            client = self._registry_client()
            for model in manifest.values():
                model_version_number = _registry_version(model["version"])
                client.set_registered_model_alias(
                    model["name"],
                    "champion",
                    model_version_number,
                )
                client.set_model_version_tag(
                    model["name"],
                    model_version_number,
                    "deployment_status",
                    "champion",
                )
        target = self._active_path(knowledge_base)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
        self._cache.pop(knowledge_base, None)
        return target


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels
    negative = ~labels
    tpr = float(np.mean(predictions[positive])) if np.any(positive) else 1.0
    tnr = float(np.mean(~predictions[negative])) if np.any(negative) else 1.0
    return (tpr + tnr) / 2


def calibrate_abstention_threshold(
    scores: np.ndarray,
    answerable: np.ndarray,
    *,
    fallback: float,
) -> float:
    finite = np.isfinite(scores)
    scores = scores[finite]
    answerable = answerable[finite]
    if len(scores) < 2 or len(np.unique(answerable)) < 2:
        return fallback
    unique_scores = np.unique(scores)
    candidates = [
        float(unique_scores[0] - 1e-9),
        *[
            float((left + right) / 2)
            for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=True)
        ],
        float(unique_scores[-1] + 1e-9),
    ]
    return max(
        candidates,
        key=lambda threshold: (
            balanced_accuracy(answerable, scores >= threshold),
            -threshold,
        ),
    )


def _wilson_bound(successes: int, total: int, *, upper: bool) -> float:
    if total == 0:
        return 1.0 if upper else 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total) / denominator
    return min(1.0, center + margin) if upper else max(0.0, center - margin)


def _safe_segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or value in {".", ".."}:
        raise ValueError("Identifier contains unsafe path characters")
    return value
