"""Evidence gate for promoting a learned Woof compatibility scorer.

The gate consumes aggregate calibration reports, cluster-aware uncertainty
evidence produced by ``uncertainty.py``, and optional shadow-service telemetry.
It emits a machine-readable receipt and exits non-zero when authoritative
promotion is not supported.

Default thresholds are conservative beta policy defaults, not universal
scientific constants. Override them explicitly and preserve the resulting
receipt when policy changes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMOTION_SCHEMA = "woof-model-promotion-receipt-v2"
UNCERTAINTY_SCHEMA = "woof-compatibility-uncertainty-v1"
EXPECTED_CLUSTER_COLUMNS = {
    "future_test": "owner_pair_key",
    "cold_pair": "pair_key",
    "cold_owner": "owner_pair_key",
    "safety": "owner_pair_key",
}


@dataclass(frozen=True)
class Thresholds:
    min_test_rows: int = 500
    min_cold_rows: int = 75
    min_test_clusters: int = 25
    min_cold_clusters: int = 10
    min_brier_improvement: float = 0.005
    max_ece: float = 0.08
    max_cold_brier_regression: float = 0.01
    max_auc_regression: float = 0.01
    min_bootstrap_resamples: int = 2000
    min_confidence_level: float = 0.95
    min_shadow_attempts: int = 200
    max_fallback_rate: float = 0.05
    max_p95_latency_ms: float = 1200.0


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def model_metrics(report: dict[str, Any], column: str) -> dict[str, Any]:
    models = report.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(column), dict):
        raise ValueError(f"calibration report is missing model column {column!r}")
    return models[column]


def telemetry_values(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    if telemetry is None:
        return {"attempts": None, "fallbackRate": None, "p95LatencyMs": None}

    attempts_block = telemetry.get("attempts", {})
    latency_block = telemetry.get("latencyMs", {})
    attempts = finite_number(
        attempts_block.get("total") if isinstance(attempts_block, dict) else None
    )
    fallbacks = finite_number(
        attempts_block.get("fallbacks") if isinstance(attempts_block, dict) else None
    )
    fallback_rate = None
    if attempts is not None and attempts > 0 and fallbacks is not None:
        fallback_rate = fallbacks / attempts
    if fallback_rate is None:
        fallback_rate = finite_number(telemetry.get("fallbackRate"))

    p95 = finite_number(
        latency_block.get("p95") if isinstance(latency_block, dict) else None
    )
    if p95 is None:
        p95 = finite_number(telemetry.get("p95LatencyMs"))

    return {
        "attempts": int(attempts) if attempts is not None else None,
        "fallbackRate": fallback_rate,
        "p95LatencyMs": p95,
    }


def compare_slice(
    name: str,
    report: dict[str, Any] | None,
    baseline_column: str,
    learned_column: str,
    thresholds: Thresholds,
) -> tuple[dict[str, Any], list[str]]:
    if report is None:
        return {"name": name, "available": False}, [f"{name}_report_missing"]

    baseline = model_metrics(report, baseline_column)
    learned = model_metrics(report, learned_column)
    rows = int(min(baseline.get("rows", 0), learned.get("rows", 0)))
    baseline_brier = finite_number(baseline.get("brier"))
    learned_brier = finite_number(learned.get("brier"))
    baseline_auc = finite_number(baseline.get("rocAuc"))
    learned_auc = finite_number(learned.get("rocAuc"))

    failures: list[str] = []
    if rows < thresholds.min_cold_rows:
        failures.append(f"{name}_rows_below_minimum")

    brier_delta = None
    if baseline_brier is None or learned_brier is None:
        failures.append(f"{name}_brier_missing")
    else:
        brier_delta = learned_brier - baseline_brier
        if brier_delta > thresholds.max_cold_brier_regression:
            failures.append(f"{name}_brier_regression")

    auc_delta = None
    if baseline_auc is not None and learned_auc is not None:
        auc_delta = learned_auc - baseline_auc
        if auc_delta < -thresholds.max_auc_regression:
            failures.append(f"{name}_auc_regression")

    return (
        {
            "name": name,
            "available": True,
            "rows": rows,
            "baselineBrier": baseline_brier,
            "learnedBrier": learned_brier,
            "learnedMinusBaselineBrier": brier_delta,
            "baselineRocAuc": baseline_auc,
            "learnedRocAuc": learned_auc,
            "learnedMinusBaselineRocAuc": auc_delta,
            "passed": not failures,
        },
        failures,
    )


def _same_metric(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    left_number = finite_number(left)
    right_number = finite_number(right)
    if left_number is None or right_number is None:
        return False
    return abs(left_number - right_number) <= tolerance


def _interval_bound(block: Any, bound: str) -> float | None:
    if not isinstance(block, dict):
        return None
    return finite_number(block.get(bound))


def _uncertainty_policy_failures(
    uncertainty_report: dict[str, Any], thresholds: Thresholds
) -> list[str]:
    policy = uncertainty_report.get("policy")
    if not isinstance(policy, dict):
        return ["uncertainty_policy_missing"]

    failures: list[str] = []
    if policy.get("pairedRows") is not True:
        failures.append("uncertainty_paired_rows_policy_missing")
    if policy.get("relationshipClusterBootstrap") is not True:
        failures.append("uncertainty_cluster_bootstrap_policy_missing")
    if policy.get("clusterIdentityFromEvaluationSplits") is not True:
        failures.append("uncertainty_split_cluster_authority_missing")
    if policy.get("splitLabelAgreementRequired") is not True:
        failures.append("uncertainty_split_label_authority_missing")

    resamples = finite_number(policy.get("resamples"))
    if resamples is None or resamples < thresholds.min_bootstrap_resamples:
        failures.append("uncertainty_policy_resamples_below_minimum")
    confidence = finite_number(policy.get("confidenceLevel"))
    if confidence is None or confidence < thresholds.min_confidence_level:
        failures.append("uncertainty_policy_confidence_below_minimum")
    return failures


def validate_uncertainty_slice(
    *,
    name: str,
    uncertainty: dict[str, Any] | None,
    aggregate: dict[str, Any],
    thresholds: Thresholds,
    baseline_column: str,
    learned_column: str,
    future_test: bool,
) -> tuple[dict[str, Any], list[str]]:
    if uncertainty is None or uncertainty.get("available") is not True:
        return {"name": name, "available": False}, [f"{name}_uncertainty_missing"]

    failures: list[str] = []
    source = uncertainty.get("source")
    point = uncertainty.get("point")
    bootstrap = uncertainty.get("bootstrap")
    if not isinstance(source, dict):
        failures.append(f"{name}_uncertainty_source_missing")
        source = {}
    if not isinstance(point, dict):
        failures.append(f"{name}_uncertainty_point_missing")
        point = {}
    if not isinstance(bootstrap, dict):
        failures.append(f"{name}_uncertainty_bootstrap_missing")
        bootstrap = {}

    if not _sha256_hex(source.get("predictionSha256")):
        failures.append(f"{name}_prediction_hash_missing")
    if not _sha256_hex(source.get("clusterSourceSha256")):
        failures.append(f"{name}_cluster_source_hash_missing")
    if source.get("clusterJoinKey") != "outcome_id":
        failures.append(f"{name}_cluster_join_key_invalid")
    if source.get("clusterLabelVerified") is not True:
        failures.append(f"{name}_cluster_label_verification_missing")
    if source.get("clusterColumn") != EXPECTED_CLUSTER_COLUMNS[name]:
        failures.append(f"{name}_cluster_column_invalid")
    if source.get("baselineColumn") != baseline_column:
        failures.append(f"{name}_baseline_column_mismatch")
    if source.get("learnedColumn") != learned_column:
        failures.append(f"{name}_learned_column_mismatch")

    if bootstrap.get("method") != "paired_cluster_percentile_bootstrap":
        failures.append(f"{name}_bootstrap_method_invalid")

    resamples = finite_number(bootstrap.get("resamples"))
    if resamples is None or resamples < thresholds.min_bootstrap_resamples:
        failures.append(f"{name}_bootstrap_resamples_below_minimum")

    confidence = finite_number(bootstrap.get("confidenceLevel"))
    if confidence is None or confidence < thresholds.min_confidence_level:
        failures.append(f"{name}_confidence_level_below_minimum")

    cluster_count = finite_number(bootstrap.get("clusterCount"))
    minimum_clusters = (
        thresholds.min_test_clusters if future_test else thresholds.min_cold_clusters
    )
    if cluster_count is None or cluster_count < minimum_clusters:
        failures.append(f"{name}_clusters_below_minimum")

    row_count = finite_number(bootstrap.get("rowCount"))
    aggregate_rows = finite_number(aggregate.get("rows"))
    if row_count is None or aggregate_rows is None or int(row_count) != int(aggregate_rows):
        failures.append(f"{name}_uncertainty_row_count_mismatch")

    if not _same_metric(point.get("baselineBrier"), aggregate.get("baselineBrier")):
        failures.append(f"{name}_baseline_brier_evidence_mismatch")
    if not _same_metric(point.get("learnedBrier"), aggregate.get("learnedBrier")):
        failures.append(f"{name}_learned_brier_evidence_mismatch")

    if future_test:
        if not _same_metric(point.get("learnedEce10"), aggregate.get("learnedEce10")):
            failures.append("future_test_ece_evidence_mismatch")
        improvement_lower = _interval_bound(bootstrap.get("brierImprovement"), "lower")
        if improvement_lower is None:
            failures.append("future_test_brier_improvement_lower_missing")
        elif improvement_lower < thresholds.min_brier_improvement:
            failures.append("future_test_brier_improvement_lower_below_minimum")

        ece_upper = _interval_bound(bootstrap.get("learnedEce10"), "upper")
        if ece_upper is None:
            failures.append("future_test_ece_upper_missing")
        elif ece_upper > thresholds.max_ece:
            failures.append("future_test_ece_upper_above_maximum")
    else:
        regression_upper = _interval_bound(
            bootstrap.get("learnedMinusBaselineBrier"), "upper"
        )
        if regression_upper is None:
            failures.append(f"{name}_brier_regression_upper_missing")
        elif regression_upper > thresholds.max_cold_brier_regression:
            failures.append(f"{name}_brier_regression_upper_above_maximum")

    return (
        {
            "name": name,
            "available": True,
            "source": source,
            "point": point,
            "bootstrap": bootstrap,
            "passed": not failures,
        },
        failures,
    )


def evaluate_gate(
    report: dict[str, Any],
    baseline_column: str,
    learned_column: str,
    thresholds: Thresholds,
    cold_pair_report: dict[str, Any] | None = None,
    cold_owner_report: dict[str, Any] | None = None,
    safety_report: dict[str, Any] | None = None,
    uncertainty_report: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    require_uncertainty: bool = True,
    require_telemetry: bool = True,
    require_safety_slice: bool = False,
) -> dict[str, Any]:
    baseline = model_metrics(report, baseline_column)
    learned = model_metrics(report, learned_column)

    failures: list[str] = []
    rows = int(min(baseline.get("rows", 0), learned.get("rows", 0)))
    baseline_brier = finite_number(baseline.get("brier"))
    learned_brier = finite_number(learned.get("brier"))
    learned_ece = finite_number(learned.get("ece10"))
    baseline_auc = finite_number(baseline.get("rocAuc"))
    learned_auc = finite_number(learned.get("rocAuc"))

    if rows < thresholds.min_test_rows:
        failures.append("test_rows_below_minimum")

    brier_improvement = None
    if baseline_brier is None or learned_brier is None:
        failures.append("test_brier_missing")
    else:
        brier_improvement = baseline_brier - learned_brier
        if brier_improvement < thresholds.min_brier_improvement:
            failures.append("brier_improvement_below_minimum")

    if learned_ece is None:
        failures.append("learned_ece_missing")
    elif learned_ece > thresholds.max_ece:
        failures.append("learned_ece_above_maximum")

    auc_delta = None
    if baseline_auc is not None and learned_auc is not None:
        auc_delta = learned_auc - baseline_auc
        if auc_delta < -thresholds.max_auc_regression:
            failures.append("test_auc_regression")

    future_test = {
        "rows": rows,
        "baselineBrier": baseline_brier,
        "learnedBrier": learned_brier,
        "brierImprovement": brier_improvement,
        "learnedEce10": learned_ece,
        "baselineRocAuc": baseline_auc,
        "learnedRocAuc": learned_auc,
        "learnedMinusBaselineRocAuc": auc_delta,
    }

    cold_pair, cold_pair_failures = compare_slice(
        "cold_pair", cold_pair_report, baseline_column, learned_column, thresholds
    )
    cold_owner, cold_owner_failures = compare_slice(
        "cold_owner", cold_owner_report, baseline_column, learned_column, thresholds
    )
    failures.extend(cold_pair_failures)
    failures.extend(cold_owner_failures)

    if safety_report is not None:
        safety, safety_failures = compare_slice(
            "safety", safety_report, baseline_column, learned_column, thresholds
        )
        failures.extend(safety_failures)
    else:
        safety = {"name": "safety", "available": False}
        if require_safety_slice:
            failures.append("safety_report_missing")

    uncertainty_failures: list[str] = []
    uncertainty_evidence: dict[str, Any] = {
        "required": require_uncertainty,
        "available": uncertainty_report is not None,
        "schemaVersion": (
            uncertainty_report.get("schemaVersion")
            if isinstance(uncertainty_report, dict)
            else None
        ),
    }

    if uncertainty_report is None:
        if require_uncertainty:
            uncertainty_failures.append("uncertainty_report_missing")
    elif uncertainty_report.get("schemaVersion") != UNCERTAINTY_SCHEMA:
        uncertainty_failures.append("uncertainty_schema_invalid")
    else:
        uncertainty_failures.extend(
            _uncertainty_policy_failures(uncertainty_report, thresholds)
        )
        policy = uncertainty_report.get("policy")
        uncertainty_evidence["policy"] = policy

        future_uncertainty, future_uncertainty_failures = validate_uncertainty_slice(
            name="future_test",
            uncertainty=(
                uncertainty_report.get("futureTest")
                if isinstance(uncertainty_report.get("futureTest"), dict)
                else None
            ),
            aggregate=future_test,
            thresholds=thresholds,
            baseline_column=baseline_column,
            learned_column=learned_column,
            future_test=True,
        )
        cold_pair_uncertainty, cold_pair_uncertainty_failures = validate_uncertainty_slice(
            name="cold_pair",
            uncertainty=(
                uncertainty_report.get("coldPair")
                if isinstance(uncertainty_report.get("coldPair"), dict)
                else None
            ),
            aggregate=cold_pair,
            thresholds=thresholds,
            baseline_column=baseline_column,
            learned_column=learned_column,
            future_test=False,
        )
        cold_owner_uncertainty, cold_owner_uncertainty_failures = validate_uncertainty_slice(
            name="cold_owner",
            uncertainty=(
                uncertainty_report.get("coldOwner")
                if isinstance(uncertainty_report.get("coldOwner"), dict)
                else None
            ),
            aggregate=cold_owner,
            thresholds=thresholds,
            baseline_column=baseline_column,
            learned_column=learned_column,
            future_test=False,
        )
        uncertainty_failures.extend(future_uncertainty_failures)
        uncertainty_failures.extend(cold_pair_uncertainty_failures)
        uncertainty_failures.extend(cold_owner_uncertainty_failures)

        safety_uncertainty: dict[str, Any] = {"name": "safety", "available": False}
        if safety_report is not None:
            safety_uncertainty, safety_uncertainty_failures = validate_uncertainty_slice(
                name="safety",
                uncertainty=(
                    uncertainty_report.get("safety")
                    if isinstance(uncertainty_report.get("safety"), dict)
                    else None
                ),
                aggregate=safety,
                thresholds=thresholds,
                baseline_column=baseline_column,
                learned_column=learned_column,
                future_test=False,
            )
            uncertainty_failures.extend(safety_uncertainty_failures)
        elif require_safety_slice:
            uncertainty_failures.append("safety_uncertainty_missing")

        uncertainty_evidence.update(
            {
                "futureTest": future_uncertainty,
                "coldPair": cold_pair_uncertainty,
                "coldOwner": cold_owner_uncertainty,
                "safety": safety_uncertainty,
            }
        )

    uncertainty_evidence["passed"] = not uncertainty_failures
    uncertainty_evidence["failures"] = list(dict.fromkeys(uncertainty_failures))
    failures.extend(uncertainty_failures)

    shadow = telemetry_values(telemetry)
    if require_telemetry:
        if shadow["attempts"] is None:
            failures.append("shadow_attempt_count_missing")
        elif shadow["attempts"] < thresholds.min_shadow_attempts:
            failures.append("shadow_attempts_below_minimum")

        if shadow["fallbackRate"] is None:
            failures.append("fallback_rate_missing")
        elif shadow["fallbackRate"] > thresholds.max_fallback_rate:
            failures.append("fallback_rate_above_maximum")

        if shadow["p95LatencyMs"] is None:
            failures.append("p95_latency_missing")
        elif shadow["p95LatencyMs"] > thresholds.max_p95_latency_ms:
            failures.append("p95_latency_above_maximum")

    deduplicated_failures = list(dict.fromkeys(failures))
    authoritative_eligible = require_uncertainty and require_telemetry
    if deduplicated_failures:
        decision = "hold_shadow"
    elif authoritative_eligible:
        decision = "promote"
    else:
        decision = "research_only"

    return {
        "schemaVersion": PROMOTION_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "passed": not deduplicated_failures,
        "authoritativeEligible": authoritative_eligible,
        "baselineColumn": baseline_column,
        "learnedColumn": learned_column,
        "thresholds": asdict(thresholds),
        "futureTest": future_test,
        "coldPair": cold_pair,
        "coldOwner": cold_owner,
        "safety": safety,
        "uncertaintyEvidence": uncertainty_evidence,
        "shadowTelemetry": shadow,
        "failures": deduplicated_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cold-pair-report", type=Path)
    parser.add_argument("--cold-owner-report", type=Path)
    parser.add_argument("--safety-report", type=Path)
    parser.add_argument("--uncertainty-report", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-column", default="baseline_score")
    parser.add_argument("--learned-column", default="learned_score")
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--min-cold-rows", type=int, default=75)
    parser.add_argument("--min-test-clusters", type=int, default=25)
    parser.add_argument("--min-cold-clusters", type=int, default=10)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece", type=float, default=0.08)
    parser.add_argument("--max-cold-brier-regression", type=float, default=0.01)
    parser.add_argument("--max-auc-regression", type=float, default=0.01)
    parser.add_argument("--min-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--min-confidence-level", type=float, default=0.95)
    parser.add_argument("--min-shadow-attempts", type=int, default=200)
    parser.add_argument("--max-fallback-rate", type=float, default=0.05)
    parser.add_argument("--max-p95-latency-ms", type=float, default=1200.0)
    parser.add_argument(
        "--allow-missing-uncertainty",
        action="store_true",
        help="Offline research only. The resulting receipt can never authorize promotion.",
    )
    parser.add_argument(
        "--allow-missing-telemetry",
        action="store_true",
        help="Offline research only. The resulting receipt can never authorize promotion.",
    )
    parser.add_argument("--require-safety-slice", action="store_true")
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Write the receipt but return zero even when the decision is hold_shadow.",
    )
    args = parser.parse_args()

    thresholds = Thresholds(
        min_test_rows=args.min_test_rows,
        min_cold_rows=args.min_cold_rows,
        min_test_clusters=args.min_test_clusters,
        min_cold_clusters=args.min_cold_clusters,
        min_brier_improvement=args.min_brier_improvement,
        max_ece=args.max_ece,
        max_cold_brier_regression=args.max_cold_brier_regression,
        max_auc_regression=args.max_auc_regression,
        min_bootstrap_resamples=args.min_bootstrap_resamples,
        min_confidence_level=args.min_confidence_level,
        min_shadow_attempts=args.min_shadow_attempts,
        max_fallback_rate=args.max_fallback_rate,
        max_p95_latency_ms=args.max_p95_latency_ms,
    )
    receipt = evaluate_gate(
        load_json(args.report) or {},
        args.baseline_column,
        args.learned_column,
        thresholds,
        cold_pair_report=load_json(args.cold_pair_report),
        cold_owner_report=load_json(args.cold_owner_report),
        safety_report=load_json(args.safety_report),
        uncertainty_report=load_json(args.uncertainty_report),
        telemetry=load_json(args.telemetry),
        require_uncertainty=not args.allow_missing_uncertainty,
        require_telemetry=not args.allow_missing_telemetry,
        require_safety_slice=args.require_safety_slice,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    if receipt["decision"] == "hold_shadow" and not args.non_blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
