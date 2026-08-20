"""Evidence gate for promoting a learned Woof compatibility scorer.

The gate consumes calibration reports produced by ``evaluate_calibration.py``
and optional shadow-service telemetry. It emits a machine-readable release
receipt and exits non-zero when an authoritative promotion is not supported.

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


@dataclass(frozen=True)
class Thresholds:
    min_test_rows: int = 500
    min_cold_rows: int = 75
    min_brier_improvement: float = 0.005
    max_ece: float = 0.08
    max_cold_brier_regression: float = 0.01
    max_auc_regression: float = 0.01
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


def model_metrics(report: dict[str, Any], column: str) -> dict[str, Any]:
    models = report.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(column), dict):
        raise ValueError(f"calibration report is missing model column {column!r}")
    return models[column]


def telemetry_values(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    if telemetry is None:
        return {
            "attempts": None,
            "fallbackRate": None,
            "p95LatencyMs": None,
        }

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
    if baseline_brier is not None and learned_brier is not None:
        brier_delta = learned_brier - baseline_brier
        if brier_delta > thresholds.max_cold_brier_regression:
            failures.append(f"{name}_brier_regression")
    else:
        failures.append(f"{name}_brier_missing")

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


def evaluate_gate(
    report: dict[str, Any],
    baseline_column: str,
    learned_column: str,
    thresholds: Thresholds,
    cold_pair_report: dict[str, Any] | None = None,
    cold_owner_report: dict[str, Any] | None = None,
    safety_report: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
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

    cold_pair, cold_pair_failures = compare_slice(
        "cold_pair",
        cold_pair_report,
        baseline_column,
        learned_column,
        thresholds,
    )
    cold_owner, cold_owner_failures = compare_slice(
        "cold_owner",
        cold_owner_report,
        baseline_column,
        learned_column,
        thresholds,
    )
    failures.extend(cold_pair_failures)
    failures.extend(cold_owner_failures)

    safety: dict[str, Any]
    if safety_report is not None:
        safety, safety_failures = compare_slice(
            "safety",
            safety_report,
            baseline_column,
            learned_column,
            thresholds,
        )
        failures.extend(safety_failures)
    else:
        safety = {"name": "safety", "available": False}
        if require_safety_slice:
            failures.append("safety_report_missing")

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
    return {
        "schemaVersion": "woof-model-promotion-receipt-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "decision": "promote" if not deduplicated_failures else "hold_shadow",
        "passed": not deduplicated_failures,
        "baselineColumn": baseline_column,
        "learnedColumn": learned_column,
        "thresholds": asdict(thresholds),
        "futureTest": {
            "rows": rows,
            "baselineBrier": baseline_brier,
            "learnedBrier": learned_brier,
            "brierImprovement": brier_improvement,
            "learnedEce10": learned_ece,
            "baselineRocAuc": baseline_auc,
            "learnedRocAuc": learned_auc,
            "learnedMinusBaselineRocAuc": auc_delta,
        },
        "coldPair": cold_pair,
        "coldOwner": cold_owner,
        "safety": safety,
        "shadowTelemetry": shadow,
        "failures": deduplicated_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cold-pair-report", type=Path)
    parser.add_argument("--cold-owner-report", type=Path)
    parser.add_argument("--safety-report", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-column", default="baseline_score")
    parser.add_argument("--learned-column", default="learned_score")
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--min-cold-rows", type=int, default=75)
    parser.add_argument("--min-brier-improvement", type=float, default=0.005)
    parser.add_argument("--max-ece", type=float, default=0.08)
    parser.add_argument("--max-cold-brier-regression", type=float, default=0.01)
    parser.add_argument("--max-auc-regression", type=float, default=0.01)
    parser.add_argument("--min-shadow-attempts", type=int, default=200)
    parser.add_argument("--max-fallback-rate", type=float, default=0.05)
    parser.add_argument("--max-p95-latency-ms", type=float, default=1200.0)
    parser.add_argument(
        "--allow-missing-telemetry",
        action="store_true",
        help="For offline research only. Authoritative release review should require telemetry.",
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
        min_brier_improvement=args.min_brier_improvement,
        max_ece=args.max_ece,
        max_cold_brier_regression=args.max_cold_brier_regression,
        max_auc_regression=args.max_auc_regression,
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
        telemetry=load_json(args.telemetry),
        require_telemetry=not args.allow_missing_telemetry,
        require_safety_slice=args.require_safety_slice,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    if not receipt["passed"] and not args.non_blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
