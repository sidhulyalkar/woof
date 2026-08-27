"""Cluster-aware uncertainty evidence for Woof compatibility promotion.

This module consumes paired prediction CSVs emitted by canonical compatibility
training and the leakage-resistant split files that own relationship identity.
It bootstraps relationship clusters rather than pretending repeated outcomes
from the same pair are independent observations.

The resulting report is evidence, not authority. ``promotion_gate.py`` decides
whether its bounds satisfy the current release policy, and artifact attestation
then binds that statistical decision to exact model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = "woof-compatibility-uncertainty-v1"


@dataclass(frozen=True)
class PredictionRow:
    label: int
    baseline: float
    learned: float
    cluster: str


@dataclass(frozen=True)
class ClusterIdentity:
    cluster: str
    label: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probability(value: str, *, column: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain numeric probabilities") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
    return min(max(result, 1e-6), 1.0 - 1e-6)


def _label(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("label must contain binary integers") from exc
    if result not in (0, 1):
        raise ValueError("label must contain only 0 or 1")
    return result


def load_cluster_map(path: Path, *, cluster_column: str) -> dict[str, ClusterIdentity]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"outcome_id", "label", cluster_column}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        result: dict[str, ClusterIdentity] = {}
        for index, raw in enumerate(reader, start=2):
            outcome_id = str(raw.get("outcome_id", "")).strip()
            cluster = str(raw.get(cluster_column, "")).strip()
            label = _label(str(raw.get("label", "")))
            if not outcome_id:
                raise ValueError(f"{path}:{index} has an empty outcome_id")
            if not cluster:
                raise ValueError(f"{path}:{index} has an empty {cluster_column}")
            identity = ClusterIdentity(cluster=cluster, label=label)
            existing = result.get(outcome_id)
            if existing is not None and existing != identity:
                raise ValueError(
                    f"{path}:{index} maps outcome {outcome_id!r} to conflicting evidence"
                )
            result[outcome_id] = identity
    if not result:
        raise ValueError(f"{path} contains no cluster identities")
    return result


def load_predictions(
    path: Path,
    *,
    cluster_column: str,
    cluster_source: Path | None = None,
    baseline_column: str = "baseline_score",
    learned_column: str = "learned_score",
) -> list[PredictionRow]:
    cluster_map = (
        load_cluster_map(cluster_source, cluster_column=cluster_column)
        if cluster_source is not None
        else None
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"label", baseline_column, learned_column}
        if cluster_map is None:
            required.add(cluster_column)
        else:
            required.add("outcome_id")
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows: list[PredictionRow] = []
        seen_outcomes: set[str] = set()
        for index, raw in enumerate(reader, start=2):
            label = _label(str(raw.get("label", "")))
            if cluster_map is None:
                cluster = str(raw.get(cluster_column, "")).strip()
            else:
                outcome_id = str(raw.get("outcome_id", "")).strip()
                if not outcome_id:
                    raise ValueError(f"{path}:{index} has an empty outcome_id")
                if outcome_id in seen_outcomes:
                    raise ValueError(f"{path}:{index} duplicates outcome {outcome_id!r}")
                seen_outcomes.add(outcome_id)
                identity = cluster_map.get(outcome_id)
                if identity is None:
                    raise ValueError(
                        f"{path}:{index} outcome {outcome_id!r} is absent from {cluster_source}"
                    )
                if identity.label != label:
                    raise ValueError(
                        f"{path}:{index} label for outcome {outcome_id!r} disagrees with {cluster_source}"
                    )
                cluster = identity.cluster
            if not cluster:
                raise ValueError(f"{path}:{index} has no {cluster_column} identity")
            rows.append(
                PredictionRow(
                    label=label,
                    baseline=_probability(
                        str(raw.get(baseline_column, "")), column=baseline_column
                    ),
                    learned=_probability(
                        str(raw.get(learned_column, "")), column=learned_column
                    ),
                    cluster=cluster,
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no evaluable prediction rows")
    return rows


def brier(labels_and_scores: Iterable[tuple[int, float]]) -> float:
    values = [(score - label) ** 2 for label, score in labels_and_scores]
    if not values:
        raise ValueError("Brier score requires at least one row")
    return sum(values) / len(values)


def expected_calibration_error(rows: Sequence[PredictionRow], bins: int = 10) -> float:
    if not rows:
        raise ValueError("ECE requires at least one row")
    total = len(rows)
    result = 0.0
    for index in range(bins):
        left = index / bins
        right = (index + 1) / bins
        bucket = [
            row
            for row in rows
            if row.learned >= left
            and (row.learned <= right if index == bins - 1 else row.learned < right)
        ]
        if not bucket:
            continue
        mean_score = sum(row.learned for row in bucket) / len(bucket)
        positive_rate = sum(row.label for row in bucket) / len(bucket)
        result += (len(bucket) / total) * abs(mean_score - positive_rate)
    return result


def point_estimates(rows: Sequence[PredictionRow]) -> dict[str, float]:
    baseline_brier = brier((row.label, row.baseline) for row in rows)
    learned_brier = brier((row.label, row.learned) for row in rows)
    improvement = baseline_brier - learned_brier
    return {
        "baselineBrier": baseline_brier,
        "learnedBrier": learned_brier,
        "brierImprovement": improvement,
        "learnedMinusBaselineBrier": -improvement,
        "learnedEce10": expected_calibration_error(rows),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    rows: Sequence[PredictionRow],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    if resamples < 200:
        raise ValueError("resamples must be at least 200")
    if not 0.80 <= confidence_level < 1.0:
        raise ValueError("confidence_level must be in [0.80, 1.0)")

    grouped: dict[str, list[PredictionRow]] = {}
    for row in rows:
        grouped.setdefault(row.cluster, []).append(row)
    clusters = sorted(grouped)
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap requires at least two distinct clusters")

    rng = random.Random(seed)
    improvements: list[float] = []
    regressions: list[float] = []
    learned_eces: list[float] = []
    for _ in range(resamples):
        sampled: list[PredictionRow] = []
        for _cluster_index in range(len(clusters)):
            sampled.extend(grouped[clusters[rng.randrange(len(clusters))]])
        estimates = point_estimates(sampled)
        improvements.append(estimates["brierImprovement"])
        regressions.append(estimates["learnedMinusBaselineBrier"])
        learned_eces.append(estimates["learnedEce10"])

    alpha = 1.0 - confidence_level
    lower_probability = alpha / 2.0
    upper_probability = 1.0 - lower_probability
    return {
        "method": "paired_cluster_percentile_bootstrap",
        "seed": seed,
        "resamples": resamples,
        "confidenceLevel": confidence_level,
        "clusterCount": len(clusters),
        "rowCount": len(rows),
        "brierImprovement": {
            "lower": percentile(improvements, lower_probability),
            "upper": percentile(improvements, upper_probability),
        },
        "learnedMinusBaselineBrier": {
            "lower": percentile(regressions, lower_probability),
            "upper": percentile(regressions, upper_probability),
        },
        "learnedEce10": {
            "lower": percentile(learned_eces, lower_probability),
            "upper": percentile(learned_eces, upper_probability),
        },
    }


def evaluate_prediction_file(
    path: Path,
    *,
    name: str,
    cluster_column: str,
    cluster_source: Path | None,
    resamples: int,
    confidence_level: float,
    seed: int,
    baseline_column: str = "baseline_score",
    learned_column: str = "learned_score",
) -> dict[str, object]:
    rows = load_predictions(
        path,
        cluster_column=cluster_column,
        cluster_source=cluster_source,
        baseline_column=baseline_column,
        learned_column=learned_column,
    )
    source: dict[str, object] = {
        "predictionPath": str(path),
        "predictionSha256": sha256_file(path),
        "clusterColumn": cluster_column,
        "baselineColumn": baseline_column,
        "learnedColumn": learned_column,
    }
    if cluster_source is not None:
        source["clusterSourcePath"] = str(cluster_source)
        source["clusterSourceSha256"] = sha256_file(cluster_source)
        source["clusterJoinKey"] = "outcome_id"
        source["clusterLabelVerified"] = True
    return {
        "name": name,
        "available": True,
        "source": source,
        "point": point_estimates(rows),
        "bootstrap": cluster_bootstrap(
            rows,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
    }


def build_report(
    *,
    future_path: Path,
    cold_pair_path: Path,
    cold_owner_path: Path,
    future_cluster_source: Path | None = None,
    cold_pair_cluster_source: Path | None = None,
    cold_owner_cluster_source: Path | None = None,
    safety_path: Path | None = None,
    safety_cluster_source: Path | None = None,
    resamples: int = 4000,
    confidence_level: float = 0.95,
    seed: int = 20260827,
) -> dict[str, object]:
    required_cluster_sources = (
        future_cluster_source,
        cold_pair_cluster_source,
        cold_owner_cluster_source,
    )
    cluster_identity_from_splits = all(source is not None for source in required_cluster_sources)
    if safety_path is not None:
        cluster_identity_from_splits = (
            cluster_identity_from_splits and safety_cluster_source is not None
        )

    report: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "pairedRows": True,
            "relationshipClusterBootstrap": True,
            "clusterIdentityFromEvaluationSplits": cluster_identity_from_splits,
            "splitLabelAgreementRequired": cluster_identity_from_splits,
            "resamples": resamples,
            "confidenceLevel": confidence_level,
            "seed": seed,
        },
        "futureTest": evaluate_prediction_file(
            future_path,
            name="future_test",
            cluster_column="owner_pair_key",
            cluster_source=future_cluster_source,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
        "coldPair": evaluate_prediction_file(
            cold_pair_path,
            name="cold_pair",
            cluster_column="pair_key",
            cluster_source=cold_pair_cluster_source,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed + 1,
        ),
        "coldOwner": evaluate_prediction_file(
            cold_owner_path,
            name="cold_owner",
            cluster_column="owner_pair_key",
            cluster_source=cold_owner_cluster_source,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed + 2,
        ),
    }
    report["safety"] = (
        evaluate_prediction_file(
            safety_path,
            name="safety",
            cluster_column="owner_pair_key",
            cluster_source=safety_cluster_source,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed + 3,
        )
        if safety_path is not None
        else {"name": "safety", "available": False}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--future-predictions", type=Path, required=True)
    parser.add_argument("--future-clusters", type=Path)
    parser.add_argument("--cold-pair-predictions", type=Path, required=True)
    parser.add_argument("--cold-pair-clusters", type=Path)
    parser.add_argument("--cold-owner-predictions", type=Path, required=True)
    parser.add_argument("--cold-owner-clusters", type=Path)
    parser.add_argument("--safety-predictions", type=Path)
    parser.add_argument("--safety-clusters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    if args.safety_clusters is not None and args.safety_predictions is None:
        raise SystemExit("--safety-clusters requires --safety-predictions")

    report = build_report(
        future_path=args.future_predictions,
        future_cluster_source=args.future_clusters,
        cold_pair_path=args.cold_pair_predictions,
        cold_pair_cluster_source=args.cold_pair_clusters,
        cold_owner_path=args.cold_owner_predictions,
        cold_owner_cluster_source=args.cold_owner_clusters,
        safety_path=args.safety_predictions,
        safety_cluster_source=args.safety_clusters,
        resamples=args.resamples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
