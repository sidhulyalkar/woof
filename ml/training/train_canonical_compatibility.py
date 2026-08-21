"""Train Woof's first canonical learned compatibility model.

The model intentionally operates on the same behavior/outcome feature contract as
the deterministic API baseline. It is order-invariant, CPU-friendly and trained
with a future validation/calibration split. Optional synthetic rows augment only
the training partition and retain reduced sample weights.

This script produces a shadow candidate. Authoritative serving still requires a
passing ``promotion_gate.py`` receipt tied to real temporal outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION = "compatibility-features-v1"
MODEL_FAMILY = "canonical-tabular-ensemble-v1"
TRAITS = ("energy", "sociability", "caution", "excitability", "trainability")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def observed(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").notna().astype(float)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build order-invariant features matching the v1 serving contract."""

    output: dict[str, pd.Series] = {}
    coverage_a = pd.Series(0.0, index=frame.index)
    coverage_b = pd.Series(0.0, index=frame.index)

    for trait in TRAITS:
        column_a = f"pet_a_{trait}"
        column_b = f"pet_b_{trait}"
        a = numeric(frame, column_a, 0.5).clip(0.0, 1.0)
        b = numeric(frame, column_b, 0.5).clip(0.0, 1.0)
        coverage_a += observed(frame, column_a)
        coverage_b += observed(frame, column_b)
        output[f"{trait}_mean"] = (a + b) / 2.0
        output[f"{trait}_gap"] = (a - b).abs()

    risk_a = numeric(frame, "pet_a_social_risk", 0.25).clip(0.0, 1.0)
    risk_b = numeric(frame, "pet_b_social_risk", 0.25).clip(0.0, 1.0)
    output["social_risk_mean"] = (risk_a + risk_b) / 2.0
    output["social_risk_max"] = pd.concat([risk_a, risk_b], axis=1).max(axis=1)

    age_a = numeric(frame, "pet_a_age_years", 5.0).clip(0.0, 25.0)
    age_b = numeric(frame, "pet_b_age_years", 5.0).clip(0.0, 25.0)
    output["age_mean_scaled"] = ((age_a + age_b) / 2.0 / 15.0).clip(0.0, 1.5)
    output["age_gap_scaled"] = ((age_a - age_b).abs() / 10.0).clip(0.0, 1.5)

    output["coverage_mean"] = ((coverage_a + coverage_b) / (2.0 * len(TRAITS))).clip(0.0, 1.0)
    output["coverage_min"] = pd.concat(
        [coverage_a / len(TRAITS), coverage_b / len(TRAITS)], axis=1
    ).min(axis=1)

    prior_count = numeric(frame, "prior_outcome_count", 0.0).clip(lower=0.0)
    output["prior_outcome_log_count"] = np.log1p(prior_count)
    output["prior_mean_rating_scaled"] = (
        numeric(frame, "prior_mean_rating", 0.0).clip(0.0, 5.0) / 5.0
    )
    output["prior_positive_rate"] = numeric(
        frame, "prior_positive_rate", 0.0
    ).clip(0.0, 1.0)
    output["prior_repeat_log_count"] = np.log1p(
        numeric(frame, "prior_repeat_meetups", 0.0).clip(lower=0.0)
    )
    days = numeric(frame, "days_since_prior_outcome", -1.0)
    output["has_prior_outcome"] = (days >= 0).astype(float)
    output["days_since_prior_scaled"] = days.clip(lower=0.0, upper=365.0) / 365.0

    features = pd.DataFrame(output, index=frame.index)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("canonical feature matrix contains non-finite values")
    return features.astype(float)


def behavior_baseline_probability(frame: pd.DataFrame) -> np.ndarray:
    """Transparent Python research baseline approximating API baseline behavior."""

    features = build_features(frame)
    gap = (
        features["energy_gap"] * 0.24
        + features["sociability_gap"] * 0.30
        + features["caution_gap"] * 0.18
        + features["excitability_gap"] * 0.18
        + features["trainability_gap"] * 0.10
    )
    behavior = 1.0 - gap
    behavior -= features["social_risk_max"] * 0.16
    age = 1.0 - features["age_gap_scaled"].clip(0.0, 1.0) * 0.54
    has_prior = features["has_prior_outcome"]
    prior = features["prior_positive_rate"] * 0.75 + 0.25 * 0.65

    raw = behavior * 0.68 + age * 0.14 + 0.18 * (
        has_prior * prior + (1.0 - has_prior) * 0.65
    )
    confidence = features["coverage_min"]
    # Sparse profiles shrink toward a conservative prior rather than extreme scores.
    raw = raw * (0.60 + 0.40 * confidence) + 0.62 * (0.40 * (1.0 - confidence))
    return np.clip(raw.to_numpy(dtype=float), 0.02, 0.98)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (scores >= left) & (scores <= right if index == bins - 1 else scores < right)
        count = int(mask.sum())
        if count:
            result += (count / len(labels)) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return float(result)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=int)
    result: dict[str, Any] = {
        "rows": int(len(labels)),
        "positiveRate": float(labels.mean()) if len(labels) else None,
    }
    if not len(labels):
        return result
    result.update(
        {
            "brier": float(brier_score_loss(labels, scores)),
            "logLoss": float(log_loss(labels, scores, labels=[0, 1])),
            "ece10": expected_calibration_error(labels, scores),
        }
    )
    if len(np.unique(labels)) > 1:
        result["rocAuc"] = float(roc_auc_score(labels, scores))
        result["prAuc"] = float(average_precision_score(labels, scores))
    else:
        result["rocAuc"] = None
        result["prAuc"] = None
    return result


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" not in frame:
        raise ValueError(f"{path} must contain label")
    return frame


def split_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 40:
        raise ValueError("validation split needs at least 40 rows for tuning + calibration")
    ordered = frame.copy()
    if "occurred_at" in ordered:
        ordered = ordered.sort_values("occurred_at").reset_index(drop=True)
    midpoint = len(ordered) // 2
    return ordered.iloc[:midpoint].copy(), ordered.iloc[midpoint:].copy()


def append_synthetic_training(
    train: pd.DataFrame,
    synthetic_path: Path | None,
    max_synthetic_fraction: float,
    default_weight: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    real = train.copy()
    real["__source"] = "real_or_primary"
    real["__weight"] = 1.0
    if synthetic_path is None:
        return real, np.ones(len(real), dtype=float), {"rows": 0, "path": None}

    synthetic = load_split(synthetic_path)
    if "data_source" in synthetic and not (synthetic["data_source"].astype(str) == "synthetic").all():
        raise ValueError("synthetic training file contains non-synthetic provenance")

    max_rows = int(len(real) * max_synthetic_fraction / max(1e-9, 1.0 - max_synthetic_fraction))
    if max_rows <= 0:
        return real, np.ones(len(real), dtype=float), {"rows": 0, "path": str(synthetic_path)}
    if len(synthetic) > max_rows:
        synthetic = synthetic.sample(n=max_rows, random_state=seed).sort_index()

    synthetic = synthetic.copy()
    synthetic["__source"] = "synthetic"
    if "sample_weight" in synthetic:
        synthetic["__weight"] = pd.to_numeric(synthetic["sample_weight"], errors="coerce").fillna(default_weight)
    else:
        synthetic["__weight"] = default_weight
    synthetic["__weight"] = synthetic["__weight"].clip(0.01, default_weight)

    combined = pd.concat([real, synthetic], ignore_index=True, sort=False)
    weights = combined["__weight"].to_numpy(dtype=float)
    return (
        combined,
        weights,
        {
            "rows": int(len(synthetic)),
            "path": str(synthetic_path),
            "sha256": sha256_file(synthetic_path),
            "effectiveWeightSum": float(weights[len(real) :].sum()),
        },
    )


def predict_raw(
    scaler: StandardScaler,
    logistic: LogisticRegression,
    booster: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    blend_weight_booster: float,
) -> np.ndarray:
    matrix = features.to_numpy(dtype=float)
    logistic_scores = logistic.predict_proba(scaler.transform(matrix))[:, 1]
    booster_scores = booster.predict_proba(matrix)[:, 1]
    return (
        blend_weight_booster * booster_scores
        + (1.0 - blend_weight_booster) * logistic_scores
    )


def choose_blend(
    labels: np.ndarray,
    logistic_scores: np.ndarray,
    booster_scores: np.ndarray,
) -> float:
    candidates = np.linspace(0.0, 1.0, 21)
    losses = []
    for weight in candidates:
        blended = weight * booster_scores + (1.0 - weight) * logistic_scores
        losses.append(brier_score_loss(labels, blended))
    return float(candidates[int(np.argmin(losses))])


def evaluate_and_write(
    name: str,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    logistic: LogisticRegression,
    booster: HistGradientBoostingClassifier,
    calibrator: IsotonicRegression,
    blend_weight: float,
    output_dir: Path,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "name": name,
            "rows": 0,
            "baseline": {"rows": 0},
            "learned": {"rows": 0},
        }
    features = build_features(frame)
    raw = predict_raw(scaler, logistic, booster, features, blend_weight)
    learned = np.clip(calibrator.predict(raw), 1e-4, 1 - 1e-4)
    baseline = behavior_baseline_probability(frame)
    labels = frame["label"].astype(int).to_numpy()

    predictions = pd.DataFrame(
        {
            "label": labels,
            "baseline_score": baseline,
            "learned_score": learned,
        }
    )
    if "outcome_id" in frame:
        predictions.insert(0, "outcome_id", frame["outcome_id"].astype(str).to_numpy())
    prediction_path = output_dir / f"{name}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    return {
        "name": name,
        "rows": int(len(frame)),
        "predictionPath": str(prediction_path),
        "predictionSha256": sha256_file(prediction_path),
        "baseline": metrics(labels, baseline),
        "learned": metrics(labels, learned),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    split_dir: Path = args.split_dir
    paths = {
        "train": split_dir / "train.csv",
        "validation": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
        "cold_pair_test": split_dir / "cold_pair_test.csv",
        "cold_owner_test": split_dir / "cold_owner_test.csv",
    }
    for required in ("train", "validation", "test"):
        if not paths[required].exists():
            raise ValueError(f"missing required split: {paths[required]}")

    train_frame = load_split(paths["train"])
    validation_frame = load_split(paths["validation"])
    test_frame = load_split(paths["test"])
    cold_pair = load_split(paths["cold_pair_test"]) if paths["cold_pair_test"].exists() else pd.DataFrame()
    cold_owner = load_split(paths["cold_owner_test"]) if paths["cold_owner_test"].exists() else pd.DataFrame()
    tune_frame, calibration_frame = split_validation(validation_frame)

    training_frame, sample_weight, synthetic_manifest = append_synthetic_training(
        train_frame,
        args.synthetic_train,
        args.max_synthetic_fraction,
        args.synthetic_weight,
        args.seed,
    )
    train_features = build_features(training_frame)
    labels = training_frame["label"].astype(int).to_numpy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features.to_numpy(dtype=float))
    logistic = LogisticRegression(
        C=0.75,
        class_weight=None,
        max_iter=1000,
        random_state=args.seed,
    )
    logistic.fit(train_scaled, labels, sample_weight=sample_weight)

    booster = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_depth=3,
        max_iter=220,
        min_samples_leaf=24,
        l2_regularization=1.25,
        random_state=args.seed,
    )
    booster.fit(train_features.to_numpy(dtype=float), labels, sample_weight=sample_weight)

    tune_features = build_features(tune_frame)
    tune_matrix = tune_features.to_numpy(dtype=float)
    tune_labels = tune_frame["label"].astype(int).to_numpy()
    logistic_tune = logistic.predict_proba(scaler.transform(tune_matrix))[:, 1]
    booster_tune = booster.predict_proba(tune_matrix)[:, 1]
    blend_weight = choose_blend(tune_labels, logistic_tune, booster_tune)

    calibration_features = build_features(calibration_frame)
    calibration_raw = predict_raw(
        scaler,
        logistic,
        booster,
        calibration_features,
        blend_weight,
    )
    calibration_labels = calibration_frame["label"].astype(int).to_numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrator.fit(calibration_raw, calibration_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_version = f"{MODEL_FAMILY}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = args.output_dir / "compatibility_model.joblib"
    joblib.dump(
        {
            "modelVersion": model_version,
            "featureVersion": FEATURE_VERSION,
            "featureNames": list(train_features.columns),
            "scaler": scaler,
            "logistic": logistic,
            "booster": booster,
            "blendWeightBooster": blend_weight,
            "calibrator": calibrator,
            "servingPolicy": "shadow_until_promotion_receipt",
        },
        model_path,
    )

    evaluations = {
        "test": evaluate_and_write(
            "test", test_frame, scaler, logistic, booster, calibrator, blend_weight, args.output_dir
        ),
        "coldPair": evaluate_and_write(
            "cold_pair",
            cold_pair,
            scaler,
            logistic,
            booster,
            calibrator,
            blend_weight,
            args.output_dir,
        ),
        "coldOwner": evaluate_and_write(
            "cold_owner",
            cold_owner,
            scaler,
            logistic,
            booster,
            calibrator,
            blend_weight,
            args.output_dir,
        ),
    }

    manifest = {
        "schemaVersion": "woof-canonical-model-training-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "modelVersion": model_version,
        "featureVersion": FEATURE_VERSION,
        "modelFamily": MODEL_FAMILY,
        "modelPath": str(model_path),
        "modelSha256": sha256_file(model_path),
        "seed": args.seed,
        "blendWeightBooster": blend_weight,
        "trainingRows": int(len(training_frame)),
        "realPrimaryTrainingRows": int(len(train_frame)),
        "syntheticTraining": synthetic_manifest,
        "validationPolicy": {
            "validationTuneRows": int(len(tune_frame)),
            "calibrationRows": int(len(calibration_frame)),
            "promotionHoldoutContainsSyntheticRows": False,
        },
        "splitHashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.exists()
        },
        "evaluations": evaluations,
        "releaseStatus": "shadow_candidate_only",
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-train", type=Path)
    parser.add_argument("--max-synthetic-fraction", type=float, default=0.35)
    parser.add_argument("--synthetic-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if not 0.0 <= args.max_synthetic_fraction <= 0.5:
        raise SystemExit("max-synthetic-fraction must be between 0 and 0.5")
    if not 0.0 < args.synthetic_weight <= 0.5:
        raise SystemExit("synthetic-weight must be in (0, 0.5]")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
