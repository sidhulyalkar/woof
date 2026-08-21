"""Build leakage-resistant compatibility evaluation splits.

Input rows must be ordered events with an `occurred_at` timestamp and pair IDs.
Rolling outcome features are computed *before* incorporating the current row, so
future meetup feedback can never leak into a historical prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ID_COLUMNS = ["owner_a_id", "owner_b_id", "pet_a_id", "pet_b_id"]
REQUIRED_COLUMNS = ["outcome_id", "occurred_at", "label", "rating", *ID_COLUMNS]


def canonical_pair(a: str, b: str) -> str:
    return "::".join(sorted((str(a), str(b))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_prior_only_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["occurred_at", "outcome_id"]).reset_index(drop=True).copy()
    pair_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "rating_sum": 0.0,
            "rated_count": 0,
            "positive_count": 0,
            "completed_count": 0,
            "last_time": None,
        }
    )

    prior_rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        pair_key = canonical_pair(row["pet_a_id"], row["pet_b_id"])
        state = pair_state[pair_key]
        current_time = pd.Timestamp(row["occurred_at"])
        last_time = state["last_time"]
        prior_rows.append(
            {
                "pair_key": pair_key,
                "owner_pair_key": canonical_pair(row["owner_a_id"], row["owner_b_id"]),
                "prior_outcome_count": state["count"],
                "prior_mean_rating": (
                    state["rating_sum"] / state["rated_count"] if state["rated_count"] else 0.0
                ),
                "prior_positive_rate": (
                    state["positive_count"] / state["rated_count"] if state["rated_count"] else 0.0
                ),
                "prior_repeat_meetups": max(0, state["completed_count"] - 1),
                "days_since_prior_outcome": (
                    (current_time - last_time).total_seconds() / 86400.0 if last_time is not None else -1.0
                ),
            }
        )

        state["count"] += 1
        rating = row.get("rating")
        if pd.notna(rating):
            state["rating_sum"] += float(rating)
            state["rated_count"] += 1
            state["positive_count"] += int(float(rating) >= 4)
        if int(row.get("occurred", 1)) == 1:
            state["completed_count"] += 1
        state["last_time"] = current_time

    return pd.concat([frame, pd.DataFrame(prior_rows)], axis=1)


def validate_no_future_leakage(frame: pd.DataFrame) -> None:
    for _, group in frame.groupby("pair_key"):
        ordered = group.sort_values("occurred_at")
        expected = list(range(len(ordered)))
        observed = ordered["prior_outcome_count"].astype(int).tolist()
        if observed != expected:
            raise AssertionError(
                f"Prior outcome counts reveal future state for pair {ordered.iloc[0]['pair_key']}"
            )


def build(input_path: Path, output_dir: Path, train_fraction: float, validation_fraction: float) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], utc=True, errors="raise")
    frame = add_prior_only_features(frame)
    validate_no_future_leakage(frame)

    if len(frame) < 20:
        raise ValueError("At least 20 outcome rows are required for leakage-resistant evaluation")

    train_end_index = max(1, min(len(frame) - 2, int(len(frame) * train_fraction)))
    validation_end_index = max(
        train_end_index + 1,
        min(len(frame) - 1, int(len(frame) * (train_fraction + validation_fraction))),
    )
    train_cutoff = frame.iloc[train_end_index - 1]["occurred_at"]
    validation_cutoff = frame.iloc[validation_end_index - 1]["occurred_at"]

    train = frame.iloc[:train_end_index].copy()
    validation = frame.iloc[train_end_index:validation_end_index].copy()
    test = frame.iloc[validation_end_index:].copy()

    train_pairs = set(train["pair_key"])
    train_owners = set(train["owner_a_id"]) | set(train["owner_b_id"])
    cold_pair_test = test[~test["pair_key"].isin(train_pairs)].copy()
    cold_owner_test = test[
        ~test["owner_a_id"].isin(train_owners) & ~test["owner_b_id"].isin(train_owners)
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
        "cold_pair_test": output_dir / "cold_pair_test.csv",
        "cold_owner_test": output_dir / "cold_owner_test.csv",
    }
    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
        ("cold_pair_test", cold_pair_test),
        ("cold_owner_test", cold_owner_test),
    ]:
        split.to_csv(paths[name], index=False)

    manifest = {
        "schemaVersion": "woof-compatibility-eval-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": len(frame),
        },
        "policy": {
            "globalTemporalSplit": True,
            "currentOutcomeExcludedFromFeatures": True,
            "trainFraction": train_fraction,
            "validationFraction": validation_fraction,
            "trainCutoff": train_cutoff.isoformat(),
            "validationCutoff": validation_cutoff.isoformat(),
        },
        "splits": {
            name: {
                "rows": len(split),
                "sha256": sha256_file(paths[name]),
            }
            for name, split in [
                ("train", train),
                ("validation", validation),
                ("test", test),
                ("cold_pair_test", cold_pair_test),
                ("cold_owner_test", cold_owner_test),
            ]
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 0.9:
        raise SystemExit("train-fraction must be between 0.5 and 0.9")
    if not 0.05 <= args.validation_fraction <= 0.25:
        raise SystemExit("validation-fraction must be between 0.05 and 0.25")
    if args.train_fraction + args.validation_fraction >= 0.95:
        raise SystemExit("leave at least 5% of rows for future test data")
    print(
        json.dumps(
            build(args.input, args.output_dir, args.train_fraction, args.validation_fraction),
            indent=2,
        )
    )
