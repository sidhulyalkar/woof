"""Evaluate compatibility score discrimination and calibration.

Prediction files must contain `label` plus one or more score columns. This tool
produces a JSON report and per-bin calibration tables suitable for dashboards and
model-promotion review.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    table = []
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (scores >= left) & (scores <= right if index == bins - 1 else scores < right)
        count = int(mask.sum())
        if count == 0:
            table.append(
                {
                    "bin": index,
                    "lower": float(left),
                    "upper": float(right),
                    "count": 0,
                    "meanScore": None,
                    "positiveRate": None,
                    "absoluteGap": None,
                }
            )
            continue
        mean_score = float(scores[mask].mean())
        positive_rate = float(labels[mask].mean())
        gap = abs(mean_score - positive_rate)
        ece += (count / total) * gap
        table.append(
            {
                "bin": index,
                "lower": float(left),
                "upper": float(right),
                "count": count,
                "meanScore": mean_score,
                "positiveRate": positive_rate,
                "absoluteGap": gap,
            }
        )
    return float(ece), table


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    scores = np.clip(scores.astype(float), 1e-6, 1 - 1e-6)
    labels = labels.astype(int)
    ece, calibration = expected_calibration_error(labels, scores)
    result: dict[str, Any] = {
        "rows": int(len(labels)),
        "positiveRate": float(labels.mean()),
        "brier": float(brier_score_loss(labels, scores)),
        "logLoss": float(log_loss(labels, scores, labels=[0, 1])),
        "ece10": ece,
        "calibration": calibration,
    }
    if len(np.unique(labels)) > 1:
        result["rocAuc"] = float(roc_auc_score(labels, scores))
        result["prAuc"] = float(average_precision_score(labels, scores))
    else:
        result["rocAuc"] = None
        result["prAuc"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=["baseline_score", "learned_score"],
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "label" not in frame:
        raise SystemExit("input must contain a label column")

    report = {
        "schemaVersion": "woof-calibration-report-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "models": {},
    }
    for column in args.score_columns:
        if column not in frame:
            continue
        subset = frame[["label", column]].dropna()
        if subset.empty:
            continue
        report["models"][column] = evaluate(
            subset["label"].to_numpy(),
            subset[column].to_numpy(),
        )

    if not report["models"]:
        raise SystemExit("none of the requested score columns contained evaluable predictions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
