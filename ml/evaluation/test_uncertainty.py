from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ml.evaluation.uncertainty import (
    PredictionRow,
    build_report,
    cluster_bootstrap,
    load_predictions,
    point_estimates,
)


class CompatibilityUncertaintyTests(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_paired(self):
        rows = []
        for cluster_index in range(8):
            cluster = f"owner-{cluster_index}"
            for label in (0, 1, 0, 1):
                baseline = 0.58 if label else 0.42
                learned = 0.82 if label else 0.18
                rows.append(PredictionRow(label, baseline, learned, cluster))

        first = cluster_bootstrap(
            rows,
            resamples=500,
            confidence_level=0.95,
            seed=7,
        )
        second = cluster_bootstrap(
            rows,
            resamples=500,
            confidence_level=0.95,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["method"], "paired_cluster_percentile_bootstrap")
        self.assertEqual(first["clusterCount"], 8)
        self.assertGreater(first["brierImprovement"]["lower"], 0.0)
        self.assertLess(first["learnedMinusBaselineBrier"]["upper"], 0.0)

    def test_point_estimate_uses_same_rows_for_baseline_and_learned(self):
        rows = [
            PredictionRow(1, 0.55, 0.90, "a"),
            PredictionRow(0, 0.45, 0.10, "b"),
        ]
        estimates = point_estimates(rows)
        self.assertAlmostEqual(estimates["baselineBrier"], 0.2025)
        self.assertAlmostEqual(estimates["learnedBrier"], 0.01)
        self.assertAlmostEqual(estimates["brierImprovement"], 0.1925)

    def test_prediction_loader_rejects_missing_cluster_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                "label,baseline_score,learned_score\n1,0.5,0.8\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "owner_pair_key"):
                load_predictions(path, cluster_column="owner_pair_key")

    def test_prediction_loader_joins_cluster_identity_from_split_by_outcome_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            clusters = root / "split.csv"
            predictions.write_text(
                "outcome_id,label,baseline_score,learned_score\n"
                "o1,1,0.55,0.85\n"
                "o2,0,0.45,0.15\n",
                encoding="utf-8",
            )
            clusters.write_text(
                "outcome_id,label,owner_pair_key\n"
                "o1,1,owner-a::owner-b\n"
                "o2,0,owner-c::owner-d\n",
                encoding="utf-8",
            )

            rows = load_predictions(
                predictions,
                cluster_column="owner_pair_key",
                cluster_source=clusters,
            )
            self.assertEqual([row.cluster for row in rows], ["owner-a::owner-b", "owner-c::owner-d"])

    def test_prediction_loader_fails_closed_when_split_label_disagrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            clusters = root / "split.csv"
            predictions.write_text(
                "outcome_id,label,baseline_score,learned_score\no1,1,0.55,0.85\n",
                encoding="utf-8",
            )
            clusters.write_text(
                "outcome_id,label,owner_pair_key\no1,0,owner-a::owner-b\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_predictions(
                    predictions,
                    cluster_column="owner_pair_key",
                    cluster_source=clusters,
                )

    def test_prediction_loader_fails_closed_when_prediction_is_absent_from_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            clusters = root / "split.csv"
            predictions.write_text(
                "outcome_id,label,baseline_score,learned_score\nmissing,1,0.55,0.85\n",
                encoding="utf-8",
            )
            clusters.write_text(
                "outcome_id,label,pair_key\no1,1,pet-a::pet-b\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                load_predictions(
                    predictions,
                    cluster_column="pair_key",
                    cluster_source=clusters,
                )

    def test_report_binds_prediction_and_split_hashes_to_cluster_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_prediction(name: str) -> Path:
                path = root / f"{name}-predictions.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "outcome_id",
                            "label",
                            "baseline_score",
                            "learned_score",
                        ],
                    )
                    writer.writeheader()
                    for cluster_index in range(4):
                        for row_index, label in enumerate((0, 1)):
                            writer.writerow(
                                {
                                    "outcome_id": f"{name}-{cluster_index}-{row_index}",
                                    "label": label,
                                    "baseline_score": 0.4 if label == 0 else 0.6,
                                    "learned_score": 0.15 if label == 0 else 0.85,
                                }
                            )
                return path

            def write_split(name: str) -> Path:
                path = root / f"{name}-split.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["outcome_id", "label", "pair_key", "owner_pair_key"],
                    )
                    writer.writeheader()
                    for cluster_index in range(4):
                        for row_index, label in enumerate((0, 1)):
                            writer.writerow(
                                {
                                    "outcome_id": f"{name}-{cluster_index}-{row_index}",
                                    "label": label,
                                    "pair_key": f"pet-pair-{cluster_index}",
                                    "owner_pair_key": f"owner-pair-{cluster_index}",
                                }
                            )
                return path

            future = write_prediction("future")
            future_split = write_split("future")
            cold_pair = write_prediction("cold-pair")
            cold_pair_split = write_split("cold-pair")
            cold_owner = write_prediction("cold-owner")
            cold_owner_split = write_split("cold-owner")
            report = build_report(
                future_path=future,
                future_cluster_source=future_split,
                cold_pair_path=cold_pair,
                cold_pair_cluster_source=cold_pair_split,
                cold_owner_path=cold_owner,
                cold_owner_cluster_source=cold_owner_split,
                resamples=300,
                seed=99,
            )

            self.assertEqual(report["schemaVersion"], "woof-compatibility-uncertainty-v1")
            self.assertTrue(report["policy"]["clusterIdentityFromEvaluationSplits"])
            self.assertTrue(report["policy"]["splitLabelAgreementRequired"])
            self.assertEqual(
                report["futureTest"]["source"]["clusterColumn"],
                "owner_pair_key",
            )
            self.assertEqual(report["coldPair"]["source"]["clusterColumn"], "pair_key")
            self.assertEqual(
                report["coldOwner"]["source"]["clusterColumn"],
                "owner_pair_key",
            )
            self.assertEqual(len(report["futureTest"]["source"]["predictionSha256"]), 64)
            self.assertEqual(len(report["futureTest"]["source"]["clusterSourceSha256"]), 64)
            self.assertEqual(report["futureTest"]["source"]["clusterJoinKey"], "outcome_id")
            self.assertTrue(report["futureTest"]["source"]["clusterLabelVerified"])
            self.assertFalse(report["safety"]["available"])


if __name__ == "__main__":
    unittest.main()
