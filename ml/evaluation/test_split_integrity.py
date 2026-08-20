from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.evaluation.build_compatibility_dataset import add_prior_only_features, build
from ml.evaluation.generate_seeded_outcomes import generate


class CompatibilityEvaluationIntegrityTest(unittest.TestCase):
    def test_prior_features_never_include_current_outcome(self):
        frame = pd.DataFrame(
            [
                {
                    "outcome_id": "1",
                    "occurred_at": "2026-01-01T00:00:00Z",
                    "owner_a_id": "owner-a",
                    "owner_b_id": "owner-b",
                    "pet_a_id": "pet-a",
                    "pet_b_id": "pet-b",
                    "rating": 5,
                    "occurred": 1,
                    "label": 1,
                },
                {
                    "outcome_id": "2",
                    "occurred_at": "2026-01-02T00:00:00Z",
                    "owner_a_id": "owner-b",
                    "owner_b_id": "owner-a",
                    "pet_a_id": "pet-b",
                    "pet_b_id": "pet-a",
                    "rating": 1,
                    "occurred": 1,
                    "label": 0,
                },
            ]
        )
        frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], utc=True)
        result = add_prior_only_features(frame)
        self.assertEqual(result.loc[0, "prior_outcome_count"], 0)
        self.assertEqual(result.loc[0, "prior_mean_rating"], 0)
        self.assertEqual(result.loc[1, "prior_outcome_count"], 1)
        self.assertEqual(result.loc[1, "prior_mean_rating"], 5)

    def test_seeded_pipeline_emits_strict_future_test_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seeded.csv"
            output = root / "splits"
            generate(source, seed=42, owners=40, interactions=160)
            manifest = build(source, output, 0.7, 0.15)

            train = pd.read_csv(output / "train.csv")
            validation = pd.read_csv(output / "validation.csv")
            test = pd.read_csv(output / "test.csv")
            self.assertLess(pd.to_datetime(train["occurred_at"], utc=True).max(), pd.to_datetime(validation["occurred_at"], utc=True).min())
            self.assertLess(pd.to_datetime(validation["occurred_at"], utc=True).max(), pd.to_datetime(test["occurred_at"], utc=True).min())
            self.assertGreater(manifest["splits"]["test"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()
