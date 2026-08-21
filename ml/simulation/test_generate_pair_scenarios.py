from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml.simulation.generate_pair_scenarios import SIMULATION_VERSION, generate


class PairScenarioSimulationTests(unittest.TestCase):
    def test_generation_is_explicitly_training_only_and_privacy_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "scenarios.csv"
            manifest_path = root / "manifest.json"
            manifest = generate(
                output,
                manifest_path,
                seed=17,
                pets=30,
                interactions=150,
                rare_safety_fraction=0.2,
                sample_weight=0.35,
            )

            self.assertEqual(manifest["simulationVersion"], SIMULATION_VERSION)
            self.assertTrue(manifest["policy"]["trainingOnly"])
            self.assertFalse(manifest["policy"]["promotionHoldoutAllowed"])
            self.assertFalse(manifest["policy"]["containsProtectedHumanAttributes"])

            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_manifest["sha256"], manifest["sha256"])

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 150)
            self.assertTrue(all(row["data_source"] == "synthetic" for row in rows))
            self.assertTrue(all(float(row["sample_weight"]) <= 0.35 for row in rows))

            forbidden = {
                "lat",
                "lng",
                "latitude",
                "longitude",
                "address",
                "owner_age",
                "owner_gender",
                "owner_race",
                "owner_income",
            }
            self.assertFalse(forbidden.intersection(rows[0].keys()))

    def test_same_seed_produces_same_scenario_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            generate(first, root / "first.json", 2026, 24, 120, 0.25, 0.3)
            generate(second, root / "second.json", 2026, 24, 120, 0.25, 0.3)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
